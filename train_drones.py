
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import itertools
import json
import os
import random
from pathlib import Path
import signal
import time

import numpy as np
import torch

from training.drone_env import DroneNavigationEnv
from controls.drone_dqn import PATH_FOLLOWING_REWARD_VERSION, REWARD_VERSION
from training.layouts import CURRICULUM_STAGES, LAYOUT_KINDS
from utils.config import drone_model_path


EPISODES = 2000
MAX_EPISODE_STEPS = None
EPSILON_START = 0.4
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 100_000
EPSILON_STAGE_EPISODES = 200
GREEDY_EPISODE_EVERY = 5
REPORT_EVERY = 25
EVALUATE_EVERY = 50
EVALUATION_EPISODES = 20
PROMOTION_SUCCESS = 0.8
NAVIGATION_SUCCESS = 0.95
NAVIGATION_EVALUATION_EPISODES = 100
VALIDATION_VERSION = 2
VALIDATION_SEED_START = 10_000_000
VALIDATION_STAGE_STRIDE = 100_000_000
FAILURE_RETRY_FRACTION = 0.3
RETRY_EXPLORATION = 0.15
REHEARSAL_EVERY = 5


class TrainingScenarioSampler:

    def __init__(self, seed, retry_fraction=FAILURE_RETRY_FRACTION, capacity=128):
        if not np.isfinite(retry_fraction) or not 0 <= retry_fraction <= 1:
            raise ValueError("failure retry fraction must be between 0 and 1")
        self.rng = random.Random(seed)
        self.retry_fraction = retry_fraction
        self.capacity = capacity
        self.pools = {}
        self.fresh_episodes = 0
        self.retry_episodes = 0

    @staticmethod
    def validate_seed(seed):
        if not 0 <= seed < VALIDATION_SEED_START:
            raise ValueError(f"Training scenario seeds must be below {VALIDATION_SEED_START}; higher seeds are reserved for validation")

    def choose(self, stage, preferred=None):
        pool = self.pools.setdefault(stage or "manual", {})
        if preferred is not None:
            self.validate_seed(preferred)
            self.fresh_episodes += 1
            return preferred, False, False
        if pool and self.rng.random() < self.retry_fraction:
            least_retries = min(item["retries"] for item in pool.values())
            seed = self.rng.choice([seed for seed, item in pool.items() if item["retries"] == least_retries])
            pool[seed]["retries"] += 1
            self.retry_episodes += 1
            return seed, True, pool[seed]["retries"] % 2 == 0
        seed = self.rng.randrange(VALIDATION_SEED_START)
        self.fresh_episodes += 1
        return seed, False, False

    def record(self, stage, seed, *, success, greedy):
        self.validate_seed(seed)
        pool = self.pools.setdefault(stage or "manual", {})
        if not success:
            if seed not in pool:
                if len(pool) >= self.capacity:
                    pool.pop(next(iter(pool)))
                pool[seed] = dict(failures=0, retries=0, greedy_successes=0)
            pool[seed]["failures"] += 1
            pool[seed]["greedy_successes"] = 0
        elif seed in pool and greedy:
            pool[seed]["greedy_successes"] += 1
            if pool[seed]["greedy_successes"] >= 2:
                del pool[seed]

    def state(self):
        return dict(rng_state=self.rng.getstate(),
                    pools={stage: {seed: dict(item) for seed, item in pool.items()} for stage, pool in self.pools.items()},
                    fresh_episodes=self.fresh_episodes, retry_episodes=self.retry_episodes)

    def restore(self, state):
        pools = state.get("pools", {})
        for pool in pools.values():
            for seed in pool:
                self.validate_seed(seed)
        self.pools = {stage: {seed: dict(item) for seed, item in pool.items()} for stage, pool in pools.items()}
        self.fresh_episodes = state.get("fresh_episodes", 0)
        self.retry_episodes = state.get("retry_episodes", 0)
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])


def epsilon_for(total_steps, stage=None, *, stage_episode=None, continuing=False):
    if stage_episode is not None:
        start = 0.1 if continuing else 0.4 if stage in (None, "basic") else 0.25
        progress = min(stage_episode / EPSILON_STAGE_EPISODES, 1.0)
        return start + progress * (EPSILON_END - start)
    progress = min(total_steps / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def next_stage(stage, success_rate, threshold=PROMOTION_SUCCESS):
    index = CURRICULUM_STAGES.index(stage)
    if success_rate >= threshold and index < len(CURRICULUM_STAGES) - 1:
        return CURRICULUM_STAGES[index + 1]
    return stage


@contextmanager
def preserve_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def evaluate(training_env, stage, *, episodes=EVALUATION_EPISODES, observer=None, should_stop=None,
             scenario_seed=None, seed_offset=0, include_regression=True, suite_label=None):
    outcomes = []
    stopped = False
    with preserve_random_state():
        env = DroneNavigationEnv(
            max_episode_steps=training_env.fixed_episode_steps,
            layout_mode=training_env.layout_mode, training=False,
            path_only=training_env.path_only,
            path_following=training_env.controller.path_following,
        )
        try:
            env.controller.policy_network.load_state_dict(training_env.controller.policy_network.state_dict())
            env.controller.epsilon = 0.0
            env.controller.max_training_q = training_env.controller.max_training_q
            env.evaluation_suite = suite_label or ("fixed" if scenario_seed is not None else "varied")
            env.evacuee_controller.q_table = training_env.evacuee_controller.q_table.copy()
            for trial in range(episodes):
                regression_trials = min(20, episodes // 5) if training_env.path_only and include_regression else 0
                offset = 0 if trial < regression_trials else seed_offset
                trial_seed = (scenario_seed if scenario_seed is not None else
                              VALIDATION_SEED_START + CURRICULUM_STAGES.index(stage) * VALIDATION_STAGE_STRIDE + offset + trial)
                random.seed(trial_seed)
                np.random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                observation, _ = env.reset(seed=trial_seed, options={"curriculum_stage": stage})
                total_reward = 0.0
                pacing = stationary = 0
                while True:
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                    observation, reward, terminated, truncated, info = env.step(env.select_action(observation))
                    total_reward += reward
                    pacing += info["oscillation"]
                    stationary += info["stationary"]
                    callback = getattr(observer, "on_evaluation_step", None)
                    stopped = callback is not None and callback(env, trial + 1, episodes, total_reward, info) is False
                    if terminated or truncated or stopped:
                        break
                if stopped:
                    break
                outcomes.append((float(info["success"]), info["evacuation_rate"], env.step_count,
                                 pacing / env.step_count, stationary / env.step_count))
        finally:
            env.close()
    count = len(outcomes)
    successful_steps = [item[2] for item in outcomes if item[0]]
    successful = [item for item in outcomes if item[0]]
    return {
        "stage": stage, "trials": count, "stopped": stopped,
        "suite": "fixed" if scenario_seed is not None else "varied",
        "success_rate": sum(item[0] for item in outcomes) / count if count else 0.0,
        "evacuation_rate": sum(item[1] for item in outcomes) / count if count else 0.0,
        "mean_steps": sum(item[2] for item in outcomes) / count if count else 0.0,
        "mean_success_steps": sum(successful_steps) / len(successful_steps) if successful_steps else None,
        "mean_pacing": sum(item[3] for item in outcomes) / count if count else 0.0,
        "mean_stationary": sum(item[4] for item in outcomes) / count if count else 0.0,
        "mean_success_pacing": sum(item[3] for item in successful) / len(successful) if successful else None,
        "mean_success_stationary": sum(item[4] for item in successful) / len(successful) if successful else None,
    }


def combine_evaluations(regression, fresh):
    trials = regression["trials"] + fresh["trials"]
    result = dict(stage=regression["stage"], suite="varied", trials=trials,
                  stopped=regression["stopped"] or fresh["stopped"],
                  qualification_rate=min(regression["success_rate"], fresh["success_rate"]),
                  regression=regression, fresh=fresh)
    for metric in ("success_rate", "evacuation_rate", "mean_steps", "mean_pacing", "mean_stationary"):
        result[metric] = sum(item.get(metric, 0) * item["trials"] for item in (regression, fresh)) / trials if trials else 0.0
    for metric in ("mean_success_steps", "mean_success_pacing", "mean_success_stationary"):
        available = [item for item in (regression, fresh) if item.get(metric) is not None]
        successes = sum(item["success_rate"] * item["trials"] for item in available)
        result[metric] = (sum(item[metric] * item["success_rate"] * item["trials"] for item in available)
                          / successes if successes else None)
    return result


def save_checkpoint(controller, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    controller.save(temporary)
    if metadata is not None:
        checkpoint = torch.load(temporary, map_location="cpu", weights_only=True)
        checkpoint["training"] = metadata
        checkpoint["optimizer_state_dict"] = controller.optimizer.state_dict() if controller.optimizer else None
        if controller.target_network is not None:
            checkpoint["target_state_dict"] = controller.target_network.state_dict()
        torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def write_status(directory, **status):
    if directory is None:
        return
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = directory / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(temporary, directory / "status.json")


def check_evacuee_following(q_table):
    from controls.action import Action
    from controls.evacuee_rl import QLearningFollowController
    from entities.drone import Drone
    from entities.evacuee import Evacuee
    from grids.map_grid import Map_Grid
    from map.cell import CellType
    from simulation import Simulation
    from training.scenario import reset_simulation

    passed = 0
    with preserve_random_state():
        random.seed(12345)
        simulation = Simulation()
        grid = Map_Grid(41, 41, CellType.EMPTY)
        grid.set(40, 40, CellType.EXIT)
        simulation.grid_world.add_floor_from_map(grid)
        controller = QLearningFollowController()
        controller.q_table = q_table.copy()
        for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            reset_simulation(simulation)
            controller.reset()
            drone = Drone(20 + 5 * dx, 20 + 5 * dy, drone_id=0)
            person = Evacuee(20, 20, assigned_drone_id=0)
            simulation.drones.append(drone)
            simulation.evacuees.append(person)
            simulation.grid_world.add_drone(drone)
            simulation.grid_world.add_evacuee(person)
            simulation.started = True
            for _ in range(32):
                simulation.step(controller.get_actions(simulation), [Action.STAY])
            passed += abs(person.grid_x - drone.grid_x) + abs(person.grid_y - drone.grid_y) <= 2
    return passed


def report_lines(episode, stage, recent, epsilon):
    count = len(recent)
    mean = lambda key: sum(item[key] for item in recent) / count
    summary = (
        f"episode {episode:5d}  stage {stage or 'manual'}  reward {mean('reward'):8.2f}  "
        f"success {mean('success'):6.1%}  evacuated {mean('evacuation_rate'):6.1%}  "
        f"steps {mean('steps'):6.1f}  timeout {mean('timeout'):6.1%}  epsilon {epsilon:.3f}"
    )
    components = {
        key: sum(item["components"][key] for item in recent) / count
        for key in recent[0]["components"]
    }
    details = "  mean reward: " + "  ".join(f"{key} {value:+.2f}" for key, value in components.items())
    details += (f" | stay {mean('stay_rate'):.1%}  stationary {mean('stationary_rate'):.1%}"
                f"  pacing (A-B-A) {mean('oscillation_rate'):.1%}")
    return summary, details


def train(seed=0, *, episodes=EPISODES, max_episode_steps=MAX_EPISODE_STEPS,
          model_path=drone_model_path, layout_mode="curriculum", observer=None,
          evaluate_every=EVALUATE_EVERY, evaluation_episodes=EVALUATION_EPISODES,
          resume_from=None, max_seconds=None, run_dir=None, save_every_seconds=300,
          stop_file=None, path_only=False, max_training_q=None, stop_after_stage=None,
          target_sync_steps=None, fixed_route_seed=None, init_weights=None,
          failure_retry_fraction=None, path_following=False):
    if (episodes is not None and episodes < 1) or evaluate_every < 1 or evaluation_episodes < 1:
        raise ValueError("Episode and evaluation counts must be at least 1")
    if max_episode_steps is not None and max_episode_steps < 1:
        raise ValueError("The episode limit must be at least 1")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("The time budget must be positive")
    if episodes is None and max_seconds is None:
        raise ValueError("Set an episode count or a time budget")
    if stop_after_stage is not None and stop_after_stage not in CURRICULUM_STAGES:
        raise ValueError("Unknown final curriculum stage")
    if path_only and Path(model_path).resolve() == Path(drone_model_path).resolve():
        raise ValueError("Choose a separate --output checkpoint for the path-only experiment")
    if target_sync_steps is not None and target_sync_steps < 1:
        raise ValueError("Target sync interval must be positive")
    if max_training_q is not None and (not np.isfinite(max_training_q) or max_training_q <= 0):
        raise ValueError("The Q-value limit must be positive and finite")
    if resume_from is not None and init_weights is not None:
        raise ValueError("Choose either resume or weights-only initialization")
    if path_following and not path_only:
        raise ValueError("Path-following rewards require --path-only")
    sampler = TrainingScenarioSampler(seed, retry_fraction=FAILURE_RETRY_FRACTION if failure_retry_fraction is None else failure_retry_fraction)
    if path_only:
        sampler.validate_seed(seed)
        if fixed_route_seed is not None:
            sampler.validate_seed(fixed_route_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started_at = time.monotonic()
    deadline = started_at + max_seconds if max_seconds is not None else None
    model_path = Path(model_path)
    directory = Path(run_dir) if run_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    latest_path = directory / "latest.pt" if directory is not None else model_path
    stop_path = Path(stop_file) if stop_file else directory / "STOP" if directory else None
    curriculum = layout_mode == "curriculum"
    stage = CURRICULUM_STAGES[0] if curriculum else None
    navigation_phase = "fixed" if path_only and curriculum else None
    fixed_seed = seed if fixed_route_seed is None else fixed_route_seed
    qualification_success = NAVIGATION_SUCCESS if path_only else PROMOTION_SUCCESS
    env = DroneNavigationEnv(max_episode_steps=max_episode_steps,
                             layout_mode="mixed" if curriculum else layout_mode,
                             path_only=path_only, path_following=path_following)
    controller = env.controller
    if max_training_q is not None:
        controller.max_training_q = max_training_q
    if target_sync_steps is not None:
        controller.target_sync_steps = target_sync_steps
    total_steps = 0
    stage_episodes = 0
    previous_episodes = 0
    completed_episodes = 0
    rehearsal_episodes = 0
    continuing_stage = resume_from is not None or init_weights is not None
    recent = deque(maxlen=100)
    best_scores = {}
    validation_round = 0
    last_evaluation = None
    last_save = time.monotonic()
    stop_reason = "episode limit"
    failure = None
    qualification_reached = False

    def emit(lines):
        if observer is None:
            print("\n".join(lines), flush=True)
        elif getattr(observer, "on_report", None):
            observer.on_report(lines)

    def requested_stop():
        if deadline is not None and time.monotonic() >= deadline:
            return "time limit"
        if stop_path is not None and stop_path.exists():
            return "stop file"
        return None

    def metadata():
        return dict(stage=stage, stage_episodes=stage_episodes, total_steps=total_steps,
                    episodes_completed=previous_episodes + completed_episodes,
                    continuing_stage=continuing_stage, best_scores=best_scores,
                    training_steps=controller.training_steps, replay_saved=False, path_only=path_only,
                    target_sync_steps=controller.target_sync_steps, navigation_phase=navigation_phase,
                    n_step=controller.n_step,
                    discount=controller.discount,
                    learning_objective=controller.learning_objective,
                    validation_version=VALIDATION_VERSION, validation_round=validation_round,
                    rehearsal_episodes=rehearsal_episodes,
                    scenario_sampler=sampler.state(), failure_retry_fraction=sampler.retry_fraction,
                    resumed_from=str(resume_from) if resume_from is not None else None,
                    fixed_route_seed=fixed_seed)

    def status(state="running"):
        write_status(directory, state=state, stage=stage, stage_episodes=stage_episodes,
                     path_only=path_only, loss=controller.last_loss,
                     navigation_phase=navigation_phase, fixed_route_seed=fixed_seed,
                     qualification_success=qualification_success,
                     successful_replay_size=len(controller.successful_replay),
                     reward_version=controller.reward_version, n_step=controller.n_step,
                     discount=controller.discount,
                     learning_objective=controller.learning_objective,
                     validation_version=VALIDATION_VERSION, validation_round=validation_round,
                     rehearsal_episodes=rehearsal_episodes,
                     failure_retry_fraction=sampler.retry_fraction,
                     failed_training_scenarios=len(sampler.pools.get(stage or "manual", {})),
                     retry_episodes=sampler.retry_episodes, fresh_episodes=sampler.fresh_episodes,
                     resumed_from=str(resume_from) if resume_from is not None else None,
                     q_abs_max=controller.last_q_abs_max, target_abs_max=controller.last_target_abs_max,
                     episodes_completed=previous_episodes + completed_episodes,
                     steps=total_steps, elapsed_seconds=time.monotonic() - started_at,
                     remaining_seconds=max(0, deadline - time.monotonic()) if deadline else None,
                     epsilon=controller.epsilon, best_scores=best_scores,
                     last_evaluation=last_evaluation, latest_checkpoint=str(latest_path),
                     best_model=str(model_path), stop_reason=stop_reason if state != "running" else None,
                     error=str(failure) if failure else None)

    def checkpoint():
        nonlocal last_save
        save_checkpoint(controller, latest_path, metadata())
        last_save = time.monotonic()
        status()

    def record_evaluation(result):
        nonlocal last_evaluation
        last_evaluation = result
        key = f"fixed_{stage}" if result.get("suite") == "fixed" else stage or "manual"
        successful_time = result.get("mean_success_steps", result["mean_steps"])
        score = ((result["qualification_rate"], result["regression"]["success_rate"], -(successful_time or 0))
                 if "qualification_rate" in result else
                 (result["success_rate"], -(successful_time or 0), result["evacuation_rate"]))
        if not result["stopped"] and score > tuple(best_scores.get(key, (-1, -1, -float("inf")))):
            best_scores[key] = list(score)
            if directory is not None:
                save_checkpoint(controller, directory / f"best_{key}.pt", metadata())
            save_checkpoint(controller, model_path, metadata())
            emit((f"Best {key}: qualifying success {score[0]:.0%}; saved {model_path}",))

    def run_evaluation():
        nonlocal validation_round
        fixed = navigation_phase == "fixed"
        count = 1 if fixed else max(NAVIGATION_EVALUATION_EPISODES, evaluation_episodes) if path_only else evaluation_episodes
        if path_only and not fixed:
            validation_round += 1
            emit((f"Evaluating {stage}: {count} fixed regression cases, then {count} fresh greedy cases...",))
            regression = dict(evaluate(env, stage, episodes=count, observer=observer,
                                      should_stop=requested_stop, seed_offset=0,
                                      include_regression=False, suite_label="regression"))
            if regression["stopped"]:
                record_evaluation(regression)
                return regression
            fresh = dict(evaluate(env, stage, episodes=count, observer=observer,
                                 should_stop=requested_stop, seed_offset=1_000_000 + validation_round * count,
                                 include_regression=False, suite_label="fresh"))
            result = combine_evaluations(regression, fresh)
            record_evaluation(result)
            emit((f"  Fixed regression {regression['success_rate']:.0%}; fresh {fresh['success_rate']:.0%}; "
                  f"both must reach {qualification_success:.0%}.",))
            return result
        label = "fixed route" if fixed else "fresh cases" if path_only else "fixed cases"
        emit((f"Evaluating {stage}: {count} greedy {label}...",))
        result = dict(evaluate(env, stage, episodes=count, observer=observer,
                              should_stop=requested_stop, scenario_seed=fixed_seed if fixed else None,
                              seed_offset=(stage_episodes // evaluate_every) * NAVIGATION_EVALUATION_EPISODES if path_only and not fixed else 0))
        result["suite"] = "fixed" if fixed else "varied"
        record_evaluation(result)
        return result

    try:
        if init_weights is not None:
            if not controller.load(init_weights):
                raise ValueError(f"Cannot load compatible drone weights from {init_weights}")
            initialized = torch.load(init_weights, map_location="cpu", weights_only=True)
            if initialized.get("reward_version") not in (REWARD_VERSION, PATH_FOLLOWING_REWARD_VERSION):
                raise ValueError("This checkpoint uses the old reward scheme; train fresh weights")
            emit((f"Initialized weights from {init_weights}; optimizer, replay and curriculum start fresh.",))
        if resume_from is not None:
            if not controller.load(resume_from):
                raise ValueError(f"Cannot load compatible drone weights from {resume_from}")
            saved = torch.load(resume_from, map_location="cpu", weights_only=True)
            if saved.get("learning_objective", "mission") != controller.learning_objective:
                raise ValueError("Checkpoint learning objective differs; match --path-following or use --init-weights to start a fresh learner")
            if saved.get("reward_version") not in (REWARD_VERSION, PATH_FOLLOWING_REWARD_VERSION):
                raise ValueError("This checkpoint uses the old reward scheme; train fresh weights")
            if saved.get("reward_version") != controller.reward_version:
                raise ValueError("This checkpoint uses a different reward scheme; use --init-weights for a fresh learner")
            info = saved.get("training", {})
            if info and info.get("path_only", False) != path_only:
                raise ValueError("The resumed checkpoint was trained for a different task")
            if curriculum and info.get("stage") in CURRICULUM_STAGES:
                stage = info["stage"]
            navigation_phase = info.get("navigation_phase", navigation_phase)
            if fixed_route_seed is None:
                fixed_seed = info.get("fixed_route_seed", fixed_seed)
            total_steps = info.get("total_steps", 0)
            stage_episodes = info.get("stage_episodes", 0)
            previous_episodes = info.get("episodes_completed", 0)
            rehearsal_episodes = info.get("rehearsal_episodes", 0)
            continuing_stage = info.get("continuing_stage", continuing_stage)
            best_scores = info.get("best_scores", {})
            if path_only and info.get("validation_version") != VALIDATION_VERSION:
                best_scores = {}
                emit(("Rechecking the resumed best model with fixed and fresh validation; previous scores used a different validation protocol.",))
            validation_round = info.get("validation_round", 0)
            sampler.restore(info.get("scenario_sampler", {}))
            if failure_retry_fraction is None:
                sampler.retry_fraction = info.get("failure_retry_fraction", sampler.retry_fraction)
            if path_only:
                sampler.validate_seed(fixed_seed)
            controller.training_steps = info.get("training_steps", 0)
            if target_sync_steps is None:
                controller.target_sync_steps = max(controller.target_sync_steps, info.get("target_sync_steps", 0))
            if saved.get("optimizer_state_dict") is not None:
                controller.optimizer.load_state_dict(saved["optimizer_state_dict"])
            if "target_state_dict" in saved:
                controller.target_network.load_state_dict(saved["target_state_dict"])
            emit((f"Loaded {resume_from}; continuing {stage or 'manual'} with a fresh replay buffer.",))
        if directory is not None:
            if not (directory / "initial.pt").exists():
                save_checkpoint(controller, directory / "initial.pt", metadata())
        if path_only:
            emit(("Task: path following only. One drone, zero evacuees; success requires the assigned exit.",))
            if path_following:
                emit(("Path following: +10 per forward cluster, +1 per closer step, -1 per farther step. One-step rewards, discount 0.",))
            if curriculum:
                emit((f"Each new stage starts with a fixed route; every {REHEARSAL_EVERY}th episode practices an earlier stage.",))
            emit((f"Varied training: {sampler.retry_fraction:.0%} failure retries when available; "
                  f"exploratory retries use epsilon >= {RETRY_EXPLORATION:g}. Validation seeds are separate.",))
            if navigation_phase == "fixed":
                emit((f"First master one fixed route (seed {fixed_seed}); then qualify on 100 fresh cases at 95% success.",))
        else:
            passed = check_evacuee_following(env.evacuee_controller.q_table)
            emit((f"Fixed evacuee policy: {passed}/8 basic guide directions passed.",))
        if curriculum and directory is not None:
            result = run_evaluation()
            checkpoint()
            if result["stopped"]:
                stop_reason = requested_stop() or "observer"
            elif navigation_phase != "fixed" and result.get("qualification_rate", result["success_rate"]) >= qualification_success:
                if stop_after_stage == stage:
                    qualification_reached = True
                    stop_reason = "target score reached"
                else:
                    promoted_stage = next_stage(stage, result.get("qualification_rate", result["success_rate"]), qualification_success)
                    if promoted_stage != stage:
                        stage, stage_episodes = promoted_stage, 0
                        continuing_stage = False
                        if path_only:
                            navigation_phase = "fixed"
        if max_seconds is not None:
            emit((f"Time budget: {max_seconds / 3600:.2f} hours; episode count is unlimited.",))
        if stop_path is not None:
            emit((f"To stop safely, create {stop_path}",))
        status()

        for episode in itertools.count():
            if qualification_reached:
                break
            if requested_stop() or (episodes is not None and episode >= episodes):
                stop_reason = requested_stop() or "episode limit"
                break
            if last_evaluation is not None and last_evaluation["stopped"]:
                break
            if (navigation_phase == "fixed" and last_evaluation is not None
                    and last_evaluation.get("suite") == "fixed"
                    and last_evaluation.get("stage", stage) == stage
                    and last_evaluation["success_rate"] == 1.0):
                navigation_phase = "varied"
                stage_episodes = 0
                recent.clear()
                emit(("Fixed route passed. Training varied routes; promotion now requires 95/100 greedy successes.",))
                last_evaluation = None
            rehearsal = (path_only and curriculum and stage != CURRICULUM_STAGES[0]
                         and (previous_episodes + completed_episodes + 1) % REHEARSAL_EVERY == 0)
            lesson_stage = (sampler.rng.choice(CURRICULUM_STAGES[:CURRICULUM_STAGES.index(stage)])
                            if rehearsal else stage)
            options = {"curriculum_stage": lesson_stage} if curriculum else None
            use_sampler = path_only and (rehearsal or navigation_phase != "fixed")
            retry = greedy_retry = False
            if use_sampler:
                episode_seed, retry, greedy_retry = sampler.choose(
                    lesson_stage, preferred=seed if episode == 0 and resume_from is None else None)
            else:
                episode_seed = fixed_seed if navigation_phase == "fixed" else seed if episode == 0 else None
            observation, _ = env.reset(seed=episode_seed, options=options)
            env.navigation_phase = "rehearsal" if rehearsal else navigation_phase
            env.training_stage = stage
            env.is_rehearsal = rehearsal
            env.training_scenario_seed = episode_seed
            env.scenario_source = "retry" if retry else "fixed" if navigation_phase == "fixed" and not rehearsal else "fresh"
            episode_reward = 0.0
            components = {}
            stays = stationary = oscillations = 0
            greedy_episode = greedy_retry if retry else (stage_episodes + 1) % GREEDY_EPISODE_EVERY == 0
            scheduled_epsilon = epsilon_for(total_steps, stage, stage_episode=stage_episodes,
                                            continuing=continuing_stage)
            controller.epsilon = 0.0 if greedy_episode else scheduled_epsilon
            if rehearsal:
                controller.epsilon = min(controller.epsilon, 0.1)
            if retry and not greedy_episode:
                controller.epsilon = max(controller.epsilon, RETRY_EXPLORATION)
            if observer is not None:
                observer.on_episode_start(env, episode + 1, episodes)
            stop_requested = False

            while True:
                action = env.select_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                total_steps += 1
                for key, value in info["reward_components"].items():
                    components[key] = components.get(key, 0.0) + value
                stays += info["stay_action"]
                stationary += info["stationary"]
                oscillations += info["oscillation"]
                if observer is not None:
                    stop_requested = observer.on_step(env, episode_reward, total_steps, info) is False
                stop_requested = stop_requested or bool(requested_stop())
                if time.monotonic() - last_save >= save_every_seconds:
                    checkpoint()
                if total_steps % 100 == 0:
                    status()
                if terminated or truncated or stop_requested:
                    break

            completed = terminated or truncated
            info.update(episode_reward_components=components, greedy_episode=greedy_episode,
                        stay_rate=stays / env.step_count, stationary_rate=stationary / env.step_count,
                        oscillation_rate=oscillations / env.step_count)
            if observer is not None:
                observer.on_episode_end(env, episode_reward, info, completed)
            if completed:
                if use_sampler:
                    sampler.record(lesson_stage, episode_seed, success=bool(info["success"]), greedy=greedy_episode)
                completed_episodes += 1
                if rehearsal:
                    rehearsal_episodes += 1
                else:
                    stage_episodes += 1
                    recent.append(dict(reward=episode_reward, success=float(info["success"]),
                                       evacuation_rate=info["evacuation_rate"], steps=env.step_count,
                                       timeout=float(truncated), components=components,
                                       stay_rate=info["stay_rate"], stationary_rate=info["stationary_rate"],
                                       oscillation_rate=info["oscillation_rate"], greedy=greedy_episode))
            status()
            if recent and (episode + 1) % REPORT_EVERY == 0:
                lines = report_lines(previous_episodes + completed_episodes, stage, recent, scheduled_epsilon)
                if path_only:
                    lines = tuple(line.replace("evacuated", "exit reached") for line in lines)
                emit(lines)
                if controller.last_loss is not None:
                    emit((f"  DQN loss {controller.last_loss:.4g} | max |Q| {controller.last_q_abs_max:.4g} | "
                          f"max |target| {controller.last_target_abs_max:.4g}",))
                if use_sampler:
                    emit((f"  Failed training maps: {len(sampler.pools.get(stage or 'manual', {}))}; "
                          f"retry episodes {sampler.retry_episodes}, fresh episodes {sampler.fresh_episodes}.",))
            if stop_requested:
                stop_reason = requested_stop() or "observer"
                break
            evaluation_interval = min(evaluate_every, 10) if navigation_phase == "fixed" else evaluate_every
            if curriculum and not rehearsal and stage_episodes % evaluation_interval == 0:
                result = run_evaluation()
                qualifying_rate = result.get("qualification_rate", result["success_rate"])
                passed_requested_stage = (not result["stopped"] and navigation_phase != "fixed"
                                          and stop_after_stage == stage and qualifying_rate >= qualification_success)
                promoted_stage = stage if result["stopped"] or navigation_phase == "fixed" else next_stage(
                    stage, qualifying_rate, qualification_success)
                outcome = "exit reached" if path_only else "evacuated"
                message = (f"Evaluation {stage}: success {result['success_rate']:.0%}, "
                           f"{outcome} {result['evacuation_rate']:.0%}, "
                           f"steps {result['mean_steps']:.1f}, {result['trials']} greedy trials.")
                if result.get("mean_success_steps") is not None:
                    message += f" Successful missions: {result['mean_success_steps']:.1f} steps."
                if result.get("mean_success_pacing") is not None:
                    message += (f" Successful-run pacing {result['mean_success_pacing']:.1%}, "
                                f"stationary {result['mean_success_stationary']:.1%}.")
                message += (f" Target score reached; stopping at {stage}." if passed_requested_stage else
                            f" Advancing to {promoted_stage}." if promoted_stage != stage else
                            " Stopped." if result["stopped"] else " Continuing this stage.")
                emit((message,))
                if result["stopped"]:
                    stop_reason = requested_stop() or "observer"
                    break
                if passed_requested_stage:
                    stop_reason = "target score reached"
                    if directory is not None:
                        checkpoint()
                    break
                if promoted_stage != stage:
                    stage = promoted_stage
                    stage_episodes = 0
                    continuing_stage = False
                    if path_only:
                        navigation_phase = "fixed"
                    recent.clear()
                if directory is not None:
                    checkpoint()
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        emit(("\nTraining interrupted; saving current drone weights.",))
    except Exception as exc:
        stop_reason = "error"
        failure = exc
    finally:
        try:
            if failure is None or total_steps > 0:
                save_checkpoint(controller, latest_path, metadata())
                emit((f"saved {latest_path}",))
            status("failed" if failure else "finished")
        finally:
            env.close()
    if failure is not None:
        raise failure


def interrupt_training(*_):
    raise KeyboardInterrupt


def add_training_arguments(parser):
    parser.add_argument("--path-only", action="store_true", help="one drone, no evacuees; requires a separate --output checkpoint")
    parser.add_argument("--path-following", action="store_true",
                        help="path-only pretraining with strong cluster/progress rewards and immediate one-step learning")
    parser.add_argument("--max-q", type=float, help="stop above this Q magnitude (default: 100)")
    parser.add_argument("--target-sync-steps", type=int, help="training steps between target network copies (default: 2000)")
    parser.add_argument("--fixed-route-seed", type=int, help="initial fixed path-only scenario seed (default: training seed)")
    parser.add_argument("--failure-retry-fraction", type=float,
                        help="fraction of varied path-only episodes retrying failed training maps (default: 0.3)")
    parser.add_argument("--stop-after-stage", choices=CURRICULUM_STAGES, help="stop once this stage passes greedy evaluation")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--hours", type=float, help="run for this many hours instead of limiting the episode count")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", nargs="?", const="", help="continue a checkpoint; defaults to the active drone model")
    initialization.add_argument("--init-weights", help="initialize policy weights only, e.g. to fine-tune a navigation model with evacuees")
    parser.add_argument("--run-dir", help="directory for latest checkpoint, best model per stage, and live status")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads used by PyTorch (default: 2)")
    parser.add_argument("--max-steps", type=int, help="override every lesson's episode budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=("curriculum", "mixed", "image", *LAYOUT_KINDS), default="curriculum")
    parser.add_argument("--output", help="checkpoint destination; defaults to best.pt in the run directory when provided")


def training_options(args):
    if args.path_following and not args.path_only:
        raise ValueError("path-following requires path-only pretraining")
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("episodes and max-steps must be at least 1")
    if args.hours is not None and args.hours <= 0:
        raise ValueError("hours must be positive")
    if args.threads < 1:
        raise ValueError("threads must be at least 1")
    if args.failure_retry_fraction is not None and (not np.isfinite(args.failure_retry_fraction) or not 0 <= args.failure_retry_fraction <= 1):
        raise ValueError("failure-retry-fraction must be between 0 and 1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, interrupt_training)
    directory = args.run_dir
    if directory is None and (args.hours is not None or args.path_only):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        directory = str(Path("training_runs") / f"drone_{'path_' if args.path_only else ''}{stamp}")
    output = args.output or (str(Path(directory) / "best.pt") if directory else drone_model_path)
    resume = drone_model_path if args.resume == "" else args.resume
    return dict(seed=args.seed, episodes=None if args.hours is not None else args.episodes,
                max_episode_steps=args.max_steps, model_path=output,
                layout_mode=args.layout, resume_from=resume, run_dir=directory,
                path_only=args.path_only, max_training_q=args.max_q, stop_after_stage=args.stop_after_stage,
                target_sync_steps=args.target_sync_steps,
                fixed_route_seed=args.fixed_route_seed,
                init_weights=args.init_weights,
                failure_retry_fraction=args.failure_retry_fraction,
                path_following=args.path_following,
                max_seconds=args.hours * 3600 if args.hours is not None else None)


def main():
    parser = argparse.ArgumentParser(description='Train drone escorting, advancing from small lessons to complete buildings.')
    add_training_arguments(parser)
    args = parser.parse_args()
    try:
        options = training_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    train(**options)


if __name__ == "__main__":
    main()