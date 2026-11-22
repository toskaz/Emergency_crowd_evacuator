
import argparse
from collections import deque

from tqdm import tqdm

from train_drones import add_training_arguments, train, training_options


class TrainingProgress:
    def __init__(self, episodes, render=False, fps=0, render_enabled=True):
        self.episodes = tqdm(total=episodes, desc="Training", unit="episode", dynamic_ncols=True)
        self.steps = tqdm(total=0, desc="Preparing", unit="step", position=1, leave=False, dynamic_ncols=True)
        self.recent = deque(maxlen=100)
        self.viewer = None
        self.episode = 0
        self.total_episodes = episodes
        self.stage = None
        if render:
            from training.viewer import TrainingFloorViewer
            self.viewer = TrainingFloorViewer(fps=fps, enabled=render_enabled)

    def on_episode_start(self, env, episode, total_episodes):
        if self.episode == 0 and episode > 1:
            self.episodes.update(episode - 1)
        self.episode, self.total_episodes = episode, total_episodes
        phase = getattr(env, "navigation_phase", None)
        if not getattr(env, "is_rehearsal", False) and self.stage != (env.curriculum_stage, phase):
            self.recent.clear()
            self.stage = (env.curriculum_stage, phase)
        self.steps.reset(total=env.max_episode_steps)
        self.steps.set_postfix_str("", refresh=False)
        lesson = env.curriculum_stage or env.layout_name
        if getattr(env.controller, "solo_episode", False) and not env.path_only:
            lesson += " / solo rehearsal"
        if phase:
            lesson += f"/{phase}"
        if getattr(env, "scenario_source", None):
            lesson += f" ({env.scenario_source})"
        self.steps.set_description(f"Episode {episode}: {lesson}, {len(env.simulation.grid_world.floors)} floors")
        if self.viewer is not None:
            self.viewer.draw(env, episode, total_episodes, reward=0.0)

    def on_step(self, env, reward, total_steps, info):
        outcome = "exit reached" if env.path_only else "evacuated"
        self.steps.set_postfix({"reward": f"{reward:.1f}", outcome: f"{info['evacuation_rate']:.0%}"}, refresh=False)
        self.steps.update(1)
        if self.viewer is not None:
            return self.viewer.update(env, self.episode, self.total_episodes, reward)
        return True

    def on_episode_end(self, env, reward, info, completed):
        self.steps.refresh()
        if completed:
            if not getattr(env, "is_rehearsal", False):
                self.recent.append((reward, info["success"], info["evacuation_rate"]))
            count = len(self.recent)
            postfix = dict(
                stage=getattr(env, "training_stage", env.curriculum_stage) or "manual",
                phase=getattr(env, "navigation_phase", None) or "escort",
                epsilon=f"{env.controller.epsilon:.3f}")
            if count:
                postfix.update(reward=f"{sum(item[0] for item in self.recent) / count:.1f}",
                               success=f"{sum(item[1] for item in self.recent) / count:.0%}",
                               **{"exit reached" if env.path_only else "evacuated": f"{sum(item[2] for item in self.recent) / count:.0%}"})
            self.episodes.set_postfix(postfix, refresh=False)
            self.episodes.update(1)

    def on_report(self, lines):
        for line in lines:
            tqdm.write(line)

    def on_evaluation_step(self, env, trial, trials, reward, info):
        if env.step_count == 1:
            self.steps.reset(total=env.max_episode_steps)
            self.steps.set_description(f"Eval {getattr(env, 'evaluation_suite', '')} {env.curriculum_stage} {trial}/{trials}")
        outcome = "exit reached" if env.path_only else "evacuated"
        self.steps.set_postfix({"reward": f"{reward:.1f}", outcome: f"{info['evacuation_rate']:.0%}"}, refresh=False)
        self.steps.update(1)
        if self.viewer is not None:
            return self.viewer.update(env, trial, trials, reward)
        return True

    def close(self):
        self.steps.close()
        self.episodes.close()
        if self.viewer is not None:
            self.viewer.close()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 (unlimited) or a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description='Train with terminal progress, optionally showing every floor in its own window.')
    add_training_arguments(parser)
    parser.add_argument("--render", action="store_true", help="show live floors; V toggles the view, Space pauses, Esc stops and saves")
    parser.add_argument("--render-off", action="store_true", help="start fast with a small control window; press V or click to show live floors")
    parser.add_argument("--fps", type=nonnegative_int, default=0, help="maximum training steps per second with rendering; 0 means unlimited (default)")
    args = parser.parse_args()
    try:
        options = training_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    progress = TrainingProgress(options["episodes"], render=args.render or args.render_off,
                                fps=args.fps, render_enabled=not args.render_off)
    try:
        train(**options, observer=progress)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
