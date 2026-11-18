import argparse
import random

from controls.evacuee_rl import QLearningFollowController
from controls.drone_astar import AStarDroneController
from controls.action import Action
from entities.drone import Drone
from entities.evacuee import Evacuee
from grids.map_grid import Map_Grid
from map.cell import CellType
from simulation import Simulation
from training.layouts import LAYOUT_KINDS, generate_training_building
from training.scenario import (
    add_training_stairs, configure_training_exits, load_default_map,
    place_building_scenario, reset_simulation,
)
from utils.config import q_table_path


EPISODES = 2000
MAX_STEPS = 2000
BASIC_EPISODES = 1024
START_EPSILON = 0.35
END_EPSILON = 0.03
REPORT_EVERY = 100


def run_episode(simulation, evacuees, drones, max_steps=MAX_STEPS):
    total_reward = 0.0
    for _ in range(max_steps):
        if not simulation.evacuees:
            break
        evacuee_actions = evacuees.get_actions(simulation)
        drone_actions = drones.get_actions(simulation)
        simulation.step(evacuee_actions, drone_actions)
        total_reward += evacuees.learn(simulation)
    return total_reward


def epsilon_for(episode, episodes=EPISODES):
    progress = episode / max(episodes - 1, 1)
    return START_EPSILON + (END_EPSILON - START_EPSILON) * progress


def pretrain_following(evacuees, rng, episodes=BASIC_EPISODES, max_steps=24):
    simulation = Simulation()
    grid = Map_Grid(31, 31, CellType.EMPTY)
    grid.set(30, 30, CellType.EXIT)
    simulation.grid_world.add_floor_from_map(grid)
    world = simulation.grid_world
    directions = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    old_epsilon = evacuees.epsilon
    try:
        for episode in range(episodes):
            reset_simulation(simulation)
            evacuees.reset()
            evacuees.epsilon = 0.8 - 0.7 * episode / max(episodes - 1, 1)
            dx, dy = directions[episode % len(directions)]
            distance = rng.choice((1, 2, 3, 4, 6))
            drone = Drone(15, 15, drone_id=0)
            person = Evacuee(15 + dx * distance, 15 + dy * distance, assigned_drone_id=0)
            simulation.drones.append(drone)
            simulation.evacuees.append(person)
            world.add_drone(drone)
            world.add_evacuee(person)
            simulation.started = True
            for _ in range(max_steps):
                actions = evacuees.get_actions(simulation)
                simulation.step(actions, [Action.STAY])
                evacuees.learn(simulation)
    finally:
        evacuees.epsilon = old_epsilon
        evacuees.reset()


def prepare_episode(rng, image_world_data=None, options=None):
    options = options or {}
    simulation = Simulation()
    kind = options.get("layout") or rng.choices(("image", *LAYOUT_KINDS), weights=(2, 3, 2, 3))[0]
    if kind == "image":
        if image_world_data is None:
            load_default_map(simulation)
            add_training_stairs(simulation.grid_world)
        else:
            simulation.grid_world.from_dict(image_world_data)
    else:
        generate_training_building(simulation, rng, kind, floor_count=options.get("floor_count"))
    world = simulation.grid_world
    floor_ids = list(world.floors)
    source = options.get("floor_id", rng.choice(floor_ids))
    use_stairs = options.get("require_stairs", rng.random() < 0.8)
    if use_stairs:
        destination = options.get("exit_floor_id", max(floor_ids, key=lambda floor: abs(floor - source)))
        if destination == source or destination not in floor_ids:
            raise ValueError("Stair episodes need an exit on a different floor")
        originals = {
            floor_id: [cell for cells in floor.map_grid.exits for cell in cells]
            for floor_id, floor in world.floors.items()
        }
        configure_training_exits(world, originals, destination)
    count = options.get("evacuee_count", rng.choice((1, 3, 6, 12, 20, 32, 48)))
    drone_count = options.get("drone_count", rng.randint(1, min(4, count)))
    place_building_scenario(
        simulation, rng, count, source, drone_count,
        distribution=options.get("distribution", rng.choice(("clustered", "scattered", "line"))),
    )
    longest = max(
        world.floors[person.floor_id].distance_map[person.grid_y][person.grid_x]
        for person in simulation.evacuees
    )
    return simulation, max(MAX_STEPS, int(8 * longest + 20 * count))


def train(seed=0, episodes=EPISODES, model_path=q_table_path, basic_episodes=BASIC_EPISODES, max_steps=None):
    random.seed(seed)
    rng = random.Random(seed)
    simulation = Simulation()
    load_default_map(simulation)
    add_training_stairs(simulation.grid_world)
    image_world_data = simulation.grid_world.to_dict()
    evacuees = QLearningFollowController(learning=True)
    drones = AStarDroneController()
    pretrain_following(evacuees, rng, episodes=basic_episodes)
    print(f"Completed {basic_episodes} basic following episodes across all eight directions.")

    rewards = []
    evacuated = []
    for episode in range(episodes):
        evacuees.epsilon = epsilon_for(episode, episodes)
        evacuees.reset()
        drones.reset()
        simulation, budget = prepare_episode(rng, image_world_data)
        start_count = len(simulation.evacuees)
        rewards.append(run_episode(simulation, evacuees, drones, max_steps=budget if max_steps is None else max_steps))
        evacuated.append(simulation.evacuated_count / max(start_count, 1))

        if (episode + 1) % REPORT_EVERY == 0:
            average_reward = sum(rewards[-REPORT_EVERY:]) / REPORT_EVERY
            average_evacuated = 100 * sum(evacuated[-REPORT_EVERY:]) / REPORT_EVERY
            print(
                f"episode {episode + 1:5d}  epsilon {evacuees.epsilon:.2f}  "
                f"reward {average_reward:8.2f}  evacuated {average_evacuated:5.1f}%"
            )

    evacuees.save(model_path)
    print(f"saved {model_path}")
    return evacuees


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the existing evacuee Q-table on following and varied buildings.")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--basic-episodes", type=int, default=BASIC_EPISODES)
    parser.add_argument("--max-steps", type=int, help="fixed episode limit; default scales with route length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=q_table_path)
    args = parser.parse_args()
    if args.episodes < 0 or args.basic_episodes < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episode counts must be nonnegative and --max-steps must be positive")
    train(args.seed, args.episodes, args.output, args.basic_episodes, args.max_steps)
