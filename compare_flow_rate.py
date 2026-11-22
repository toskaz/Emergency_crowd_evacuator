import json
import random
import sys
from pathlib import Path

from controls.drone_assignment import assign_drone_groups
from controls.drone_astar import AStarDroneController
from controls.evacuee_rl import QLearningFollowController
from controls.flow_forecast import FlowRerouter
from controls.global_coordinator import GlobalCoordinator
from entities.drone import Drone
from entities.evacuee import Evacuee
from grids.map_grid import Map_Grid
from map.cell import CellType
from simulation import Simulation
from utils import config


SEEDS = range(1000, 1020)
MAX_STEPS = 600
PROJECT_PATH = Path(__file__).resolve().parent / "assets" / "project.json"

TWO_EXITS = (
    ["#" * 31]
    + ["#" + "." * 29 + "#" for _ in range(5)]
    + ["#####.##################..#####"]
    + ["#" + "." * 14 + "#" + "." * 14 + "#" for _ in range(6)]
    + ["#####E##################EE#####"]
)

MODES = (
    ("no flow limit", False, False),
    ("flow limit", True, False),
    ("flow limit + reroute", True, True),
)


def two_exit_simulation(rng):
    grid = Map_Grid(len(TWO_EXITS[0]), len(TWO_EXITS), CellType.EMPTY)
    for y, row in enumerate(TWO_EXITS):
        for x, symbol in enumerate(row):
            grid.set(x, y, {"#": CellType.WALL, "E": CellType.EXIT}.get(symbol, CellType.EMPTY))
    simulation = Simulation()
    simulation.grid_world.add_floor_from_map(grid)
    world = simulation.grid_world
    for x, y in ((4, 3), (11, 3)):
        drone = Drone(x, y, 0)
        simulation.drones.append(drone)
        world.add_drone(drone)
    cells = [(x, y) for y in range(1, 6) for x in range(1, 15) if (x, y) not in ((4, 3), (11, 3))]
    for x, y in rng.sample(cells, 35):
        evacuee = Evacuee(x, y, 0)
        simulation.evacuees.append(evacuee)
        world.add_evacuee(evacuee)
    assign_drone_groups(world, simulation.drones, simulation.evacuees)
    world.rebuild_simulation_grid(simulation.evacuees, simulation.drones)
    simulation.reset_door_flow()
    simulation.prepared = True
    return simulation


def project_simulation(rng):
    simulation = Simulation()
    simulation.grid_world.from_dict(json.loads(PROJECT_PATH.read_text()))
    simulation.prepare()
    return simulation


def run(build, seed, flow, reroute):
    config.door_flow_enabled = flow
    config.door_reroute_enabled = reroute
    rng = random.Random(seed)
    random.seed(seed)
    simulation = build(rng)
    world = simulation.grid_world
    evacuees = QLearningFollowController()
    evacuees.load(config.q_table_path)
    drones = AStarDroneController()
    coordinator = GlobalCoordinator(world)
    routes = coordinator.plan_initial_routes(simulation.drones)
    rerouter = FlowRerouter(coordinator)
    start_count = len(simulation.evacuees)
    exit_steps = []
    for step in range(1, MAX_STEPS + 1):
        if not simulation.evacuees:
            break
        rerouter.step(simulation, routes, drones)
        before = simulation.evacuated_count
        evacuee_actions = evacuees.get_actions(simulation)
        drone_actions = drones.get_actions(simulation, routes)
        simulation.step(evacuee_actions, drone_actions)
        exit_steps += [step] * (simulation.evacuated_count - before)
    door_flow = world.door_flow
    max_queue = max((doorway.max_queue for doorway in door_flow.doorways), default=0) if door_flow else 0
    total_wait = sum(doorway.total_wait for doorway in door_flow.doorways) if door_flow else 0
    return start_count, exit_steps, max_queue, total_wait, len(rerouter.reroutes)


def summarize(name, results):
    people = sum(result[0] for result in results)
    exit_steps = [step for result in results for step in result[1]]
    finished = [max(result[1]) for result in results if result[1] and len(result[1]) == result[0]]
    evacuated = 100 * len(exit_steps) / max(people, 1)
    mean_exit = sum(exit_steps) / len(exit_steps) if exit_steps else float("nan")
    everyone_out = sum(finished) / len(finished) if finished else float("nan")
    max_queue = max(result[2] for result in results)
    mean_wait = sum(result[3] for result in results) / len(results)
    reroutes = sum(result[4] for result in results) / len(results)
    print(
        f"{name:22s} evacuated {evacuated:5.1f}%  mean exit {mean_exit:6.1f}  "
        f"everyone out {everyone_out:6.1f} ({len(finished)}/{len(results)})  "
        f"max queue {max_queue:3d}  queue ticks {mean_wait:7.1f}  reroutes {reroutes:4.1f}"
    )


def main():
    maps = {"two exits": two_exit_simulation, "project": project_simulation}
    selected = sys.argv[1:] or list(maps)
    for map_name in selected:
        print(map_name)
        for name, flow, reroute in MODES:
            summarize(name, [run(maps[map_name], seed, flow, reroute) for seed in SEEDS])


if __name__ == "__main__":
    main()
