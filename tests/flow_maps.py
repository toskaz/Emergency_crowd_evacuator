import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controls.evacuee_rl import step_towards_exit
from entities.drone import Drone
from entities.evacuee import Evacuee
from grids.map_grid import Map_Grid
from map.cell import CellType
from simulation import Simulation


SYMBOLS = {
    ".": CellType.EMPTY,
    "#": CellType.WALL,
    "E": CellType.EXIT,
}


def map_from_rows(rows):
    grid = Map_Grid(len(rows[0]), len(rows), CellType.EMPTY)
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            grid.set(x, y, SYMBOLS[symbol])
    return grid


def simulation_from_rows(rows):
    simulation = Simulation()
    simulation.grid_world.add_floor_from_map(map_from_rows(rows))
    return simulation


def place_people(simulation, cells, floor_id=0):
    for x, y in cells:
        evacuee = Evacuee(x, y, floor_id)
        simulation.evacuees.append(evacuee)
        simulation.grid_world.add_evacuee(evacuee)
    simulation.grid_world.rebuild_simulation_grid(simulation.evacuees, simulation.drones)
    simulation.prepared = True


def place_drone(simulation, x, y, floor_id=0):
    drone = Drone(x, y, floor_id)
    simulation.drones.append(drone)
    simulation.grid_world.add_drone(drone)
    return drone


def walk_to_exit(simulation, max_steps=500):
    exit_steps = []
    for step in range(1, max_steps + 1):
        if not simulation.evacuees:
            break
        before = simulation.evacuated_count
        world = simulation.grid_world
        actions = [step_towards_exit(world, evacuee) for evacuee in simulation.evacuees]
        simulation.step(actions)
        exit_steps += [step] * (simulation.evacuated_count - before)
    return exit_steps


DOOR_ROOM = [
    "##############",
    "#.....#......#",
    "#.....#......#",
    "#............#",
    "#............E",
    "#.....#......#",
    "#.....#......#",
    "##############",
]

CORRIDOR = [
    "###############",
    "#....#####....#",
    "#.............#",
    "#.............#",
    "#.............E",
    "#....#####....#",
    "###############",
]

WIDE_EXIT = [
    "##########",
    "#........E",
    "#........E",
    "#........E",
    "#........#",
    "##########",
]

ONE_CELL_EXIT = [
    "##########",
    "#........#",
    "#........E",
    "#........#",
    "#........#",
    "#........#",
    "##########",
]

