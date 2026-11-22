import math
import random
import unittest

from flow_maps import (
    CORRIDOR,
    DOOR_ROOM,
    ONE_CELL_EXIT,
    WIDE_EXIT,
    place_people,
    simulation_from_rows,
    walk_to_exit,
)

import compare_flow_rate
from controls.flow_forecast import door_timeline
from grids.door_flow import DoorFlow, Doorway
from utils import config


SETTINGS = (
    "door_flow_enabled",
    "door_flow_per_cell",
    "door_reroute_enabled",
    "reroute_min_gain",
    "min_ticks_between_reroutes",
)


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.saved = {name: getattr(config, name) for name in SETTINGS}

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(config, name, value)


def doorways_of(rows):
    simulation = simulation_from_rows(rows)
    return DoorFlow(simulation.grid_world).doorways


class DoorwayWidthTest(unittest.TestCase):
    def test_two_cell_door_has_width_two(self):
        doors = [doorway for doorway in doorways_of(DOOR_ROOM) if doorway.kind == "door"]
        self.assertEqual([doorway.width for doorway in doors], [2])
        self.assertAlmostEqual(doors[0].capacity, 2 * config.door_flow_per_cell)

    def test_corridor_is_one_doorway_with_its_width(self):
        corridor = max(doorways_of(CORRIDOR), key=lambda doorway: len(doorway.cells))
        self.assertEqual(corridor.kind, "door")
        self.assertEqual(corridor.width, 3)
        self.assertEqual(len(corridor.cells), 15)
        self.assertGreater(len(corridor.cluster_ids), 1)

    def test_exit_width_is_its_long_side(self):
        exits = [doorway for doorway in doorways_of(WIDE_EXIT) if doorway.kind == "exit"]
        self.assertEqual([doorway.width for doorway in exits], [3])


class BudgetTest(unittest.TestCase):
    def make(self, capacity):
        return Doorway(0, 0, "door", [0], [(0, 0)], 1, capacity)

    def test_whole_capacity_is_the_same_every_tick(self):
        doorway = self.make(2)
        self.assertEqual([doorway.allowed(tick) for tick in range(5)], [2] * 5)

    def test_fractional_capacity_keeps_the_average(self):
        doorway = self.make(0.5)
        self.assertEqual([doorway.allowed(tick) for tick in range(6)], [0, 1, 0, 1, 0, 1])
        doorway = self.make(0.4)
        self.assertEqual(sum(doorway.allowed(tick) for tick in range(50)), 20)


class DoorLimitTest(ConfigTestCase):
    def crowd(self, rows, cells):
        simulation = simulation_from_rows(rows)
        place_people(simulation, cells)
        return simulation

    def test_door_lets_through_at_most_its_capacity(self):
        simulation = self.crowd(DOOR_ROOM, [(x, y) for x in range(1, 6) for y in range(1, 7)][:20])
        door = None
        for tick in range(1, 40):
            walk_to_exit(simulation, max_steps=1)
            door = next(d for d in simulation.grid_world.door_flow.doorways if d.kind == "door")
            self.assertLessEqual(door.passed, math.floor(door.capacity * tick + 1e-9))
        self.assertGreater(door.passed, 0)

    def test_limit_slows_the_crowd_down(self):
        cells = [(x, y) for x in range(1, 6) for y in range(1, 7)][:20]
        config.door_flow_enabled = False
        free = walk_to_exit(self.crowd(DOOR_ROOM, cells))
        config.door_flow_enabled = True
        limited = walk_to_exit(self.crowd(DOOR_ROOM, cells))
        self.assertEqual(len(free), 20)
        self.assertEqual(len(limited), 20)
        self.assertGreater(max(limited), max(free))

    def test_one_cell_exit_needs_two_and_a_half_ticks_per_person(self):
        simulation = self.crowd(ONE_CELL_EXIT, [(x, y) for x in range(6, 9) for y in range(1, 6)][:10])
        steps = walk_to_exit(simulation)
        self.assertEqual(len(steps), 10)
        self.assertGreaterEqual(max(steps) - min(steps), 10 / 0.4 - 3)

    def test_disabled_flow_changes_nothing(self):
        config.door_flow_enabled = False
        simulation = self.crowd(DOOR_ROOM, [(1, 1), (2, 2)])
        walk_to_exit(simulation)
        self.assertIsNone(simulation.grid_world.door_flow)
        self.assertEqual(simulation.evacuated_count, 2)


class FairnessTest(unittest.TestCase):
    def test_longest_waiting_person_leaves_first(self):
        simulation = simulation_from_rows(WIDE_EXIT)
        place_people(simulation, [(9, 1), (9, 2), (9, 3)])
        door_flow = DoorFlow(simulation.grid_world)
        exit_doorway = next(d for d in door_flow.doorways if d.kind == "exit")
        exit_doorway.capacity = 1
        first, second, third = simulation.evacuees
        door_flow.waiting = {third: 5, second: 2}
        self.assertEqual(door_flow.exit_departures(simulation.evacuees), [third])


class TimelineTest(unittest.TestCase):
    def test_worked_example(self):
        timeline = door_timeline([("A", 100, 20), ("B", 104, 15)], 1.2)
        (_, _, a_start, a_end), (_, b_arrival, b_start, b_end) = timeline
        self.assertEqual(a_start, 100)
        self.assertAlmostEqual(a_end, 116.67, places=2)
        self.assertAlmostEqual(b_start - b_arrival, 12.67, places=2)
        self.assertAlmostEqual(b_end, 129.17, places=2)

    def test_group_of_twenty_through_two_cell_door(self):
        timeline = door_timeline([("A", 0, 20)], 2.0)
        self.assertEqual(timeline[0][3], 10)


class RerouteTest(ConfigTestCase):
    def test_second_group_goes_round_to_the_other_exit(self):
        seeds = range(1000, 1005)
        flow_only = [compare_flow_rate.run(compare_flow_rate.two_exit_simulation, seed, True, False) for seed in seeds]
        rerouted = [compare_flow_rate.run(compare_flow_rate.two_exit_simulation, seed, True, True) for seed in seeds]
        self.assertTrue(all(result[4] >= 1 for result in rerouted))
        self.assertTrue(all(len(result[1]) == result[0] for result in rerouted))
        self.assertLess(
            sum(max(result[1]) for result in rerouted),
            sum(max(result[1]) for result in flow_only),
        )

    def test_no_reroute_when_the_gain_is_too_small(self):
        config.reroute_min_gain = 10000
        result = compare_flow_rate.run(compare_flow_rate.two_exit_simulation, 1000, True, True)
        self.assertEqual(result[4], 0)

    def test_drone_does_not_switch_twice_in_a_row(self):
        config.reroute_min_gain = 0
        random.seed(0)
        simulation = compare_flow_rate.two_exit_simulation(random.Random(1000))
        from controls.drone_astar import AStarDroneController
        from controls.evacuee_rl import QLearningFollowController
        from controls.flow_forecast import FlowRerouter
        from controls.global_coordinator import GlobalCoordinator

        evacuees = QLearningFollowController()
        evacuees.load(config.q_table_path)
        drones = AStarDroneController()
        coordinator = GlobalCoordinator(simulation.grid_world)
        routes = coordinator.plan_initial_routes(simulation.drones)
        rerouter = FlowRerouter(coordinator)
        for _ in range(150):
            if not simulation.evacuees:
                break
            rerouter.step(simulation, routes, drones)
            simulation.step(evacuees.get_actions(simulation), drones.get_actions(simulation, routes))
        by_drone = {}
        for reroute in rerouter.reroutes:
            by_drone.setdefault(reroute["drone"], []).append(reroute["step"])
        for steps in by_drone.values():
            for earlier, later in zip(steps, steps[1:]):
                self.assertGreaterEqual(later - earlier, config.min_ticks_between_reroutes)


if __name__ == "__main__":
    unittest.main()
