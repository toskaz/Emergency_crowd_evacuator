import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pygame

from controls.action import Action
from controls.evacuee_rl import (
    MOVES,
    QLearningFollowController,
    movement_options,
    panic,
    step_towards_exit,
)
from controls.global_coordinator import GlobalCoordinator
from editor.editor import handle_map_editor
from editor.tool import Tool
from entities.drone import Drone
from entities.evacuee import Evacuee
from grids.cluster import ClusterType
from grids.grid import Grid
from grids.grid_world import Grid_World
from grids.map_grid import Map_Grid
from map.cell import CellType
from map.image_extractor import _mark_exits_on_grid
from simulation import Simulation
from utils.program_context import ProgramContext, ViewOptions
from utils.program_state import EditorState, SimulationPausedState
from utils.program import BASE_SIMULATION_INTERVAL, simulation_tick_interval


def map_with(**cells):
    grid = Map_Grid(12, 12, CellType.EMPTY)
    for location, cell_type in cells.items():
        x, y = (int(value) for value in location.split("_"))
        grid.set(x, y, cell_type)
    return grid


def linked_world():
    world = Grid_World()
    world.add_floor_from_map(map_with(**{"6_6": CellType.STAIRS}))
    world.add_floor_from_map(
        map_with(
            **{
                "6_6": CellType.STAIRS,
                "7_6": CellType.STAIRS,
                "9_6": CellType.EXIT,
            }
        )
    )
    first = world.stair_cluster_at(0, 6, 6)
    second = world.stair_cluster_at(1, 6, 6)
    world.connect_stair_clusters(first, second)
    return world


class EvacueeStairTests(unittest.TestCase):
    def test_existing_q_table_is_migrated_with_a_stairs_action(self):
        controller = QLearningFollowController()
        legacy = np.ones(controller.q_table.shape[:-1] + (len(MOVES) - 1,))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npy"
            np.save(path, legacy)

            self.assertTrue(controller.load(path))

        self.assertEqual(controller.q_table.shape[-1], len(MOVES))
        self.assertTrue(np.all(controller.q_table[..., :-1] == 1))
        self.assertTrue(np.all(controller.q_table[..., -1] == 0))

    def test_following_leader_across_floors_routes_to_linked_stairs(self):
        world = linked_world()
        evacuee = Evacuee(4, 6, 0)
        controller = QLearningFollowController()
        leader = Evacuee(9, 6, 1)
        controller.remember_drone_position(world, evacuee, leader)
        self.assertEqual(controller.last_seen[evacuee], (0, 6, 6))

        controller.last_seen[evacuee] = (1, 9, 6)

        self.assertEqual(controller.follow_last_seen(world, evacuee), Action.RIGHT)
        self.assertEqual(controller.last_seen[evacuee], (0, 6, 6))

        evacuee.grid_x, evacuee.grid_y = 6, 6
        self.assertEqual(
            controller.follow_last_seen(world, evacuee), Action.USE_STAIRS
        )
        self.assertIn(Action.USE_STAIRS, movement_options(world, evacuee))

        state = (0, 0, 0)
        controller.q_table[state][int(Action.USE_STAIRS)] = 1.0
        self.assertEqual(
            controller.choose_action(state, movement_options(world, evacuee)),
            Action.USE_STAIRS,
        )
        self.assertEqual(controller.q_table.shape[-1], len(MOVES))

    def test_exit_routing_uses_stairs_to_reach_another_floor(self):
        world = linked_world()
        evacuee = Evacuee(6, 6, 0)
        self.assertEqual(step_towards_exit(world, evacuee), Action.USE_STAIRS)

    def test_following_leader_does_not_immediately_return_on_stairs(self):
        world = linked_world()
        simulation = Simulation()
        simulation.grid_world = world
        evacuee = Evacuee(6, 6, 0)
        simulation.evacuees = [evacuee]
        world.add_evacuee(evacuee)
        controller = QLearningFollowController()
        controller.last_seen[evacuee] = (0, 6, 6)

        action = controller.decide(
            simulation, evacuee, Evacuee(8, 6, 1)
        )
        self.assertEqual(action, Action.USE_STAIRS)
        self.assertNotIn(evacuee, controller.last_seen)

        world.step_evacuees([evacuee], [action])
        next_action = controller.decide(simulation, evacuee, None)

        self.assertEqual(evacuee.floor_id, 1)
        self.assertNotEqual(next_action, Action.USE_STAIRS)

    def test_stair_transfer_keeps_occupancy_and_simulation_grid_consistent(self):
        world = linked_world()
        simulation = Simulation()
        simulation.grid_world = world
        entering = Evacuee(6, 6, 0)
        standing = Evacuee(6, 6, 1)
        simulation.evacuees = [entering, standing]
        world.add_evacuee(entering)
        world.add_evacuee(standing)
        world.rebuild_simulation_grid(simulation.evacuees)
        simulation.prepared = True
        simulation.started = True

        simulation.step(
            [Action.USE_STAIRS, Action.STAY],
            drone_actions=[],
        )

        self.assertEqual((entering.floor_id, entering.grid_x, entering.grid_y), (1, 7, 6))
        self.assertIsNone(world.floors[0].evacuee_occupancy_grid.get(6, 6))
        self.assertIs(world.floors[1].evacuee_occupancy_grid.get(6, 6), standing)
        self.assertIs(world.floors[1].evacuee_occupancy_grid.get(7, 6), entering)
        self.assertEqual(world.floors[0].simulation_grid.get(6, 6), CellType.STAIRS)
        self.assertEqual(world.floors[1].simulation_grid.get(6, 6), CellType.EVACUEE)
        self.assertEqual(world.floors[1].simulation_grid.get(7, 6), CellType.EVACUEE)


class MapAndResetTests(unittest.TestCase):
    def test_wide_external_doorway_is_detected(self):
        grid = Grid(40, 40, CellType.EMPTY)
        for x in range(5, 35):
            if not 14 <= x <= 20:
                grid.set(x, 5, CellType.WALL)
            grid.set(x, 34, CellType.WALL)
        for y in range(5, 35):
            if not 14 <= y <= 20:
                grid.set(5, y, CellType.WALL)
            grid.set(34, y, CellType.WALL)

        _mark_exits_on_grid(grid, 9)

        self.assertTrue(
            any(grid.get(5, y) == CellType.EXIT for y in range(14, 21))
        )

    def test_interior_doorway_near_recessed_exterior_is_not_an_exit(self):
        grid = Grid(32, 30, CellType.EMPTY)
        for x in range(5, 27):
            grid.set(x, 5, CellType.WALL)
            grid.set(x, 24, CellType.WALL)
        for y in range(5, 25):
            if not 14 <= y <= 20:
                grid.set(5, y, CellType.WALL)
            grid.set(26, y, CellType.WALL)
        for y in range(5, 25):
            if y != 16:
                grid.set(8, y, CellType.WALL)

        _mark_exits_on_grid(grid, 9)

        exits = {
            (x, y)
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.get(x, y) == CellType.EXIT
        }
        self.assertIn((5, 16), exits)
        self.assertNotIn((8, 16), exits)
        self.assertTrue(all(x == 5 for x, _ in exits))

    def test_closed_thin_exterior_wall_is_not_marked_as_an_exit(self):
        grid = Grid(30, 30, CellType.EMPTY)
        for x in range(5, 25):
            grid.set(x, 5, CellType.WALL)
            grid.set(x, 24, CellType.WALL)
        for y in range(5, 25):
            grid.set(5, y, CellType.WALL)
            grid.set(24, y, CellType.WALL)

        _mark_exits_on_grid(grid, 5)

        self.assertFalse(
            any(
                grid.get(x, y) == CellType.EXIT
                for y in range(grid.height)
                for x in range(grid.width)
            )
        )

    def test_external_exit_is_marked_on_the_threshold_not_outside(self):
        grid = Grid(20, 20, CellType.EMPTY)
        for x in range(4, 16):
            grid.set(x, 4, CellType.WALL)
            grid.set(x, 15, CellType.WALL)
        for y in range(4, 16):
            if y != 9:
                grid.set(4, y, CellType.WALL)
            grid.set(15, y, CellType.WALL)

        _mark_exits_on_grid(grid, 5)

        exits = {
            (x, y)
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.get(x, y) == CellType.EXIT
        }
        self.assertIn((4, 9), exits)
        self.assertTrue(all(x == 4 and y == 9 for x, y in exits))

    def test_editor_rebuilds_exit_clusters_and_reset_restores_base_map(self):
        simulation = Simulation()
        simulation.grid_world.add_floor_from_map(Map_Grid(12, 12, CellType.EMPTY))
        simulation.prepare()

        with patch("editor.editor.pixel_to_grid", return_value=(3, 4)):
            handle_map_editor(simulation, (0, 0), 1, Tool.EXIT)

        world = simulation.grid_world
        cluster_id = world.cluster_grid.get(3, 4)
        self.assertGreaterEqual(cluster_id, 0)
        self.assertEqual(world.cluster_grid.clusters[cluster_id].type, ClusterType.EXIT)
        self.assertFalse(simulation.prepared)

        context = ProgramContext.__new__(ProgramContext)
        context.simulation = simulation
        context.drone_controller = None
        context.rl_evacuee_controller = None
        context.coordinator = GlobalCoordinator(world)
        context.drone_routes = {}
        context.link_stairs_mode = False
        context.simulation_options = SimpleNamespace(action=Action.RIGHT)
        context.reset_simulation(restore_base_map=True)

        self.assertEqual(world.map_grid.get(3, 4), CellType.EMPTY)
        self.assertFalse(
            any(cluster.type is ClusterType.EXIT for cluster in world.cluster_grid.clusters)
        )

    def test_editor_drag_interpolates_cells_and_batches_cluster_rebuilds(self):
        simulation = Simulation()
        world = simulation.grid_world
        world.add_floor_from_map(Map_Grid(12, 12, CellType.EMPTY))
        simulation.prepare()
        program = SimpleNamespace(
            simulation=simulation,
            screen=SimpleNamespace(get_size=lambda: (640, 480)),
        )
        state = EditorState()
        state.selected_tool = Tool.STAIRS
        rebuild = patch.object(
            world, "rebuild_changed_map_data", wraps=world.rebuild_changed_map_data
        )

        with rebuild as rebuild_map, patch(
            "utils.program_state.pixel_to_grid",
            side_effect=[(1, 1), (4, 1), (4, 1)],
        ):
            state.handle_mouse(
                program,
                SimpleNamespace(
                    type=pygame.MOUSEBUTTONDOWN,
                    button=1,
                    pos=(100, 100),
                ),
            )
            state.handle_mouse(
                program,
                SimpleNamespace(
                    type=pygame.MOUSEMOTION,
                    pos=(120, 100),
                ),
            )
            self.assertEqual(rebuild_map.call_count, 2)

            state.handle_mouse(
                program,
                SimpleNamespace(
                    type=pygame.MOUSEMOTION,
                    pos=(121, 100),
                ),
            )
            self.assertEqual(rebuild_map.call_count, 2)

            state.handle_mouse(
                program,
                SimpleNamespace(type=pygame.MOUSEBUTTONUP, button=1, pos=(121, 100)),
            )

        self.assertIsNone(state.active_mouse_button)
        self.assertTrue(
            all(world.map_grid.get(x, 1) == CellType.STAIRS for x in range(1, 5))
        )

    def test_simulation_reset_preserves_edits_and_stair_links(self):
        world = linked_world()
        simulation = Simulation()
        simulation.grid_world = world
        world.set_map_cell(0, 9, 9, CellType.EXIT)
        world.rebuild_changed_map_data()
        world.set_temporary_wall(0, 1, 1, True)
        world.pending_stair_endpoint = world.stair_cluster_at(0, 6, 6)
        simulation.started = True
        simulation.step_count = 8
        simulation.evacuated_count = 2

        context = ProgramContext.__new__(ProgramContext)
        context.simulation = simulation
        context.drone_controller = None
        context.rl_evacuee_controller = None
        context.coordinator = GlobalCoordinator(world)
        context.drone_routes = {}
        context.link_stairs_mode = True
        context.simulation_options = SimpleNamespace(action=Action.RIGHT)
        context.reset_simulation()

        self.assertFalse(simulation.started)
        self.assertEqual(simulation.step_count, 0)
        self.assertEqual(simulation.evacuated_count, 0)
        self.assertEqual(context.simulation_options.action, Action.STAY)
        self.assertEqual(world.map_grid.get(9, 9), CellType.EXIT)
        self.assertFalse(world.temporary_wall_grid.get(1, 1))
        self.assertIsNone(world.pending_stair_endpoint)
        self.assertTrue(context.link_stairs_mode)
        self.assertEqual(
            world._get_stair_cluster(world.stair_cluster_at(0, 6, 6)).linked_to,
            world.stair_cluster_at(1, 6, 6),
        )


class PlaybackAndResultsTests(unittest.TestCase):
    def test_single_step_advances_once_and_remains_paused(self):
        simulation = Simulation()
        simulation.grid_world.add_floor_from_map(
            Map_Grid(4, 4, CellType.EMPTY)
        )
        simulation.prepare()

        class EmptyEvacueeController:
            def get_actions(self, _simulation):
                return []

        class EmptyDroneController:
            def get_actions(self, _simulation, _routes, _cell_routes=None):
                return []

            def learn(self, _simulation, _routes):
                pass

        context = ProgramContext.__new__(ProgramContext)
        context.simulation = simulation
        context.simulation_options = SimpleNamespace(
            rl_evacuees=True,
            autonomous_drones=True,
            action=Action.STAY,
        )
        context.rl_evacuee_controller = EmptyEvacueeController()
        context.drone_controller = EmptyDroneController()
        context.drone_routes = {}
        context.coordinator = None
        context.state = SimulationPausedState()
        paused_state = context.state

        context.step_simulation(keep_paused=True)

        self.assertEqual(simulation.step_count, 1)
        self.assertIs(context.state, paused_state)
        self.assertFalse(simulation.started)

    def test_speed_intervals_and_evacuation_history(self):
        context = ProgramContext.__new__(ProgramContext)
        context.speed_multiplier = 1.0
        context.simulation_tick_accumulator = 0.2
        for speed in (0.5, 0.75, 2.0, 4.0):
            context.set_speed_multiplier(speed)
            self.assertAlmostEqual(
                simulation_tick_interval(context.speed_multiplier),
                BASE_SIMULATION_INTERVAL / speed,
            )
            self.assertEqual(context.simulation_tick_accumulator, 0.0)

        grid = Map_Grid(4, 4, CellType.EMPTY)
        grid.set(1, 1, CellType.EVACUEE)
        grid.set(2, 1, CellType.EXIT)
        simulation = Simulation()
        simulation.grid_world.add_floor_from_map(grid)
        simulation.prepare()
        simulation.started = True

        simulation.step([Action.RIGHT], drone_actions=[])
        simulation.step([], drone_actions=[])

        self.assertEqual(simulation.evacuation_history, [(0, 0), (1, 1), (2, 1)])
        self.assertEqual(simulation.evacuation_times, [1])


class VisualAndEvacuationGuidanceTests(unittest.TestCase):
    def test_overlays_are_disabled_by_default(self):
        options = ViewOptions()

        self.assertFalse(options.show_grid)
        self.assertFalse(options.show_clusters)
        self.assertFalse(options.show_cluster_connections)
        self.assertFalse(options.show_drone_routes)

    def test_reaching_last_seen_target_continues_toward_exit(self):
        grid = Map_Grid(20, 12, CellType.EMPTY)
        grid.set(18, 5, CellType.EXIT)
        for y in range(12):
            if y != 9:
                grid.set(10, y, CellType.WALL)

        world = Grid_World()
        world.add_floor_from_map(grid)
        evacuee = Evacuee(3, 5)
        world.add_evacuee(evacuee)
        controller = QLearningFollowController()
        controller.last_seen[evacuee] = (0, evacuee.grid_x, evacuee.grid_y)

        expected = step_towards_exit(world, evacuee)
        with patch("controls.evacuee_rl.panic", side_effect=AssertionError):
            action = controller.follow_last_seen(world, evacuee)

        self.assertEqual(action, expected)
        self.assertNotEqual(action, Action.STAY)
        self.assertNotIn(evacuee, controller.last_seen)

    def test_exit_gradient_uses_sidestep_when_doorway_cell_is_occupied(self):
        grid = Map_Grid(8, 8, CellType.EMPTY)
        grid.set(6, 3, CellType.EXIT)
        world = Grid_World()
        world.add_floor_from_map(grid)
        evacuee = Evacuee(3, 3)
        blocker = Evacuee(4, 3)
        world.add_evacuee(evacuee)
        world.add_evacuee(blocker)

        action = step_towards_exit(world, evacuee)

        self.assertNotEqual(action, Action.RIGHT)
        self.assertIn(action, (Action.UP, Action.DOWN))

    def test_drone_route_terminates_at_an_exit_cluster(self):
        grid = Map_Grid(20, 20, CellType.EMPTY)
        grid.set(18, 18, CellType.EXIT)
        world = Grid_World()
        world.add_floor_from_map(grid)
        drone = Drone(2, 2)
        coordinator = GlobalCoordinator(world)

        route = coordinator.plan_initial_routes([drone])[drone]

        self.assertTrue(route)
        self.assertIn(route[-1], coordinator.get_exit_nodes())

    def test_drone_astar_uses_doorway_and_detours_around_temporary_wall(self):
        grid = Map_Grid(24, 22, CellType.EMPTY)
        for x in range(3, 11):
            grid.set(x, 3, CellType.WALL)
            grid.set(x, 10, CellType.WALL)
        for y in range(3, 11):
            grid.set(3, y, CellType.WALL)
            grid.set(10, y, CellType.WALL)
        grid.set(6, 3, CellType.EMPTY)
        grid.set(22, 20, CellType.EXIT)

        world = Grid_World()
        world.add_floor_from_map(grid)
        drone = Drone(6, 7)
        world.add_drone(drone)
        coordinator = GlobalCoordinator(world)
        coordinator.plan_initial_routes([drone])

        original_route = coordinator.cell_routes[drone]
        self.assertTrue(original_route)
        self.assertIn((0, 6, 3), original_route)
        self.assertEqual(
            coordinator.action_for_cell_route(drone, original_route), Action.UP
        )
        self.assertTrue(
            all(
                world.can_move_to(x, y, floor_id)
                for floor_id, x, y in original_route
            )
        )
        self.assertTrue(
            all(
                first[0] != second[0]
                or abs(first[1] - second[1]) + abs(first[2] - second[2]) == 1
                for first, second in zip(original_route, original_route[1:])
            )
        )

        blocked = next(
            (floor_id, x, y)
            for floor_id, x, y in original_route[2:-1]
            if grid.get(x, y) == CellType.EMPTY and (x, y) != (6, 3)
        )
        self.assertTrue(world.set_temporary_wall(blocked[0], blocked[1], blocked[2], True))
        coordinator.on_environment_changed([drone])
        detour_route = coordinator.cell_routes[drone]

        self.assertTrue(detour_route)
        self.assertNotEqual(detour_route, original_route)
        self.assertNotIn(blocked, detour_route)
        self.assertTrue(
            all(
                world.can_move_to(x, y, floor_id)
                for floor_id, x, y in detour_route
            )
        )

        second_blocked = next(
            (floor_id, x, y)
            for floor_id, x, y in detour_route[2:-1]
            if grid.get(x, y) == CellType.EMPTY
            and (floor_id, x, y) != blocked
            and (x, y) != (6, 3)
        )
        self.assertTrue(
            world.set_temporary_wall(
                second_blocked[0], second_blocked[1], second_blocked[2], True
            )
        )
        coordinator.refresh_cell_routes([drone])
        self.assertNotIn(second_blocked, coordinator.cell_routes[drone])

    def test_evacuee_exits_enclosed_room_through_threshold(self):
        grid = Map_Grid(14, 14, CellType.EMPTY)
        for x in range(2, 12):
            grid.set(x, 2, CellType.WALL)
            grid.set(x, 11, CellType.WALL)
        for y in range(2, 12):
            grid.set(2, y, CellType.WALL)
            grid.set(11, y, CellType.WALL)
        grid.set(2, 7, CellType.EXIT)
        grid.set(7, 7, CellType.EVACUEE)

        simulation = Simulation()
        simulation.grid_world.add_floor_from_map(grid)
        simulation.prepare()
        simulation.started = True
        controller = QLearningFollowController()

        for _ in range(8):
            actions = [
                controller.decide(simulation, evacuee, None)
                for evacuee in simulation.evacuees
            ]
            simulation.step(actions, drone_actions=[])
            if not simulation.evacuees:
                break

        self.assertEqual(simulation.evacuated_count, 1)
        self.assertEqual(simulation.evacuees, [])

    def test_doorway_cluster_edges_require_unblocked_orthogonal_cells(self):
        grid = Map_Grid(16, 16, CellType.EMPTY)
        for x in range(3, 12):
            grid.set(x, 3, CellType.WALL)
            grid.set(x, 11, CellType.WALL)
        for y in range(3, 12):
            grid.set(3, y, CellType.WALL)
            grid.set(11, y, CellType.WALL)
        grid.set(7, 3, CellType.EMPTY)

        world = Grid_World()
        world.add_floor_from_map(grid)
        cluster_grid = world.active_floor.cluster_grid
        doorway_id = cluster_grid.get(7, 3)

        self.assertEqual(cluster_grid.clusters[doorway_id].type, ClusterType.DOOR)
        doorway_edges = {
            edge for edge in cluster_grid.adjacent_cluster_pairs
            if doorway_id in edge
        }
        self.assertGreaterEqual(len(doorway_edges), 2)

        self.assertTrue(world.set_temporary_wall(0, 7, 3, True))
        self.assertFalse(any(
            doorway_id in edge
            for edge in cluster_grid.adjacent_cluster_pairs
        ))
        self.assertTrue(world.set_temporary_wall(0, 7, 3, False))
        self.assertTrue(any(
            doorway_id in edge
            for edge in cluster_grid.adjacent_cluster_pairs
        ))

    def test_panic_near_exit_follows_exit_gradient(self):
        grid = Map_Grid(12, 12, CellType.EMPTY)
        grid.set(8, 5, CellType.EXIT)
        world = Grid_World()
        world.add_floor_from_map(grid)
        evacuee = Evacuee(5, 5)

        with patch("controls.evacuee_rl.random.random", return_value=0):
            action = panic(world, evacuee)

        self.assertEqual(action, Action.RIGHT)


if __name__ == "__main__":
    unittest.main()
