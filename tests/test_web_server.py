import json
import unittest
from unittest.mock import patch

from grids.map_grid import Map_Grid
from map.cell import CellType
from server import SUPPORTED_SPEEDS, SimulationSession


class SimulationSessionTests(unittest.TestCase):
    def make_session(self):
        return SimulationSession(load_default_map=False)

    def test_snapshot_is_json_serializable_and_contains_floor_telemetry(self):
        session = self.make_session()

        snapshot = session.get_state_snapshot()

        self.assertEqual(snapshot["mode"], "editor")
        self.assertEqual(len(snapshot["floors"]), 1)
        self.assertEqual(len(snapshot["floors"][0]["cells"]), 24)
        self.assertEqual(snapshot["telemetry"]["speed_multiplier"], 1.0)
        json.dumps(snapshot)

    def test_snapshot_serializes_clusters_for_every_floor(self):
        session = self.make_session()
        session.edit_cells([[2, 2]], 1, "EXIT")
        session.world.add_floor_from_map(Map_Grid(8, 8, CellType.EMPTY))
        session.world.set_active_floor(1)
        session.edit_cells([[3, 3]], 1, "STAIRS")

        floors = session.get_state_snapshot()["floors"]

        self.assertEqual(len(floors), 2)
        for floor in floors:
            self.assertTrue(floor["clusters"])
            for cluster in floor["clusters"]:
                self.assertIsInstance(cluster["id"], int)
                self.assertIsInstance(cluster["type"], str)
                self.assertIsInstance(cluster["is_doorway"], bool)
                self.assertTrue(cluster["cells"])
                for cell in cluster["cells"]:
                    self.assertEqual(len(cell), 2)
                    self.assertTrue(all(isinstance(coordinate, int) for coordinate in cell))
        self.assertIn("EXIT", {cluster["type"] for cluster in floors[0]["clusters"]})
        self.assertIn("STAIRS", {cluster["type"] for cluster in floors[1]["clusters"]})

    def test_batched_cell_edits_rebuild_map_data_once(self):
        session = self.make_session()
        with patch.object(
            session.world, "rebuild_map_data", wraps=session.world.rebuild_map_data
        ) as rebuild:
            changed = session.edit_cells([[2, 3], [3, 3]], 1, "EXIT")

        self.assertEqual(changed, 2)
        self.assertEqual(session.world.map_grid.get(2, 3), CellType.EXIT)
        self.assertEqual(session.world.map_grid.get(3, 3), CellType.EXIT)
        rebuild.assert_called_once()

    def test_edit_undo_restores_map_and_disables_when_empty(self):
        session = self.make_session()

        session.edit_cells([[2, 3]], 1, "EXIT")
        self.assertTrue(session.get_state_snapshot()["can_undo"])
        self.assertEqual(session.world.map_grid.get(2, 3), CellType.EXIT)

        snapshot = session.handle_command({"type": "undo"})

        self.assertEqual(session.world.map_grid.get(2, 3), CellType.EMPTY)
        self.assertFalse(snapshot["can_undo"])
        self.assertNotIn(
            "EXIT",
            {cluster["type"] for cluster in snapshot["floors"][0]["clusters"]},
        )

    def test_undo_history_is_per_floor_and_capped(self):
        session = self.make_session()
        session.world.add_floor_from_map(Map_Grid(8, 8, CellType.EMPTY))
        session.edit_cells([[1, 1]], 1, "WALL")
        session.world.set_active_floor(1)
        session.edit_cells([[2, 2]], 1, "STAIRS")

        session.handle_command({"type": "undo"})

        self.assertEqual(session.world.map_grid.get(2, 2), CellType.EMPTY)
        self.assertFalse(session.get_state_snapshot()["can_undo"])
        session.world.set_active_floor(0)
        self.assertTrue(session.get_state_snapshot()["can_undo"])

        for x in range(31):
            session.edit_cells([[x, 3]], 1, "WALL")
        self.assertEqual(len(session.undo_stack[0]), 30)

    def test_sparse_edit_cells_are_interpolated(self):
        session = self.make_session()

        changed = session.edit_cells([[2, 2], [5, 2]], 1, "WALL")

        self.assertEqual(changed, 4)
        self.assertTrue(
            all(
                session.world.map_grid.get(x, 2) == CellType.WALL
                for x in range(2, 6)
            )
        )

    def test_stair_clusters_can_be_linked_and_unlinked(self):
        session = self.make_session()
        session.world.add_floor_from_map(Map_Grid(8, 8, CellType.EMPTY))
        session.world.set_active_floor(0)
        session.edit_cells([[3, 3]], 1, "STAIRS")
        session.world.set_active_floor(1)
        session.edit_cells([[3, 3]], 1, "STAIRS")

        session.link_stairs(0, [3, 3], 1, [3, 3])
        first_node = session.world.stair_cluster_at(0, 3, 3)
        self.assertIsNotNone(session.world._get_stair_cluster(first_node).linked_to)

        self.assertTrue(session.unlink_stairs(*first_node))
        self.assertIsNone(session.world._get_stair_cluster(first_node).linked_to)

    def test_websocket_stair_link_uses_cluster_ids(self):
        session = self.make_session()
        session.world.add_floor_from_map(Map_Grid(8, 8, CellType.EMPTY))
        session.edit_cells([[3, 3]], 1, "STAIRS")
        first_node = session.world.stair_cluster_at(0, 3, 3)
        session.world.set_active_floor(1)
        session.edit_cells([[3, 3]], 1, "STAIRS")
        second_node = session.world.stair_cluster_at(1, 3, 3)

        session.handle_command({
            "type": "link_stairs",
            "floor_a": first_node[0],
            "cluster_a": first_node[1],
            "floor_b": second_node[0],
            "cluster_b": second_node[1],
        })

        self.assertEqual(
            session.world._get_stair_cluster(first_node).linked_to, second_node
        )

    def test_hazards_and_dynamic_evacuees_update_runtime_counts(self):
        session = self.make_session()

        self.assertTrue(session.spawn_evacuee(1, 1))
        self.assertTrue(session.toggle_hazard(2, 2))
        self.assertFalse(session.toggle_hazard(1, 1))
        telemetry = session.get_state_snapshot()["telemetry"]

        self.assertEqual(telemetry["total_evacuees"], 1)
        self.assertEqual(telemetry["active_hazards"], 1)

    def test_single_step_advances_once_and_remains_paused(self):
        session = self.make_session()

        session.playback("step")

        self.assertEqual(session.simulation.step_count, 1)
        self.assertFalse(session.playing)
        self.assertTrue(session.get_state_snapshot()["telemetry"]["paused"])

    def test_completed_evacuation_enters_results_mode(self):
        session = self.make_session()
        session.total_evacuees = 1
        session.simulation.evacuated_count = 1

        session.playback("step")

        self.assertEqual(session.mode, "results")
        self.assertFalse(session.playing)

    def test_supported_speed_multipliers_are_applied(self):
        session = self.make_session()

        for speed in SUPPORTED_SPEEDS:
            session.playback("pause", speed)
            self.assertEqual(session.speed_multiplier, speed)

        with self.assertRaises(ValueError):
            session.playback("pause", 3.0)

    def test_project_round_trip_preserves_floor_cells(self):
        session = self.make_session()
        session.edit_cells([[4, 5]], 1, "STAIRS")
        project = session.world.to_dict()

        restored = self.make_session()
        restored.load_project(project)

        self.assertEqual(
            restored.world.floors[0].map_grid.get(4, 5), CellType.STAIRS
        )
        self.assertEqual(len(restored.get_state_snapshot()["floors"]), 1)


if __name__ == "__main__":
    unittest.main()
