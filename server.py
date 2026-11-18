import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from controls.drone_dqn import DQNDroneController
from controls.evacuee_rl import QLearningFollowController, find_leaders
from controls.global_coordinator import GlobalCoordinator
from controls.scripted_drone import ScriptedDroneController
from editor.editor import grid_line, handle_map_editor_cell
from editor.tool import Tool
from grids.cluster import ClusterType
from grids.grid_world import Grid_World
from grids.map_grid import Map_Grid
from map.cell import CellType
from map.image_extractor import extract_map_image
from simulation import Simulation
from utils import config


logger = logging.getLogger(__name__)

BASE_TICK_MS = 200
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SUPPORTED_SPEEDS = (0.5, 0.75, 1.0, 2.0, 4.0)
VALID_MODES = {"editor", "overview", "simulate", "results"}
UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UNDO_STEPS = 30
TOOL_CELL_TYPES = {
    Tool.WALL: CellType.WALL,
    Tool.EXIT: CellType.EXIT,
    Tool.STAIRS: CellType.STAIRS,
    Tool.EVACUEE: CellType.EVACUEE,
    Tool.DRONE: CellType.DRONE,
}


class SimulationSession:
    """Owns the simulation and its controllers without creating a Pygame window."""

    def __init__(self, *, load_default_map=True):
        self.simulation = Simulation()
        if load_default_map:
            floorplan = self._extract_floorplan(config.image_path)
        else:
            floorplan = Map_Grid(32, 24, CellType.EMPTY)
        self.simulation.grid_world.add_floor_from_map(floorplan)

        self.evacuee_controller = QLearningFollowController()
        self.evacuee_controller.load(config.q_table_path)
        drone_controller = DQNDroneController(training=False)
        self.drone_model_loaded = drone_controller.load(config.drone_model_path)
        self.drone_controller = (
            drone_controller
            if self.drone_model_loaded
            else ScriptedDroneController()
        )
        self.coordinator = GlobalCoordinator(self.simulation.grid_world)
        self.drone_routes = {}

        self.mode = "editor"
        self.speed_multiplier = 1.0
        self.playing = False
        self.undo_stack = {}
        self.total_evacuees = 0
        self._entity_ids = {"evacuee": {}, "drone": {}}
        self._next_entity_ids = {"evacuee": 0, "drone": 0}
        self.simulation.prepare()
        self._reset_entity_ids()
        self._plan_drone_routes()
        self._update_total_evacuees()

    @staticmethod
    def _extract_floorplan(image_path):
        extracted = extract_map_image(image_path, config.block_size)
        floorplan = Map_Grid(extracted.width, extracted.height, CellType.EMPTY)
        floorplan.cells = extracted.cells
        return floorplan

    @property
    def world(self) -> Grid_World:
        return self.simulation.grid_world

    def _plan_drone_routes(self):
        self.drone_routes = self.coordinator.plan_initial_routes(
            self.simulation.drones
        )

    def _replan_after_environment_change(self):
        if self.simulation.drones:
            self.drone_routes, _ = self.coordinator.on_environment_changed(
                self.simulation.drones
            )
        else:
            self.drone_routes = {}
            self.coordinator.cell_routes.clear()

    def _reset_controllers(self):
        self.evacuee_controller.reset()
        self.drone_controller.reset()
        self._reset_entity_ids()
        self._plan_drone_routes()

    def _reset_entity_ids(self):
        self._entity_ids = {
            "evacuee": {
                entity: index
                for index, entity in enumerate(self.simulation.evacuees)
            },
            "drone": {
                entity: index
                for index, entity in enumerate(self.simulation.drones)
            },
        }
        self._next_entity_ids = {
            kind: len(entities) for kind, entities in self._entity_ids.items()
        }

    def _entity_id(self, kind, entity):
        ids = self._entity_ids[kind]
        if entity not in ids:
            ids[entity] = self._next_entity_ids[kind]
            self._next_entity_ids[kind] += 1
        return ids[entity]

    def _update_total_evacuees(self):
        self.total_evacuees = (
            self.simulation.evacuated_count + len(self.simulation.evacuees)
        )

    def set_mode(self, mode):
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if mode == "editor" and self.mode != "editor":
            self.playing = False
            self.world.clear_temporary_walls()
            self.simulation.prepare()
            self._reset_controllers()
            self._update_total_evacuees()
        elif mode != "simulate":
            self.playing = False
        self.mode = mode

    def reset(self):
        self.playing = False
        self.world.clear_temporary_walls()
        if self.mode == "editor":
            self.world.restore_base_maps()
            self.undo_stack.clear()
        self.world.pending_stair_endpoint = None
        self.world.stair_link_error = None
        self.world.stair_link_notice = None
        self.simulation.prepare()
        self._reset_controllers()
        self._update_total_evacuees()

    def _prepare_if_needed(self):
        if not self.simulation.prepared:
            self.simulation.prepare()
            self._plan_drone_routes()
            self._update_total_evacuees()

    def _step_once(self, *, keep_paused):
        self._prepare_if_needed()
        cell_routes = self.coordinator.refresh_cell_routes(
            self.simulation.drones
        )
        was_started = self.simulation.started

        evacuee_actions = self.evacuee_controller.get_actions(self.simulation)
        if self.drone_model_loaded:
            drone_actions = self.drone_controller.get_actions(
                self.simulation, self.drone_routes, cell_routes
            )
        else:
            drone_actions = self.drone_controller.get_actions(
                self.simulation, cell_routes
            )
        self.simulation.step(evacuee_actions, drone_actions)
        if keep_paused and not was_started:
            self.simulation.started = False
        if self.drone_model_loaded:
            self.drone_controller.learn(self.simulation, self.drone_routes)
        self._update_total_evacuees()
        if (
            self.total_evacuees > 0
            and self.simulation.evacuated_count >= self.total_evacuees
        ):
            self.playing = False
            self.mode = "results"

    def playback(self, action, speed=None):
        if speed is not None:
            speed = float(speed)
            if speed not in SUPPORTED_SPEEDS:
                raise ValueError(f"Unsupported speed multiplier: {speed}")
            self.speed_multiplier = speed

        if action == "start":
            self._prepare_if_needed()
            self.simulation.start()
            self.playing = True
        elif action == "pause":
            self.playing = False
        elif action == "step":
            self.playing = False
            self._step_once(keep_paused=True)
        elif action == "reset":
            self.reset()
        else:
            raise ValueError(f"Unsupported playback action: {action}")

    def add_floor_from_image(self, image_path):
        self.playing = False
        floorplan = self._extract_floorplan(image_path)
        self.world.add_floor_from_map(floorplan)
        self.world.rebuild_changed_map_data()
        floor_id = len(self.world.floors) - 1
        self.undo_stack.setdefault(floor_id, [])
        self.world.set_active_floor(floor_id)
        self.simulation.prepare()
        self._reset_controllers()
        self._update_total_evacuees()
        return self.world.active_floor_id

    def load_project(self, data):
        if not isinstance(data, dict) or not isinstance(data.get("floors"), list):
            raise ValueError("Project JSON must contain a floors array")
        replacement = Simulation()
        replacement.grid_world.from_dict(data)
        replacement.grid_world.rebuild_changed_map_data()
        if not replacement.grid_world.floors:
            raise ValueError("Project must contain at least one floor")
        self.playing = False
        self.simulation = replacement
        self.undo_stack = {}
        self.coordinator = GlobalCoordinator(self.world)
        self.simulation.prepare()
        self._reset_controllers()
        self._update_total_evacuees()
        self.mode = "editor"

    def edit_cells(self, cells, button, tool_name):
        if self.mode != "editor":
            raise ValueError("Map cells can only be edited in Edit Map mode")
        if button not in (1, 3):
            raise ValueError("Edit button must be 1 (paint) or 3 (erase)")
        try:
            tool = Tool[tool_name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unsupported map tool: {tool_name}") from error

        normalized = []
        for cell in cells:
            self._validate_cell(cell)
            x, y = cell
            normalized.append((x, y))

        interpolated = []
        seen = set()
        for index, cell in enumerate(normalized):
            line = grid_line(normalized[index - 1], cell) if index else (cell,)
            for point in line:
                if point not in seen:
                    interpolated.append(point)
                    seen.add(point)

        floor_id = self.world.active_floor_id
        target_type = (
            CellType.EMPTY if button == 3 else TOOL_CELL_TYPES[tool]
        )
        changed_cells = [
            (x, y)
            for x, y in interpolated
            if self.world.map_grid.in_bounds(x, y)
            and self.world.map_grid.get(x, y) != target_type
        ]
        if changed_cells:
            stack = self.undo_stack.setdefault(floor_id, [])
            stack.append(self.world.map_grid.copy())
            del stack[:-MAX_UNDO_STEPS]

        changed_count = 0
        for x, y in changed_cells:
            if not self.world.map_grid.in_bounds(x, y):
                continue
            before = self.world.map_grid.get(x, y)
            if handle_map_editor_cell(
                self.simulation, x, y, button, tool, rebuild_clusters=False
            ) or before != self.world.map_grid.get(x, y):
                changed_count += 1

        map_changed = bool(self.world.changed_floor_ids)
        self.world.rebuild_changed_map_data()
        if map_changed:
            self.simulation.prepare()
            self._reset_controllers()
            self._update_total_evacuees()
        return changed_count

    def _resolve_stair_node(self, floor_id, endpoint):
        if (
            not isinstance(floor_id, int)
            or isinstance(floor_id, bool)
            or floor_id not in self.world.floors
        ):
            raise ValueError("Stair endpoint must reference an existing floor")
        if isinstance(endpoint, (list, tuple)):
            self._validate_cell(endpoint)
            if not self.world.floors[floor_id].map_grid.in_bounds(*endpoint):
                raise ValueError("Stair endpoint is outside its floor")
            node = self.world.stair_cluster_at(floor_id, *endpoint)
        else:
            if not isinstance(endpoint, int) or isinstance(endpoint, bool):
                raise ValueError("Stair cluster ID must be an integer")
            node = (floor_id, endpoint)
            cluster = self.world._get_stair_cluster(node)
            if cluster is None or cluster.type is not ClusterType.STAIRS:
                node = None
        if node is None:
            raise ValueError("Both endpoints must be on a STAIRS cluster")
        return node

    def link_stairs(self, floor_a, endpoint_a, floor_b, endpoint_b):
        node_a = self._resolve_stair_node(floor_a, endpoint_a)
        node_b = self._resolve_stair_node(floor_b, endpoint_b)
        self.world.connect_stair_clusters(node_a, node_b)
        self._plan_drone_routes()

    def undo(self):
        if self.mode != "editor":
            raise ValueError("Map edits can only be undone in Edit Map mode")
        floor_id = self.world.active_floor_id
        stack = self.undo_stack.get(floor_id, [])
        if not stack:
            return False
        floor = self.world.floors[floor_id]
        floor.map_grid = stack.pop()
        self.world.changed_floor_ids.add(floor_id)
        self.world.rebuild_changed_map_data()
        self.simulation.prepare()
        self._reset_controllers()
        self._update_total_evacuees()
        return True

    @staticmethod
    def _validate_cell(cell):
        if (
            not isinstance(cell, (list, tuple))
            or len(cell) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in cell
            )
        ):
            raise ValueError("Cell coordinates must be integer [x, y] pairs")

    def unlink_stairs(self, floor_id, cluster_id):
        cluster = self.world._get_stair_cluster((floor_id, cluster_id))
        if cluster is None or cluster.linked_to is None:
            return False
        linked_node = cluster.linked_to
        cluster.linked_to = None
        linked_cluster = self.world._get_stair_cluster(linked_node)
        if linked_cluster is not None and linked_cluster.linked_to == (floor_id, cluster_id):
            linked_cluster.linked_to = None
        self.world.rebuild_navigation()
        self._plan_drone_routes()
        return True

    def toggle_hazard(self, x, y):
        self._validate_cell([x, y])
        floor_id = self.world.active_floor_id
        if not self.world.map_grid.in_bounds(x, y):
            return False
        enabled = not self.world.temporary_wall_grid.get(x, y)
        if not self.world.set_temporary_wall(floor_id, x, y, enabled):
            return False
        self._replan_after_environment_change()
        return True

    def spawn_evacuee(self, x, y):
        from entities.evacuee import Evacuee

        self._validate_cell([x, y])
        floor = self.world.active_floor
        if (
            not floor.map_grid.in_bounds(x, y)
            or floor.map_grid.get(x, y) != CellType.EMPTY
            or floor.temporary_wall_grid.get(x, y)
            or floor.evacuee_occupancy_grid.get(x, y) is not None
            or floor.drone_occupancy_grid.get(x, y) is not None
        ):
            return False
        evacuee = Evacuee(x, y, floor.id)
        self.simulation.evacuees.append(evacuee)
        self.world.add_evacuee(evacuee)
        self._entity_id("evacuee", evacuee)
        self.world.rebuild_simulation_grid(
            self.simulation.evacuees, self.simulation.drones
        )
        self.total_evacuees += 1
        return True

    def _evacuee_state(self, evacuee, leaders):
        floor = self.world.floors[evacuee.floor_id]
        distance = floor.distance_map[evacuee.grid_y][evacuee.grid_x]
        if distance != float("inf") and distance <= 10:
            return "evacuating"
        if evacuee in leaders or evacuee in self.evacuee_controller.last_seen:
            return "following"
        return "panicking"

    def _drone_route(self, drone):
        points = []
        for floor_id, x, y in self.coordinator.cell_routes.get(drone, ()):
            if floor_id != drone.floor_id:
                break
            points.append([x, y])
        return points

    def get_state_snapshot(self):
        floors = []
        active_hazards = 0
        for floor_id, floor in self.world.floors.items():
            temporary_walls = [
                [x, y]
                for y, row in enumerate(floor.temporary_wall_grid.cells)
                for x, enabled in enumerate(row)
                if enabled
            ]
            active_hazards += len(temporary_walls)
            clusters = []
            adjacent_counts = [0] * len(floor.cluster_grid.clusters)
            for first, second in floor.cluster_grid.adjacent_cluster_pairs:
                adjacent_counts[first] += 1
                adjacent_counts[second] += 1
            for cluster_index, cluster in enumerate(floor.cluster_grid.clusters):
                linked_to = getattr(cluster, "linked_to", None)
                is_doorway = (
                    cluster.type is ClusterType.DOOR
                    or (
                        cluster.type not in (ClusterType.EXIT, ClusterType.STAIRS)
                        and len(cluster.cells) <= 6
                        and adjacent_counts[cluster_index] >= 2
                    )
                )
                clusters.append({
                    "id": int(getattr(cluster, "id", cluster_index)),
                    "type": cluster.type.name,
                    "is_doorway": is_doorway,
                    "cells": [
                        [int(x), int(y)]
                        for x, y in cluster.cells
                    ],
                    "linked_to": (
                        [int(linked_to[0]), int(linked_to[1])]
                        if linked_to is not None
                        else None
                    ),
                })
            floors.append({
                "id": floor_id,
                "name": f"Floor {floor_id + 1}",
                "width": floor.map_grid.width,
                "height": floor.map_grid.height,
                "cells": [
                    [int(floor.map_grid.get(x, y)) for x in range(floor.map_grid.width)]
                    for y in range(floor.map_grid.height)
                ],
                "temporary_walls": temporary_walls,
                "clusters": clusters,
            })

        leaders = find_leaders(self.simulation)
        evacuees = [
            {
                "id": self._entity_id("evacuee", evacuee),
                "x": evacuee.grid_x,
                "y": evacuee.grid_y,
                "floor_id": evacuee.floor_id,
                "state": self._evacuee_state(evacuee, leaders),
            }
            for evacuee in self.simulation.evacuees
        ]
        drones = [
            {
                "id": self._entity_id("drone", drone),
                "x": drone.grid_x,
                "y": drone.grid_y,
                "floor_id": drone.floor_id,
                "route": self._drone_route(drone),
            }
            for drone in self.simulation.drones
        ]
        evacuated = self.simulation.evacuated_count
        total = max(self.total_evacuees, evacuated + len(evacuees))
        history = [
            {"step": int(step), "evacuated": int(count)}
            for step, count in self.simulation.evacuation_history
        ]
        return {
            "mode": self.mode,
            "active_floor_id": self.world.active_floor_id,
            "can_undo": bool(self.undo_stack.get(self.world.active_floor_id)),
            "floors": floors,
            "evacuees": evacuees,
            "drones": drones,
            "telemetry": {
                "step_count": self.simulation.step_count,
                "evacuated_count": evacuated,
                "total_evacuees": total,
                "remaining_count": len(evacuees),
                "evacuation_pct": round(
                    100.0 * evacuated / total if total else 0.0, 1
                ),
                "average_evacuation_time": round(
                    sum(self.simulation.evacuation_times)
                    / len(self.simulation.evacuation_times),
                    1,
                )
                if self.simulation.evacuation_times
                else None,
                "active_hazards": active_hazards,
                "speed_multiplier": self.speed_multiplier,
                "started": self.simulation.started,
                "paused": not self.playing,
                "history": history,
            },
        }

    def handle_command(self, command):
        if not isinstance(command, dict):
            raise ValueError("WebSocket commands must be JSON objects")
        command_type = command.get("type")
        if command_type == "set_mode":
            self.set_mode(command.get("mode"))
        elif command_type == "edit_cells":
            self.edit_cells(
                command.get("cells", []),
                command.get("button"),
                command.get("tool"),
            )
        elif command_type == "link_stairs":
            self.link_stairs(
                command["floor_a"],
                command.get("cluster_a", command.get("cell_a")),
                command["floor_b"],
                command.get("cluster_b", command.get("cell_b")),
            )
        elif command_type == "unlink_stairs":
            self.unlink_stairs(command["floor_id"], command["cluster_id"])
        elif command_type == "undo":
            self.undo()
        elif command_type == "toggle_hazard":
            self.toggle_hazard(command["x"], command["y"])
        elif command_type == "spawn_evacuee":
            self.spawn_evacuee(command["x"], command["y"])
        elif command_type == "playback":
            self.playback(command.get("action"), command.get("speed"))
        else:
            raise ValueError(f"Unsupported command type: {command_type}")
        return self.get_state_snapshot()


session = SimulationSession()
clients = set()
session_lock = asyncio.Lock()
background_task = None


async def broadcast_state(snapshot=None):
    message = {"type": "state", "state": snapshot or session.get_state_snapshot()}
    disconnected = []
    for client in tuple(clients):
        try:
            await client.send_json(message)
        except (RuntimeError, WebSocketDisconnect):
            disconnected.append(client)
    for client in disconnected:
        clients.discard(client)


async def simulation_loop():
    while True:
        delay = BASE_TICK_MS / 1000 / session.speed_multiplier
        await asyncio.sleep(delay)
        if not session.playing:
            continue
        try:
            snapshot = None
            async with session_lock:
                if session.playing:
                    session._step_once(keep_paused=False)
                    snapshot = session.get_state_snapshot()
            await broadcast_state(snapshot)
        except Exception:
            logger.exception("Simulation tick failed; pausing playback")
            session.playing = False
            await broadcast_state()


@asynccontextmanager
async def lifespan(_app):
    global background_task
    background_task = asyncio.create_task(simulation_loop())
    yield
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Emergency Crowd Evacuator", lifespan=lifespan)
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
async def get_state():
    async with session_lock:
        return session.get_state_snapshot()


@app.post("/api/floors/upload")
async def upload_floor(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(415, "Upload a PNG, JPG, or JPEG floorplan")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Floorplan upload exceeds 20 MB")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        async with session_lock:
            floor_id = session.add_floor_from_image(temp_path)
            snapshot = session.get_state_snapshot()
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
    await broadcast_state(snapshot)
    return {"floor_id": floor_id, "state": snapshot}


@app.post("/api/project/save")
async def save_project():
    async with session_lock:
        project = session.world.to_dict()
    return JSONResponse(
        project,
        headers={"Content-Disposition": 'attachment; filename="evacuation-project.json"'},
    )


@app.post("/api/project/load")
async def load_project(project: dict):
    try:
        async with session_lock:
            session.load_project(project)
            snapshot = session.get_state_snapshot()
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, f"Invalid project file: {error}") from error
    await broadcast_state(snapshot)
    return snapshot


@app.post("/api/floors/{floor_id}/select")
async def select_floor(floor_id: int):
    async with session_lock:
        if not session.world.set_active_floor(floor_id):
            raise HTTPException(404, "Floor not found")
        snapshot = session.get_state_snapshot()
    await broadcast_state(snapshot)
    return snapshot


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json({"type": "state", "state": session.get_state_snapshot()})
    try:
        while True:
            command = await websocket.receive_json()
            try:
                async with session_lock:
                    snapshot = session.handle_command(command)
                await broadcast_state(snapshot)
            except (KeyError, TypeError, ValueError) as error:
                await websocket.send_json({
                    "type": "error",
                    "message": str(error),
                    "state": session.get_state_snapshot(),
                })
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
