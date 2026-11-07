import pygame

from controls.action import Action
from controls.action_controller import get_actions
from editor.editor import handle_map_editor
from editor.tool import Tool
from grids.coordinates import pixel_to_grid


class ProgramState:
    name = "state"
    render_options_enabled = False
    floor_switch_enabled = False
    selected_tool = None
    key_bindings = ()

    def enter(self, program):
        pass

    def exit(self, program):
        pass

    def handle_key(self, program, key):
        pass

    def handle_mouse(self, program, event):
        pass

    def update(self, program):
        pass

    def status_lines(self, context):
        mode = (
            "CNN-DQN + cluster route"
            if context.simulation_options.autonomous_drones
            else "manual"
        )
        evacuee_mode = (
            "RL followers"
            if context.simulation_options.rl_evacuees
            else "manual"
        )
        evacuated = context.simulation.evacuated_count
        total = evacuated + len(context.simulation.evacuees)
        floor = context.simulation.grid_world.active_floor_id + 1
        floor_count = len(context.simulation.grid_world.floors)

        lines = [
            f"State: {self.name}",
            f"Floor: {floor}/{floor_count}",
            f"Drones: {mode}",
            f"Evacuees: {evacuee_mode}",
            f"Step: {context.simulation.step_count}",
            f"Evacuated: {evacuated}/{total}",
        ]
        world = context.simulation.grid_world
        if context.link_stairs_mode and world.pending_stair_endpoint is not None:
            pending_floor = world.pending_stair_endpoint[0] + 1
            lines.append(f"Stair link selected on floor {pending_floor}")
        if world.stair_link_error:
            lines.append(world.stair_link_error)
        return lines

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)

    def handle_key_event(self, program, key):
        world = program.simulation.grid_world
        if self.floor_switch_enabled:
            if key == pygame.K_UP:
                world.set_active_floor(
                    min(len(world.floors) - 1, world.active_floor_id + 1)
                )
                return
            if key == pygame.K_DOWN:
                world.set_active_floor(max(0, world.active_floor_id - 1))
                return
        if key == pygame.K_g:
            program.view_options.show_grid = not program.view_options.show_grid
            return
        if self.render_options_enabled:
            if key == pygame.K_c:
                program.view_options.show_clusters = not program.view_options.show_clusters
                return
            if key == pygame.K_b:
                program.view_options.show_cluster_connections = not program.view_options.show_cluster_connections
                return
            if key == pygame.K_v:
                program.view_options.show_drone_routes = not program.view_options.show_drone_routes
                return
        self.handle_key(program, key)


class EditorState(ProgramState):
    name = "editor"
    key_bindings = (
        "E: overview",
        "1-5: select tool",
        "Left click: place",
        "Right click: erase",
        "G: toggle grid",
    )

    def __init__(self):
        self.selected_tool = Tool.WALL

    def enter(self, program):
        program.view_options.show_clusters = False
        program.view_options.show_cluster_connections = False
        program.view_options.show_drone_routes = False

    def exit(self, program):
        program.simulation.grid_world.rebuild_changed_map_data()

    def handle_key(self, program, key):
        if key == pygame.K_e:
            program.transition_to(OverviewState())
        elif key == pygame.K_1:
            self.selected_tool = Tool.WALL
        elif key == pygame.K_2:
            self.selected_tool = Tool.EVACUEE
        elif key == pygame.K_3:
            self.selected_tool = Tool.EXIT
        elif key == pygame.K_4:
            self.selected_tool = Tool.DRONE
        elif key == pygame.K_5:
            self.selected_tool = Tool.STAIRS

    def handle_mouse(self, program, event):
        handle_map_editor(
            program.simulation,
            event.pos,
            event.button,
            self.selected_tool,
        )

    def render(self, program):
        program.window.draw_editor(program, self.selected_tool, self.key_bindings)


class OverviewState(ProgramState):
    name = "overview"
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "P: start simulation",
        "E: edit map",
        "Up/Down: change floor",
        "L: link stairs, click one cluster on each floor",
        "C: toggle clusters",
        "B: toggle connections",
        "V: toggle drone routes",
        "G: toggle grid",
    )

    def enter(self, program):
        program.simulation.prepare()
        program.drone_routes = {}
        program.drone_controller.pending_transitions.clear()
        program.drone_controller.route_cursors.clear()
        program.rl_evacuee_controller.reset()
        program.plan_drone_routes()

    def handle_key(self, program, key):
        if key == pygame.K_e:
            program.transition_to(EditorState())
        elif key == pygame.K_p:
            program.transition_to(SimulationRunningState())
        elif key == pygame.K_l:
            program.link_stairs_mode = not program.link_stairs_mode
            program.simulation.grid_world.pending_stair_endpoint = None
            program.simulation.grid_world.stair_link_error = None

    def handle_mouse(self, program, event):
        if not program.link_stairs_mode:
            return

        world = program.simulation.grid_world
        if event.button == 3:
            world.pending_stair_endpoint = None
            world.stair_link_error = None
            return
        if event.button != 1:
            return

        grid_x, grid_y = pixel_to_grid(*event.pos)
        endpoint = world.stair_cluster_at(world.active_floor_id, grid_x, grid_y)
        if endpoint is None:
            world.stair_link_error = "Select a stair cluster"
            return

        pending = world.pending_stair_endpoint
        if pending is None:
            world.pending_stair_endpoint = endpoint
            world.stair_link_error = None
        elif pending[0] == endpoint[0]:
            world.pending_stair_endpoint = endpoint
            world.stair_link_error = "Choose the other floor"
        else:
            try:
                world.connect_stair_clusters(pending, endpoint)
            except ValueError as error:
                world.stair_link_error = str(error)
            else:
                world.pending_stair_endpoint = None
                world.stair_link_error = None

    def render(self, program):
        program.window.draw_overview(program, self.key_bindings)


class SimulationRunningState(ProgramState):
    name = "simulation_running"
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "P: pause",
        "X: finish simulation",
        "WASD: move evacuees",
        "T: use stairs",
        "F: toggle drone mode",
        "R: toggle RL evacuees",
        "C/B/V: toggle overlays",
        "G: toggle grid",
    )

    def enter(self, program):
        program.simulation.start()
        if not program.drone_routes:
            program.plan_drone_routes()

    def handle_key(self, program, key):
        if key == pygame.K_p:
            program.transition_to(SimulationPausedState())
        elif key == pygame.K_x:
            program.transition_to(ResultsState())
        elif key == pygame.K_f:
            program.simulation_options.autonomous_drones = not program.simulation_options.autonomous_drones
        elif key == pygame.K_r:
            program.simulation_options.rl_evacuees = not program.simulation_options.rl_evacuees
        elif key == pygame.K_w:
            program.simulation_options.action = Action.UP
        elif key == pygame.K_s:
            program.simulation_options.action = Action.DOWN
        elif key == pygame.K_a:
            program.simulation_options.action = Action.LEFT
        elif key == pygame.K_d:
            program.simulation_options.action = Action.RIGHT
        elif key == pygame.K_t:
            program.simulation_options.action = Action.USE_STAIRS

    def update(self, program):
        options = program.simulation_options
        if options.rl_evacuees:
            evacuee_actions = program.rl_evacuee_controller.get_actions(program.simulation)
        else:
            evacuee_actions = get_actions(program.simulation.evacuees, options.action)

        if options.autonomous_drones:
            drone_actions = program.drone_controller.get_actions(
                program.simulation,
                program.drone_routes,
            )
        else:
            drone_actions = get_actions(program.simulation.drones, options.action)

        program.simulation.step(evacuee_actions, drone_actions)
        if options.action is Action.USE_STAIRS:
            options.action = Action.STAY
        if options.autonomous_drones:
            program.drone_controller.learn(program.simulation, program.drone_routes)


class SimulationPausedState(ProgramState):
    name = "simulation_paused"
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "P: resume",
        "M: simulation editor",
        "C: toggle clusters",
        "B: toggle connections",
        "V: toggle drone routes",
        "G: toggle grid",
    )

    def handle_key(self, program, key):
        if key == pygame.K_p:
            program.transition_to(SimulationRunningState())
        elif key == pygame.K_m:
            program.transition_to(SimulationEditorState())


class SimulationEditorState(ProgramState):
    name = "simulation_editor"
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "Enter: return to pause",
        "C: toggle clusters",
        "B: toggle connections",
        "V: toggle drone routes",
        "G: toggle grid",
    )

    def handle_key(self, program, key):
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(SimulationPausedState())


class ResultsState(ProgramState):
    name = "results"
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "Enter: return to overview",
        "C: toggle clusters",
        "B: toggle connections",
        "V: toggle drone routes",
        "G: toggle grid",
    )

    def handle_key(self, program, key):
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(OverviewState())