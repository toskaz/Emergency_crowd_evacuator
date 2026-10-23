import pygame

from controls.action import Action
from controls.action_controller import get_actions
from editor.editor import handle_map_editor
from editor.tool import Tool


class ProgramState:
    name = "state"
    uses_simulation_grid = False
    render_options_enabled = False
    show_tool = False
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

    def render(self, program):
        program.window.draw(
            program.simulation.grid_world,
            program.simulation.evacuees,
            program.simulation.drones,
            program.selected_tool,
            program.show_grid,
            program.status_lines(),
            program.show_drone_routes and self.render_options_enabled,
            program.drone_routes,
            program.show_cluster_overlay and self.render_options_enabled,
            program.show_cluster_connections and self.render_options_enabled,
            self.uses_simulation_grid,
            self.key_bindings,
            self.show_tool,
        )

    def handle_key_event(self, program, key):
        if key == pygame.K_g:
            program.show_grid = not program.show_grid
            return
        if self.render_options_enabled:
            if key == pygame.K_c:
                program.show_cluster_overlay = not program.show_cluster_overlay
                return
            if key == pygame.K_b:
                program.show_cluster_connections = not program.show_cluster_connections
                return
            if key == pygame.K_v:
                program.show_drone_routes = not program.show_drone_routes
                return
        self.handle_key(program, key)


class EditorState(ProgramState):
    name = "editor"
    show_tool = True
    key_bindings = (
        "E: overview",
        "1-4: select tool",
        "Left click: place",
        "Right click: erase",
        "G: toggle grid",
    )

    def enter(self, program):
        program.show_cluster_overlay = False
        program.show_cluster_connections = False
        program.show_drone_routes = False

    def handle_key(self, program, key):
        if key == pygame.K_e:
            program.transition_to(OverviewState())
        elif key == pygame.K_1:
            program.selected_tool = Tool.WALL
        elif key == pygame.K_2:
            program.selected_tool = Tool.EVACUEE
        elif key == pygame.K_3:
            program.selected_tool = Tool.EXIT
        elif key == pygame.K_4:
            program.selected_tool = Tool.DRONE

    def handle_mouse(self, program, event):
        handle_map_editor(
            program.simulation,
            event.pos,
            event.button,
            program.selected_tool,
        )


class OverviewState(ProgramState):
    name = "overview"
    render_options_enabled = True
    key_bindings = (
        "P: start simulation",
        "E: edit map",
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
        program.plan_drone_routes()

    def handle_key(self, program, key):
        if key == pygame.K_e:
            program.transition_to(EditorState())
        elif key == pygame.K_p:
            program.transition_to(SimulationRunningState())


class SimulationRunningState(ProgramState):
    name = "simulation_running"
    uses_simulation_grid = True
    render_options_enabled = True
    key_bindings = (
        "P: pause",
        "X: finish simulation",
        "WASD: move evacuees",
        "F: toggle drone mode",
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
            program.autonomous_drones = not program.autonomous_drones
        elif key == pygame.K_w:
            program.action = Action.UP
        elif key == pygame.K_s:
            program.action = Action.DOWN
        elif key == pygame.K_a:
            program.action = Action.LEFT
        elif key == pygame.K_d:
            program.action = Action.RIGHT

    def update(self, program):
        evacuee_actions = get_actions(program.simulation.evacuees, program.action)

        if program.autonomous_drones:
            drone_actions = program.drone_controller.get_actions(
                program.simulation,
                program.drone_routes,
            )
        else:
            drone_actions = get_actions(program.simulation.drones, program.action)

        program.simulation.step(evacuee_actions, drone_actions)
        if program.autonomous_drones:
            program.drone_controller.learn(program.simulation, program.drone_routes)


class SimulationPausedState(ProgramState):
    name = "simulation_paused"
    uses_simulation_grid = True
    render_options_enabled = True
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
    uses_simulation_grid = True
    render_options_enabled = True
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
    uses_simulation_grid = True
    render_options_enabled = True
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