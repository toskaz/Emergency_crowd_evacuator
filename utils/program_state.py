from google.protobuf import text_format
from networkx.generators import spectral_graph_forge
import platform
import subprocess
import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pygame
import pygame_gui

from controls.action import Action
from controls.action_controller import get_actions
from editor.editor import handle_map_editor
from editor.tool import Tool
from grids.coordinates import pixel_to_grid
from grids.map_grid import Map_Grid
from simulation import Simulation
from utils.config import block_size, acceptance_threshold

MENU_WINDOW_SIZE = (500, 350)
CALIBRATION_WINDOW_SIZE = (1024, 768)


logger = logging.getLogger(__name__)


def _open_file_dialog(title, filetypes):
    # Use native macOS dialog to avoid Pygame/Tkinter thread crash on macOS
    if platform.system() == "Darwin":
        script = f'POSIX path of (choose file with prompt "{title}")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "" # User cancelled dialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

def _save_file_dialog(title, default_ext, filetypes):
    if platform.system() == "Darwin":
        script = f'POSIX path of (choose file name with prompt "{title}" default name "project{default_ext}")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "" # User cancelled dialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title=title, defaultextension=default_ext, filetypes=filetypes
    )
    root.destroy()
    return path


def _save_project(context):
    path = _save_file_dialog(
        "Zapisz projekt",
        ".json",
        [("JSON files", "*.json")],
    )
    if not path:
        return
    context.simulation.grid_world.rebuild_changed_map_data()
    data = context.simulation.grid_world.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Project saved to %s", path)


class ProgramState:
    name = "state"
    render_options_enabled = False
    floor_switch_enabled = False
    selected_tool = None
    key_bindings = ()
    can_pan = False

    def enter(self, program):
        pass

    def exit(self, program):
        pass

    def handle_key(self, program, key):
        pass

    def handle_mouse(self, program, event):
        pass

    def handle_ui_event(self, program, event):
        pass

    def update(self, program):
        if self.can_pan:
            import utils.config as config
            keys = pygame.key.get_pressed()
            pan_speed = 15
            if keys[pygame.K_LEFT]: config.camera_offset_x += pan_speed
            if keys[pygame.K_RIGHT]: config.camera_offset_x -= pan_speed
            if keys[pygame.K_UP]: config.camera_offset_y += pan_speed
            if keys[pygame.K_DOWN]: config.camera_offset_y -= pan_speed

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
        floor_label = (
            f"Piętro: {floor} / {floor_count} (Góra/Dół: zmiana)"
            if floor_count > 1
            else f"Piętro: {floor} / {floor_count}"
        )

        lines = [
            "Skala: 1 px (siatki) = ~60 cm",
            f"State: {self.name}",
            floor_label,
            f"Drones: {mode}",
            f"Evacuees: {evacuee_mode}",
            f"Step: {context.simulation.step_count}",
            f"Evacuated: {evacuated}/{total}",
        ]
        world = context.simulation.grid_world
        if context.link_stairs_mode:
            if world.pending_stair_endpoint is not None:
                pending_floor = world.pending_stair_endpoint[0] + 1
                lines.append(f"★ Schody na P{pending_floor} wybrane. Zmień piętro (Góra/Dół) i kliknij drugie!")
            else:
                lines.append("★ Tryb łączenia: Kliknij pierwsze schody (L: wyłącz, PPM: anuluj)")
        if world.stair_link_error:
            lines.append(f"Błąd: {world.stair_link_error}")
        return lines

    def on_floor_changed(self, program):
        pass

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)

    def handle_key_event(self, program, key):
        world = program.simulation.grid_world if program.simulation else None
        import utils.config as config

        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            if getattr(self, 'can_pan', False):
                config.drawing_size = min(100, config.drawing_size + 2)
            return
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            if getattr(self, 'can_pan', False):
                config.drawing_size = max(2, config.drawing_size - 2)
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


class MainMenuState(ProgramState):
    name = "main_menu"

    def enter(self, program):
        self._import_button = None
        self._open_button = None
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

        button_width = 280
        button_height = 50
        gap = 20
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        center_x = (screen_w - button_width) // 2
        center_y = (screen_h - (2 * button_height + gap)) // 2

        self._import_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(center_x, center_y, button_width, button_height),
            text="Importuj zdjęcie",
            manager=program.ui_manager,
        )
        self._open_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                center_x, center_y + button_height + gap, button_width, button_height
            ),
            text="Otwórz projekt (.json)",
            manager=program.ui_manager,
        )

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        self._import_button = None
        self._open_button = None

    def handle_ui_event(self, program, event):
        if event.ui_element == self._import_button:
            path = _open_file_dialog(
                "Wybierz zdjęcie planu",
                [("Images", "*.jpg *.png *.jpeg")],
            )
            if path:
                program.imported_image_path = path
                program.transition_to(CalibrationState())

        elif event.ui_element == self._open_button:
            path = _open_file_dialog(
                "Otwórz projekt",
                [("JSON files", "*.json")],
            )
            if path:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                program.initialize_world_from_dict(data)
                program.transition_to(OverviewState())

    def handle_key_event(self, program, key):
        pass

    def update(self, program):
        pass

    def render(self, program):
        program.screen.fill((30, 30, 30))

    def status_lines(self, context):
        return ["State: main_menu"]


class CalibrationState(ProgramState):
    name = "calibration"

    def enter(self, program):
        self.image_path = program.imported_image_path
        if not self.image_path:
            logger.warning("No image path provided for calibration; returning to main menu")
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
            return

        self.screen_size = CALIBRATION_WINDOW_SIZE
        program.screen = pygame.display.set_mode(self.screen_size)
        if program.ui_manager is not None:
            program.ui_manager.set_window_resolution(self.screen_size)
            program.ui_manager.clear_and_reset()

        self.control_panel_height = 148
        self.control_panel_rect = pygame.Rect(
            0,
            self.screen_size[1] - self.control_panel_height,
            self.screen_size[0],
            self.control_panel_height,
        )

        self.current_block_size = 25
        self._slider = None
        self._info_label = None
        self._sub_label = None
        self._generate_button = None
        self._add_floor_button = None
        self._back_button = None

        self.raw_image = None
        self.display_image = None
        self.overlay_surface = None

        self._load_image(self.image_path)
        self._build_ui(program)
        self._update_grid_overlay(self.current_block_size)

    def _load_image(self, path):
        self.raw_image = pygame.image.load(path).convert()
        self.img_w, self.img_h = self.raw_image.get_size()

        margin = 14
        avail_w = self.screen_size[0] - (2 * margin)
        avail_h = (self.screen_size[1] - self.control_panel_height) - (2 * margin)

        scale = min(avail_w / self.img_w, avail_h / self.img_h)
        self.disp_scale = scale
        self.disp_w = max(1, int(self.img_w * scale))
        self.disp_h = max(1, int(self.img_h * scale))

        self.disp_x = (self.screen_size[0] - self.disp_w) // 2
        preview_area_h = self.screen_size[1] - self.control_panel_height
        self.disp_y = (preview_area_h - self.disp_h) // 2

        self.display_image = pygame.transform.smoothscale(
            self.raw_image, (self.disp_w, self.disp_h)
        )

    def _build_ui(self, program):
        if program.ui_manager is None:
            return

        program.ui_manager.clear_and_reset()

        top_y = self.control_panel_rect.top + 14

        self._info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(24, top_y, 520, 26),
            text="Rozmiar bloku: 25px | Wymiary siatki: -- x --",
            manager=program.ui_manager,
        )

        self._slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(24, top_y + 32, 520, 28),
            start_value=self.current_block_size,
            value_range=(10, 100),
            manager=program.ui_manager,
            click_increment=1,
        )

        floor_count = len(program.simulation.grid_world.floors) if program.simulation else 0
        file_name = Path(self.image_path).name
        self._sub_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(24, top_y + 68, 520, 22),
            text=f"Plik: {file_name}  |  Obraz: {self.img_w}x{self.img_h}px  |  Zapisane piętra: {floor_count}",
            manager=program.ui_manager,
        )

        btn_w = 210
        btn_h = 42
        gap = 16
        col1_x = 565
        col2_x = col1_x + btn_w + gap

        self._add_floor_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(col1_x, top_y + 12, btn_w, btn_h),
            text="Dodaj kolejne piętro z obrazka",
            manager=program.ui_manager,
        )

        self._generate_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(col2_x, top_y + 12, btn_w, btn_h),
            text="Generuj piętro i otwórz edytor",
            manager=program.ui_manager,
        )

        self._back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(col2_x, top_y + 62, btn_w, 28),
            text="Wróć do menu",
            manager=program.ui_manager,
        )

    def _update_grid_overlay(self, block_size):
        self.overlay_surface = pygame.Surface(
            (self.disp_w, self.disp_h), pygame.SRCALPHA
        )
        self.overlay_surface.fill((0, 0, 0, 0))

        grid_w = (self.img_w + block_size - 1) // block_size
        grid_h = (self.img_h + block_size - 1) // block_size

        from map.cell import CellType
        preview_grid = Map_Grid.from_image(
            self.image_path, block_size, acceptance_threshold=acceptance_threshold
        )

        for y in range(preview_grid.height):
            for x in range(preview_grid.width):
                cell_type = preview_grid.get(x, y)
                if cell_type == CellType.WALL:
                    color = (0, 0, 255, 180)  
                elif cell_type == CellType.EXIT:
                    color = (0, 255, 0, 180)  
                else:
                    continue

                rect_x = int(x * block_size * self.disp_scale)
                rect_y = int(y * block_size * self.disp_scale)
                next_x = int((x + 1) * block_size * self.disp_scale)
                next_y = int((y + 1) * block_size * self.disp_scale)

                rect_w = max(1, next_x - rect_x)
                rect_h = max(1, next_y - rect_y)

                if rect_x < self.disp_w and rect_y < self.disp_h:
                    rect_w = min(rect_w, self.disp_w - rect_x)
                    rect_h = min(rect_h, self.disp_h - rect_y)
                    pygame.draw.rect(self.overlay_surface, color, (rect_x, rect_y, rect_w, rect_h))

        line_color = (0, 190, 255, 95)
        border_color = (0, 210, 255, 180)

        for col in range(grid_w + 1):
            x = int(col * block_size * self.disp_scale)
            if x <= self.disp_w:
                pygame.draw.line(self.overlay_surface, line_color, (x, 0), (x, self.disp_h), 1)

        for row in range(grid_h + 1):
            y = int(row * block_size * self.disp_scale)
            if y <= self.disp_h:
                pygame.draw.line(self.overlay_surface, line_color, (0, y), (self.disp_w, y), 1)

        pygame.draw.rect(
            self.overlay_surface, border_color, (0, 0, self.disp_w, self.disp_h), 2
        )

        if self._info_label is not None:
            self._info_label.set_text(
                f"Rozmiar bloku: {block_size}px | Wymiary siatki: {grid_w} x {grid_h}"
            )

    def update(self, program):
        if self._slider is not None:
            val = int(round(self._slider.get_current_value()))
            val = max(10, min(100, val))
            if val != self.current_block_size:
                self.current_block_size = val
                self._update_grid_overlay(val)

    def render(self, program):
        program.screen.fill((22, 23, 27))

        if self.display_image is not None:
            program.screen.blit(self.display_image, (self.disp_x, self.disp_y))

        if self.overlay_surface is not None:
            program.screen.blit(self.overlay_surface, (self.disp_x, self.disp_y))

        pygame.draw.rect(program.screen, (32, 33, 40), self.control_panel_rect)
        pygame.draw.line(
            program.screen,
            (55, 58, 70),
            (0, self.control_panel_rect.top),
            (self.screen_size[0], self.control_panel_rect.top),
            1,
        )

    def handle_ui_event(self, program, event):
        if event.ui_element == self._generate_button:
            block_size = self.current_block_size
            map_grid = Map_Grid.from_image(
                self.image_path, block_size, acceptance_threshold=acceptance_threshold
            )
            program.initialize_world(map_grid)
            program.transition_to(EditorState())

        elif event.ui_element == self._add_floor_button:
            next_path = _open_file_dialog(
                "Wybierz zdjęcie kolejnego piętra",
                [("Images", "*.jpg *.png *.jpeg")],
            )
            if next_path:
                block_size = self.current_block_size
                map_grid = Map_Grid.from_image(
                    self.image_path, block_size, acceptance_threshold=acceptance_threshold
                )
                if program.simulation is None:
                    program.simulation = Simulation()
                program.simulation.grid_world.add_floor_from_map(map_grid)
                logger.info(
                    "Added floor %d from %s",
                    len(program.simulation.grid_world.floors) - 1,
                    self.image_path,
                )

                self.image_path = next_path
                program.imported_image_path = next_path
                self._load_image(next_path)
                self._update_grid_overlay(self.current_block_size)
                self._build_ui(program)

        elif event.ui_element == self._back_button:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())

    def handle_key_event(self, program, key):
        if key == pygame.K_ESCAPE:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif key == pygame.K_LEFT:
            if self._slider is not None:
                self._slider.set_current_value(max(10, self.current_block_size - 1))
        elif key == pygame.K_RIGHT:
            if self._slider is not None:
                self._slider.set_current_value(min(100, self.current_block_size + 1))
        elif key == pygame.K_RETURN:
            block_size = self.current_block_size
            map_grid = Map_Grid.from_image(
                self.image_path, block_size, acceptance_threshold=acceptance_threshold
            )
            program.initialize_world(map_grid)
            program.transition_to(EditorState())

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        self._slider = None
        self._info_label = None
        self._sub_label = None
        self._generate_button = None
        self._add_floor_button = None
        self._back_button = None
        self.raw_image = None
        self.display_image = None
        self.overlay_surface = None

    def status_lines(self, context):
        floor_count = (
            len(context.simulation.grid_world.floors) if context.simulation else 0
        )
        return [
            f"State: {self.name}",
            f"Block size: {self.current_block_size}px",
            f"Floors loaded: {floor_count}",
        ]


class EditorState(ProgramState):
    name = "editor"
    can_pan = True
    floor_switch_enabled = True
    key_bindings = (
        "Strzałki: przesuwanie mapy (Pan)",
        "G: toggle grid",
        "LPM: postaw, PPM: usuń",
    )

    def __init__(self):
        self.selected_tool = Tool.WALL
        #self._save_button = None
        #self._menu_button = None

    def enter(self, program):
        program.view_options.show_clusters = False
        program.view_options.show_cluster_connections = False
        program.view_options.show_drone_routes = False
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        screen_w, screen_h = program.screen.get_size()
        btn_w, btn_h = 180, 32
        panel_x = screen_w - btn_w - 15
        y = 220  
        
        self._btn_wall = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Narzędzie: Ściana", manager=program.ui_manager)
        y += btn_h + 4
        self._btn_evacuee = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Narzędzie: Ewakuowany", manager=program.ui_manager)
        y += btn_h + 4
        self._btn_exit = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Narzędzie: Wyjście", manager=program.ui_manager)
        y += btn_h + 4
        self._btn_drone = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Narzędzie: Dron", manager=program.ui_manager)
        y += btn_h + 4
        self._btn_stairs = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Narzędzie: Schody", manager=program.ui_manager)
        y += btn_h + 15
        self._btn_zoom_in = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w//2 - 2, btn_h), "Zoom +", manager=program.ui_manager)
        self._btn_zoom_out = pygame_gui.elements.UIButton(pygame.Rect(panel_x + btn_w//2 + 2, y, btn_w//2 - 2, btn_h), "Zoom -", manager=program.ui_manager)
        y += btn_h + 15
        self._btn_floor_up = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w//2 - 2, btn_h), "Piętro +", manager=program.ui_manager)
        self._btn_floor_down = pygame_gui.elements.UIButton(pygame.Rect(panel_x + btn_w//2 + 2, y, btn_w//2 - 2, btn_h), "Piętro -", manager=program.ui_manager)
        y += btn_h + 15
        self._btn_overview = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Overview", manager=program.ui_manager)
        y += btn_h + 4
        self._save_button = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Zapisz (Ctrl+S)", manager=program.ui_manager)
        y += btn_h + 4
        self._menu_button = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Menu", manager=program.ui_manager)

    def exit(self, program):
        program.simulation.grid_world.rebuild_changed_map_data()
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        #self._save_button = None
        #self._menu_button = None

    def handle_ui_event(self, program, event):
        if event.ui_element == self._save_button:
            _save_project(program)
        elif event.ui_element == self._btn_wall: self.selected_tool = Tool.WALL
        elif event.ui_element == self._btn_evacuee: self.selected_tool = Tool.EVACUEE
        elif event.ui_element == self._btn_exit: self.selected_tool = Tool.EXIT
        elif event.ui_element == self._btn_drone: self.selected_tool = Tool.DRONE
        elif event.ui_element == self._btn_stairs: self.selected_tool = Tool.STAIRS
        elif event.ui_element == self._btn_overview: program.transition_to(OverviewState())
        elif getattr(self, '_btn_floor_up', None) and event.ui_element == self._btn_floor_up:
            world = program.simulation.grid_world
            new_floor = min(len(world.floors) - 1, world.active_floor_id + 1)
            if new_floor != world.active_floor_id:
                world.set_active_floor(new_floor)
                program.update_window_for_active_floor()
                self.on_floor_changed(program)
        elif getattr(self, '_btn_floor_down', None) and event.ui_element == self._btn_floor_down:
            world = program.simulation.grid_world
            new_floor = max(0, world.active_floor_id - 1)
            if new_floor != world.active_floor_id:
                world.set_active_floor(new_floor)
                program.update_window_for_active_floor()
                self.on_floor_changed(program)
        elif event.ui_element == self._menu_button:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager: program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif getattr(self, '_btn_zoom_in', None) and event.ui_element == self._btn_zoom_in:
            import utils.config as config
            config.drawing_size = min(100, config.drawing_size + 2)
        elif getattr(self, '_btn_zoom_out', None) and event.ui_element == self._btn_zoom_out:
            import utils.config as config
            config.drawing_size = max(2, config.drawing_size - 2)

    def handle_key(self, program, key):
        
        if key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)

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
    can_pan = True
    key_bindings = (
        "Strzałki: przesuwanie mapy (Pan)",
        "L: tryb łączenia schodów",
        "C: toggle clusters",
        "B: toggle connections",
        "V: toggle drone routes",
        "G: toggle grid"
    )

    def __init__(self):
        self._save_button = None
        self._menu_button = None

    def enter(self, program):
        program.simulation.grid_world.clear_temporary_walls()
        program.simulation.prepare()
        program.drone_routes = {}
        program.drone_controller.pending_transitions.clear()
        program.drone_controller.route_cursors.clear()
        program.rl_evacuee_controller.reset()
        program.plan_drone_routes()
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        program.plan_drone_routes()
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        screen_w, screen_h = program.screen.get_size()
        btn_w, btn_h = 180, 32
        panel_x = screen_w - btn_w - 15
        y = 220 
        
        self._btn_sim = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Start Symulacji", manager=program.ui_manager)
        y += btn_h + 4
        self._btn_edit = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Edytor Mapy", manager=program.ui_manager)
        y += btn_h + 4
        self._btn_link = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Łączenie Schodów", manager=program.ui_manager)
        y += btn_h + 15
        self._btn_zoom_in = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w//2 - 2, btn_h), "Zoom +", manager=program.ui_manager)
        self._btn_zoom_out = pygame_gui.elements.UIButton(pygame.Rect(panel_x + btn_w//2 + 2, y, btn_w//2 - 2, btn_h), "Zoom -", manager=program.ui_manager)
        y += btn_h + 15
        self._btn_floor_up = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w//2 - 2, btn_h), "Piętro +", manager=program.ui_manager)
        self._btn_floor_down = pygame_gui.elements.UIButton(pygame.Rect(panel_x + btn_w//2 + 2, y, btn_w//2 - 2, btn_h), "Piętro -", manager=program.ui_manager)
        y += btn_h + 15
        self._save_button = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Zapisz (Ctrl+S)", manager=program.ui_manager)
        y += btn_h + 4
        self._menu_button = pygame_gui.elements.UIButton(pygame.Rect(panel_x, y, btn_w, btn_h), "Menu", manager=program.ui_manager)

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        #self._save_button = None
        #self._menu_button = None

    def handle_ui_event(self, program, event):
        if event.ui_element == self._btn_sim: program.transition_to(SimulationRunningState())
        elif event.ui_element == self._btn_edit: program.transition_to(EditorState())
        elif event.ui_element == self._btn_link:
            program.link_stairs_mode = not program.link_stairs_mode
            program.simulation.grid_world.pending_stair_endpoint = None
            program.simulation.grid_world.stair_link_error = None
        elif event.ui_element == self._save_button: _save_project(program)
        elif getattr(self, '_btn_floor_up', None) and event.ui_element == self._btn_floor_up:
            world = program.simulation.grid_world
            new_floor = min(len(world.floors) - 1, world.active_floor_id + 1)
            if new_floor != world.active_floor_id:
                world.set_active_floor(new_floor)
                program.update_window_for_active_floor()
                self.on_floor_changed(program)
        elif getattr(self, '_btn_floor_down', None) and event.ui_element == self._btn_floor_down:
            world = program.simulation.grid_world
            new_floor = max(0, world.active_floor_id - 1)
            if new_floor != world.active_floor_id:
                world.set_active_floor(new_floor)
                program.update_window_for_active_floor()
                self.on_floor_changed(program)
        elif event.ui_element == self._menu_button:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager: program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif getattr(self, '_btn_zoom_in', None) and event.ui_element == self._btn_zoom_in:
            import utils.config as config
            config.drawing_size = min(100, config.drawing_size + 2)
        elif getattr(self, '_btn_zoom_out', None) and event.ui_element == self._btn_zoom_out:
            import utils.config as config
            config.drawing_size = max(2, config.drawing_size - 2)

    def handle_key(self, program, key):
        if key == pygame.K_l:
            program.link_stairs_mode = not program.link_stairs_mode
            program.simulation.grid_world.pending_stair_endpoint = None
            program.simulation.grid_world.stair_link_error = None
        elif key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)

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
            world.stair_link_error = "Wybierz klastry schodów"
            return

        pending = world.pending_stair_endpoint
        if pending is None:
            world.pending_stair_endpoint = endpoint
            world.stair_link_error = None
        elif pending[0] == endpoint[0]:
            world.pending_stair_endpoint = endpoint
            world.stair_link_error = "Wybierz schody na innym piętrze"
        else:
            try:
                world.connect_stair_clusters(pending, endpoint)
            except ValueError as error:
                world.stair_link_error = str(error)
            else:
                world.pending_stair_endpoint = None
                world.stair_link_error = None
                program.plan_drone_routes()

    def render(self, program):
        program.window.draw_overview(program, self.key_bindings)


class SimulationRunningState(ProgramState):
    name = "simulation_running"
    can_pan = True
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
    can_pan = True
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
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "Enter: return to pause",
        "Left click: place temporary wall",
        "Right click: erase temporary wall",
        "C: toggle clusters",
        "B: toggle connections",
        "V: toggle drone routes",
        "G: toggle grid",
    )

    def exit(self, program):
        program.replan_drone_routes_after_hazard()

    def handle_key(self, program, key):
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(SimulationPausedState())

    def handle_mouse(self, program, event):
        if event.button not in (1, 3):
            return

        world = program.simulation.grid_world
        grid_x, grid_y = pixel_to_grid(*event.pos)
        world.set_temporary_wall(
            world.active_floor_id, grid_x, grid_y, event.button == 1
        )


class ResultsState(ProgramState):
    name = "results"
    can_pan = True
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