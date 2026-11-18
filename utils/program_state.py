from torch.export import exported_program
from torch.export import exported_program
from torch.export import exported_program
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
from editor.editor import grid_line, handle_map_editor_cell
from editor.tool import Tool
from grids.coordinates import pixel_to_grid
from grids.map_grid import Map_Grid
from map.cell import CellType
from simulation import Simulation
from utils.config import block_size, acceptance_threshold

MENU_WINDOW_SIZE = (1280, 800)
CALIBRATION_WINDOW_SIZE = (1024, 768)
MAIN_WINDOW_SIZE = (1280, 800)
SIDEBAR_WIDTH = 280
TOP_BAR_HEIGHT = 88

logger = logging.getLogger(__name__)


def _open_file_dialog(title, filetypes):
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
            return ""

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
            return ""

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
        "Save Project",
        ".json",
        [("JSON files", "*.json")],
    )
    if not path:
        return
    context.simulation.grid_world.rebuild_changed_map_data()
    data = context.simulation.grid_world.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    for floor in context.simulation.grid_world.floors.values():
        floor.base_map_grid = floor.map_grid.copy()
    logger.info("Project saved to %s", path)


def _build_playback_controls(
    state, program, tool_x, tool_w, playback_y, primary_label, *, step_enabled=True
):
    primary = pygame_gui.elements.UIButton(
        pygame.Rect(tool_x, playback_y, tool_w, 38),
        primary_label,
        manager=program.ui_manager,
        object_id="#btn_start_sim",
    )
    state._btn_start_sim = primary
    state._btn_pause = primary
    state._btn_step = pygame_gui.elements.UIButton(
        pygame.Rect(tool_x, playback_y + 44, 120, 34),
        "+1 Step",
        manager=program.ui_manager,
        object_id="#btn_playback",
    )
    if not step_enabled:
        state._btn_step.disable()
    state._btn_reset = pygame_gui.elements.UIButton(
        pygame.Rect(tool_x + 128, playback_y + 44, 120, 34),
        "Reset",
        manager=program.ui_manager,
        object_id="#btn_playback",
    )
    state._speed_buttons = {}
    labels = ("0.5x", "0.75x", "1x", "2x", "4x")
    for index, (speed, label) in enumerate(
        zip(program.speed_multipliers, labels)
    ):
        button = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + 4 + index * 49, playback_y + 84, 44, 26),
            label,
            manager=program.ui_manager,
            object_id=(
                "#speed_btn_active"
                if speed == program.speed_multiplier
                else "#speed_btn"
            ),
        )
        state._speed_buttons[button] = speed


def _handle_speed_button(state, program, event):
    speed = state._speed_buttons.get(event.ui_element)
    if speed is None:
        return False
    program.set_speed_multiplier(speed)
    state._rebuild_ui(program)
    return True


def _build_overlay_controls(state, program, tool_x, y):
    labels = (
        ("show_grid", "Grid (G)"),
        ("show_clusters", "Clusters (C)"),
        ("show_drone_routes", "Routes (V)"),
    )
    state._overlay_buttons = {}
    for index, (option, label) in enumerate(labels):
        button = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + index * 83, y, 78, 28),
            label,
            manager=program.ui_manager,
            object_id=(
                "#overlay_btn_active"
                if getattr(program.view_options, option)
                else "#overlay_btn"
            ),
        )
        state._overlay_buttons[button] = option


def _handle_overlay_button(state, program, event):
    option = getattr(state, "_overlay_buttons", {}).get(event.ui_element)
    if option is None:
        return False
    setattr(program.view_options, option, not getattr(program.view_options, option))
    state._rebuild_ui(program)
    return True


def _step_once(state, program):
    program.step_simulation(keep_paused=True)
    if state.name == "overview":
        program.transition_to(SimulationPausedState())
    else:
        state._rebuild_ui(program)


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
            if keys[pygame.K_LEFT]:
                config.camera_offset_x += pan_speed
            if keys[pygame.K_RIGHT]:
                config.camera_offset_x -= pan_speed
            if keys[pygame.K_UP]:
                config.camera_offset_y += pan_speed
            if keys[pygame.K_DOWN]:
                config.camera_offset_y -= pan_speed

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
        evacuated = context.simulation.evacuated_count if context.simulation else 0
        total = evacuated + len(context.simulation.evacuees) if context.simulation else 0
        floor = (context.simulation.grid_world.active_floor_id + 1) if context.simulation else 1
        floor_count = len(context.simulation.grid_world.floors) if context.simulation else 1
        floor_label = f"Floor: {floor} / {floor_count}"

        lines = [
            "Scale: 1 cell = ~60 cm",
            f"State: {self.name}",
            floor_label,
            f"Drones: {mode}",
            f"Evacuees: {evacuee_mode}",
            f"Step: {context.simulation.step_count if context.simulation else 0}",
            f"Evacuated: {evacuated}/{total}",
        ]
        world = context.simulation.grid_world if context.simulation else None
        if world and context.link_stairs_mode:
            if world.pending_stair_endpoint is not None:
                pending_floor = world.pending_stair_endpoint[0] + 1
                lines.append(f"  Stairs on Floor {pending_floor} selected. Switch floor and click target stairs!")
            else:
                lines.append("  Stair Link Mode: Click first stairs (L to exit, RMB to cancel)")
        if world and world.stair_link_notice:
            lines.append(f"Notice: {world.stair_link_notice}")
        if world and world.stair_link_error:
            lines.append(f"Error: {world.stair_link_error}")
        return lines

    def on_floor_changed(self, program):
        pass

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)

    def handle_key_event(self, program, key):
        import utils.config as config

        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            if getattr(self, "can_pan", False):
                config.drawing_size = min(100, config.drawing_size + 2)
            return
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            if getattr(self, "can_pan", False):
                config.drawing_size = max(2, config.drawing_size - 2)
            return
        if key == pygame.K_g:
            program.view_options.show_grid = not program.view_options.show_grid
            self._refresh_overlay_controls(program)
            return
        if self.render_options_enabled:
            if key == pygame.K_c:
                program.view_options.show_clusters = not program.view_options.show_clusters
                self._refresh_overlay_controls(program)
                return
            if key == pygame.K_b:
                program.view_options.show_cluster_connections = not program.view_options.show_cluster_connections
                return
            if key == pygame.K_v:
                program.view_options.show_drone_routes = not program.view_options.show_drone_routes
                self._refresh_overlay_controls(program)
                return
        self.handle_key(program, key)

    def _refresh_overlay_controls(self, program):
        rebuild = getattr(self, "_rebuild_ui", None)
        if rebuild is not None:
            rebuild(program)


class MainMenuState(ProgramState):
    name = "main_menu"

    def enter(self, program):
        program.simulation = None
        program.imported_image_path = None
        program.drone_routes = {}
        program.link_stairs_mode = False
        self._import_button = None
        self._blank_button = None
        self._open_button = None
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

        screen_w, screen_h = program.screen.get_size()
        card_w = 280
        card_h = 248
        gap = 20
        total_w = card_w * 3 + gap * 2
        start_x = (screen_w - total_w) // 2
        cards_y = 250

        self._import_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + 18, cards_y + 188, card_w - 36, 38),
            text="Import Image...",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

        self._blank_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + card_w + gap + 18, cards_y + 188, card_w - 36, 38),
            text="Create Blank Map",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )


        self._open_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + (card_w + gap) * 2 + 18, cards_y + 188, card_w - 36, 38),
            text="Browse JSON...",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        self._import_button = None
        self._blank_button = None
        self._open_button = None

    def handle_ui_event(self, program, event):
        if event.ui_element == self._import_button:
            path = _open_file_dialog(
                "Select Floorplan Image",
                [("Images", "*.jpg *.png *.jpeg")],
            )
            if path:
                program.imported_image_path = path
                program.transition_to(CalibrationState())

        elif event.ui_element == self._blank_button:
            w, h = 300, 300
            blank_grid = Map_Grid(w,h)

            program.initialize_world(blank_grid)
            program.transition_to(EditorState())

        elif event.ui_element == self._open_button:
            path = _open_file_dialog(
                "Open Project",
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
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        screen.fill((22, 22, 22))

        header_font = pygame.font.Font(None, 34)
        sub_font = pygame.font.Font(None, 18)
        card_title_font = pygame.font.Font(None, 20)
        card_desc_font = pygame.font.Font(None, 15)

        badge_rect = pygame.Rect(screen_w // 2 - 26, 72, 52, 52)
        pygame.draw.rect(screen, (29, 44, 67), badge_rect, border_radius=12)
        pygame.draw.rect(screen, (45, 70, 109), badge_rect, 1, border_radius=12)
        pygame.draw.polygon(screen, (59, 130, 246), [
            (badge_rect.centerx - 7, badge_rect.centery - 10),
            (badge_rect.centerx + 7, badge_rect.centery),
            (badge_rect.centerx - 7, badge_rect.centery + 10)
        ])

        logo_box = pygame.Rect(screen_w // 2 - 44, 52, 88, 88)
        pygame.draw.rect(screen, (255, 255, 255), logo_box, border_radius=16)

        logo_path = Path(__file__).resolve().parent.parent / "logo" / "logoSquare.png"
        if logo_path.exists():
            logo_img = pygame.image.load(str(logo_path)).convert_alpha()
            logo_img = pygame.transform.smoothscale(logo_img, (72, 72))
            screen.blit(logo_img, (logo_box.centerx - 36, logo_box.centery - 36))
        else:
            badge_rect = pygame.Rect(screen_w // 2 - 26, 72, 52, 52)
            pygame.draw.rect(screen, (18, 48, 32), badge_rect, border_radius=12)
            pygame.draw.rect(screen, (28, 82, 54), badge_rect, 1, border_radius=12)
            pygame.draw.polygon(screen, (34, 139, 84), [
                (badge_rect.centerx - 7, badge_rect.centery - 10),
                (badge_rect.centerx + 7, badge_rect.centery),
                (badge_rect.centerx - 7, badge_rect.centery + 10)
            ])

        title_surf = header_font.render("ARIADNE - Decision-Support System for 2D Evacuation Planning", True, (244, 244, 245))
        screen.blit(title_surf, (screen_w // 2 - title_surf.get_width() // 2, 142))

        sub_surf = sub_font.render(
            "Import a floorplan image, build a grid from scratch, or open a saved project",
            True,
            (150, 153, 159),
        )
        screen.blit(sub_surf, (screen_w // 2 - sub_surf.get_width() // 2, 182))

        card_w = 280
        card_h = 248
        gap = 20
        total_w = card_w * 3 + gap * 2
        start_x = (screen_w - total_w) // 2
        cards_y = 250

        cards_info = [
            (
                "New from Floorplan Image",
                "Convert a PNG or JPG floorplan into a walkable grid and calibrate scale.",
                (43, 43, 43),
            ),
            (
                "New Blank Grid",
                "Start with an empty architectural grid and draw walls, stairs, and exits manually.",
                (43, 43, 43),
            ),
            (
                "Open Project (.json)",
                "Load an existing multi-floor evacuation map, stair links, and simulation setup.",
                (43, 43, 43),
            ),
        ]

        for i, (title, desc, border_col) in enumerate(cards_info):
            cx = start_x + i * (card_w + gap)
            card_rect = pygame.Rect(cx, cards_y, card_w, card_h)
            pygame.draw.rect(screen, (30, 30, 30), card_rect, border_radius=9)
            pygame.draw.rect(screen, border_col, card_rect, 1, border_radius=9)

            icon_rect = pygame.Rect(cx + 20, cards_y + 20, 44, 44)

            t_surf = card_title_font.render(title, True, (240, 240, 242))
            screen.blit(t_surf, (cx + 20, cards_y + 82))

            words = desc.split(" ")
            lines = []
            cur_line = ""
            for w in words:
                test = cur_line + (" " if cur_line else "") + w
                if card_desc_font.size(test)[0] <= card_w - 40:
                    cur_line = test
                else:
                    lines.append(cur_line)
                    cur_line = w
            if cur_line:
                lines.append(cur_line)

            dy = cards_y + 114
            for l in lines:
                l_surf = card_desc_font.render(l, True, (165, 167, 173))
                screen.blit(l_surf, (cx + 20, dy))
                dy += 18

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
            text="Block size: 25px | Grid dimensions: -- x --",
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
            text=f"File: {file_name}  |  Image: {self.img_w}x{self.img_h}px  |  Floors loaded: {floor_count}",
            manager=program.ui_manager,
        )

        btn_w = 210
        btn_h = 42
        gap = 16
        col1_x = 565
        col2_x = col1_x + btn_w + gap

        self._add_floor_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(col1_x, top_y + 12, btn_w, btn_h),
            text="Add Another Floor from Image",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

        self._generate_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(col2_x, top_y + 12, btn_w, btn_h),
            text="Generate Floor & Open Editor",
            manager=program.ui_manager,
            object_id="#btn_primary",
        )

        self._back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(24, top_y + 98, 120, 28),
            text="Back to Menu",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def _update_grid_overlay(self, block_size):
        cols = self.img_w // block_size
        rows = self.img_h // block_size

        if self._info_label is not None:
            self._info_label.set_text(
                f"Block size: {block_size}px | Grid dimensions: {cols} x {rows}"
            )

        self.overlay_surface = pygame.Surface(
            (self.disp_w, self.disp_h), pygame.SRCALPHA
        )

        grid_color = (0, 210, 255, 110)

        step_x = max(1.0, block_size * self.disp_scale)
        step_y = max(1.0, block_size * self.disp_scale)

        x = 0.0
        while x <= self.disp_w:
            ix = int(x)
            pygame.draw.line(
                self.overlay_surface, grid_color, (ix, 0), (ix, self.disp_h)
            )
            x += step_x

        y = 0.0
        while y <= self.disp_h:
            iy = int(y)
            pygame.draw.line(
                self.overlay_surface, grid_color, (0, iy), (self.disp_w, iy)
            )
            y += step_y

    def update(self, program):
        if self._slider is not None:
            new_value = int(self._slider.get_current_value())
            if new_value != self.current_block_size:
                self.current_block_size = new_value
                self._update_grid_overlay(new_value)

    def render(self, program):
        screen = program.screen
        screen.fill((20, 20, 20))

        if self.display_image is not None:
            screen.blit(self.display_image, (self.disp_x, self.disp_y))

        if self.overlay_surface is not None:
            screen.blit(self.overlay_surface, (self.disp_x, self.disp_y))

        pygame.draw.rect(screen, (32, 32, 32), self.control_panel_rect)
        pygame.draw.line(
            screen,
            (60, 60, 60),
            (0, self.control_panel_rect.top),
            (self.screen_size[0], self.control_panel_rect.top),
            2,
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
            block_size = self.current_block_size
            map_grid = Map_Grid.from_image(
                self.image_path, block_size, acceptance_threshold=acceptance_threshold
            )
            if program.simulation is None:
                program.simulation = Simulation()
            program.simulation.grid_world.add_floor_from_map(map_grid)
            next_path = _open_file_dialog(
                "Select Another Floor Image",
                [("Images", "*.jpg *.png *.jpeg")],
            )
            if next_path:
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
    tracks_mouse_drag = True
    render_options_enabled = True
    can_pan = True
    floor_switch_enabled = True
    key_bindings = (
        "Arrow keys: Pan canvas",
        "G: Toggle grid",
        "+ / -: Zoom",
        "LMB: Place, RMB: Erase",
    )

    def __init__(self):
        self.selected_tool = Tool.WALL
        self.active_mouse_button = None
        self.last_grid_position = None
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_add_floor = None
        self._btn_wall = None
        self._btn_evacuee = None
        self._btn_exit = None
        self._btn_drone = None
        self._btn_stairs = None
        self._save_project_btn = None
        self._btn_start_sim = None
        self._btn_pause = None
        self._btn_reset = None
        self._overlay_buttons = {}

    def enter(self, program):
        self.active_mouse_button = None
        self.last_grid_position = None
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        program.simulation.grid_world.rebuild_changed_map_data()
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(270, 9, 115, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(395, 9, 70, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 168
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "1. Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_active",
        )
        self._nav_link = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "2. Link Stairs",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w * 2, 9, tab_item_w, 30),
            "3. Simulate",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )

        world = program.simulation.grid_world if program.simulation else None
        floor_count = len(world.floors) if world else 1
        active_floor_id = world.active_floor_id if world else 0

        tab_x = 14
        for f_id in range(floor_count):
            is_active = (f_id == active_floor_id)
            obj_id = "#floor_tab_active" if is_active else "#floor_tab_inactive"
            tab_btn = pygame_gui.elements.UIButton(
                pygame.Rect(tab_x, 52, 100, 32),
                f"Floor {f_id + 1}",
                manager=program.ui_manager,
                object_id=obj_id,
            )
            self._floor_tabs[f_id] = tab_btn
            tab_x += 106

        self._btn_add_floor = pygame_gui.elements.UIButton(
            pygame.Rect(tab_x + 6, 54, 105, 28),
            "+ Add Floor",
            manager=program.ui_manager,
            object_id="#btn_add_floor",
        )

        tool_x = sidebar_x + 16
        _build_overlay_controls(self, program, tool_x, 248)
        tool_w = 248
        tool_h = 32
        y = 282

        self._btn_wall = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Wall" if self.selected_tool == Tool.WALL else "Wall",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.WALL else "#tool_btn",
        )
        y += 36
        self._btn_evacuee = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Evacuee" if self.selected_tool == Tool.EVACUEE else "Evacuee",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.EVACUEE else "#tool_btn",
        )
        y += 36
        self._btn_exit = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Exit" if self.selected_tool == Tool.EXIT else "Exit",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.EXIT else "#tool_btn",
        )
        y += 36
        self._btn_drone = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Agent" if self.selected_tool == Tool.DRONE else "Agent",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.DRONE else "#tool_btn",
        )
        y += 36
        self._btn_stairs = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Stairs" if self.selected_tool == Tool.STAIRS else "Stairs",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.STAIRS else "#tool_btn",
        )
        y += 40
        self._save_project_btn = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, 34),
            "Save Project",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        playback_y = 564
        self._btn_start_sim = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, playback_y, tool_w, 38),
            "Start Simulation",
            manager=program.ui_manager,
            object_id="#btn_start_sim",
        )
        self._btn_pause = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, playback_y + 44, 120, 34),
            "Pause",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        self._btn_reset = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + 128, playback_y + 44, 120, 34),
            "Reset",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )

    def exit(self, program):
        self.active_mouse_button = None
        self.last_grid_position = None
        program.simulation.grid_world.rebuild_changed_map_data()
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_ui_event(self, program, event):
        if _handle_overlay_button(self, program, event):
            return
        if event.ui_element in (self._save_btn, self._save_project_btn):
            _save_project(program)
        elif event.ui_element == self._menu_btn:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            pass
        elif event.ui_element == self._nav_link:
            program.link_stairs_mode = True
            program.transition_to(OverviewState())
        elif event.ui_element == self._nav_sim:
            program.reset_simulation(clear_link_mode=True)
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_start_sim:
            program.transition_to(SimulationRunningState())
        elif event.ui_element == self._btn_wall:
            self.selected_tool = Tool.WALL
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_evacuee:
            self.selected_tool = Tool.EVACUEE
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_exit:
            self.selected_tool = Tool.EXIT
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_drone:
            self.selected_tool = Tool.DRONE
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_stairs:
            self.selected_tool = Tool.STAIRS
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_reset:
            program.reset_simulation(
                restore_base_map=True, clear_link_mode=True
            )
        elif event.ui_element == self._btn_add_floor:
            path = _open_file_dialog("Select Floorplan Image", [("Images", "*.jpg *.png *.jpeg")])
            if path:
                program.imported_image_path = path
                program.transition_to(CalibrationState())
        else:
            for f_id, tab_btn in self._floor_tabs.items():
                if event.ui_element == tab_btn:
                    world = program.simulation.grid_world
                    if f_id != world.active_floor_id:
                        world.set_active_floor(f_id)
                        program.update_window_for_active_floor()
                        self.on_floor_changed(program)
                    break

    def handle_key(self, program, key):
        if key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)

    def handle_mouse(self, program, event):
        if event.type == pygame.MOUSEBUTTONUP:
            self.active_mouse_button = None
            self.last_grid_position = None
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button not in (1, 3):
                return
            if not self._is_canvas_position(program, event.pos):
                self.active_mouse_button = None
                self.last_grid_position = None
                return
            self.active_mouse_button = event.button
            self.last_grid_position = None
        elif event.type == pygame.MOUSEMOTION:
            if self.active_mouse_button is None:
                return
        else:
            return

        self._paint_to_position(program, event.pos)

    def _is_canvas_position(self, program, position):
        screen_w, screen_h = program.screen.get_size()
        return (
            0 <= position[0] < screen_w - SIDEBAR_WIDTH
            and TOP_BAR_HEIGHT <= position[1] < screen_h
        )

    def _paint_to_position(self, program, position):
        if not self._is_canvas_position(program, position):
            self.last_grid_position = None
            return

        world = program.simulation.grid_world
        grid_position = pixel_to_grid(*position)
        if not world.map_grid.in_bounds(*grid_position):
            self.last_grid_position = None
            return

        start = self.last_grid_position or grid_position
        clusters_changed = False
        for grid_x, grid_y in grid_line(start, grid_position):
            clusters_changed |= handle_map_editor_cell(
                program.simulation,
                grid_x,
                grid_y,
                self.active_mouse_button,
                self.selected_tool,
                rebuild_clusters=False,
            )
        if clusters_changed:
            world.rebuild_changed_map_data()
        self.last_grid_position = grid_position

    def render(self, program):
        program.window.draw_editor(program, self.selected_tool, self.key_bindings)


class OverviewState(ProgramState):
    name = "overview"
    render_options_enabled = True
    floor_switch_enabled = True
    can_pan = True
    key_bindings = (
        "Space: Start/Pause",
        "Arrow keys: Pan canvas",
        "L: Stair link mode",
        "H: Toggle assignments",
        "C: Clusters",
        "B: Connections",
        "V: Drone routes",
        "G: Grid",
    )

    def __init__(self):
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_add_floor = None
        self._btn_link_toggle = None
        self._save_project_btn = None
        self._btn_start_sim = None
        self._btn_reset = None
        self._btn_pause = None
        self._btn_step = None
        self._speed_buttons = {}
        self._overlay_buttons = {}

    def enter(self, program):
        program.reset_simulation()
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        program.plan_drone_routes()
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(270, 9, 115, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(395, 9, 70, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 168
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "1. Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_link = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "2. Link Stairs",
            manager=program.ui_manager,
            object_id="#nav_item_active",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w * 2, 9, tab_item_w, 30),
            "3. Simulate",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )

        world = program.simulation.grid_world if program.simulation else None
        floor_count = len(world.floors) if world else 1
        active_floor_id = world.active_floor_id if world else 0

        tab_x = 14
        for f_id in range(floor_count):
            is_active = (f_id == active_floor_id)
            obj_id = "#floor_tab_active" if is_active else "#floor_tab_inactive"
            tab_btn = pygame_gui.elements.UIButton(
                pygame.Rect(tab_x, 52, 100, 32),
                f"Floor {f_id + 1}",
                manager=program.ui_manager,
                object_id=obj_id,
            )
            self._floor_tabs[f_id] = tab_btn
            tab_x += 106

        self._btn_add_floor = pygame_gui.elements.UIButton(
            pygame.Rect(tab_x + 6, 54, 105, 28),
            "+ Add Floor",
            manager=program.ui_manager,
            object_id="#btn_add_floor",
        )

        tool_x = sidebar_x + 16
        tool_w = 248
        y = 248

        self._btn_link_toggle = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, 40),
            "Stair Link: Active" if program.link_stairs_mode else "Enable Stair Link",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if program.link_stairs_mode else "#tool_btn",
        )

        y += 46
        self._save_project_btn = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, 34),
            "Save Project",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        _build_overlay_controls(self, program, tool_x, 340)

        playback_y = 564
        _build_playback_controls(
            self,
            program,
            tool_x,
            tool_w,
            playback_y,
            "Resume" if program.simulation.step_count > 0 else "Start",
        )

    def exit(self, program):
        program.link_stairs_mode = False
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_ui_event(self, program, event):
        if _handle_overlay_button(self, program, event):
            return
        if _handle_speed_button(self, program, event):
            return
        if event.ui_element in (self._save_btn, self._save_project_btn):
            _save_project(program)
        elif event.ui_element == self._menu_btn:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element in (self._nav_link, self._btn_link_toggle):
            program.link_stairs_mode = not program.link_stairs_mode
            program.simulation.grid_world.pending_stair_endpoint = None
            program.simulation.grid_world.stair_link_error = None
            program.simulation.grid_world.stair_link_notice = None
            self._rebuild_ui(program)
        elif event.ui_element == self._nav_sim:
            program.reset_simulation(clear_link_mode=True)
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_start_sim:
            program.transition_to(SimulationRunningState())
        elif event.ui_element == self._btn_step:
            _step_once(self, program)
        elif event.ui_element == self._btn_reset:
            program.reset_simulation()
        elif event.ui_element == self._btn_add_floor:
            path = _open_file_dialog("Select Floorplan Image", [("Images", "*.jpg *.png *.jpeg")])
            if path:
                program.imported_image_path = path
                program.transition_to(CalibrationState())
        else:
            for f_id, tab_btn in self._floor_tabs.items():
                if event.ui_element == tab_btn:
                    world = program.simulation.grid_world
                    if f_id != world.active_floor_id:
                        world.set_active_floor(f_id)
                        program.update_window_for_active_floor()
                        self.on_floor_changed(program)
                    break


    def handle_key(self, program, key):
        if key == pygame.K_SPACE:
            program.transition_to(SimulationRunningState())
        elif key == pygame.K_l:
            program.link_stairs_mode = not program.link_stairs_mode
            program.simulation.grid_world.pending_stair_endpoint = None
            program.simulation.grid_world.stair_link_error = None
            program.simulation.grid_world.stair_link_notice = None
            self._rebuild_ui(program)
        elif key == pygame.K_h:
            self._toggle_assignments(program)
        elif key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)

    def handle_mouse(self, program, event):
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        if not program.link_stairs_mode:
            return

        world = program.simulation.grid_world
        if event.button == 3:
            world.pending_stair_endpoint = None
            world.stair_link_error = None
            world.stair_link_notice = None
            return
        if event.button != 1:
            return

        grid_x, grid_y = pixel_to_grid(*event.pos)
        endpoint = world.stair_cluster_at(world.active_floor_id, grid_x, grid_y)
        if endpoint is None:
            world.stair_link_error = "Select a stair cluster"
            world.stair_link_notice = None
            return

        pending = world.pending_stair_endpoint
        if pending is None:
            world.pending_stair_endpoint = endpoint
            world.stair_link_error = None
            world.stair_link_notice = None
        elif pending[0] == endpoint[0]:
            world.pending_stair_endpoint = endpoint
            world.stair_link_error = "Select stairs on a different floor"
            world.stair_link_notice = None
        else:
            try:
                world.connect_stair_clusters(pending, endpoint)
            except ValueError as error:
                world.stair_link_error = str(error)
                world.stair_link_notice = None
            else:
                f1 = pending[0] + 1
                c1 = pending[1]
                f2 = endpoint[0] + 1
                c2 = endpoint[1]
                world.stair_link_notice = f"Linked Floor {f1} (C{c1}) <-> Floor {f2} (C{c2})"
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
        "Space / P: Pause",
        "X: Finish",
        "WASD: Evacuees",
        "T: Stairs",
        "F: Autopilot",
        "R: RL Follow",
        "C/B/V: Overlays",
        "G: Grid",
    )

    def __init__(self):
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_pause = None
        self._btn_reset = None
        self._btn_start_sim = None
        self._btn_step = None
        self._speed_buttons = {}
        self._overlay_buttons = {}

    def enter(self, program):
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(270, 9, 115, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(395, 9, 70, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 168
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "1. Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_link = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "2. Link Stairs",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w * 2, 9, tab_item_w, 30),
            "3. Simulate",
            manager=program.ui_manager,
            object_id="#nav_item_active",
        )

        world = program.simulation.grid_world if program.simulation else None
        floor_count = len(world.floors) if world else 1
        active_floor_id = world.active_floor_id if world else 0

        tab_x = 14
        for f_id in range(floor_count):
            is_active = (f_id == active_floor_id)
            obj_id = "#floor_tab_active" if is_active else "#floor_tab_inactive"
            tab_btn = pygame_gui.elements.UIButton(
                pygame.Rect(tab_x, 52, 100, 32),
                f"Floor {f_id + 1}",
                manager=program.ui_manager,
                object_id=obj_id,
            )
            self._floor_tabs[f_id] = tab_btn
            tab_x += 106
        
        self._btn_add_floor = pygame_gui.elements.UIButton(
            pygame.Rect(tab_x + 6, 54, 105, 28),
            "+ Add Floor",
            manager=program.ui_manager,
            object_id="#btn_add_floor",
        )


        tool_x = sidebar_x + 16
        tool_w = 248
        playback_y = 564
        _build_playback_controls(
            self,
            program,
            tool_x,
            tool_w,
            playback_y,
            "Pause",
            step_enabled=False,
        )
        _build_overlay_controls(self, program, tool_x, 302)


    def handle_ui_event(self, program, event):
        if _handle_overlay_button(self, program, event):
            return
        if _handle_speed_button(self, program, event):
            return
        if event.ui_element == self._save_btn:
            _save_project(program)
        elif event.ui_element == self._btn_pause:
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._menu_btn:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element == self._nav_link:
            program.transition_to(OverviewState())
        elif event.ui_element == self._btn_reset:
            program.reset_simulation(clear_link_mode=True)
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_add_floor:
            path = _open_file_dialog("Select Floorplan Image", [("Images", "*.jpg *.png *.jpeg")])
            if path:
                program.imported_image_path = path
                program.transition_to(CalibrationState())
        else:
            for f_id, tab_btn in self._floor_tabs.items():
                if event.ui_element == tab_btn:
                    world = program.simulation.grid_world
                    if f_id != world.active_floor_id:
                        world.set_active_floor(f_id)
                        program.update_window_for_active_floor()
                        self._rebuild_ui(program)
                    break

    def handle_key(self, program, key):
        if key in (pygame.K_p, pygame.K_SPACE):
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
        program.step_simulation()

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

class SimulationPausedState(ProgramState):
    name = "simulation_paused"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "Space / P: Resume",
        "Right / .: +1 Step",
        "M: Place hazards (temp walls)",
        "C/B/V: Overlays",
        "G: Grid",
    )

    def __init__(self):
        self._floor_tabs = {}
        self._add_floor_btn = None
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_hazard = None
        self._btn_resume = None
        self._btn_reset = None
        self._btn_start_sim = None
        self._btn_step = None
        self._speed_buttons = {}
        self._overlay_buttons = {}

    def enter(self, program):
        self._rebuild_ui(program)

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(270, 9, 115, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(395, 9, 70, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 168
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "1. Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_link = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "2. Link Stairs",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w * 2, 9, tab_item_w, 30),
            "3. Simulate",
            manager=program.ui_manager,
            object_id="#nav_item_active",
        )

        world = program.simulation.grid_world if program.simulation else None
        floor_count = len(world.floors) if world else 1
        active_floor_id = world.active_floor_id if world else 0
        tab_x = 14
        for f_id in range(floor_count):
            is_active = (f_id == active_floor_id)
            obj_id = "#floor_tab_active" if is_active else "#floor_tab_inactive"
            tab_btn = pygame_gui.elements.UIButton(
                pygame.Rect(tab_x, 52, 100, 32),
                f"Floor {f_id + 1}",
                manager=program.ui_manager,
                object_id=obj_id,
            )
            self._floor_tabs[f_id] = tab_btn
            tab_x += 106
        
        self._btn_add_floor = pygame_gui.elements.UIButton(
            pygame.Rect(tab_x + 6, 54, 105, 28),
            "+ Add Floor",
            manager=program.ui_manager,
            object_id="#btn_add_floor",
        )


        tool_x = sidebar_x + 16
        tool_w = 248

        self._btn_hazard = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 248, tool_w, 40),
            "Place Hazards (M)",
            manager=program.ui_manager,
            object_id="#tool_btn",
        )
        _build_overlay_controls(self, program, tool_x, 302)

        playback_y = 564
        _build_playback_controls(
            self,
            program,
            tool_x,
            tool_w,
            playback_y,
            "Resume" if program.simulation.step_count > 0 else "Start",
        )

    def handle_ui_event(self, program, event):
        if _handle_overlay_button(self, program, event):
            return
        if _handle_speed_button(self, program, event):
            return
        if event.ui_element == self._save_btn:
            _save_project(program)
        elif event.ui_element == self._menu_btn:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element == self._nav_link:
            program.transition_to(OverviewState())
        elif event.ui_element == self._btn_hazard:
            program.transition_to(SimulationEditorState())
        elif event.ui_element in (self._btn_resume, self._btn_start_sim):
            program.transition_to(SimulationRunningState())
        elif event.ui_element == self._btn_step:
            _step_once(self, program)
        elif event.ui_element == self._btn_reset:
            program.reset_simulation(clear_link_mode=True)
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_add_floor:
            path = _open_file_dialog("Select Floorplan Image", [("Images", "*.jpg *.png *.jpeg")])
            if path:
                program.imported_image_path = path
                program.transition_to(CalibrationState())
        else:
            for f_id, tab_btn in self._floor_tabs.items():
                if event.ui_element == tab_btn:
                    world = program.simulation.grid_world
                    if f_id != world.active_floor_id:
                        world.set_active_floor(f_id)
                        program.update_window_for_active_floor()
                        self._rebuild_ui(program)
                    break

    def handle_key(self, program, key):
        if key in (pygame.K_p, pygame.K_SPACE):
            program.transition_to(SimulationRunningState())
        elif key in (pygame.K_RIGHT, pygame.K_PERIOD):
            _step_once(self, program)
        elif key == pygame.K_m:
            program.transition_to(SimulationEditorState())


class SimulationEditorState(ProgramState):
    name = "simulation_editor"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = (
        "Enter: Return to pause",
        "LMB: Place temporary wall",
        "RMB: Erase temporary wall",
        "G: Grid",
    )

    def exit(self, program):
        program.replan_drone_routes_after_hazard()

    def handle_key(self, program, key):
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(SimulationPausedState())

    def handle_mouse(self, program, event):
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
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
        "Enter: Return to overview",
        "G: Grid",
    )

    def __init__(self):
        self._btn_reset = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None

    def enter(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        screen_w, _ = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(395, 9, 70, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )
        switcher_w = 510
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 168
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "1. Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_link = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "2. Link Stairs",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._btn_reset = pygame_gui.elements.UIButton(
            pygame.Rect(sidebar_x + 16, 564, 248, 38),
            "Reset Simulation",
            manager=program.ui_manager,
            object_id="#btn_start_sim",
        )

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_ui_event(self, program, event):
        if event.ui_element == self._btn_reset:
            program.reset_simulation(clear_link_mode=True)
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._nav_link:
            program.link_stairs_mode = False
            program.transition_to(OverviewState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element == self._menu_btn:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())

    def handle_key(self, program, key):
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(OverviewState())