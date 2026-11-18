import platform
import subprocess
import json
import logging
import random
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pygame
import pygame_gui

from controls.action import Action
from controls.action_controller import get_actions
from editor.editor import handle_map_editor
from editor.tool import Tool
from graphics.colors import Color
from grids.coordinates import pixel_to_grid
from grids.map_grid import Map_Grid
from map.cell import CellType
from simulation import Simulation
from utils.config import block_size, acceptance_threshold
from map.image_extractor import (
    preprocess_floorplan_image,
    detect_grid_cells,
)

MENU_WINDOW_SIZE = (1280, 800)
CALIBRATION_WINDOW_SIZE = (1280, 800)
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

    def handle_raw_mouse(self, program, event):
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
                lines.append("  Stair Link Mode: Click first stairs")
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

    def __init__(self):
        self._import_button = None
        self._blank_button = None
        self._open_button = None

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
                program.calibrated_floors = []
                program.simulation = None
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


class FloorCalibrationData:
    def __init__(self, image_path, block_size=25, acceptance_threshold=0.40, px_per_m=None):
        self.image_path = image_path
        self.raw_image = pygame.image.load(image_path).convert()
        self.img_w, self.img_h = self.raw_image.get_size()
        self.preprocessed = preprocess_floorplan_image(image_path)
        self.block_size = block_size
        self.acceptance_threshold = float(acceptance_threshold)
        if px_per_m is not None and float(px_per_m) > 0:
            self.px_per_m = float(px_per_m)
        else:
            self.px_per_m = 25.0 / 0.6
        self.cell_size_m = self.block_size / self.px_per_m
        self.update_cell_size()
        self.detect_exits = True
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.fit_scale = 1.0
        self.wall_cells = set()
        self.exit_cells = set()
        self.grid_dimensions = (0, 0)
        self.display_image = None
        self.overlay_surface = None
        self.disp_x = 0
        self.disp_y = 0
        self.disp_w = 0
        self.disp_h = 0
        self.is_added_to_simulation = False

    def update_cell_size(self):
        if getattr(self, "px_per_m", 0) > 0:
            self.cell_size_m = self.block_size / self.px_per_m
        else:
            self.cell_size_m = (self.block_size / 25.0) * 0.6


class CalibrationState(ProgramState):
    name = "calibration"

    def __init__(self):
        self.is_panning = False
        self.pan_start = (0, 0)
        self.lock_grid_size = True
        self.is_measuring = False
        self.measure_p1 = None
        self.measure_p2 = None
        self.last_measured_dist_px = 0.0

        self._slider_block_size = None
        self._slider_sensitivity = None
        self._btn_coarse = None
        self._btn_standard = None
        self._btn_fine = None
        self._btn_measure_scale = None
        self._btn_reset_measure = None
        self._scale_input_window = None
        self._scale_entry_meters = None
        self._scale_btn_apply = None
        self._scale_btn_cancel = None
        self._btn_lock_grid = None
        self._btn_toggle_exits = None
        self._generate_button = None
        self._btn_back = None
        self._btn_load_project = None
        self._btn_add_floor = None
        self._btn_zoom_in = None
        self._btn_zoom_out = None
        self._floor_tab_buttons = []
        self.floors = []

    def enter(self, program):
        if not hasattr(program, 'calibrated_floors') or program.calibrated_floors is None:
            program.calibrated_floors = []

        new_path = program.imported_image_path
        program.imported_image_path = None

        if not program.calibrated_floors and not new_path:
            logger.warning("No image path provided for calibration; returning to main menu")
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
            return

        self.screen_size = CALIBRATION_WINDOW_SIZE
        program.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        if program.ui_manager is not None:
            program.ui_manager.set_window_resolution(self.screen_size)
            program.ui_manager.clear_and_reset()

        self.title_font = pygame.font.Font(None, 20)
        self.font = pygame.font.Font(None, 18)
        self.small_font = pygame.font.Font(None, 16)
        self.tiny_font = pygame.font.Font(None, 14)

        self.sidebar_width = SIDEBAR_WIDTH
        self.top_nav_height = 48
        self.tabs_height = 40
        self.header_height = self.top_nav_height + self.tabs_height

        self.canvas_rect = pygame.Rect(
            0,
            self.header_height,
            self.screen_size[0] - self.sidebar_width,
            self.screen_size[1] - self.header_height,
        )
        self.sidebar_rect = pygame.Rect(
            self.canvas_rect.right,
            self.top_nav_height,
            self.sidebar_width,
            self.screen_size[1] - self.top_nav_height,
        )

        self.is_panning = False
        self.pan_start = (0, 0)
        self.lock_grid_size = True
        self.is_measuring = False
        self.measure_p1 = None
        self.measure_p2 = None
        self.last_measured_dist_px = 0.0

        self._slider_block_size = None
        self._slider_sensitivity = None
        self._btn_coarse = None
        self._btn_standard = None
        self._btn_fine = None
        self._btn_measure_scale = None
        self._btn_reset_measure = None
        self._scale_input_window = None
        self._scale_entry_meters = None
        self._scale_btn_apply = None
        self._scale_btn_cancel = None
        self._btn_lock_grid = None
        self._btn_toggle_exits = None
        self._generate_button = None
        self._btn_back = None
        self._btn_load_project = None
        self._btn_add_floor = None
        self._btn_zoom_in = None
        self._btn_zoom_out = None
        self._floor_tab_buttons = []

        self.floors = program.calibrated_floors

        if new_path:
            cur_bs = self.floors[0].block_size if (self.lock_grid_size and self.floors) else 25
            cur_sens = self.floors[-1].acceptance_threshold if self.floors else 0.40
            cur_px = self.floors[0].px_per_m if (self.lock_grid_size and self.floors) else None
            new_fl = FloorCalibrationData(
                new_path,
                block_size=cur_bs,
                acceptance_threshold=cur_sens,
                px_per_m=cur_px,
            )
            self._init_floor_fit(new_fl)
            self.floors.append(new_fl)
            self.active_floor_index = len(self.floors) - 1
        elif not self.floors:
            first_floor = FloorCalibrationData(
                new_path,
                block_size=25,
                acceptance_threshold=0.40,
            )
            self._init_floor_fit(first_floor)
            self.floors.append(first_floor)
            self.active_floor_index = 0
        else:
            self.active_floor_index = min(getattr(self, 'active_floor_index', 0), len(self.floors) - 1)

        self._build_ui(program)
        self._recompute_detection()

    @property
    def active_floor(self):
        if 0 <= self.active_floor_index < len(self.floors):
            return self.floors[self.active_floor_index]
        return None

    def _init_floor_fit(self, fl):
        margin = 32
        avail_w = self.canvas_rect.width - (2 * margin)
        avail_h = self.canvas_rect.height - (2 * margin)
        fl.fit_scale = min(avail_w / fl.img_w, avail_h / fl.img_h)

    def handle_resize(self, program, new_size):
        self.screen_size = new_size
        self.canvas_rect = pygame.Rect(
            0,
            self.header_height,
            self.screen_size[0] - self.sidebar_width,
            self.screen_size[1] - self.header_height,
        )
        self.sidebar_rect = pygame.Rect(
            self.canvas_rect.right,
            self.top_nav_height,
            self.sidebar_width,
            self.screen_size[1] - self.top_nav_height,
        )
        self._build_ui(program)
        self._update_overlay_surface()

    def _build_ui(self, program):
        if program.ui_manager is None:
            return

        program.ui_manager.clear_and_reset()
        mgr = program.ui_manager
        screen_w = self.screen_size[0]
        sb_x = self.sidebar_rect.x

        container_w = 260
        container_x = screen_w - container_w - 16

        self._btn_back = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(container_x + 4, 9, 85, 30),
            text="Menu",
            manager=mgr,
            object_id="#menu_button",
        )
        self._btn_load_project = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(container_x + 95, 9, 161, 30),
            text="Load Project (.json)",
            manager=mgr,
            object_id="#save_button",
        )

        self._floor_tab_buttons = []
        tab_x = 14
        for idx in range(len(self.floors)):
            is_active = (idx == self.active_floor_index)
            btn = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(tab_x, 52, 100, 32),
                text=f"Floor {idx + 1}",
                manager=mgr,
                object_id="#floor_tab_active" if is_active else "#floor_tab_inactive",
            )
            self._floor_tab_buttons.append(btn)
            tab_x += 106

        self._btn_add_floor = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(tab_x + 6, 54, 110, 28),
            text="+ Add Floor",
            manager=mgr,
            object_id="#btn_add_floor",
        )

        zoom_pill_x = self.canvas_rect.right - 236
        zoom_pill_y = self.screen_size[1] - 54
        self._btn_zoom_out = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(zoom_pill_x + 4, zoom_pill_y + 4, 68, 30),
            text="Zoom -",
            manager=mgr,
            object_id="#menu_button",
        )
        self._btn_zoom_in = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(zoom_pill_x + 144, zoom_pill_y + 4, 68, 30),
            text="Zoom +",
            manager=mgr,
            object_id="#menu_button",
        )

        cur = self.active_floor
        bs = cur.block_size if cur else 25
        sens = int(cur.acceptance_threshold * 100) if cur else 40
        dex = cur.detect_exits if cur else True

        self._slider_block_size = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(sb_x + 16, 100, 248, 18),
            start_value=bs,
            value_range=(10, 50),
            manager=mgr,
            click_increment=1,
        )

        self._btn_coarse = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 16, 148, 66, 28),
            text="Coarse",
            manager=mgr,
            object_id="#btn_preset_active" if bs == 35 else "#btn_preset_inactive",
        )
        self._btn_standard = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 86, 148, 116, 28),
            text="Standard (60cm)",
            manager=mgr,
            object_id="#btn_preset_active" if bs == 25 else "#btn_preset_inactive",
        )
        self._btn_fine = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 206, 148, 58, 28),
            text="Fine",
            manager=mgr,
            object_id="#btn_preset_active" if bs == 15 else "#btn_preset_inactive",
        )

        self._btn_measure_scale = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 16, 184, 168, 28),
            text="Calibrate Scale (Tape)",
            manager=mgr,
            object_id="#tool_btn_selected" if self.is_measuring else "#tool_btn",
        )
        self._btn_reset_measure = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 188, 184, 76, 28),
            text="Reset",
            manager=mgr,
            object_id="#menu_button",
        )

        self._slider_sensitivity = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(sb_x + 16, 364, 248, 18),
            start_value=sens,
            value_range=(15, 65),
            manager=mgr,
            click_increment=1,
        )

        self._btn_toggle_exits = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 16, 432, 248, 34),
            text="Detect Exits: ON" if dex else "Detect Exits: OFF",
            manager=mgr,
            object_id="#tool_btn_selected" if dex else "#tool_btn",
        )

        self._btn_lock_grid = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 16, 472, 248, 34),
            text="Lock Grid Size on All Floors: ON" if self.lock_grid_size else "Lock Grid Size on All Floors: OFF",
            manager=mgr,
            object_id="#tool_btn_selected" if self.lock_grid_size else "#tool_btn",
        )

        self._generate_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(sb_x + 16, self.screen_size[1] - 58, 248, 42),
            text="Open in Editor",
            manager=mgr,
            object_id="#btn_start_sim",
        )

    def _recompute_detection(self):
        cur = self.active_floor
        if cur is None or cur.preprocessed is None:
            return

        cur.wall_cells, cur.exit_cells, gw, gh = detect_grid_cells(
            cur.preprocessed,
            cur.block_size,
            cur.acceptance_threshold,
            detect_exits=cur.detect_exits,
        )
        cur.grid_dimensions = (gw, gh)
        self._update_overlay_surface()

    def _update_overlay_surface(self):
        cur = self.active_floor
        if cur is None or cur.raw_image is None:
            return

        scale = cur.fit_scale * cur.zoom_level
        cur.disp_w = max(1, int(cur.img_w * scale))
        cur.disp_h = max(1, int(cur.img_h * scale))

        cur.disp_x = self.canvas_rect.x + (self.canvas_rect.width - cur.disp_w) // 2 + cur.pan_offset_x
        cur.disp_y = self.canvas_rect.y + (self.canvas_rect.height - cur.disp_h) // 2 + cur.pan_offset_y

        cur.display_image = pygame.transform.smoothscale(
            cur.raw_image, (cur.disp_w, cur.disp_h)
        )

        cur.overlay_surface = pygame.Surface((cur.disp_w, cur.disp_h), pygame.SRCALPHA)

        cell_w = cur.block_size * scale
        cell_h = cur.block_size * scale
        gw, gh = cur.grid_dimensions

        wall_color = (59, 130, 246, 185)
        for cx, cy in cur.wall_cells:
            rx = int(cx * cell_w)
            ry = int(cy * cell_h)
            rw = max(1, int((cx + 1) * cell_w) - rx)
            rh = max(1, int((cy + 1) * cell_h) - ry)
            pygame.draw.rect(cur.overlay_surface, wall_color, (rx, ry, rw, rh))

        exit_color = (16, 185, 129, 220)
        for cx, cy in cur.exit_cells:
            rx = int(cx * cell_w)
            ry = int(cy * cell_h)
            rw = max(1, int((cx + 1) * cell_w) - rx)
            rh = max(1, int((cy + 1) * cell_h) - ry)
            pygame.draw.rect(cur.overlay_surface, exit_color, (rx, ry, rw, rh))
  
        grid_line_color = (200, 215, 230, 45)
        for x_idx in range(gw + 1):
            ix = int(x_idx * cell_w)
            if ix < cur.disp_w:
                pygame.draw.line(cur.overlay_surface, grid_line_color, (ix, 0), (ix, cur.disp_h))
        for y_idx in range(gh + 1):
            iy = int(y_idx * cell_h)
            if iy < cur.disp_h:
                pygame.draw.line(cur.overlay_surface, grid_line_color, (0, iy), (cur.disp_w, iy))

    def update(self, program):
        cur = self.active_floor
        if cur is None:
            return

        if self._slider_block_size is not None:
            new_val = int(self._slider_block_size.get_current_value())
            if new_val != cur.block_size:
                cur.block_size = new_val
                cur.update_cell_size()
                if self.lock_grid_size:
                    for fl in self.floors:
                        fl.block_size = new_val
                        fl.update_cell_size()
                self._recompute_detection()

        if self._slider_sensitivity is not None:
            new_sens = int(self._slider_sensitivity.get_current_value()) / 100.0
            if abs(new_sens - cur.acceptance_threshold) >= 0.01:
                cur.acceptance_threshold = new_sens
                self._recompute_detection()

        if self.is_panning:
            buttons = pygame.mouse.get_pressed()
            if not any(buttons):
                self.is_panning = False
            else:
                m_pos = pygame.mouse.get_pos()
                dx = m_pos[0] - self.pan_start[0]
                dy = m_pos[1] - self.pan_start[1]
                if dx != 0 or dy != 0:
                    self.pan_start = m_pos
                    cur.pan_offset_x += dx
                    cur.pan_offset_y += dy
                    self._update_overlay_surface()

    def render(self, program):
        screen = program.screen
        screen_w, screen_h = self.screen_size
        sb_x = self.sidebar_rect.x
        cur = self.active_floor

        screen.fill(Color.BACKGROUND.value)

        pygame.draw.rect(screen, Color.BACKGROUND.value, self.canvas_rect)
        screen.set_clip(self.canvas_rect)
        if cur and cur.display_image is not None:
            screen.blit(cur.display_image, (cur.disp_x, cur.disp_y))
        if cur and cur.overlay_surface is not None:
            screen.blit(cur.overlay_surface, (cur.disp_x, cur.disp_y))

        if self.is_measuring and self.measure_p1 is not None:
            p2 = self.measure_p2 if self.measure_p2 is not None else pygame.mouse.get_pos()
            pygame.draw.line(screen, (245, 158, 11), self.measure_p1, p2, 2)
            pygame.draw.circle(screen, (245, 158, 11), self.measure_p1, 5)
            pygame.draw.circle(screen, (0, 0, 0), self.measure_p1, 2)
            pygame.draw.circle(screen, (245, 158, 11), p2, 5)
            pygame.draw.circle(screen, (0, 0, 0), p2, 2)

            dist_scr = ((p2[0] - self.measure_p1[0]) ** 2 + (p2[1] - self.measure_p1[1]) ** 2) ** 0.5
            cur_scale = (cur.fit_scale * cur.zoom_level) if cur else 1.0
            dist_raw_val = dist_scr / cur_scale if cur_scale > 0 else dist_scr
            mid_x = (self.measure_p1[0] + p2[0]) // 2
            mid_y = (self.measure_p1[1] + p2[1]) // 2 - 14
            b_txt = self.tiny_font.render(f"{dist_raw_val:.1f} px", True, (255, 255, 255))
            b_rect = pygame.Rect(mid_x - b_txt.get_width() // 2 - 4, mid_y - 2, b_txt.get_width() + 8, b_txt.get_height() + 4)
            pygame.draw.rect(screen, (20, 20, 20), b_rect, border_radius=3)
            pygame.draw.rect(screen, (245, 158, 11), b_rect, 1, border_radius=3)
            screen.blit(b_txt, (b_rect.x + 4, b_rect.y + 2))

        screen.set_clip(None)

        if self.is_measuring:
            banner_w = 420
            banner_h = 32
            bx = self.canvas_rect.x + (self.canvas_rect.width - banner_w) // 2
            by = 96
            banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
            banner_surf.fill((20, 20, 20, 230))
            pygame.draw.rect(banner_surf, (245, 158, 11), banner_surf.get_rect(), 1, border_radius=6)
            screen.blit(banner_surf, (bx, by))
            if self.measure_p1 is None:
                guide_txt = "Click 1st point (e.g. door edge) on map (ESC to cancel)"
            elif self.measure_p2 is None:
                guide_txt = "Click 2nd point to finish distance measurement"
            else:
                guide_txt = "Enter real-world length in meters"
            t_surf = self.small_font.render(guide_txt, True, (255, 230, 150))
            screen.blit(t_surf, (bx + (banner_w - t_surf.get_width()) // 2, by + 8))

        legend_hdr = self.tiny_font.render("DETECTION PREVIEW", True, Color.TEXT_MUTED.value)
        screen.blit(legend_hdr, (24, 102))

        pygame.draw.rect(screen, (59, 130, 246), (24, 122, 12, 12), border_radius=2)
        pygame.draw.rect(screen, (147, 197, 253), (24, 122, 12, 12), 1, border_radius=2)
        wall_txt = self.small_font.render("Detected walls", True, Color.TEXT_PRIMARY.value)
        screen.blit(wall_txt, (42, 120))

        pygame.draw.rect(screen, Color.EXIT.value, (24, 142, 12, 12), border_radius=2)
        pygame.draw.rect(screen, (80, 240, 130), (24, 142, 12, 12), 1, border_radius=2)
        exit_txt = self.small_font.render("Detected exits", True, Color.TEXT_PRIMARY.value)
        screen.blit(exit_txt, (42, 140))

        zoom_pill_rect = pygame.Rect(self.canvas_rect.right - 246, screen_h - 56, 226, 42)
        pygame.draw.rect(screen, Color.CARD_BG.value, zoom_pill_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, zoom_pill_rect, 1, border_radius=7)
        zoom_val = int(cur.zoom_level * 100) if cur else 100
        zoom_txt = self.font.render(f"{zoom_val}%", True, (240, 240, 240))
        screen.blit(
            zoom_txt,
            (zoom_pill_rect.centerx - zoom_txt.get_width() // 2, zoom_pill_rect.centery - zoom_txt.get_height() // 2),
        )

        pygame.draw.rect(screen, Color.TOPNAV_BG.value, (0, 0, screen_w, self.top_nav_height))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 47), (screen_w, 47), 1)

        mark_rect = pygame.Rect(16, 12, 24, 24)
        pygame.draw.rect(screen, Color.ACCENT_BLUE.value, mark_rect, border_radius=5)
        pygame.draw.polygon(screen, (255, 255, 255), [
            (mark_rect.centerx - 4, mark_rect.centery - 5),
            (mark_rect.centerx + 4, mark_rect.centery),
            (mark_rect.centerx - 4, mark_rect.centery + 5)
        ])

        title_surf = self.title_font.render(
            "New Project - Floorplan Calibration", True, Color.TEXT_PRIMARY.value
        )
        screen.blit(title_surf, (48, 15))

        container_w = 260
        container_x = screen_w - container_w - 16
        nav_label = self.tiny_font.render("PROJECT", True, Color.TEXT_MUTED.value)
        screen.blit(nav_label, (container_x - nav_label.get_width() - 10, 18))
        pygame.draw.rect(screen, (21, 21, 21), (container_x, 7, container_w, 34), border_radius=6)

        pygame.draw.rect(screen, Color.FLOORTABS_BG.value, (0, 48, self.canvas_rect.width, self.tabs_height))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 87), (self.canvas_rect.width, 87), 1)

        floor_plan_surf = self.tiny_font.render("FLOOR PLAN", True, Color.TEXT_MUTED.value)
        screen.blit(floor_plan_surf, (self.canvas_rect.width - 260, 62))

        if cur:
            file_name = Path(cur.image_path).name if cur.image_path else "floorplan"
            source_info = f"{cur.img_w} × {cur.img_h} px ({file_name})"
            source_surf = self.tiny_font.render(source_info, True, (170, 170, 170))
            screen.blit(source_surf, (self.canvas_rect.width - source_surf.get_width() - 16, 62))

        pygame.draw.rect(screen, Color.SIDEBAR_BG.value, self.sidebar_rect)
        pygame.draw.line(screen, Color.DIVIDER.value, (sb_x, 48), (sb_x, screen_h), 1)

        hdr1 = self.tiny_font.render("GRID RESOLUTION", True, Color.TEXT_MUTED.value)
        screen.blit(hdr1, (sb_x + 20, 58))

        cur_bs = cur.block_size if cur else 25
        lbl_bs = self.font.render("Block Size:", True, (170, 170, 170))
        val_bs = self.font.render(f"{cur_bs} px", True, (240, 240, 240))
        screen.blit(lbl_bs, (sb_x + 20, 78))
        screen.blit(val_bs, (sb_x + 264 - val_bs.get_width(), 78))

        sub_l1 = self.font.render("10 px", True, (170, 170, 170))
        sub_r1 = self.font.render("50 px", True, (170, 170, 170))
        screen.blit(sub_l1, (sb_x + 20, 120))
        screen.blit(sub_r1, (sb_x + 264 - sub_r1.get_width(), 120))

        card_rect = pygame.Rect(sb_x + 16, 220, 248, 82)
        pygame.draw.rect(screen, Color.CARD_BG.value, card_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)

        gw, gh = cur.grid_dimensions if cur else (0, 0)
        lbl_dim = self.font.render("Grid Dimensions:", True, (170, 170, 170))
        val_dim = self.font.render(f"{gw} × {gh}", True, (240, 240, 240))
        screen.blit(lbl_dim, (card_rect.x + 14, card_rect.y + 11))
        screen.blit(val_dim, (card_rect.right - val_dim.get_width() - 14, card_rect.y + 11))

        pygame.draw.line(screen, (52, 52, 52), (card_rect.x + 14, card_rect.y + 37), (card_rect.right - 14, card_rect.y + 37), 1)

        cell_m = cur.cell_size_m if cur else 0.6
        lbl_scale = self.font.render("Real Scale:", True, (170, 170, 170))
        if cell_m >= 1.0:
            val_scale = self.font.render(f"1 cell = {cell_m:.2f} m", True, (240, 240, 240))
        else:
            val_scale = self.font.render(f"1 cell = {round(cell_m * 100)} cm", True, (240, 240, 240))
        screen.blit(lbl_scale, (card_rect.x + 14, card_rect.y + 48))
        screen.blit(val_scale, (card_rect.right - val_scale.get_width() - 14, card_rect.y + 48))

        pygame.draw.line(screen, Color.DIVIDER.value, (sb_x + 16, 310), (sb_x + 264, 310), 1)

        hdr2 = self.tiny_font.render("DETECTION SETTINGS", True, Color.TEXT_MUTED.value)
        screen.blit(hdr2, (sb_x + 20, 320))

        sens_pct = round((cur.acceptance_threshold if cur else 0.40) * 100)
        lbl_sens = self.font.render("Wall Sensitivity:", True, (170, 170, 170))
        val_sens = self.font.render(f"{sens_pct}%", True, (240, 240, 240))
        screen.blit(lbl_sens, (sb_x + 20, 342))
        screen.blit(val_sens, (sb_x + 264 - val_sens.get_width(), 342))

        sub_l2 = self.font.render("Low (15%)", True, (170, 170, 170))
        sub_r2 = self.font.render("High (65%)", True, (170, 170, 170))
        screen.blit(sub_l2, (sb_x + 20, 384))
        screen.blit(sub_r2, (sb_x + 264 - sub_r2.get_width(), 384))

        pygame.draw.line(screen, Color.DIVIDER.value, (sb_x + 16, 412), (sb_x + 264, 412), 1)

        hdr3 = self.tiny_font.render("CONTROLS", True, Color.TEXT_MUTED.value)
        screen.blit(hdr3, (sb_x + 20, 420))

        pygame.draw.line(screen, Color.DIVIDER.value, (sb_x + 16, screen_h - 72), (sb_x + 264, screen_h - 72), 1)

    def handle_mouse_wheel(self, program, event):
        mx, my = pygame.mouse.get_pos()
        if self.canvas_rect.collidepoint(mx, my):
            cur = self.active_floor
            if cur:
                old_zoom = cur.zoom_level
                cur.zoom_level = max(0.5, min(2.5, round(cur.zoom_level + event.y * 0.1, 2)))
                if cur.zoom_level != old_zoom:
                    self._update_overlay_surface()

    def _show_scale_input_dialog(self, program, dist_raw_px):
        if self._scale_input_window is not None:
            self._scale_input_window.kill()
        self.last_measured_dist_px = dist_raw_px
        win_w, win_h = 280, 140
        win_x = self.canvas_rect.centerx - win_w // 2
        win_y = self.canvas_rect.centery - win_h // 2
        self._scale_input_window = pygame_gui.elements.UIWindow(
            pygame.Rect(win_x, win_y, win_w, win_h),
            manager=program.ui_manager,
            window_display_title="Calibrate Scale",
        )
        self._scale_entry_meters = pygame_gui.elements.UITextEntryLine(
            pygame.Rect(14, 18, 140, 32),
            manager=program.ui_manager,
            container=self._scale_input_window,
            initial_text="1.0",
        )
        self._scale_btn_apply = pygame_gui.elements.UIButton(
            pygame.Rect(160, 18, 96, 32),
            text="Apply",
            manager=program.ui_manager,
            container=self._scale_input_window,
            object_id="#tool_btn_selected",
        )
        self._scale_btn_cancel = pygame_gui.elements.UIButton(
            pygame.Rect(14, 58, 242, 30),
            text="Cancel / Reset",
            manager=program.ui_manager,
            container=self._scale_input_window,
            object_id="#menu_button",
        )

    def handle_raw_mouse(self, program, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._slider_block_size is not None and self._slider_block_size.rect.collidepoint(event.pos):
                r = self._slider_block_size.rect
                ratio = max(0.0, min(1.0, (event.pos[0] - r.x) / max(1, r.width)))
                val = int(round(10 + ratio * 40))
                self._slider_block_size.set_current_value(val)
                cur = self.active_floor
                if cur is not None and cur.block_size != val:
                    cur.block_size = val
                    cur.update_cell_size()
                    if self.lock_grid_size:
                        for fl in self.floors:
                            fl.block_size = val
                            fl.update_cell_size()
                    self._recompute_detection()
            elif self._slider_sensitivity is not None and self._slider_sensitivity.rect.collidepoint(event.pos):
                r = self._slider_sensitivity.rect
                ratio = max(0.0, min(1.0, (event.pos[0] - r.x) / max(1, r.width)))
                val = int(round(15 + ratio * 50))
                self._slider_sensitivity.set_current_value(val)
                cur = self.active_floor
                if cur is not None:
                    cur.acceptance_threshold = val / 100.0
                    self._recompute_detection()

    def handle_mouse(self, program, event):
        if self.is_measuring and event.button == 1:
            mx, my = event.pos
            if self.canvas_rect.collidepoint(mx, my):
                cur = self.active_floor
                if cur and cur.raw_image:
                    if self.measure_p1 is None:
                        self.measure_p1 = (mx, my)
                    elif self.measure_p2 is None:
                        self.measure_p2 = (mx, my)
                        scale = cur.fit_scale * cur.zoom_level
                        dx = (self.measure_p2[0] - self.measure_p1[0]) / scale
                        dy = (self.measure_p2[1] - self.measure_p1[1]) / scale
                        dist_raw = (dx * dx + dy * dy) ** 0.5
                        if dist_raw >= 3:
                            self._show_scale_input_dialog(program, dist_raw)
                        else:
                            self.measure_p1 = None
                            self.measure_p2 = None
                return

        if event.button in (1, 2, 3):
            mx, my = event.pos
            if self.canvas_rect.collidepoint(mx, my) and my < self.screen_size[1] - 60:
                self.is_panning = True
                self.pan_start = event.pos

    def _finish_and_generate(self, program):
        if not self.floors:
            return

        if program.simulation is None:
            program.simulation = Simulation()
            first = True
            for fl in self.floors:
                gw, gh = fl.grid_dimensions
                if gw > 0 and gh > 0 and fl.wall_cells:
                    grid = Map_Grid(gw, gh, CellType.EMPTY)
                    for cx, cy in fl.wall_cells:
                        grid.add_wall(cx, cy)
                    if fl.detect_exits:
                        for cx, cy in fl.exit_cells:
                            grid.add_exit(cx, cy)
                else:
                    grid = Map_Grid.from_image(
                        fl.image_path,
                        fl.block_size,
                        acceptance_threshold=fl.acceptance_threshold,
                    )
                fl.update_cell_size()
                grid.cell_size_m = fl.cell_size_m
                if first:
                    program.initialize_world(grid)
                    first = False
                else:
                    program.simulation.grid_world.add_floor_from_map(grid)
                fl.is_added_to_simulation = True

            program.simulation.grid_world.set_active_floor(0)
            program.update_window_for_active_floor()
        else:
            for fl in self.floors:
                if not getattr(fl, 'is_added_to_simulation', False):
                    gw, gh = fl.grid_dimensions
                    if gw > 0 and gh > 0 and fl.wall_cells:
                        grid = Map_Grid(gw, gh, CellType.EMPTY)
                        for cx, cy in fl.wall_cells:
                            grid.add_wall(cx, cy)
                        if fl.detect_exits:
                            for cx, cy in fl.exit_cells:
                                grid.add_exit(cx, cy)
                    else:
                        grid = Map_Grid.from_image(
                            fl.image_path,
                            fl.block_size,
                            acceptance_threshold=fl.acceptance_threshold,
                        )
                    fl.update_cell_size()
                    grid.cell_size_m = fl.cell_size_m
                    program.simulation.grid_world.add_floor_from_map(grid)
                    fl.is_added_to_simulation = True

            new_floor_id = len(program.simulation.grid_world.floors) - 1
            program.simulation.grid_world.set_active_floor(new_floor_id)
            program.update_window_for_active_floor()

        program.transition_to(EditorState())

    def handle_ui_event(self, program, event):
        if event.ui_element == self._generate_button:
            self._finish_and_generate(program)

        elif event.ui_element == self._btn_add_floor:
            next_path = _open_file_dialog(
                "Select Another Floor Image",
                [("Images", "*.jpg *.png *.jpeg")],
            )
            if next_path:
                cur = self.active_floor
                cur_bs = cur.block_size if (self.lock_grid_size and cur) else 25
                cur_sens = cur.acceptance_threshold if cur else 0.40
                cur_px = cur.px_per_m if (self.lock_grid_size and cur) else None
                new_floor = FloorCalibrationData(
                    next_path,
                    block_size=cur_bs,
                    acceptance_threshold=cur_sens,
                    px_per_m=cur_px,
                )
                self._init_floor_fit(new_floor)
                self.floors.append(new_floor)
                self.active_floor_index = len(self.floors) - 1
                self._build_ui(program)
                self._recompute_detection()

        elif event.ui_element == self._btn_back:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())

        elif event.ui_element == self._btn_load_project:
            path = _open_file_dialog(
                "Open project",
                [("JSON files", "*.json")],
            )
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    program.initialize_world_from_dict(data)
                    program.transition_to(EditorState())
                except Exception as e:
                    logger.error("Failed to load project from %s: %s", path, e)

        elif event.ui_element == self._btn_lock_grid:
            self.lock_grid_size = not self.lock_grid_size
            if self.lock_grid_size:
                cur = self.active_floor
                if cur:
                    for fl in self.floors:
                        fl.block_size = cur.block_size
                        fl.px_per_m = cur.px_per_m
                        fl.update_cell_size()
                    self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == self._btn_coarse:
            cur = self.active_floor
            if cur:
                cur.block_size = 35
                cur.update_cell_size()
                if self.lock_grid_size:
                    for fl in self.floors:
                        fl.block_size = 35
                        fl.update_cell_size()
                if self._slider_block_size:
                    self._slider_block_size.set_current_value(35)
                self._recompute_detection()
                self._build_ui(program)

        elif event.ui_element == self._btn_standard:
            cur = self.active_floor
            if cur:
                cur.block_size = 25
                cur.update_cell_size()
                if self.lock_grid_size:
                    for fl in self.floors:
                        fl.block_size = 25
                        fl.update_cell_size()
                if self._slider_block_size:
                    self._slider_block_size.set_current_value(25)
                self._recompute_detection()
                self._build_ui(program)

        elif event.ui_element == self._btn_fine:
            cur = self.active_floor
            if cur:
                cur.block_size = 15
                cur.update_cell_size()
                if self.lock_grid_size:
                    for fl in self.floors:
                        fl.block_size = 15
                        fl.update_cell_size()
                if self._slider_block_size:
                    self._slider_block_size.set_current_value(15)
                self._recompute_detection()
                self._build_ui(program)

        elif event.ui_element == self._btn_measure_scale:
            self.is_measuring = not self.is_measuring
            self.measure_p1 = None
            self.measure_p2 = None
            if self._scale_input_window:
                self._scale_input_window.kill()
                self._scale_input_window = None
            self._build_ui(program)

        elif event.ui_element in (self._btn_reset_measure, self._scale_btn_cancel):
            self.is_measuring = False
            self.measure_p1 = None
            self.measure_p2 = None
            if self._scale_input_window:
                self._scale_input_window.kill()
                self._scale_input_window = None
            self._build_ui(program)

        elif event.ui_element == self._scale_btn_apply:
            cur = self.active_floor
            if cur and hasattr(self, "last_measured_dist_px") and self._scale_entry_meters:
                try:
                    val_str = self._scale_entry_meters.get_text().strip().replace(",", ".")
                    meters = float(val_str)
                    if meters > 0:
                        px_per_m = self.last_measured_dist_px / meters
                        new_bs = int(round(0.6 * px_per_m))
                        new_bs = max(10, min(50, new_bs))
                        cur.px_per_m = px_per_m
                        cur.block_size = new_bs
                        cur.update_cell_size()
                        if self.lock_grid_size:
                            for fl in self.floors:
                                fl.px_per_m = px_per_m
                                fl.block_size = new_bs
                                fl.update_cell_size()
                        if self._slider_block_size:
                            self._slider_block_size.set_current_value(new_bs)
                        self._recompute_detection()
                except Exception as e:
                    logger.warning("Invalid scale meters input: %s", e)
            self.is_measuring = False
            self.measure_p1 = None
            self.measure_p2 = None
            if self._scale_input_window:
                self._scale_input_window.kill()
                self._scale_input_window = None
            self._build_ui(program)

        elif event.ui_element == self._btn_toggle_exits:
            cur = self.active_floor
            if cur:
                cur.detect_exits = not cur.detect_exits
                self._recompute_detection()
                self._build_ui(program)

        elif event.ui_element == self._btn_zoom_in:
            cur = self.active_floor
            if cur:
                cur.zoom_level = min(2.5, round(cur.zoom_level + 0.25, 2))
                self._update_overlay_surface()

        elif event.ui_element == self._btn_zoom_out:
            cur = self.active_floor
            if cur:
                cur.zoom_level = max(0.5, round(cur.zoom_level - 0.25, 2))
                self._update_overlay_surface()

        else:
            for idx, tab_btn in enumerate(self._floor_tab_buttons):
                if event.ui_element == tab_btn:
                    if self.active_floor_index != idx:
                        self.active_floor_index = idx
                        self._build_ui(program)
                        if self.active_floor and not self.active_floor.wall_cells:
                            self._recompute_detection()
                        else:
                            self._update_overlay_surface()
                    break

    def handle_key_event(self, program, key):
        if key == pygame.K_ESCAPE:
            if self.is_measuring or self._scale_input_window:
                self.is_measuring = False
                self.measure_p1 = None
                self.measure_p2 = None
                if self._scale_input_window:
                    self._scale_input_window.kill()
                    self._scale_input_window = None
                self._build_ui(program)
                return
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif key == pygame.K_LEFT:
            cur = self.active_floor
            if cur:
                cur.pan_offset_x += 30
                self._update_overlay_surface()
        elif key == pygame.K_RIGHT:
            cur = self.active_floor
            if cur:
                cur.pan_offset_x -= 30
                self._update_overlay_surface()
        elif key == pygame.K_UP:
            cur = self.active_floor
            if cur:
                cur.pan_offset_y += 30
                self._update_overlay_surface()
        elif key == pygame.K_DOWN:
            cur = self.active_floor
            if cur:
                cur.pan_offset_y -= 30
                self._update_overlay_surface()
        elif key == pygame.K_RETURN:
            if self._scale_input_window and hasattr(self, "_scale_btn_apply"):
                event_dummy = pygame.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=self._scale_btn_apply)
                self.handle_ui_event(program, event_dummy)
                return
            self._finish_and_generate(program)

    def exit(self, program):
        if getattr(self, "_scale_input_window", None) is not None:
            self._scale_input_window.kill()
            self._scale_input_window = None
        self.is_measuring = False
        self.measure_p1 = None
        self.measure_p2 = None
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        self._slider_block_size = None
        self._slider_sensitivity = None
        self._btn_coarse = None
        self._btn_standard = None
        self._btn_fine = None
        self._btn_measure_scale = None
        self._btn_reset_measure = None
        self._scale_btn_apply = None
        self._scale_btn_cancel = None
        self._scale_entry_meters = None
        self._btn_lock_grid = None
        self._btn_toggle_exits = None
        self._generate_button = None
        self._btn_back = None
        self._btn_load_project = None
        self._btn_add_floor = None
        self._btn_zoom_in = None
        self._btn_zoom_out = None
        self._floor_tab_buttons = []
        self.floors = []

    def status_lines(self, context):
        cur = self.active_floor
        bs = cur.block_size if cur else 25
        return [
            f"State: {self.name}",
            f"Active floor: {self.active_floor_index + 1}/{len(self.floors)}",
            f"Block size: {bs}px",
        ]


class EditorState(ProgramState):
    name = "editor"
    can_pan = True
    floor_switch_enabled = True
    key_bindings = ()

    def __init__(self):
        self.selected_tool = Tool.WALL
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
        self._btn_random_crowd = None
        self._crowd_window = None
        self._crowd_entry = None
        self._crowd_btn_apply = None
        self._crowd_btn_cancel = None
        self._save_project_btn = None
        self._btn_toggle_grid = None

    def enter(self, program):
        program.view_options.show_heatmap = False
        program.view_options.show_clusters = False
        program.view_options.show_cluster_connections = False
        program.view_options.show_drone_routes = False
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
        tool_w = 248
        tool_h = 34
        y = 166

        self._btn_wall = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Wall",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.WALL else "#tool_btn",
        )
        y += 38
        self._btn_evacuee = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Evacuee",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.EVACUEE else "#tool_btn",
        )
        y += 38
        self._btn_exit = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Exit",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.EXIT else "#tool_btn",
        )
        y += 38
        self._btn_drone = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Agent",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.DRONE else "#tool_btn",
        )
        y += 38
        self._btn_stairs = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, tool_h),
            "Stairs",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.STAIRS else "#tool_btn",
        )

        btn_w_half = 122
        gap = 4
        action_y = 402
        self._btn_random_crowd = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, action_y, btn_w_half, 34),
            "Random Crowd",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        self._save_project_btn = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, action_y, btn_w_half, 34),
            "Save Project",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, action_y + 40, tool_w, 32),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if program.view_options.show_grid else "#tool_btn",
        )

    def _show_crowd_dialog(self, program):
        if self._crowd_window is not None:
            self._crowd_window.kill()
            self._crowd_window = None

        screen_w, screen_h = program.screen.get_size()
        win_w = 300
        win_h = 160
        win_x = max(20, (screen_w - SIDEBAR_WIDTH - win_w) // 2)
        win_y = max(20, (screen_h - win_h) // 2)

        self._crowd_window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=program.ui_manager,
            window_display_title="Generate Random Crowd",
            object_id="#scale_window",
        )
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, 12, win_w - 32, 24),
            text="Evacuees per floor:",
            manager=program.ui_manager,
            container=self._crowd_window,
        )
        self._crowd_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(16, 40, win_w - 32, 30),
            manager=program.ui_manager,
            container=self._crowd_window,
            initial_text="30",
        )
        btn_w = (win_w - 40) // 2
        self._crowd_btn_apply = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 80, btn_w, 32),
            text="Generate",
            manager=program.ui_manager,
            container=self._crowd_window,
            object_id="#btn_start_sim",
        )
        self._crowd_btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(24 + btn_w, 80, btn_w, 32),
            text="Cancel",
            manager=program.ui_manager,
            container=self._crowd_window,
            object_id="#menu_button",
        )

    def _apply_random_crowd(self, program):
        if self._crowd_entry is None or not program.simulation:
            return
        try:
            count = int(self._crowd_entry.get_text().strip())
        except (ValueError, TypeError):
            count = 30
        count = max(1, min(1000, count))

        world = program.simulation.grid_world
        world.build_distance_maps()

        for f_id, floor in world.floors.items():
            for y in range(floor.map_grid.height):
                for x in range(floor.map_grid.width):
                    if floor.map_grid.get(x, y) == CellType.EVACUEE:
                        floor.map_grid.set(x, y, CellType.EMPTY)

            w, h = floor.map_grid.width, floor.map_grid.height
            wall_barrier = np.zeros((h, w), dtype=np.uint8)
            wall_pts = []
            for y in range(h):
                for x in range(w):
                    if floor.map_grid.get(x, y) in (CellType.WALL, CellType.EXIT):
                        wall_barrier[y, x] = 255
                        wall_pts.append((x, y))

            inside_mask = None
            if wall_pts:
                min_x = min(pt[0] for pt in wall_pts)
                max_x = max(pt[0] for pt in wall_pts)
                min_y = min(pt[1] for pt in wall_pts)
                max_y = max(pt[1] for pt in wall_pts)

                for k_size in (5, 7, 9):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    closed = cv2.morphologyEx(wall_barrier, cv2.MORPH_CLOSE, kernel)
                    pad = cv2.copyMakeBorder(closed, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
                    cv2.floodFill(pad, None, (0, 0), 128, flags=8)
                    cv2.floodFill(pad, None, (pad.shape[1] - 1, 0), 128, flags=8)
                    cv2.floodFill(pad, None, (0, pad.shape[0] - 1), 128, flags=8)
                    cv2.floodFill(pad, None, (pad.shape[1] - 1, pad.shape[0] - 1), 128, flags=8)
                    outside = (pad[4:-4, 4:-4] == 128)
                    raw_inside = (~outside) & (wall_barrier == 0)
                    clean_inside = cv2.erode(raw_inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    if np.sum(clean_inside) > 0:
                        inside_mask = (clean_inside > 0)
                        break

            dist_map = getattr(floor, "distance_map", None)
            candidates = []
            for y in range(h):
                for x in range(w):
                    if floor.map_grid.get(x, y) != CellType.EMPTY:
                        continue
                    if hasattr(floor, "temporary_wall_grid") and floor.temporary_wall_grid and floor.temporary_wall_grid.get(x, y):
                        continue
                    if getattr(floor, "drone_occupancy_grid", None) and floor.drone_occupancy_grid.get(x, y) is not None:
                        continue
                    if wall_pts and not (min_x < x < max_x and min_y < y < max_y):
                        continue
                    is_inside = bool(inside_mask[y, x]) if inside_mask is not None else False
                    has_route = (dist_map is not None) and (dist_map[y][x] < float("inf"))
                    if is_inside and has_route:
                        candidates.append((x, y))

            if candidates:
                chosen = random.sample(candidates, min(count, len(candidates)))
                for cx, cy in chosen:
                    floor.map_grid.set(cx, cy, CellType.EVACUEE)

        if self._crowd_window:
            self._crowd_window.kill()
            self._crowd_window = None

        program.simulation.prepare()
        program.plan_drone_routes()
        self._rebuild_ui(program)

    def exit(self, program):
        if getattr(self, "_crowd_window", None) is not None:
            self._crowd_window.kill()
            self._crowd_window = None
        program.simulation.grid_world.rebuild_changed_map_data()
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_ui_event(self, program, event):
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
            program.simulation.prepare()
            program.plan_drone_routes()
            program.transition_to(SimulationPausedState())
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
        elif event.ui_element == self._btn_random_crowd:
            self._show_crowd_dialog(program)
        elif event.ui_element == self._crowd_btn_apply or (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == self._crowd_entry):
            self._apply_random_crowd(program)
        elif event.ui_element == self._crowd_btn_cancel:
            if self._crowd_window:
                self._crowd_window.kill()
                self._crowd_window = None
        elif event.ui_element == self._btn_toggle_grid:
            program.view_options.show_grid = not program.view_options.show_grid
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
                        self.on_floor_changed(program)
                    break

    def handle_key(self, program, key):
        if key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)

    def handle_mouse(self, program, event):
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        handle_map_editor(
            program.simulation,
            event.pos,
            event.button,
            self.selected_tool,
        )

    def handle_mouse_motion(self, program, event):
        buttons = event.buttons
        if not buttons[0] and not buttons[2]:
            return
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        btn = 1 if buttons[0] else 3
        handle_map_editor(
            program.simulation,
            event.pos,
            btn,
            self.selected_tool,
        )

    def status_lines(self, context):
        lines = []
        if context.simulation:
            msg = getattr(context.simulation, "last_editor_error", None)
            timer = getattr(context.simulation, "last_editor_error_timer", 0)
            if msg and pygame.time.get_ticks() < timer:
                lines.append(f"Error: {msg}")
        return lines

    def render(self, program):
        program.window.draw_editor(program, self.selected_tool, self.key_bindings)


class OverviewState(ProgramState):
    name = "overview"
    render_options_enabled = True
    floor_switch_enabled = True
    can_pan = True
    key_bindings = ()

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
        self._btn_toggle_clusters = None
        self._btn_toggle_connections = None
        self._btn_toggle_routes = None
        self._btn_toggle_assignments = None
        self._btn_toggle_grid = None

    def enter(self, program):
        program.view_options.show_heatmap = False
        program.simulation.prepare()
        program.drone_routes = {}
        if program.drone_controller is not None:
            program.drone_controller.reset()
        if program.rl_evacuee_controller is not None:
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

        self._btn_link_toggle = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 276, tool_w, 32),
            "Stair Link: Active" if program.link_stairs_mode else "Enable Stair Link",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if program.link_stairs_mode else "#tool_btn",
        )

        y = 352
        opts = program.view_options
        btn_w_half = 122
        gap = 4

        self._btn_toggle_clusters = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Clusters",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_clusters else "#tool_btn",
        )
        self._btn_toggle_connections = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Links",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_cluster_connections else "#tool_btn",
        )

        y += 32
        self._btn_toggle_routes = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Agent Routes",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_drone_routes else "#tool_btn",
        )
        self._btn_toggle_assignments = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Assign",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_assignments else "#tool_btn",
        )

        y += 32
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, 28),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_grid else "#tool_btn",
        )

    def exit(self, program):
        program.link_stairs_mode = False
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_ui_event(self, program, event):
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
            program.simulation.prepare()
            program.plan_drone_routes()
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_toggle_clusters:
            program.view_options.show_clusters = not program.view_options.show_clusters
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_connections:
            program.view_options.show_cluster_connections = not program.view_options.show_cluster_connections
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_routes:
            program.view_options.show_drone_routes = not program.view_options.show_drone_routes
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_assignments:
            program.view_options.show_assignments = not program.view_options.show_assignments
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_grid:
            program.view_options.show_grid = not program.view_options.show_grid
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
                        self.on_floor_changed(program)
                    break


    def handle_key(self, program, key):
        if key == pygame.K_l:
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
        if not world.map_grid.in_bounds(grid_x, grid_y):
            return
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
    key_bindings = ()

    def __init__(self):
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_add_floor = None
        self._btn_pause = None
        self._btn_reset = None
        self._btn_start_sim = None
        self._btn_toggle_clusters = None
        self._btn_toggle_connections = None
        self._btn_toggle_routes = None
        self._btn_toggle_assignments = None
        self._btn_toggle_grid = None
        self._btn_toggle_heatmap = None
        self._btn_finish = None

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
        btn_w_half = 122
        gap = 4
        opts = program.view_options
        y = 246

        self._btn_toggle_clusters = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Clusters",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_clusters else "#tool_btn",
        )
        self._btn_toggle_connections = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Links",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_cluster_connections else "#tool_btn",
        )
        y += 32
        self._btn_toggle_routes = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Agent Routes",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_drone_routes else "#tool_btn",
        )
        self._btn_toggle_assignments = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Assign",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_assignments else "#tool_btn",
        )
        y += 32
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_grid else "#tool_btn",
        )
        self._btn_toggle_heatmap = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Heatmap",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_heatmap else "#tool_btn",
        )

        playback_y = 384
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
        self._btn_finish = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, playback_y + 86, tool_w, 32),
            "Finish (Results)",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def handle_ui_event(self, program, event):
        if event.ui_element == self._save_btn:
            _save_project(program)
        elif event.ui_element == self._btn_start_sim:
            if not program.simulation.started:
                program.simulation.start()
            else:
                program.simulation.paused = False
                self._rebuild_ui(program)
        elif event.ui_element == self._btn_finish:
            program.transition_to(ResultsState())
        elif event.ui_element == self._menu_btn:
            program.screen = pygame.display.set_mode(MENU_WINDOW_SIZE)
            if program.ui_manager:
                program.ui_manager.set_window_resolution(MENU_WINDOW_SIZE)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element == self._nav_link:
            program.transition_to(OverviewState())
        elif event.ui_element == self._btn_pause:
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_toggle_clusters:
            program.view_options.show_clusters = not program.view_options.show_clusters
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_connections:
            program.view_options.show_cluster_connections = not program.view_options.show_cluster_connections
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_routes:
            program.view_options.show_drone_routes = not program.view_options.show_drone_routes
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_assignments:
            program.view_options.show_assignments = not program.view_options.show_assignments
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_grid:
            program.view_options.show_grid = not program.view_options.show_grid
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_heatmap:
            program.view_options.show_heatmap = not program.view_options.show_heatmap
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_reset:
            if program.simulation:
                program.simulation.grid_world.clear_temporary_walls()
                program.simulation.prepare()
                program.drone_routes = {}
                if program.drone_controller is not None:
                    program.drone_controller.reset()
                if program.rl_evacuee_controller is not None:
                    program.rl_evacuee_controller.reset()
                program.plan_drone_routes()
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
        if program.simulation.started and len(program.simulation.evacuees) == 0:
            program.transition_to(SimulationPausedState())
            return

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

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

class SimulationPausedState(ProgramState):
    name = "simulation_paused"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = ()

    def __init__(self):
        self._floor_tabs = {}
        self._btn_add_floor = None
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_hazard = None
        self._btn_resume = None
        self._btn_reset = None
        self._btn_start_sim = None
        self._btn_toggle_clusters = None
        self._btn_toggle_connections = None
        self._btn_toggle_routes = None
        self._btn_toggle_assignments = None
        self._btn_toggle_grid = None
        self._btn_toggle_heatmap = None
        self._btn_finish = None

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
        btn_w_half = 122
        gap = 4
        opts = program.view_options

        self._btn_hazard = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 246, tool_w, 34),
            "Place Hazards (M)",
            manager=program.ui_manager,
            object_id="#tool_btn",
        )

        y = 322
        self._btn_toggle_clusters = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Clusters",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_clusters else "#tool_btn",
        )
        self._btn_toggle_connections = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Links",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_cluster_connections else "#tool_btn",
        )
        y += 32
        self._btn_toggle_routes = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Agent Routes",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_drone_routes else "#tool_btn",
        )
        self._btn_toggle_assignments = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Assign",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_assignments else "#tool_btn",
        )
        y += 32
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 28),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_grid else "#tool_btn",
        )
        self._btn_toggle_heatmap = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 28),
            "Heatmap",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_heatmap else "#tool_btn",
        )
        playback_y = 458
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
        self._btn_finish = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, playback_y + 86, tool_w, 32),
            "Finish (Results)",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def handle_ui_event(self, program, event):
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
        elif event.ui_element == self._btn_finish:
            program.transition_to(ResultsState())
        elif event.ui_element == self._btn_toggle_clusters:
            program.view_options.show_clusters = not program.view_options.show_clusters
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_connections:
            program.view_options.show_cluster_connections = not program.view_options.show_cluster_connections
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_routes:
            program.view_options.show_drone_routes = not program.view_options.show_drone_routes
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_assignments:
            program.view_options.show_assignments = not program.view_options.show_assignments
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_grid:
            program.view_options.show_grid = not program.view_options.show_grid
            self._rebuild_ui(program)
        elif event.ui_element == self._btn_toggle_heatmap:
            program.view_options.show_heatmap = not program.view_options.show_heatmap
            self._rebuild_ui(program)
        elif event.ui_element in (self._btn_resume, self._btn_start_sim):
            program.transition_to(SimulationRunningState())
        elif event.ui_element == self._btn_reset:
            if program.simulation:
                program.simulation.grid_world.clear_temporary_walls()
                program.simulation.prepare()
                program.drone_routes = {}
                if program.drone_controller is not None:
                    program.drone_controller.reset()
                if program.rl_evacuee_controller is not None:
                    program.rl_evacuee_controller.reset()
                program.plan_drone_routes()
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
        if key == pygame.K_p:
            program.transition_to(SimulationRunningState())
        elif key == pygame.K_m:
            program.transition_to(SimulationEditorState())


class SimulationEditorState(ProgramState):
    name = "simulation_editor"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = ()

    def __init__(self):
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_link = None
        self._nav_sim = None
        self._btn_done = None
        self._btn_clear_hazards = None

    def enter(self, program):
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        self._rebuild_ui(program)

    def exit(self, program):
        program.replan_drone_routes_after_hazard()
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

        tool_x = sidebar_x + 16
        tool_w = 248

        self._btn_done = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 248, tool_w, 38),
            "Done (Return to Simulation)",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        self._btn_clear_hazards = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 294, tool_w, 34),
            "Clear All Hazards",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def handle_ui_event(self, program, event):
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
        elif event.ui_element in (self._nav_sim, self._btn_done):
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_clear_hazards:
            program.simulation.grid_world.clear_temporary_walls()
            program.replan_drone_routes_after_hazard()
            self._rebuild_ui(program)
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
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(SimulationPausedState())
        elif key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)

    def handle_mouse(self, program, event):
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        if event.button not in (1, 3):
            return

        world = program.simulation.grid_world
        grid_x, grid_y = pixel_to_grid(*event.pos)
        if not world.map_grid.in_bounds(grid_x, grid_y):
            return
        placed = world.set_temporary_wall(
            world.active_floor_id, grid_x, grid_y, event.button == 1
        )
        if placed:
            program.replan_drone_routes_after_hazard()


class ResultsState(ProgramState):
    name = "results"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = ()

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
            pygame.Rect(sidebar_x + 16, 216, 248, 38),
            "Reset Simulation",
            manager=program.ui_manager,
            object_id="#btn_start_sim",
        )

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_ui_event(self, program, event):
        if event.ui_element == self._btn_reset:
            if program.simulation:
                program.simulation.grid_world.clear_temporary_walls()
                program.simulation.prepare()
                program.drone_routes = {}
                if program.drone_controller is not None:
                    program.drone_controller.reset()
                if program.rl_evacuee_controller is not None:
                    program.rl_evacuee_controller.reset()
                program.plan_drone_routes()
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