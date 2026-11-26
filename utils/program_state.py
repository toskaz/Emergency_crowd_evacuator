import os
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
from editor.editor import handle_map_editor, place
from editor.tool import Tool
from graphics.colors import Color
from graphics.fonts import get_font
from graphics.renderer import draw_building_elevation
from grids.coordinates import grid_to_pixel, grid_to_pixel_center, pixel_to_grid
from grids.map_grid import Map_Grid
from map.cell import CellType
from simulation import Simulation
from utils.config import block_size, acceptance_threshold
from map.image_extractor import (
    preprocess_floorplan_image,
    detect_grid_cells,
)
from map.floorplan_loader import (
    ImportSettings,
    SUPPORTED_FLOORPLAN_FILETYPES,
    build_aligned_reference_surface,
    build_preprocessed_for_floorplan,
    compute_dxf_px_per_m,
    compute_pdf_px_per_m,
    compute_smart_block_size,
    crop_loaded_floorplan,
    load_floorplan,
    numpy_bgr_to_pygame_surface,
    reprocess_floorplan,
    reset_crop_floorplan,
    resolve_plan_source_path,
    restore_reference_image,
    rotate_loaded_floorplan,
    sample_wall_at_loaded_coords,
)
from utils.ui_dialogs import (
    AddFloorChoiceDialog,
    DeleteFloorConfirmDialog,
    DxfLayersDialog,
    NoticeDialog,
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


def _save_file_dialog(title, default_ext, filetypes, initialfile=None):
    if platform.system() == "Darwin":
        init_name = initialfile or f"project{default_ext}"
        script = f'POSIX path of (choose file name with prompt "{title}" default name "{init_name}")'
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
    kwargs = {"title": title, "defaultextension": default_ext, "filetypes": filetypes}
    if initialfile:
        kwargs["initialfile"] = initialfile
    path = filedialog.asksaveasfilename(**kwargs)
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

    def __init__(self):
        self.title_font = get_font(22, bold=True)
        self.font = get_font(15)
        self.small_font = get_font(13)
        self.tiny_font = get_font(11, bold=True)

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
            ("DQN" if context.simulation_options.rl_drones else "A*")
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
        if getattr(context, "drone_mode_notice", ""):
            lines.append(context.drone_mode_notice)
        world = context.simulation.grid_world if context.simulation else None
        if world and context.link_stairs_mode:
            if world.pending_stair_endpoint is not None:
                pending_floor = world.pending_stair_endpoint[0] + 1
                lines.append(f"  Stairs on Floor {pending_floor} selected. Switch floor and click target stairs!")
            else:
                lines.append("  Stair Link Mode: Click first stairs")
        import time
        if world and world.stair_link_notice:
            if time.time() - getattr(world, 'stair_link_notice_time', 0.0) < 3.5:
                lines.append(f"Notice: {world.stair_link_notice}")
            else:
                world.stair_link_notice = None
        if world and world.stair_link_error:
            if time.time() - getattr(world, 'stair_link_notice_time', 0.0) < 3.5:
                lines.append(f"Error: {world.stair_link_error}")
            else:
                world.stair_link_error = None
        return lines

    def exit(self, program):
        world = program.simulation.grid_world if program.simulation else None
        if world:
            world.stair_link_notice = None
            world.stair_link_error = None

    def on_floor_changed(self, program):
        pass

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)

    def _get_bottom_pill_rects(self, screen_w, screen_h):
        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        pill_y = screen_h - 52
        zoom_rect = pygame.Rect(canvas_w - 212 - 20, pill_y, 212, 38)
        overlay_rect = pygame.Rect(zoom_rect.x - 12 - 192, pill_y, 192, 38)
        return overlay_rect, zoom_rect

    def _build_bottom_canvas_controls(self, mgr, screen_w, screen_h, overlay_on, has_ref, zoom_btn_id="#menu_button"):
        overlay_rect, zoom_rect = self._get_bottom_pill_rects(screen_w, screen_h)
        off_obj = "#nav_item_inactive" if (not has_ref or overlay_on) else "#btn_secondary"
        on_obj = "#nav_item_active" if (has_ref and overlay_on) else "#nav_item_inactive"
        self._btn_plan_overlay_off = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(overlay_rect.x + 96, overlay_rect.y + 4, 44, 30),
            text="OFF",
            manager=mgr,
            object_id=off_obj,
        )
        self._btn_plan_overlay_on = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(overlay_rect.x + 144, overlay_rect.y + 4, 44, 30),
            text="ON",
            manager=mgr,
            object_id=on_obj,
        )
        if not has_ref:
            self._btn_plan_overlay_off.disable()
            self._btn_plan_overlay_on.disable()
        self._btn_zoom_out = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(zoom_rect.x + 4, zoom_rect.y + 4, 68, 30),
            text="Zoom -",
            manager=mgr,
            object_id=zoom_btn_id,
        )
        self._btn_zoom_in = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(zoom_rect.x + 140, zoom_rect.y + 4, 68, 30),
            text="Zoom +",
            manager=mgr,
            object_id=zoom_btn_id,
        )
        return overlay_rect, zoom_rect

    def _draw_bottom_canvas_pills(self, screen, zoom_pct, has_ref):
        screen_w, screen_h = screen.get_size()
        overlay_rect, zoom_rect = self._get_bottom_pill_rects(screen_w, screen_h)
        for r in (overlay_rect, zoom_rect):
            pygame.draw.rect(screen, Color.CARD_BG.value, r, border_radius=7)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, r, 1, border_radius=7)
        lbl_col = (220, 220, 220) if has_ref else (115, 115, 120)
        ov_txt = self.small_font.render("Plan Overlay", True, lbl_col)
        screen.blit(ov_txt, ov_txt.get_rect(left=overlay_rect.x + 10, centery=overlay_rect.centery))
        zoom_txt = self.font.render(f"{zoom_pct}%", True, (240, 240, 240))
        screen.blit(zoom_txt, zoom_txt.get_rect(center=zoom_rect.center))

    def handle_key_event(self, program, key):
        import utils.config as config

        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            if getattr(self, "can_pan", False):
                config.drawing_size = min(100, config.drawing_size + 2)
                if hasattr(program, "save_active_floor_camera"):
                    program.save_active_floor_camera()
            return
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            if getattr(self, "can_pan", False):
                config.drawing_size = max(2, config.drawing_size - 2)
                if hasattr(program, "save_active_floor_camera"):
                    program.save_active_floor_camera()
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
        super().__init__()
        self._import_button = None
        self._blank_button = None
        self._open_button = None
        self._notice_dialog = None

    def enter(self, program):
        program.simulation = None
        program.imported_image_path = None
        program.pending_floor_clone = None
        program.view_options.show_plan_overlay = False
        program.drone_routes = {}
        program.link_stairs_mode = False
        self._import_button = None
        self._blank_button = None
        self._open_button = None
        self._notice_dialog = None
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

        screen_w, screen_h = program.screen.get_size()
        card_w = 280
        card_h = 192
        gap = 20
        total_w = card_w * 3 + gap * 2
        start_x = (screen_w - total_w) // 2
        cards_y = 236

        self._import_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + 18, cards_y + 138, card_w - 36, 36),
            text="Import Floorplan...",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

        self._blank_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + card_w + gap + 18, cards_y + 138, card_w - 36, 36),
            text="Create Blank Map",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

        self._open_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + (card_w + gap) * 2 + 18, cards_y + 138, card_w - 36, 36),
            text="Open Project (.json)",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def exit(self, program):
        if getattr(self, "_notice_dialog", None) is not None:
            self._notice_dialog.kill()
            self._notice_dialog = None
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        self._import_button = None
        self._blank_button = None
        self._open_button = None

    def handle_ui_event(self, program, event):
        if getattr(self, "_notice_dialog", None) is not None:
            if self._notice_dialog.handle_event(event) == "close":
                self._notice_dialog.kill()
                self._notice_dialog = None
            return

        if event.ui_element == self._import_button:
            path = _open_file_dialog(
                "Select Floorplan",
                SUPPORTED_FLOORPLAN_FILETYPES,
            )
            if path:
                program.calibrated_floors = []
                program.simulation = None
                program.imported_image_path = path
                program.pending_floor_clone = None
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
                program.initialize_world_from_dict(data, project_path=path)
                program.transition_to(EditorState())

    def handle_key_event(self, program, key):
        pass

    def update(self, program):
        pass

    def render(self, program):
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        screen.fill((22, 22, 22))

        header_font = get_font(34, bold=True)
        sub_font = get_font(18)
        card_title_font = get_font(20, bold=True)
        card_desc_font = get_font(15)

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
            "Import a DXF, PDF, or image floorplan, build a grid from scratch, or open a saved project",
            True,
            (150, 153, 159),
        )
        screen.blit(sub_surf, (screen_w // 2 - sub_surf.get_width() // 2, 182))

        card_w = 280
        card_h = 192
        gap = 20
        total_w = card_w * 3 + gap * 2
        start_x = (screen_w - total_w) // 2
        cards_y = 236

        cards_info = [
            (
                "New from Floorplan",
                "DXF  •  PDF  •  PNG / JPG",
                "Import a DXF, PDF, PNG, or JPG floorplan and calibrate scale.",
                (43, 43, 43),
            ),
            (
                "New Blank Grid",
                None,
                "Start with an empty architectural grid and draw walls, stairs, and exits manually.",
                (43, 43, 43),
            ),
            (
                "Open Project (.json)",
                None,
                "Load an existing multi-floor evacuation map, stair links, and simulation setup.",
                (43, 43, 43),
            ),
        ]

        tag_font = get_font(11, bold=True)
        for i, (title, fmt_tag, desc, border_col) in enumerate(cards_info):
            cx = start_x + i * (card_w + gap)
            card_rect = pygame.Rect(cx, cards_y, card_w, card_h)
            pygame.draw.rect(screen, (30, 30, 30), card_rect, border_radius=9)
            pygame.draw.rect(screen, border_col, card_rect, 1, border_radius=9)

            t_surf = card_title_font.render(title, True, (240, 240, 242))
            screen.blit(t_surf, (cx + 18, cards_y + 18))

            dy = cards_y + 50
            if fmt_tag:
                tag_surf = tag_font.render(fmt_tag, True, (56, 189, 248))
                screen.blit(tag_surf, (cx + 18, cards_y + 44))
                dy = cards_y + 64

            words = desc.split(" ")
            lines = []
            cur_line = ""
            for w in words:
                test = cur_line + (" " if cur_line else "") + w
                if card_desc_font.size(test)[0] <= card_w - 36:
                    cur_line = test
                else:
                    lines.append(cur_line)
                    cur_line = w
            if cur_line:
                lines.append(cur_line)

            for l in lines:
                l_surf = card_desc_font.render(l, True, (165, 167, 173))
                screen.blit(l_surf, (cx + 18, dy))
                dy += 18

    def status_lines(self, context):
        return ["State: main_menu"]


def _delete_floor_from_world(program, deleted_idx):
    sim = getattr(program, "simulation", None)
    if sim is None or sim.grid_world is None:
        return False
    if not sim.grid_world.delete_floor(deleted_idx):
        return False
    sim.evacuees = [
        ev for ev in sim.evacuees
        if getattr(ev, "floor_id", 0) != deleted_idx
    ]
    for ev in sim.evacuees:
        fid = getattr(ev, "floor_id", 0)
        if fid > deleted_idx:
            ev.floor_id = fid - 1
    sim.drones = [
        dr for dr in sim.drones
        if getattr(dr, "floor_id", 0) != deleted_idx
    ]
    for dr in sim.drones:
        fid = getattr(dr, "floor_id", 0)
        if fid > deleted_idx:
            dr.floor_id = fid - 1
    program.drone_routes = {}
    if getattr(program, "drone_controller", None) is not None:
        program.drone_controller.reset()
    return True


class FloorCalibrationData:
    def __init__(
        self,
        image_path,
        block_size=None,
        acceptance_threshold=0.40,
        px_per_m=None,
        floor_height_m=3.0,
        page_index=0,
        rotation_deg=0,
        crop_rect=None,
        settings=None,
        detect_exits=None,
        target_cell_meters=0.50,
        inherit_scale=False,
    ):
        self.image_path = image_path
        self.target_cell_meters = max(0.05, float(target_cell_meters))
        self.settings = settings if settings is not None else ImportSettings(sensitivity=float(acceptance_threshold))
        self.settings.sensitivity = float(acceptance_threshold)
        self.loaded = load_floorplan(
            image_path,
            page_index=page_index,
            crop_rect=crop_rect,
            rotation_deg=rotation_deg,
            settings=self.settings,
        )
        self.acceptance_threshold = float(self.settings.sensitivity)
        self.detect_exits = bool(detect_exits) if detect_exits is not None else False

        self.show_original_plan = False
        self.has_manual_scale = False
        self.raw_image = None
        self.ref_raw_image = None
        self.img_w = 0
        self.img_h = 0
        self.preprocessed = None

        if self.loaded.px_per_m is not None and self.loaded.px_per_m > 0:
            self.px_per_m = float(self.loaded.px_per_m)
        elif px_per_m is not None and float(px_per_m) > 0:
            self.px_per_m = float(px_per_m)
            self.loaded.px_per_m = self.px_per_m
            if inherit_scale and self.loaded.source_type in ("image", "scanned_pdf"):
                self.loaded.has_native_scale = True
        else:
            self.px_per_m = 25.0 / 0.6

        self.sync_from_loaded(recompute_preprocessed=True)

        if block_size is not None:
            min_safe = max(8, int((max(1, self.img_w, self.img_h) + 189) // 190))
            self.block_size = max(min_safe, int(block_size))
        else:
            self.block_size = compute_smart_block_size(
                self.img_w,
                self.img_h,
                self.px_per_m if self.loaded.has_native_scale else None,
                target_cell_meters=self.target_cell_meters,
            )
        self.slider_max_bs = max(50, self.block_size + 20)
        self.update_cell_size()
        if block_size is not None:
            self.target_cell_meters = float(self.cell_size_m)

        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.fit_scale = 1.0
        self.wall_cells = set()
        self.exit_cells = set()
        self.grid_dimensions = (0, 0)
        self.display_image = None
        self.ref_display_image = None
        self.overlay_surface = None
        self.disp_x = 0
        self.disp_y = 0
        self.disp_w = 0
        self.disp_h = 0
        self.is_added_to_simulation = False
        self.floor_height_m = float(floor_height_m)

    def sync_from_loaded(self, recompute_preprocessed=True):
        self.raw_image = numpy_bgr_to_pygame_surface(self.loaded.extraction_image)
        self.ref_raw_image = numpy_bgr_to_pygame_surface(self.loaded.reference_image)
        self.img_w, self.img_h = self.raw_image.get_size()
        if self.loaded.px_per_m is not None and self.loaded.px_per_m > 0:
            self.px_per_m = float(self.loaded.px_per_m)
        if recompute_preprocessed:
            self.settings.sensitivity = float(self.acceptance_threshold)
            self.preprocessed = build_preprocessed_for_floorplan(self.loaded, self.settings)

    def update_cell_size(self):
        if getattr(self, "px_per_m", 0) > 0:
            self.cell_size_m = self.block_size / self.px_per_m
        else:
            self.cell_size_m = (self.block_size / 25.0) * 0.6

    def apply_target_cell_size(self, target_m=None):
        if target_m is not None:
            self.target_cell_meters = max(0.05, float(target_m))
        min_safe = max(8, int((max(1, self.img_w, self.img_h) + 189) // 190))
        if self.loaded.has_native_scale and self.px_per_m > 0:
            raw_bs = int(round(self.target_cell_meters * self.px_per_m))
        else:
            raw_bs = int(round((self.target_cell_meters / 0.6) * 25.0))
        self.slider_max_bs = max(50, raw_bs + 20)
        self.block_size = max(min_safe, min(self.slider_max_bs, raw_bs))
        self.update_cell_size()

    def reset_manual_scale(self):
        self.has_manual_scale = False
        if self.loaded.source_type == "dxf":
            self.loaded.px_per_m = compute_dxf_px_per_m(
                self.loaded.pixels_per_source_unit,
                self.loaded.dxf_units,
            )
            self.loaded.has_native_scale = True
            self.px_per_m = self.loaded.px_per_m
        elif self.loaded.source_type == "vector_pdf" and self.settings.pdf_scale_denom:
            self.loaded.px_per_m = compute_pdf_px_per_m(
                self.loaded.pixels_per_source_unit,
                self.settings.pdf_scale_denom,
            )
            self.loaded.has_native_scale = True
            self.px_per_m = self.loaded.px_per_m
        else:
            self.loaded.px_per_m = None
            self.loaded.has_native_scale = False
            self.px_per_m = 25.0 / 0.6
        self.apply_target_cell_size()

    def refresh_smart_block_size_after_crop(self):
        min_safe = max(8, int((max(1, self.img_w, self.img_h) + 189) // 190))
        if self.loaded.has_native_scale and self.px_per_m > 0:
            self.block_size = compute_smart_block_size(
                self.img_w,
                self.img_h,
                self.px_per_m,
                target_cell_meters=self.target_cell_meters,
            )
        else:
            self.block_size = max(min_safe, self.block_size)
        self.slider_max_bs = max(50, self.block_size + 20)
        self.update_cell_size()


class CalibrationState(ProgramState):
    name = "calibration"

    def __init__(self):
        self.is_panning = False
        self.pan_start = (0, 0)
        self.screen_size = (1280, 800)
        self.lock_grid_size = True
        self.is_measuring = False
        self.measure_p1 = None
        self.measure_p2 = None
        self.last_measured_dist_px = 0.0
        self.is_cropping = False
        self.crop_p1 = None
        self.crop_p2 = None
        self.is_picking_color = False

        self._slider_block_size = None
        self._slider_sensitivity = None
        self._slider_remove_thin = None
        self._slider_pdf_thickness = None

        self._btn_coarse = None
        self._btn_standard = None
        self._btn_fine = None
        self._btn_measure_scale = None
        self._btn_reset_measure = None

        self._btn_rot_left = None
        self._btn_rot_right = None
        self._btn_crop = None
        self._btn_reset_crop = None
        self._btn_plan_overlay_off = None
        self._btn_plan_overlay_on = None

        self._btn_dxf_layers = None
        self._dxf_unit_buttons = {}
        self._btn_page_prev = None
        self._btn_page_next = None
        self._btn_pdf_fixed = None
        self._btn_pdf_filled_only = None
        self._pdf_scale_buttons = {}

        self._btn_pick_color = None
        self._btn_pick_auto = None
        self._btn_img_ignore_colors = None
        self._btn_img_gray_blocks = None

        self._scale_input_window = None
        self._scale_entry_meters = None
        self._scale_btn_apply = None
        self._scale_btn_cancel = None
        self._btn_lock_grid = None
        self._btn_toggle_exits = None
        self._btn_floor_height = None
        self._height_window = None
        self._height_entry = None
        self._height_same_all_btn = None
        self._height_btn_apply = None
        self._height_btn_cancel = None
        self._add_floor_dialog = None
        self._delete_floor_dialog = None
        self._dxf_layers_dialog = None
        self._notice_dialog = None

        self._generate_button = None
        self._btn_back = None
        self._btn_load_project = None
        self._btn_add_floor = None
        self._btn_delete_floor = None
        self._btn_zoom_in = None
        self._btn_zoom_out = None
        self._floor_tab_buttons = []
        self.floors = []
        self._panel_layout = {
            "headers": [],
            "dividers": [],
            "sublabels": [],
            "slider_headers": [],
            "summary_card": None,
            "page_label": None,
        }

        logo_path = Path(__file__).resolve().parent.parent / "logo" / "logoSquare.png"
        self.logo_surf = None
        if logo_path.exists():
            try:
                raw_logo = pygame.image.load(str(logo_path)).convert_alpha()
                self.logo_surf = pygame.transform.smoothscale(raw_logo, (20, 20))
            except Exception:
                self.logo_surf = None

    def _clone_settings(self, src_settings: ImportSettings) -> ImportSettings:
        from map.image_extractor import ImageFilterSettings
        img_f = src_settings.image_filters
        return ImportSettings(
            sensitivity=src_settings.sensitivity,
            image_filters=ImageFilterSettings(
                mode=img_f.mode,
                ignore_colors=img_f.ignore_colors,
                gray_blocks=img_f.gray_blocks,
                remove_thin_px=img_f.remove_thin_px,
                picked_rgb=img_f.picked_rgb,
                picked_lab=img_f.picked_lab,
                sensitivity=img_f.sensitivity,
            ),
            dxf_selected_layers=set(src_settings.dxf_selected_layers) if src_settings.dxf_selected_layers is not None else None,
            dxf_units=src_settings.dxf_units,
            pdf_min_thickness_pt=src_settings.pdf_min_thickness_pt,
            pdf_include_fixed=src_settings.pdf_include_fixed,
            pdf_filled_walls_only=src_settings.pdf_filled_walls_only,
            pdf_scale_denom=src_settings.pdf_scale_denom,
        )

    def _create_floor_safe(self, program, path, **kwargs):
        try:
            return FloorCalibrationData(path, **kwargs)
        except Exception as exc:
            logger.error("Failed to load floorplan %s: %s", path, exc)
            self._last_import_error = f"Could not import floorplan:<br>{exc}"
            self._notice_dialog = NoticeDialog(
                program.ui_manager,
                self.screen_size,
                "Import Error",
                self._last_import_error,
            )
            return None

    def _append_new_floor(self, program, new_fl, ref_target_m, *, has_ref_floors, build_ui=True, show_height=True):
        if new_fl is None:
            return
        if self.lock_grid_size and has_ref_floors:
            self._sync_floor_to_physical_scale(new_fl, ref_target_m)
        self._init_floor_fit(new_fl)
        self.floors.append(new_fl)
        self.active_floor_index = len(self.floors) - 1
        if build_ui:
            self._build_ui(program)
            self._recompute_detection()
            if show_height and len(self.floors) > 1:
                self._show_height_dialog(program)

    def enter(self, program):
        if not hasattr(program, 'calibrated_floors') or program.calibrated_floors is None:
            program.calibrated_floors = []

        new_path = program.imported_image_path
        program.imported_image_path = None
        clone_req = getattr(program, "pending_floor_clone", None)
        program.pending_floor_clone = None

        if not program.calibrated_floors and not new_path and not clone_req:
            logger.warning("No floorplan path provided for calibration; returning to main menu")
            cur_size = program.screen.get_size()
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())
            return

        self.screen_size = program.screen.get_size()
        if program.ui_manager is not None:
            program.ui_manager.set_window_resolution(self.screen_size)
            program.ui_manager.clear_and_reset()

        self.title_font = get_font(22, bold=True)
        self.font = get_font(15)
        self.small_font = get_font(13)
        self.tiny_font = get_font(11, bold=True)

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
        self.is_cropping = False
        self.crop_p1 = None
        self.crop_p2 = None
        self.is_picking_color = False
        self._add_floor_dialog = None
        self._delete_floor_dialog = None
        self._dxf_layers_dialog = None
        self._notice_dialog = None

        self.floors = program.calibrated_floors
        has_ref_floors = bool(
            self.floors
            or (program.simulation and program.simulation.grid_world and program.simulation.grid_world.floors)
        )
        ref_target_m, ref_px, ref_has_scale, def_h = self._resolve_reference_scale(program)

        if clone_req:
            src_path = clone_req.get("source_path")
            rot_deg = int(clone_req.get("rotation_deg", 0) or 0)
            p_idx = int(clone_req.get("page_index", 0) or 0)
            cloned_settings = clone_req.get("settings")
            if cloned_settings is not None:
                cloned_settings = self._clone_settings(cloned_settings)
            clone_px = clone_req.get("px_per_m") if clone_req.get("px_per_m") else ref_px
            clone_has_scale = bool(clone_req.get("has_native_scale")) or (ref_has_scale and bool(clone_px))
            new_fl = self._create_floor_safe(
                program,
                src_path,
                page_index=p_idx,
                rotation_deg=rot_deg,
                crop_rect=None,
                settings=cloned_settings,
                px_per_m=clone_px,
                inherit_scale=clone_has_scale,
                floor_height_m=def_h,
                detect_exits=clone_req.get("detect_exits"),
                target_cell_meters=ref_target_m,
            )
            self._append_new_floor(program, new_fl, ref_target_m, has_ref_floors=has_ref_floors, build_ui=False, show_height=False)
        elif new_path:
            src_fl = None
            if self.floors:
                act_idx = 0
                if program.simulation and program.simulation.grid_world:
                    act_idx = getattr(program.simulation.grid_world, "active_floor_id", 0)
                src_fl = self.floors[act_idx] if 0 <= act_idx < len(self.floors) else self.floors[-1]
            img_kwargs = self._inherited_image_kwargs(src_fl, new_path)
            new_fl = self._create_floor_safe(
                program,
                new_path,
                px_per_m=ref_px,
                inherit_scale=ref_has_scale,
                floor_height_m=def_h,
                target_cell_meters=ref_target_m,
                **img_kwargs,
            )
            self._append_new_floor(program, new_fl, ref_target_m, has_ref_floors=has_ref_floors, build_ui=False, show_height=False)
        else:
            self.active_floor_index = min(getattr(self, 'active_floor_index', 0), max(0, len(self.floors) - 1))

        if not self.floors:
            err_dlg = self._notice_dialog
            self._build_ui(program)
            if err_dlg is not None:
                self._notice_dialog = NoticeDialog(
                    program.ui_manager,
                    self.screen_size,
                    "Import Error",
                    getattr(self, "_last_import_error", "Could not import floorplan."),
                )
            return

        self._build_ui(program)
        self._recompute_detection()

        total_floors = len(self.floors)
        if program.simulation and program.simulation.grid_world:
            total_floors = max(total_floors, len(program.simulation.grid_world.floors))
        if total_floors > 1 and (new_path or clone_req):
            self._show_height_dialog(program)

    @property
    def active_floor(self):
        if 0 <= getattr(self, "active_floor_index", 0) < len(self.floors):
            return self.floors[self.active_floor_index]
        return None

    def _resolve_reference_scale(self, program):
        ref_target_m = 0.50
        ref_px = None
        ref_has_scale = False
        def_h = 3.0
        if self.floors:
            f0 = self.floors[0]
            if getattr(program, "same_height_for_all", True):
                def_h = getattr(f0, "floor_height_m", 3.0)
            if self.lock_grid_size:
                ref_target_m = getattr(f0, "target_cell_meters", f0.cell_size_m)
                if getattr(f0, "px_per_m", 0) > 0:
                    ref_px = float(f0.px_per_m)
                ref_has_scale = bool(f0.loaded.has_native_scale)
        elif program.simulation and program.simulation.grid_world and program.simulation.grid_world.floors:
            f0 = program.simulation.grid_world.floors[0]
            if getattr(program, "same_height_for_all", True):
                def_h = getattr(f0, "floor_height_m", 3.0)
            if self.lock_grid_size:
                ref_target_m = getattr(f0, "cell_size_m", 0.50)
                plan_ref = getattr(f0, "plan_reference", None)
                ref_bs = plan_ref.get("block_size") if isinstance(plan_ref, dict) else None
                if ref_bs and ref_target_m > 0:
                    ref_px = float(ref_bs) / float(ref_target_m)
                    ref_has_scale = True
        return ref_target_m, ref_px, ref_has_scale, def_h

    def _inherited_image_kwargs(self, src_fl, path: str) -> dict:
        if src_fl is None:
            return {"acceptance_threshold": 0.40}
        ext = Path(path).suffix.lower()
        if src_fl.loaded.source_type == "image" and ext not in (".dxf", ".pdf"):
            return {
                "acceptance_threshold": src_fl.acceptance_threshold,
                "settings": self._clone_settings(src_fl.settings),
                "detect_exits": src_fl.detect_exits,
            }
        return {"acceptance_threshold": src_fl.acceptance_threshold}

    def _sync_floor_to_physical_scale(self, target_fl, target_cell_m):
        if target_fl is None or target_cell_m <= 0:
            return
        target_fl.target_cell_meters = max(0.05, float(target_cell_m))
        ref_fl = self.floors[0] if self.floors else self.active_floor
        if not (target_fl.loaded.has_native_scale and target_fl.px_per_m > 0):
            if ref_fl and ref_fl.px_per_m > 0:
                target_fl.px_per_m = ref_fl.px_per_m
                target_fl.loaded.px_per_m = ref_fl.px_per_m
                if ref_fl.loaded.has_native_scale and target_fl.loaded.source_type in ("image", "scanned_pdf"):
                    target_fl.loaded.has_native_scale = True
        target_fl.apply_target_cell_size()

    def _sync_all_floors_scale(self, source_fl):
        if source_fl is None or not self.lock_grid_size:
            return
        target_cell_m = getattr(source_fl, "target_cell_meters", source_fl.cell_size_m)
        for fl in self.floors:
            if fl is source_fl:
                continue
            self._sync_floor_to_physical_scale(fl, target_cell_m)
            self._recompute_floor_detection(fl)

    def _apply_scale_and_refresh(self, cur, program, target_m=None, *, sync_slider=False, rebuild_ui=True):
        cur.apply_target_cell_size(target_m)
        self._sync_all_floors_scale(cur)
        if sync_slider and self._slider_block_size:
            self._slider_block_size.set_current_value(cur.block_size)
        self._recompute_detection()
        if rebuild_ui:
            self._build_ui(program)

    def _refresh_after_crop_or_page(self, cur, program):
        cur.sync_from_loaded(recompute_preprocessed=True)
        if self.lock_grid_size and len(self.floors) > 1 and cur is not self.floors[0]:
            cur.target_cell_meters = float(getattr(self.floors[0], "target_cell_meters", self.floors[0].cell_size_m))
        cur.refresh_smart_block_size_after_crop()
        cur.zoom_level = 1.0
        cur.pan_offset_x = 0
        cur.pan_offset_y = 0
        self._init_floor_fit(cur)
        self._sync_all_floors_scale(cur)
        self._recompute_detection()
        self._build_ui(program)

    def _init_floor_fit(self, fl):
        margin = 32
        avail_w = max(100, self.canvas_rect.width - (2 * margin))
        avail_h = max(100, self.canvas_rect.height - (2 * margin))
        fl.fit_scale = min(avail_w / max(1, fl.img_w), avail_h / max(1, fl.img_h))

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
        for fl in self.floors:
            self._init_floor_fit(fl)
        self._build_ui(program)
        self._update_overlay_surface()

    def _build_ui(self, program):
        if program.ui_manager is None:
            return

        program.ui_manager.clear_and_reset()
        mgr = program.ui_manager
        screen_w, screen_h = self.screen_size
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
        self._btn_delete_floor = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(tab_x + 122, 54, 100, 28),
            text="Delete Floor",
            manager=mgr,
            object_id="#btn_destructive",
        )
        if len(self.floors) <= 1:
            self._btn_delete_floor.disable()

        cur = self.active_floor
        orig_on = bool(cur and cur.show_original_plan)
        has_ref = bool(cur and cur.ref_raw_image is not None)
        self._build_bottom_canvas_controls(
            mgr,
            self.screen_size[0],
            self.screen_size[1],
            orig_on,
            has_ref,
            zoom_btn_id="#menu_button",
        )

        self._slider_block_size = None
        self._slider_sensitivity = None
        self._slider_remove_thin = None
        self._slider_pdf_thickness = None
        self._btn_dxf_layers = None
        self._dxf_unit_buttons = {}
        self._btn_page_prev = None
        self._btn_page_next = None
        self._btn_pdf_fixed = None
        self._btn_pdf_filled_only = None
        self._pdf_scale_buttons = {}
        self._btn_pick_color = None
        self._btn_pick_auto = None
        self._btn_img_ignore_colors = None
        self._btn_img_gray_blocks = None
        self._btn_rot_left = None
        self._btn_rot_right = None
        self._btn_crop = None
        self._btn_reset_crop = None
        self._btn_toggle_exits = None
        self._btn_lock_grid = None
        self._btn_floor_height = None
        self._btn_reset_measure = None

        layout = {
            "headers": [],
            "dividers": [],
            "sublabels": [],
            "slider_headers": [],
            "summary_card": None,
            "page_label": None,
        }

        x = sb_x + 16
        w = 248
        gap = 6
        half_w = (w - gap) // 2
        btn_h = 26
        src_type = cur.loaded.source_type if cur else "image"
        src_badge = {
            "dxf": "DXF",
            "vector_pdf": "VECTOR PDF",
            "scanned_pdf": "SCANNED PDF",
            "image": "IMAGE",
        }.get(src_type, "PLAN")

        cy = 58
        layout["headers"].append((cy, "PLAN", src_badge))
        cy += 20

        self._btn_rot_left = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, cy, half_w, btn_h),
            text="-90°",
            manager=mgr,
            object_id="#btn_secondary",
        )
        self._btn_rot_right = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x + half_w + gap, cy, w - half_w - gap, btn_h),
            text="+90°",
            manager=mgr,
            object_id="#btn_secondary",
        )
        cy += 30

        has_crop = bool(cur and cur.loaded.crop_rect is not None)
        if has_crop:
            self._btn_crop = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, half_w, btn_h),
                text="Crop Area",
                manager=mgr,
                object_id="#tool_btn_selected" if self.is_cropping else "#tool_btn",
            )
            self._btn_reset_crop = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x + half_w + gap, cy, w - half_w - gap, btn_h),
                text="Reset Crop",
                manager=mgr,
                object_id="#btn_secondary",
            )
        else:
            self._btn_crop = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Crop Area",
                manager=mgr,
                object_id="#tool_btn_selected" if self.is_cropping else "#tool_btn",
            )
        cy += 34

        layout["dividers"].append(cy)
        cy += 8

        layout["headers"].append((cy, "WALL DETECTION", None))
        cy += 20

        dex = cur.detect_exits if cur else False

        if src_type == "dxf":
            total_layers = len(cur.loaded.dxf_layers)
            sel_layers = sum(1 for l in cur.loaded.dxf_layers if l.selected)
            self._btn_dxf_layers = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text=f"Layers ({sel_layers}/{total_layers})...",
                manager=mgr,
                object_id="#btn_secondary",
            )
            cy += 30

            layout["sublabels"].append((cy, "Units"))
            cy += 18
            unit_widths = (47, 46, 47, 46, 46)
            ux = x
            cur_u = cur.loaded.dxf_units
            for u, uw in zip(("mm", "cm", "m", "in", "ft"), unit_widths):
                btn = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(ux, cy, uw, btn_h),
                    text=u,
                    manager=mgr,
                    object_id="#btn_preset_active" if cur_u == u else "#btn_preset_inactive",
                )
                self._dxf_unit_buttons[u] = btn
                ux += uw + 4
            cy += 30

        elif src_type == "vector_pdf":
            if cur.loaded.page_count > 1:
                cy, page_info = self._build_page_switcher(mgr, cur, x, cy, w)
                layout["page_label"] = page_info

            layout["sublabels"].append((cy, "Drawing Scale"))
            cy += 18
            scale_widths = (58, 57, 58, 57)
            sx = x
            active_denom = cur.settings.pdf_scale_denom
            for denom, sw in zip((50, 100, 200, 500), scale_widths):
                btn = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(sx, cy, sw, btn_h),
                    text=f"1:{denom}",
                    manager=mgr,
                    object_id="#btn_preset_active" if active_denom == denom else "#btn_preset_inactive",
                )
                self._pdf_scale_buttons[denom] = btn
                sx += sw + 6
            cy += 30

            thick = cur.settings.pdf_min_thickness_pt
            layout["slider_headers"].append((cy, "Min Line Width", lambda c: f"{c.settings.pdf_min_thickness_pt:.2f} pt"))
            cy += 18
            thick_step = int(round(max(0.0, min(3.0, thick)) / 0.05))
            self._slider_pdf_thickness = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(x, cy, w, 18),
                start_value=thick_step,
                value_range=(0, 60),
                manager=mgr,
                click_increment=1,
            )
            cy += 22

            fix_on = cur.settings.pdf_include_fixed
            self._btn_pdf_fixed = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Include Furniture: ON" if fix_on else "Include Furniture: OFF",
                manager=mgr,
                object_id="#tool_btn_selected" if fix_on else "#tool_btn",
            )
            cy += 30

            filled_only = bool(getattr(cur.settings, "pdf_filled_walls_only", False))
            self._btn_pdf_filled_only = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Filled Walls Only: ON" if filled_only else "Filled Walls Only: OFF",
                manager=mgr,
                object_id="#tool_btn_selected" if filled_only else "#tool_btn",
            )
            cy += 30

        else:
            if cur and cur.loaded.page_count > 1:
                cy, page_info = self._build_page_switcher(mgr, cur, x, cy, w)
                layout["page_label"] = page_info

            sens = int(round(cur.acceptance_threshold * 100)) if cur else 40
            layout["slider_headers"].append((cy, "Wall Sensitivity", lambda c: f"{int(round(c.acceptance_threshold * 100))}%"))
            cy += 18
            self._slider_sensitivity = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(x, cy, w, 18),
                start_value=max(10, min(80, sens)),
                value_range=(10, 80),
                manager=mgr,
                click_increment=1,
            )
            cy += 22

            img_f = cur.settings.image_filters if cur else None
            thin_px = int(img_f.remove_thin_px) if img_f else 0
            layout["slider_headers"].append((cy, "Remove Thin Lines", lambda c: f"{int(c.settings.image_filters.remove_thin_px)} px"))
            cy += 18
            self._slider_remove_thin = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(x, cy, w, 18),
                start_value=thin_px,
                value_range=(0, 4),
                manager=mgr,
                click_increment=1,
            )
            cy += 22

            pick_mode = bool(img_f and img_f.mode == "pick_color" and img_f.picked_rgb)
            if pick_mode:
                pick_w = 170
                reset_w = w - pick_w - gap
                self._btn_pick_color = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(x, cy, pick_w, btn_h),
                    text="Pick Wall Color",
                    manager=mgr,
                    object_id="#tool_btn_selected" if self.is_picking_color else "#tool_btn",
                )
                self._btn_pick_auto = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(x + pick_w + gap, cy, reset_w, btn_h),
                    text="Reset",
                    manager=mgr,
                    object_id="#btn_secondary",
                )
            else:
                self._btn_pick_color = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(x, cy, w, btn_h),
                    text="Pick Wall Color",
                    manager=mgr,
                    object_id="#tool_btn_selected" if self.is_picking_color else "#tool_btn",
                )
            cy += 30

            ign_col = bool(img_f.ignore_colors) if img_f else True
            self._btn_img_ignore_colors = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Ignore Colors: ON" if ign_col else "Ignore Colors: OFF",
                manager=mgr,
                object_id="#tool_btn_selected" if ign_col else "#tool_btn",
            )
            cy += 30

            gray_blk = bool(img_f.gray_blocks) if img_f else False
            self._btn_img_gray_blocks = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Include Furniture: ON" if gray_blk else "Include Furniture: OFF",
                manager=mgr,
                object_id="#tool_btn_selected" if gray_blk else "#tool_btn",
            )
            cy += 30

        self._btn_toggle_exits = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, cy, w, btn_h),
            text="Detect Exits: ON" if dex else "Detect Exits: OFF",
            manager=mgr,
            object_id="#tool_btn_selected" if dex else "#tool_btn",
        )
        cy += 32

        layout["dividers"].append(cy)
        cy += 8

        layout["headers"].append((cy, "GRID & SCALE", None))
        cy += 20

        bs = cur.block_size if cur else 25
        max_bs = cur.slider_max_bs if cur else 50
        target_m = getattr(cur, "target_cell_meters", 0.50) if cur else 0.50

        layout["slider_headers"].append((cy, "Block Size", lambda c: f"{c.block_size} px"))
        cy += 18
        self._slider_block_size = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(x, cy, w, 18),
            start_value=bs,
            value_range=(8, max_bs),
            manager=mgr,
            click_increment=1,
        )
        cy += 22

        has_phys_scale = bool(cur and cur.loaded.has_native_scale and cur.px_per_m > 0)
        preset_labels = ("60 cm", "50 cm", "40 cm") if has_phys_scale else ("Coarse", "Medium", "Fine")
        pw = 80
        pgap = 4
        self._btn_coarse = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, cy, pw, btn_h),
            text=preset_labels[0],
            manager=mgr,
            object_id="#btn_preset_active" if abs(target_m - 0.60) < 0.015 else "#btn_preset_inactive",
        )
        self._btn_standard = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x + pw + pgap, cy, pw, btn_h),
            text=preset_labels[1],
            manager=mgr,
            object_id="#btn_preset_active" if abs(target_m - 0.50) < 0.015 else "#btn_preset_inactive",
        )
        self._btn_fine = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x + 2 * (pw + pgap), cy, pw, btn_h),
            text=preset_labels[2],
            manager=mgr,
            object_id="#btn_preset_active" if abs(target_m - 0.40) < 0.015 else "#btn_preset_inactive",
        )
        cy += 30

        has_manual = bool(cur and getattr(cur, "has_manual_scale", False))
        if has_manual:
            meas_w = 176
            rst_w = w - meas_w - gap
            self._btn_measure_scale = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, meas_w, btn_h),
                text="Measure Scale",
                manager=mgr,
                object_id="#tool_btn_selected" if self.is_measuring else "#tool_btn",
            )
            self._btn_reset_measure = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x + meas_w + gap, cy, rst_w, btn_h),
                text="Reset",
                manager=mgr,
                object_id="#btn_secondary",
            )
        else:
            self._btn_measure_scale = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Measure Scale",
                manager=mgr,
                object_id="#tool_btn_selected" if self.is_measuring else "#tool_btn",
            )
        cy += 30

        layout["summary_card"] = pygame.Rect(x, cy, w, 54)
        cy += 62

        if len(self.floors) >= 2:
            layout["dividers"].append(cy)
            cy += 8
            layout["headers"].append((cy, "MULTI-FLOOR", None))
            cy += 20

            self._btn_lock_grid = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text="Sync Floor Scales: ON" if self.lock_grid_size else "Sync Floor Scales: OFF",
                manager=mgr,
                object_id="#tool_btn_selected" if self.lock_grid_size else "#tool_btn",
            )
            cy += 30

            active_idx = getattr(self, "active_floor_index", 0)
            if active_idx > 0:
                f_prev, f_num = active_idx, active_idx + 1
                cur_h = getattr(cur, "floor_height_m", 3.0) if cur else 3.0
            else:
                f_prev, f_num = 1, 2
                if getattr(program, "same_height_for_all", True):
                    cur_h = getattr(self.floors[0], "floor_height_m", 3.0)
                else:
                    cur_h = getattr(self.floors[1], "floor_height_m", 3.0)
            self._btn_floor_height = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(x, cy, w, btn_h),
                text=f"Floor Height ({f_prev}-{f_num}): {cur_h:.1f} m",
                manager=mgr,
                object_id="#btn_secondary",
            )
            cy += 26

        self._panel_layout = layout

        self._generate_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, screen_h - 58, w, 42),
            text="Open in Editor",
            manager=mgr,
            object_id="#btn_start_sim",
        )

    def _build_page_switcher(self, mgr, cur, x, cy, w):
        nav_w = 40
        self._btn_page_prev = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, cy, nav_w, 26),
            text="<",
            manager=mgr,
            object_id="#btn_secondary",
        )
        self._btn_page_next = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x + w - nav_w, cy, nav_w, 26),
            text=">",
            manager=mgr,
            object_id="#btn_secondary",
        )
        page_info = (cy, f"Page {cur.loaded.page_index + 1} / {cur.loaded.page_count}")
        return cy + 30, page_info

    def _show_height_dialog(self, program):
        if getattr(self, "_height_window", None) is not None:
            self._height_window.kill()
            self._height_window = None

        screen_w, screen_h = program.screen.get_size()
        win_w = 340
        win_h = 230
        win_x = max(20, (screen_w - SIDEBAR_WIDTH - win_w) // 2)
        win_y = max(20, (screen_h - win_h) // 2)

        f_num = max(2, self.active_floor_index + 1)
        f_prev = f_num - 1

        self._height_window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=program.ui_manager,
            window_display_title=f"Set Height (Floor {f_prev}–{f_num})",
            object_id="#scale_window",
        )
        cur = self.active_floor
        same = getattr(program, "same_height_for_all", True)
        if same and self.floors:
            cur_h = getattr(self.floors[0], "floor_height_m", 3.0)
        elif self.active_floor_index == 0 and len(self.floors) > 1:
            cur_h = getattr(self.floors[1], "floor_height_m", 3.0)
        else:
            cur_h = getattr(cur, "floor_height_m", 3.0) if cur else 3.0

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, 10, win_w - 32, 22),
            text=f"Height between Floor {f_prev} and Floor {f_num} (m):",
            manager=program.ui_manager,
            container=self._height_window,
        )
        self._height_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(16, 34, win_w - 32, 30),
            manager=program.ui_manager,
            container=self._height_window,
            initial_text=f"{cur_h:.1f}",
        )
        self._height_info_lbl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, 68, win_w - 32, 20),
            text=f"~{round(cur_h / 0.17)} stair steps (standard 17cm rise)",
            manager=program.ui_manager,
            container=self._height_window,
        )
        self._height_same_all_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 94, win_w - 32, 30),
            text="Same height for all floors: ON" if same else "Same height for all floors: OFF",
            manager=program.ui_manager,
            container=self._height_window,
            object_id="#tool_btn_selected" if same else "#tool_btn",
        )
        btn_w = (win_w - 40) // 2
        self._height_btn_apply = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 136, btn_w, 32),
            text="Apply",
            manager=program.ui_manager,
            container=self._height_window,
            object_id="#btn_start_sim",
        )
        self._height_btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(24 + btn_w, 136, btn_w, 32),
            text="Cancel",
            manager=program.ui_manager,
            container=self._height_window,
            object_id="#menu_button",
        )

    def _recompute_floor_detection(self, fl):
        if fl is None or fl.preprocessed is None:
            return
        mpp = (1.0 / fl.px_per_m) if (fl.px_per_m > 0 and fl.loaded.has_native_scale) else None
        fl.wall_cells, fl.exit_cells, gw, gh = detect_grid_cells(
            fl.preprocessed,
            fl.block_size,
            fl.acceptance_threshold,
            detect_exits=fl.detect_exits,
            meters_per_pixel=mpp,
        )
        fl.grid_dimensions = (gw, gh)
        if fl is self.active_floor:
            self._update_overlay_surface()

    def _recompute_detection(self):
        self._recompute_floor_detection(self.active_floor)

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
        if cur.ref_raw_image is not None:
            cur.ref_display_image = pygame.transform.smoothscale(
                cur.ref_raw_image, (cur.disp_w, cur.disp_h)
            )
            cur.ref_display_image.set_alpha(120)
        else:
            cur.ref_display_image = None

        cur.overlay_surface = pygame.Surface((cur.disp_w, cur.disp_h), pygame.SRCALPHA)

        cell_w = cur.block_size * scale
        cell_h = cur.block_size * scale
        gw, gh = cur.grid_dimensions

        wall_color = (59, 130, 246, 145)
        for cx, cy in cur.wall_cells:
            rx = int(cx * cell_w)
            ry = int(cy * cell_h)
            rw = max(1, int((cx + 1) * cell_w) - rx)
            rh = max(1, int((cy + 1) * cell_h) - ry)
            pygame.draw.rect(cur.overlay_surface, wall_color, (rx, ry, rw, rh))

        exit_color = (16, 185, 129, 170)
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

    def _screen_to_image_coords(self, cur, sx, sy):
        scale = cur.fit_scale * cur.zoom_level
        if scale <= 0:
            return 0.0, 0.0
        ix = (sx - cur.disp_x) / scale
        iy = (sy - cur.disp_y) / scale
        return ix, iy

    def _reprocess_and_refresh(self, cur, program):
        reprocess_floorplan(cur.loaded, cur.settings)
        cur.sync_from_loaded(recompute_preprocessed=True)
        min_safe = max(8, int((max(1, cur.img_w, cur.img_h) + 189) // 190))
        if cur.block_size < min_safe:
            cur.block_size = min_safe
            cur.slider_max_bs = max(50, cur.block_size + 20)
        cur.update_cell_size()
        self._init_floor_fit(cur)
        self._recompute_detection()

    def _apply_slider_value(self, kind, raw_val, cur, program):
        if kind == "block_size":
            val = int(round(raw_val))
            if val != cur.block_size:
                cur.block_size = val
                cur.update_cell_size()
                cur.target_cell_meters = float(cur.cell_size_m)
                self._sync_all_floors_scale(cur)
                self._recompute_detection()
        elif kind == "sensitivity":
            new_sens = int(round(raw_val)) / 100.0
            if abs(new_sens - cur.acceptance_threshold) >= 0.01:
                cur.acceptance_threshold = new_sens
                cur.settings.sensitivity = new_sens
                cur.settings.image_filters.sensitivity = new_sens
                cur.preprocessed = build_preprocessed_for_floorplan(cur.loaded, cur.settings)
                self._recompute_detection()
        elif kind == "remove_thin":
            new_thin = int(round(raw_val))
            if new_thin != cur.settings.image_filters.remove_thin_px:
                cur.settings.image_filters.remove_thin_px = new_thin
                cur.preprocessed = build_preprocessed_for_floorplan(cur.loaded, cur.settings)
                self._recompute_detection()
        elif kind == "pdf_thickness":
            step = int(round(raw_val))
            new_thick = round(step * 0.05, 2)
            if abs(new_thick - cur.settings.pdf_min_thickness_pt) >= 0.01:
                cur.settings.pdf_min_thickness_pt = new_thick
                self._reprocess_and_refresh(cur, program)

    def update(self, program):
        cur = self.active_floor
        if cur is None:
            return

        slider_bindings = (
            ("block_size", self._slider_block_size),
            ("sensitivity", self._slider_sensitivity),
            ("remove_thin", self._slider_remove_thin),
            ("pdf_thickness", self._slider_pdf_thickness),
        )
        for kind, slider in slider_bindings:
            if slider is not None:
                self._apply_slider_value(kind, slider.get_current_value(), cur, program)

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

        if cur is not None and cur.preprocessed is not None:
            expected_dims = (
                (cur.img_w + cur.block_size - 1) // cur.block_size,
                (cur.img_h + cur.block_size - 1) // cur.block_size,
            )
            if cur.grid_dimensions != expected_dims or cur.overlay_surface is None:
                self._recompute_floor_detection(cur)

        pygame.draw.rect(screen, Color.BACKGROUND.value, self.canvas_rect)
        screen.set_clip(self.canvas_rect)
        if cur and cur.display_image is not None:
            screen.blit(cur.display_image, (cur.disp_x, cur.disp_y))
        if cur and cur.show_original_plan and cur.ref_display_image is not None:
            screen.blit(cur.ref_display_image, (cur.disp_x, cur.disp_y))
        if cur and cur.overlay_surface is not None:
            screen.blit(cur.overlay_surface, (cur.disp_x, cur.disp_y))

        if self.is_cropping and self.crop_p1 is not None:
            p2 = self.crop_p2 if self.crop_p2 is not None else pygame.mouse.get_pos()
            rx = min(self.crop_p1[0], p2[0])
            ry = min(self.crop_p1[1], p2[1])
            rw = abs(p2[0] - self.crop_p1[0])
            rh = abs(p2[1] - self.crop_p1[1])
            crop_s = pygame.Surface((max(1, rw), max(1, rh)), pygame.SRCALPHA)
            crop_s.fill((56, 189, 248, 55))
            screen.blit(crop_s, (rx, ry))
            pygame.draw.rect(screen, (56, 189, 248), (rx, ry, rw, rh), 2)

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

        banner_txt = None
        banner_col = (245, 158, 11)
        if self.is_measuring:
            if self.measure_p1 is None:
                banner_txt = "Click two points of known length (Tip: dimension line or standard door = 0.9 m) — Esc to cancel"
            elif self.measure_p2 is None:
                banner_txt = "Click 2nd point to finish distance measurement (Esc to cancel)"
            else:
                banner_txt = "Enter real-world length in meters"
        elif self.is_cropping:
            banner_txt = "Drag a rectangle to crop the floorplan (Esc to cancel)"
            banner_col = (56, 189, 248)
        elif self.is_picking_color:
            banner_txt = "Click on any wall to sample its color (Esc to cancel)"
            banner_col = (16, 185, 129)

        if banner_txt:
            t_surf = self.small_font.render(banner_txt, True, (255, 240, 190))
            banner_w = min(self.canvas_rect.width - 32, max(380, t_surf.get_width() + 28))
            banner_h = 32
            bx = self.canvas_rect.x + (self.canvas_rect.width - banner_w) // 2
            by = 96
            banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
            banner_surf.fill((20, 20, 20, 230))
            pygame.draw.rect(banner_surf, banner_col, banner_surf.get_rect(), 1, border_radius=6)
            screen.blit(banner_surf, (bx, by))
            screen.blit(t_surf, (bx + (banner_w - t_surf.get_width()) // 2, by + 8))

        leg_x, leg_y = 14, 96
        leg_w, leg_h = 154, 70
        leg_bg = pygame.Surface((leg_w, leg_h), pygame.SRCALPHA)
        leg_bg.fill((20, 20, 20, 225))
        pygame.draw.rect(leg_bg, (55, 55, 55, 225), leg_bg.get_rect(), 1, border_radius=7)
        screen.blit(leg_bg, (leg_x, leg_y))

        legend_hdr = self.tiny_font.render("DETECTION PREVIEW", True, Color.TEXT_MUTED.value)
        screen.blit(legend_hdr, (leg_x + 12, leg_y + 8))

        pygame.draw.rect(screen, (59, 130, 246), (leg_x + 12, leg_y + 28, 12, 12), border_radius=2)
        pygame.draw.rect(screen, (147, 197, 253), (leg_x + 12, leg_y + 28, 12, 12), 1, border_radius=2)
        wall_txt = self.small_font.render("Detected walls", True, Color.TEXT_PRIMARY.value)
        screen.blit(wall_txt, (leg_x + 30, leg_y + 26))

        pygame.draw.rect(screen, Color.EXIT.value, (leg_x + 12, leg_y + 47, 12, 12), border_radius=2)
        pygame.draw.rect(screen, (80, 240, 130), (leg_x + 12, leg_y + 47, 12, 12), 1, border_radius=2)
        exit_txt = self.small_font.render("Detected exits", True, Color.TEXT_PRIMARY.value)
        screen.blit(exit_txt, (leg_x + 30, leg_y + 45))

        zoom_val = int(round(cur.zoom_level * 100)) if cur else 100
        self._draw_bottom_canvas_pills(screen, zoom_val, has_ref=bool(cur and cur.ref_raw_image is not None))

        if len(self.floors) > 1:
            cal_world = type("BuildingProfileData", (), {
                "floors": {i: f for i, f in enumerate(self.floors)},
                "active_floor_id": self.active_floor_index,
            })()
            draw_building_elevation(
                screen,
                self.font,
                self.small_font,
                cal_world,
                min_top_y=174,
            )

        pygame.draw.rect(screen, Color.TOPNAV_BG.value, (0, 0, screen_w, self.top_nav_height))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 47), (screen_w, 47), 1)

        mark_rect = pygame.Rect(16, (self.top_nav_height - 28) // 2, 28, 28)
        pygame.draw.rect(screen, (255, 255, 255), mark_rect, border_radius=6)
        if getattr(self, "logo_surf", None):
            lx = mark_rect.centerx - self.logo_surf.get_width() // 2
            ly = mark_rect.centery - self.logo_surf.get_height() // 2
            screen.blit(self.logo_surf, (lx, ly))

        cal_title = "New Project - Floorplan Calibration" if screen_w >= 1050 else "Floorplan Calibration"
        title_surf = self.title_font.render(cal_title, True, Color.TEXT_PRIMARY.value)
        title_rect = title_surf.get_rect(left=mark_rect.right + 12, centery=self.top_nav_height // 2)
        screen.blit(title_surf, title_rect)

        container_w = 260
        container_x = screen_w - container_w - 16
        nav_label = self.tiny_font.render("PROJECT", True, Color.TEXT_MUTED.value)
        nav_rect = nav_label.get_rect(right=container_x - 10, centery=self.top_nav_height // 2)
        screen.blit(nav_label, nav_rect)
        pygame.draw.rect(screen, (21, 21, 21), (container_x, 7, container_w, 34), border_radius=6)

        pygame.draw.rect(screen, Color.FLOORTABS_BG.value, (0, 48, self.canvas_rect.width, self.tabs_height))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 87), (self.canvas_rect.width, 87), 1)

        if cur:
            file_name = Path(cur.image_path).name if cur.image_path else "floorplan"
            source_info = f"{cur.img_w} × {cur.img_h} px ({file_name})"
            source_surf = self.tiny_font.render(source_info, True, (170, 170, 170))
            screen.blit(source_surf, (self.canvas_rect.width - source_surf.get_width() - 16, 62))

        pygame.draw.rect(screen, Color.SIDEBAR_BG.value, self.sidebar_rect)
        pygame.draw.line(screen, Color.DIVIDER.value, (sb_x, 48), (sb_x, screen_h), 1)

        x = sb_x + 16
        right = sb_x + 264
        layout = self._panel_layout

        for hy, title, badge in layout["headers"]:
            hdr_surf = self.tiny_font.render(title, True, Color.TEXT_MUTED.value)
            screen.blit(hdr_surf, (x, hy))
            if badge:
                badge_surf = self.tiny_font.render(badge, True, (56, 189, 248))
                screen.blit(badge_surf, (right - badge_surf.get_width(), hy))

        for dy in layout["dividers"]:
            pygame.draw.line(screen, Color.DIVIDER.value, (x, dy), (right, dy), 1)

        for sy, text in layout["sublabels"]:
            sub_surf = self.small_font.render(text, True, (170, 170, 170))
            screen.blit(sub_surf, (x, sy))

        if layout["page_label"] is not None:
            py, ptxt = layout["page_label"]
            p_surf = self.small_font.render(ptxt, True, (235, 235, 235))
            screen.blit(p_surf, ((x + right - p_surf.get_width()) // 2, py + 5))

        if cur:
            for sy, name_str, val_fn in layout["slider_headers"]:
                n_surf = self.small_font.render(name_str, True, (170, 170, 170))
                v_surf = self.small_font.render(val_fn(cur), True, (240, 240, 240))
                screen.blit(n_surf, (x, sy))
                screen.blit(v_surf, (right - v_surf.get_width(), sy))

        card_rect = layout["summary_card"]
        if card_rect is not None:
            pygame.draw.rect(screen, Color.CARD_BG.value, card_rect, border_radius=7)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, card_rect, 1, border_radius=7)

            gw, gh = cur.grid_dimensions if cur else (0, 0)
            lbl_dim = self.small_font.render("Grid", True, (170, 170, 170))
            val_dim = self.small_font.render(f"{gw} × {gh}", True, (240, 240, 240))
            screen.blit(lbl_dim, (card_rect.x + 12, card_rect.y + 6))
            screen.blit(val_dim, (card_rect.right - val_dim.get_width() - 12, card_rect.y + 6))

            pygame.draw.line(screen, (52, 52, 52), (card_rect.x + 12, card_rect.y + 27), (card_rect.right - 12, card_rect.y + 27), 1)

            cell_m = cur.cell_size_m if cur else 0.6
            lbl_scale = self.small_font.render("Scale", True, (170, 170, 170))
            if cell_m >= 1.0:
                val_scale = self.small_font.render(f"1 cell = {cell_m:.2f} m", True, (240, 240, 240))
            else:
                val_scale = self.small_font.render(f"1 cell = {round(cell_m * 100)} cm", True, (240, 240, 240))
            screen.blit(lbl_scale, (card_rect.x + 12, card_rect.y + 32))
            screen.blit(val_scale, (card_rect.right - val_scale.get_width() - 12, card_rect.y + 32))

        pygame.draw.line(screen, Color.DIVIDER.value, (x, screen_h - 70), (right, screen_h - 70), 1)

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
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        cur = self.active_floor
        if cur is None:
            return
        max_bs = cur.slider_max_bs if cur else 50
        slider_specs = (
            ("block_size", self._slider_block_size, 8, max_bs),
            ("sensitivity", self._slider_sensitivity, 10, 80),
            ("remove_thin", self._slider_remove_thin, 0, 4),
            ("pdf_thickness", self._slider_pdf_thickness, 0, 60),
        )
        for kind, slider, lo, hi in slider_specs:
            if slider is not None and slider.rect.collidepoint(event.pos):
                r = slider.rect
                ratio = max(0.0, min(1.0, (event.pos[0] - r.x) / max(1, r.width)))
                val = int(round(lo + ratio * (hi - lo)))
                slider.set_current_value(val)
                self._apply_slider_value(kind, val, cur, program)
                break

    def handle_mouse(self, program, event):
        mx, my = event.pos
        if self.is_picking_color and event.button == 1:
            if self.canvas_rect.collidepoint(mx, my):
                cur = self.active_floor
                if cur:
                    ix, iy = self._screen_to_image_coords(cur, mx, my)
                    if 0 <= ix < cur.img_w and 0 <= iy < cur.img_h:
                        picked_lab, picked_rgb = sample_wall_at_loaded_coords(cur.loaded, int(ix), int(iy))
                        if picked_lab is not None:
                            cur.settings.image_filters.mode = "pick_color"
                            cur.settings.image_filters.picked_lab = picked_lab
                            cur.settings.image_filters.picked_rgb = picked_rgb
                            self.is_picking_color = False
                            cur.preprocessed = build_preprocessed_for_floorplan(cur.loaded, cur.settings)
                            self._recompute_detection()
                            self._build_ui(program)
                return

        if self.is_cropping and event.button == 1:
            if self.canvas_rect.collidepoint(mx, my):
                self.crop_p1 = (mx, my)
                self.crop_p2 = (mx, my)
                return

        if self.is_measuring and event.button == 1:
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
            if self.canvas_rect.collidepoint(mx, my) and my < self.screen_size[1] - 60:
                self.is_panning = True
                self.pan_start = event.pos

    def handle_mouse_motion(self, program, event):
        if self.is_cropping and self.crop_p1 is not None and event.buttons[0]:
            self.crop_p2 = event.pos

    def handle_mouse_up(self, program, event):
        if event.button in (1, 2, 3):
            self.is_panning = False
        if self.is_cropping and event.button == 1 and self.crop_p1 is not None:
            p1 = self.crop_p1
            p2 = event.pos
            self.crop_p1 = None
            self.crop_p2 = None
            self.is_cropping = False
            cur = self.active_floor
            if cur is not None:
                ix0, iy0 = self._screen_to_image_coords(cur, p1[0], p1[1])
                ix1, iy1 = self._screen_to_image_coords(cur, p2[0], p2[1])
                if abs(ix1 - ix0) >= 12 and abs(iy1 - iy0) >= 12:
                    crop_loaded_floorplan(cur.loaded, (ix0, iy0, ix1, iy1), cur.settings)
                    self._refresh_after_crop_or_page(cur, program)
                    return
            self._build_ui(program)

    def _finish_and_generate(self, program):
        if not self.floors:
            return

        def _build_grid_for_floor(fl):
            gw, gh = fl.grid_dimensions
            if gw > 0 and gh > 0 and (fl.wall_cells or fl.loaded.source_type != "image"):
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
            grid.floor_height_m = getattr(fl, "floor_height_m", 3.0)
            ref_meta = fl.loaded.to_reference_metadata()
            ref_meta["block_size"] = int(fl.block_size)
            grid.plan_reference = ref_meta
            grid.reference_image = fl.loaded.reference_image
            return grid

        def _sync_world_floor_meta(floor_obj, grid, block_sz):
            floor_obj.floor_height_m = grid.floor_height_m
            floor_obj.plan_reference = grid.plan_reference
            floor_obj.reference_image = grid.reference_image
            floor_obj.reference_surface = build_aligned_reference_surface(
                floor_obj.reference_image, grid.width, grid.height, block_sz
            )

        if program.simulation is None:
            program.simulation = Simulation()
            first = True
            for fl in self.floors:
                grid = _build_grid_for_floor(fl)
                if first:
                    program.initialize_world(grid)
                    if program.simulation and program.simulation.grid_world.floors:
                        _sync_world_floor_meta(program.simulation.grid_world.floors[0], grid, fl.block_size)
                    first = False
                else:
                    new_id = len(program.simulation.grid_world.floors)
                    program.simulation.grid_world.add_floor_from_map(grid, floor_height_m=grid.floor_height_m)
                    if new_id in program.simulation.grid_world.floors:
                        _sync_world_floor_meta(program.simulation.grid_world.floors[new_id], grid, fl.block_size)
                fl.is_added_to_simulation = True

            if getattr(program, "same_height_for_all", True):
                shared_h = self.floors[0].floor_height_m if self.floors else 3.0
                for f in program.simulation.grid_world.floors.values():
                    f.floor_height_m = shared_h
            program.simulation.grid_world.same_height_for_all = getattr(program, "same_height_for_all", True)
            program.simulation.grid_world.set_active_floor(0)
            program.update_window_for_active_floor()
        else:
            for fl in self.floors:
                if not getattr(fl, 'is_added_to_simulation', False):
                    grid = _build_grid_for_floor(fl)
                    new_id = len(program.simulation.grid_world.floors)
                    program.simulation.grid_world.add_floor_from_map(grid, floor_height_m=grid.floor_height_m)
                    if new_id in program.simulation.grid_world.floors:
                        _sync_world_floor_meta(program.simulation.grid_world.floors[new_id], grid, fl.block_size)
                    fl.is_added_to_simulation = True

            if getattr(program, "same_height_for_all", True):
                shared_h = self.floors[0].floor_height_m if self.floors else 3.0
                for f in program.simulation.grid_world.floors.values():
                    f.floor_height_m = shared_h
            program.simulation.grid_world.same_height_for_all = getattr(program, "same_height_for_all", True)
            program.save_active_floor_camera()
            new_floor_id = len(program.simulation.grid_world.floors) - 1
            program.simulation.grid_world.set_active_floor(new_floor_id)
            program.update_window_for_active_floor()

        program.transition_to(EditorState())

    def _add_floor_from_file(self, program, path):
        cur = self.active_floor
        ref_target_m, ref_px, ref_has_scale, def_h = self._resolve_reference_scale(program)
        img_kwargs = self._inherited_image_kwargs(cur, path)
        new_floor = self._create_floor_safe(
            program,
            path,
            px_per_m=ref_px,
            inherit_scale=ref_has_scale,
            floor_height_m=def_h,
            target_cell_meters=ref_target_m,
            **img_kwargs,
        )
        self._append_new_floor(program, new_floor, ref_target_m, has_ref_floors=bool(self.floors))

    def _add_floor_from_same_file(self, program):
        cur = self.active_floor
        if cur is None or not cur.image_path:
            return
        cloned_settings = self._clone_settings(cur.settings)
        ref_target_m, ref_px, ref_has_scale, def_h = self._resolve_reference_scale(program)
        clone_px = cur.px_per_m if cur.px_per_m > 0 else ref_px
        clone_has_scale = bool(cur.loaded.has_native_scale) or ref_has_scale
        new_floor = self._create_floor_safe(
            program,
            cur.image_path,
            page_index=cur.loaded.page_index,
            rotation_deg=cur.loaded.rotation_deg,
            crop_rect=None,
            settings=cloned_settings,
            px_per_m=clone_px,
            inherit_scale=clone_has_scale,
            floor_height_m=def_h,
            detect_exits=cur.detect_exits,
            target_cell_meters=ref_target_m if self.lock_grid_size else getattr(cur, "target_cell_meters", 0.50),
        )
        self._append_new_floor(program, new_floor, ref_target_m, has_ref_floors=self.lock_grid_size)

    def handle_ui_event(self, program, event):
        if getattr(self, "_notice_dialog", None) is not None:
            if self._notice_dialog.handle_event(event) == "close":
                self._notice_dialog.kill()
                self._notice_dialog = None
            return

        if getattr(self, "_add_floor_dialog", None) is not None:
            res = self._add_floor_dialog.handle_event(event)
            if res == "cancel":
                self._add_floor_dialog.kill()
                self._add_floor_dialog = None
            elif res == "new_file":
                self._add_floor_dialog.kill()
                self._add_floor_dialog = None
                next_path = _open_file_dialog("Select Floorplan", SUPPORTED_FLOORPLAN_FILETYPES)
                if next_path:
                    self._add_floor_from_file(program, next_path)
            elif res == "same_file":
                self._add_floor_dialog.kill()
                self._add_floor_dialog = None
                self._add_floor_from_same_file(program)
            return

        if getattr(self, "_delete_floor_dialog", None) is not None:
            res = self._delete_floor_dialog.handle_event(event)
            if res == "cancel":
                self._delete_floor_dialog.kill()
                self._delete_floor_dialog = None
            elif res == "delete":
                self._delete_floor_dialog.kill()
                self._delete_floor_dialog = None
                if len(self.floors) > 1:
                    deleted_idx = self.active_floor_index
                    deleted_fl = self.floors.pop(deleted_idx)
                    if (
                        getattr(deleted_fl, "is_added_to_simulation", False)
                        and program.simulation
                        and program.simulation.grid_world
                        and deleted_idx in program.simulation.grid_world.floors
                    ):
                        _delete_floor_from_world(program, deleted_idx)
                    self.active_floor_index = min(deleted_idx, len(self.floors) - 1)
                    self.is_measuring = False
                    self.is_cropping = False
                    self.is_picking_color = False
                    self._build_ui(program)
                    self._recompute_detection()
            return

        if getattr(self, "_dxf_layers_dialog", None) is not None:
            res = self._dxf_layers_dialog.handle_event(event)
            if res == "cancel":
                self._dxf_layers_dialog.kill()
                self._dxf_layers_dialog = None
            elif res == "apply":
                cur = self.active_floor
                self._dxf_layers_dialog.kill()
                self._dxf_layers_dialog = None
                if cur is not None:
                    cur.settings.dxf_selected_layers = {l.name for l in cur.loaded.dxf_layers if l.selected}
                    self._reprocess_and_refresh(cur, program)
                    self._build_ui(program)
            return

        cur = self.active_floor

        if event.ui_element == self._generate_button:
            self._finish_and_generate(program)

        elif event.ui_element == self._btn_add_floor:
            has_src = bool(cur and cur.image_path and Path(cur.image_path).exists())
            self._add_floor_dialog = AddFloorChoiceDialog(program.ui_manager, self.screen_size, has_source_file=has_src)

        elif event.ui_element == getattr(self, "_btn_delete_floor", None):
            if len(self.floors) > 1:
                self._delete_floor_dialog = DeleteFloorConfirmDialog(
                    program.ui_manager,
                    self.screen_size,
                    self.active_floor_index + 1,
                )

        elif event.ui_element == self._btn_back:
            cur_size = program.screen.get_size()
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())

        elif event.ui_element == self._btn_load_project:
            path = _open_file_dialog("Open project", [("JSON files", "*.json")])
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    program.initialize_world_from_dict(data, project_path=path)
                    program.transition_to(EditorState())
                except Exception as e:
                    logger.error("Failed to load project from %s: %s", path, e)

        elif event.ui_element == self._btn_rot_left and cur:
            rotate_loaded_floorplan(cur.loaded, -90)
            cur.sync_from_loaded(recompute_preprocessed=True)
            self._init_floor_fit(cur)
            self._recompute_detection()

        elif event.ui_element == self._btn_rot_right and cur:
            rotate_loaded_floorplan(cur.loaded, 90)
            cur.sync_from_loaded(recompute_preprocessed=True)
            self._init_floor_fit(cur)
            self._recompute_detection()

        elif event.ui_element == self._btn_crop and cur:
            self.is_cropping = not self.is_cropping
            self.is_measuring = False
            self.is_picking_color = False
            self.crop_p1 = None
            self.crop_p2 = None
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_reset_crop", None) and cur:
            reset_crop_floorplan(cur.loaded, cur.settings)
            self._refresh_after_crop_or_page(cur, program)

        elif event.ui_element == getattr(self, "_btn_plan_overlay_off", None) and cur:
            if cur.show_original_plan:
                cur.show_original_plan = False
                self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_plan_overlay_on", None) and cur:
            if not cur.show_original_plan:
                cur.show_original_plan = True
                self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_dxf_layers", None) and cur:
            self._dxf_layers_dialog = DxfLayersDialog(
                program.ui_manager,
                self.screen_size,
                cur.loaded.dxf_layers,
            )

        elif event.ui_element in self._dxf_unit_buttons.values() and cur:
            for u, btn in self._dxf_unit_buttons.items():
                if event.ui_element == btn:
                    cur.loaded.dxf_units = u
                    cur.settings.dxf_units = u
                    cur.loaded.px_per_m = compute_dxf_px_per_m(cur.loaded.pixels_per_source_unit, u)
                    cur.loaded.has_native_scale = True
                    cur.px_per_m = cur.loaded.px_per_m
                    cur.has_manual_scale = False
                    self._apply_scale_and_refresh(cur, program)
                    break

        elif event.ui_element == getattr(self, "_btn_page_prev", None) and cur:
            if cur.loaded.page_index > 0:
                self._switch_pdf_page(program, cur, cur.loaded.page_index - 1)

        elif event.ui_element == getattr(self, "_btn_page_next", None) and cur:
            if cur.loaded.page_index + 1 < cur.loaded.page_count:
                self._switch_pdf_page(program, cur, cur.loaded.page_index + 1)

        elif event.ui_element == getattr(self, "_btn_pdf_fixed", None) and cur:
            cur.settings.pdf_include_fixed = not cur.settings.pdf_include_fixed
            self._reprocess_and_refresh(cur, program)
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_pdf_filled_only", None) and cur:
            cur.settings.pdf_filled_walls_only = not getattr(cur.settings, "pdf_filled_walls_only", False)
            self._reprocess_and_refresh(cur, program)
            self._build_ui(program)

        elif event.ui_element in self._pdf_scale_buttons.values() and cur:
            for denom, btn in self._pdf_scale_buttons.items():
                if event.ui_element == btn:
                    cur.settings.pdf_scale_denom = denom
                    cur.loaded.pdf_scale_denom = denom
                    cur.loaded.px_per_m = compute_pdf_px_per_m(cur.loaded.pixels_per_source_unit, denom)
                    cur.loaded.has_native_scale = True
                    cur.px_per_m = cur.loaded.px_per_m
                    cur.has_manual_scale = False
                    self._apply_scale_and_refresh(cur, program)
                    break

        elif event.ui_element == getattr(self, "_btn_pick_color", None) and cur:
            self.is_picking_color = not self.is_picking_color
            self.is_measuring = False
            self.is_cropping = False
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_pick_auto", None) and cur:
            self.is_picking_color = False
            cur.settings.image_filters.mode = "auto"
            cur.preprocessed = build_preprocessed_for_floorplan(cur.loaded, cur.settings)
            self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_img_ignore_colors", None) and cur:
            cur.settings.image_filters.ignore_colors = not cur.settings.image_filters.ignore_colors
            cur.preprocessed = build_preprocessed_for_floorplan(cur.loaded, cur.settings)
            self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_img_gray_blocks", None) and cur:
            cur.settings.image_filters.gray_blocks = not cur.settings.image_filters.gray_blocks
            cur.preprocessed = build_preprocessed_for_floorplan(cur.loaded, cur.settings)
            self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == self._btn_lock_grid:
            self.lock_grid_size = not self.lock_grid_size
            if self.lock_grid_size and cur:
                self._sync_all_floors_scale(cur)
                self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == self._btn_coarse and cur:
            self._apply_scale_and_refresh(cur, program, 0.60, sync_slider=True)

        elif event.ui_element == self._btn_standard and cur:
            self._apply_scale_and_refresh(cur, program, 0.50, sync_slider=True)

        elif event.ui_element == self._btn_fine and cur:
            self._apply_scale_and_refresh(cur, program, 0.40, sync_slider=True)

        elif event.ui_element == self._btn_measure_scale:
            self.is_measuring = not self.is_measuring
            self.is_cropping = False
            self.is_picking_color = False
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
            if event.ui_element == self._btn_reset_measure and cur:
                cur.reset_manual_scale()
                self._sync_all_floors_scale(cur)
                self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == self._scale_btn_apply:
            if cur and hasattr(self, "last_measured_dist_px") and self._scale_entry_meters:
                try:
                    val_str = self._scale_entry_meters.get_text().strip().replace(",", ".")
                    meters = float(val_str)
                    if meters > 0:
                        px_per_m = self.last_measured_dist_px / meters
                        cur.px_per_m = px_per_m
                        cur.loaded.px_per_m = px_per_m
                        cur.loaded.has_native_scale = True
                        cur.has_manual_scale = True
                        self._apply_scale_and_refresh(cur, program, rebuild_ui=False)
                except Exception as e:
                    logger.warning("Invalid scale meters input: %s", e)
            self.is_measuring = False
            self.measure_p1 = None
            self.measure_p2 = None
            if self._scale_input_window:
                self._scale_input_window.kill()
                self._scale_input_window = None
            self._build_ui(program)

        elif event.ui_element == self._btn_toggle_exits and cur:
            cur.detect_exits = not cur.detect_exits
            self._recompute_detection()
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_btn_floor_height", None):
            self._show_height_dialog(program)

        elif event.ui_element == getattr(self, "_height_same_all_btn", None):
            program.same_height_for_all = not getattr(program, "same_height_for_all", True)
            if program.simulation and program.simulation.grid_world:
                program.simulation.grid_world.same_height_for_all = program.same_height_for_all
            state_txt = "ON" if program.same_height_for_all else "OFF"
            self._height_same_all_btn.set_text(f"Same height for all floors: {state_txt}")
            self._height_same_all_btn.change_object_id("#tool_btn_selected" if program.same_height_for_all else "#tool_btn")

        elif event.ui_element == getattr(self, "_height_btn_apply", None) or (
            event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == getattr(self, "_height_entry", None)
        ):
            if getattr(self, "_height_entry", None):
                try:
                    val_str = self._height_entry.get_text().strip().replace(",", ".")
                    val = float(val_str)
                    if val <= 0:
                        raise ValueError("Height must be positive")
                    val = max(1.0, min(20.0, val))
                    same = getattr(program, "same_height_for_all", True)
                    if same:
                        for fl in self.floors:
                            fl.floor_height_m = val
                        if program.simulation and program.simulation.grid_world:
                            program.simulation.grid_world.set_floor_height(0, val, same_for_all=True)
                    else:
                        target_idx = 1 if (self.active_floor_index == 0 and len(self.floors) > 1) else self.active_floor_index
                        if 0 <= target_idx < len(self.floors):
                            self.floors[target_idx].floor_height_m = val
                        if program.simulation and program.simulation.grid_world:
                            world = program.simulation.grid_world
                            world.set_floor_height(target_idx, val, same_for_all=False)
                except Exception as e:
                    logger.warning("Invalid floor height input: %s", e)
            if getattr(self, "_height_window", None):
                self._height_window.kill()
                self._height_window = None
            self._build_ui(program)

        elif event.ui_element == getattr(self, "_height_btn_cancel", None):
            if getattr(self, "_height_window", None):
                self._height_window.kill()
                self._height_window = None

        elif event.ui_element == self._btn_zoom_in and cur:
            cur.zoom_level = min(2.5, round(cur.zoom_level + 0.25, 2))
            self._update_overlay_surface()

        elif event.ui_element == self._btn_zoom_out and cur:
            cur.zoom_level = max(0.5, round(cur.zoom_level - 0.25, 2))
            self._update_overlay_surface()

        else:
            for idx, tab_btn in enumerate(self._floor_tab_buttons):
                if event.ui_element == tab_btn:
                    if self.active_floor_index != idx:
                        self.active_floor_index = idx
                        self.is_measuring = False
                        self.is_cropping = False
                        self.is_picking_color = False
                        self._build_ui(program)
                        self._recompute_detection()
                    break

    def _switch_pdf_page(self, program, cur, new_page_idx):
        cur.loaded = load_floorplan(
            cur.image_path,
            page_index=new_page_idx,
            crop_rect=None,
            rotation_deg=cur.loaded.rotation_deg,
            settings=cur.settings,
        )
        self._refresh_after_crop_or_page(cur, program)

    def handle_key_event(self, program, key):
        if key == pygame.K_ESCAPE:
            if self.is_measuring or self._scale_input_window or self.is_cropping or self.is_picking_color:
                self.is_measuring = False
                self.is_cropping = False
                self.is_picking_color = False
                self.measure_p1 = None
                self.measure_p2 = None
                self.crop_p1 = None
                self.crop_p2 = None
                if self._scale_input_window:
                    self._scale_input_window.kill()
                    self._scale_input_window = None
                self._build_ui(program)
                return
            cur_size = program.screen.get_size()
            if program.ui_manager is not None:
                program.ui_manager.set_window_resolution(cur_size)
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
        for attr in (
            "_scale_input_window",
            "_height_window",
            "_add_floor_dialog",
            "_delete_floor_dialog",
            "_dxf_layers_dialog",
            "_notice_dialog",
        ):
            dlg = getattr(self, attr, None)
            if dlg is not None:
                dlg.kill()
                setattr(self, attr, None)
        self.is_measuring = False
        self.is_cropping = False
        self.is_picking_color = False
        self.measure_p1 = None
        self.measure_p2 = None
        self.crop_p1 = None
        self.crop_p2 = None
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()
        self._slider_block_size = None
        self._slider_sensitivity = None
        self._slider_remove_thin = None
        self._slider_pdf_thickness = None
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
        self._btn_delete_floor = None
        self._btn_plan_overlay = None
        self._btn_zoom_in = None
        self._btn_zoom_out = None
        self._floor_tab_buttons = []
        self.floors = []

        logo_path = Path(__file__).resolve().parent.parent / "logo" / "logoSquare.png"
        self.logo_surf = None
        if logo_path.exists():
            try:
                raw_logo = pygame.image.load(str(logo_path)).convert_alpha()
                self.logo_surf = pygame.transform.smoothscale(raw_logo, (20, 20))
            except Exception:
                self.logo_surf = None

    def status_lines(self, context):
        cur = self.active_floor
        bs = cur.block_size if cur else 25
        return [
            f"State: {self.name}",
            f"Active floor: {self.active_floor_index + 1}/{len(self.floors)}",
            f"Block size: {bs}px",
        ]


def _bresenham_cells(x0, y0, x1, y1):
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


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
        self._nav_sim = None
        self._btn_add_floor = None
        self._btn_delete_floor = None
        self._delete_floor_dialog = None
        self._btn_wall = None
        self._btn_eraser = None
        self._btn_evacuee = None
        self._btn_drone = None
        self._btn_exit = None
        self._btn_stairs = None
        self._btn_random_crowd = None
        self._btn_clear_entities = None
        self._btn_toggle_grid = None
        self._btn_plan_overlay_off = None
        self._btn_plan_overlay_on = None
        self._btn_floor_height = None
        self._stair_link_buttons = {}
        self._stair_unlink_buttons = {}
        self._btn_zoom_in = None
        self._btn_zoom_out = None
        self._is_panning = False
        self._pan_start = (0, 0)
        self._line_start = None
        self._line_end = None
        self._add_floor_dialog = None
        self.title_font = get_font(22, bold=True)
        self.font = get_font(15)
        self.small_font = get_font(13)
        self.tiny_font = get_font(11, bold=True)
        self._height_window = None
        self._height_entry = None
        self._height_same_all_btn = None
        self._height_btn_apply = None
        self._height_btn_cancel = None
        self._crowd_window = None
        self._crowd_entry = None
        self._crowd_btn_apply = None
        self._crowd_btn_cancel = None

    def _ensure_active_floor_overlay(self, program):
        if not program.simulation or not program.simulation.grid_world:
            return
        floor = program.simulation.grid_world.active_floor
        if getattr(floor, "reference_surface", None) is not None:
            return
        ref_img = getattr(floor, "reference_image", None)
        plan_ref = getattr(floor, "plan_reference", None)
        if ref_img is None and plan_ref:
            ref_img = restore_reference_image(
                plan_ref,
                project_json_dir=getattr(program, "loaded_project_dir", None),
            )
            floor.reference_image = ref_img
        if ref_img is not None:
            bs = plan_ref.get("block_size") if isinstance(plan_ref, dict) else None
            floor.reference_surface = build_aligned_reference_surface(
                ref_img,
                floor.map_grid.width,
                floor.map_grid.height,
                bs,
            )

    def _has_floor_plan_reference(self, floor, program=None):
        if floor is None:
            return False
        if getattr(floor, "reference_image", None) is not None or getattr(floor, "reference_surface", None) is not None:
            return True
        plan_ref = getattr(floor, "plan_reference", None)
        if isinstance(plan_ref, dict) and plan_ref.get("source_path"):
            return resolve_plan_source_path(
                plan_ref["source_path"],
                project_json_dir=getattr(program, "loaded_project_dir", None) if program else None,
            ) is not None
        return False

    def enter(self, program):
        program.view_options.show_heatmap = False
        program.view_options.show_clusters = False
        program.view_options.show_cluster_connections = False
        program.view_options.show_drone_routes = False
        program.update_window_for_active_floor()
        if getattr(program.view_options, "show_plan_overlay", False):
            self._ensure_active_floor_overlay(program)
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        program.simulation.grid_world.rebuild_changed_map_data()
        if getattr(program.view_options, "show_plan_overlay", False):
            self._ensure_active_floor_overlay(program)
        self._rebuild_ui(program)

    def handle_resize(self, program, *args):
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        self._stair_link_buttons = {}
        self._stair_unlink_buttons = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        title_w = 90
        save_x = max(180, 48 + title_w + 24)
        save_w = 110
        menu_x = save_x + save_w + 8
        menu_w = 70

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(save_x, 9, save_w, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(menu_x, 9, menu_w, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 260
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 128
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_active",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "Simulate",
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
        self._btn_delete_floor = pygame_gui.elements.UIButton(
            pygame.Rect(tab_x + 117, 54, 100, 28),
            "Delete Floor",
            manager=program.ui_manager,
            object_id="#btn_destructive",
        )
        if world is None or len(world.floors) <= 1:
            self._btn_delete_floor.disable()

        tool_x = sidebar_x + 16
        col_w = 120
        gap = 8
        col1_x = tool_x
        col2_x = tool_x + col_w + gap
        tool_h = 32

        self._btn_wall = pygame_gui.elements.UIButton(
            pygame.Rect(col1_x, 152, col_w, tool_h),
            "Wall",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.WALL else "#tool_btn",
            tool_tip_text="Draw walls (Hold Shift + Drag for straight line)",
        )
        self._btn_eraser = pygame_gui.elements.UIButton(
            pygame.Rect(col2_x, 152, col_w, tool_h),
            "Eraser",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.ERASER else "#tool_btn",
        )
        self._btn_evacuee = pygame_gui.elements.UIButton(
            pygame.Rect(col1_x, 188, col_w, tool_h),
            "Evacuee",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.EVACUEE else "#tool_btn",
        )
        self._btn_drone = pygame_gui.elements.UIButton(
            pygame.Rect(col2_x, 188, col_w, tool_h),
            "Guide Agent",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.DRONE else "#tool_btn",
        )
        self._btn_exit = pygame_gui.elements.UIButton(
            pygame.Rect(col1_x, 224, col_w, tool_h),
            "Exit",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.EXIT else "#tool_btn",
        )
        self._btn_stairs = pygame_gui.elements.UIButton(
            pygame.Rect(col2_x, 224, col_w, tool_h),
            "Stairs",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if self.selected_tool == Tool.STAIRS else "#tool_btn",
        )

        active_floor = world.active_floor if world else None
        cur_h = getattr(active_floor, "floor_height_m", 3.0) if active_floor else 3.0

        action_w = 248
        action_h = 30
        act_y = 296

        self._btn_random_crowd = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, act_y, action_w, action_h),
            "Random Crowd",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        act_y += 34
        self._btn_clear_entities = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, act_y, action_w, action_h),
            "Clear Evacuees & Guides",
            manager=program.ui_manager,
            object_id="#btn_secondary",
            tool_tip_text="Clear evacuees and guide agents on this floor",
        )
        act_y += 34
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, act_y, action_w, action_h),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if program.view_options.show_grid else "#tool_btn",
        )
        act_y += 34

        self._btn_floor_height = None
        if active_floor_id > 0:
            f_num = active_floor_id + 1
            f_prev = active_floor_id
            self._btn_floor_height = pygame_gui.elements.UIButton(
                pygame.Rect(tool_x, act_y, action_w, action_h),
                f"Height (Floor {f_prev}–{f_num}): {cur_h:.1f}m",
                manager=program.ui_manager,
                object_id="#btn_secondary",
                tool_tip_text=f"Configure height between Floor {f_prev} and Floor {f_num}",
            )
            act_y += 34

        if world and len(world.floors) > 1:
            stair_div_y = 438 if world.active_floor_id > 0 else 406
            stair_y = stair_div_y + 30
            stair_clusters = world.get_stair_clusters(world.active_floor_id)
            pending = world.pending_stair_endpoint

            for cid, c in stair_clusters:
                target = getattr(c, "linked_to", None)
                if target is not None:
                    btn_u = pygame_gui.elements.UIButton(
                        pygame.Rect(tool_x + action_w - 60, stair_y, 60, 26),
                        "Unlink",
                        manager=program.ui_manager,
                        object_id="#btn_destructive",
                    )
                    self._stair_unlink_buttons[cid] = btn_u
                elif pending == (world.active_floor_id, cid):
                    btn_c = pygame_gui.elements.UIButton(
                        pygame.Rect(tool_x + action_w - 64, stair_y, 64, 26),
                        "Cancel",
                        manager=program.ui_manager,
                        object_id="#tool_btn_selected",
                    )
                    self._stair_link_buttons[cid] = btn_c
                else:
                    btn_l = pygame_gui.elements.UIButton(
                        pygame.Rect(tool_x + action_w - 56, stair_y, 56, 26),
                        "Link",
                        manager=program.ui_manager,
                        object_id="#btn_secondary",
                    )
                    self._stair_link_buttons[cid] = btn_l
                stair_y += 32

        overlay_on = getattr(program.view_options, "show_plan_overlay", False)
        has_ref = self._has_floor_plan_reference(active_floor, program)
        self._build_bottom_canvas_controls(
            program.ui_manager,
            screen_w,
            screen_h,
            overlay_on and has_ref,
            has_ref,
            zoom_btn_id="#btn_secondary",
        )

    def _zoom(self, program, factor, mouse_pos):
        import utils.config as config
        old_size = getattr(config, 'drawing_size', 8)
        new_size = max(4, min(24, int(round(old_size * factor))))
        if new_size != old_size:
            mx, my = mouse_pos
            gx = (mx - getattr(config, 'camera_offset_x', 0)) / old_size
            gy = (my - getattr(config, 'camera_offset_y', 0)) / old_size
            config.drawing_size = new_size
            config.camera_offset_x = int(mx - gx * new_size)
            config.camera_offset_y = int(my - gy * new_size)
            program.save_active_floor_camera()
            self._rebuild_ui(program)

    def _handle_stair_click(self, program, endpoint):
        import time
        world = program.simulation.grid_world
        pending = world.pending_stair_endpoint
        if pending is None:
            world.pending_stair_endpoint = endpoint
            s_name = world.get_stair_name(endpoint[0], endpoint[1])
            world.stair_link_notice = f"{s_name} selected. Switch floor and click target stairs!"
            world.stair_link_notice_time = time.time()
            world.stair_link_error = None
        elif pending[0] == endpoint[0]:
            if pending[1] == endpoint[1]:
                world.pending_stair_endpoint = None
                world.stair_link_notice = None
                world.stair_link_error = None
            else:
                world.pending_stair_endpoint = endpoint
                s_name = world.get_stair_name(endpoint[0], endpoint[1])
                world.stair_link_notice = f"{s_name} selected. Switch floor and click target stairs!"
                world.stair_link_notice_time = time.time()
                world.stair_link_error = None
        else:
            try:
                world.connect_stair_clusters(pending, endpoint)
            except ValueError as error:
                world.stair_link_error = str(error)
                world.stair_link_notice = None
                world.stair_link_notice_time = time.time()
            else:
                s1_name = world.get_stair_name(pending[0], pending[1])
                s2_name = world.get_stair_name(endpoint[0], endpoint[1])
                world.stair_link_notice = f"Linked {s1_name} (Floor {pending[0] + 1}) ↔ {s2_name} (Floor {endpoint[0] + 1})"
                world.stair_link_notice_time = time.time()
                world.pending_stair_endpoint = None
                world.stair_link_error = None
                program.plan_drone_routes()
        self._rebuild_ui(program)

    def _show_height_dialog(self, program):
        if getattr(self, "_height_window", None) is not None:
            self._height_window.kill()
            self._height_window = None

        screen_w, screen_h = program.screen.get_size()
        win_w = 340
        win_h = 230
        win_x = max(20, (screen_w - SIDEBAR_WIDTH - win_w) // 2)
        win_y = max(20, (screen_h - win_h) // 2)

        world = program.simulation.grid_world if program.simulation else None
        active_floor_id = world.active_floor_id if world else 1
        active_floor = world.active_floor if world else None
        same = getattr(program, "same_height_for_all", True)
        if same and world and world.floors:
            cur_h = getattr(world.floors.get(active_floor_id, world.floors[0]), "floor_height_m", 3.0)
        else:
            cur_h = getattr(active_floor, "floor_height_m", 3.0) if active_floor else 3.0

        f_num = max(2, active_floor_id + 1)
        f_prev = f_num - 1

        self._height_window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=program.ui_manager,
            window_display_title=f"Set Height (Floor {f_prev}–{f_num})",
            object_id="#scale_window",
        )

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, 10, win_w - 32, 22),
            text=f"Height between Floor {f_prev} and Floor {f_num} (m):",
            manager=program.ui_manager,
            container=self._height_window,
        )
        self._height_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(16, 34, win_w - 32, 30),
            manager=program.ui_manager,
            container=self._height_window,
            initial_text=f"{cur_h:.1f}",
        )
        self._height_info_lbl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, 68, win_w - 32, 20),
            text=f"~{round(cur_h / 0.17)} stair steps (standard 17cm rise)",
            manager=program.ui_manager,
            container=self._height_window,
        )
        self._height_same_all_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 94, win_w - 32, 30),
            text="Same height for all floors: ON" if same else "Same height for all floors: OFF",
            manager=program.ui_manager,
            container=self._height_window,
            object_id="#tool_btn_selected" if same else "#tool_btn",
        )
        btn_w = (win_w - 40) // 2
        self._height_btn_apply = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 136, btn_w, 32),
            text="Apply",
            manager=program.ui_manager,
            container=self._height_window,
            object_id="#btn_start_sim",
        )
        self._height_btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(24 + btn_w, 136, btn_w, 32),
            text="Cancel",
            manager=program.ui_manager,
            container=self._height_window,
            object_id="#menu_button",
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
        if getattr(self, "_add_floor_dialog", None) is not None:
            self._add_floor_dialog.kill()
            self._add_floor_dialog = None
        if getattr(self, "_delete_floor_dialog", None) is not None:
            self._delete_floor_dialog.kill()
            self._delete_floor_dialog = None
        if getattr(self, "_crowd_window", None) is not None:
            self._crowd_window.kill()
            self._crowd_window = None
        if getattr(self, "_height_window", None) is not None:
            self._height_window.kill()
            self._height_window = None
        self._line_start = None
        self._line_end = None
        program.simulation.grid_world.rebuild_changed_map_data()
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def _find_active_source_for_clone(self, program):
        world = program.simulation.grid_world if program.simulation else None
        active_floor = world.active_floor if world else None
        cal_floors = getattr(program, "calibrated_floors", None) or []
        if world and 0 <= world.active_floor_id < len(cal_floors):
            cf = cal_floors[world.active_floor_id]
            if cf.image_path and Path(cf.image_path).exists():
                return {
                    "source_path": cf.image_path,
                    "rotation_deg": cf.loaded.rotation_deg,
                    "page_index": cf.loaded.page_index,
                    "settings": cf.settings,
                    "px_per_m": cf.px_per_m,
                    "has_native_scale": cf.loaded.has_native_scale,
                    "detect_exits": cf.detect_exits,
                }
        plan_ref = getattr(active_floor, "plan_reference", None) if active_floor else None
        if isinstance(plan_ref, dict) and plan_ref.get("source_path"):
            resolved = resolve_plan_source_path(
                plan_ref["source_path"],
                project_json_dir=getattr(program, "loaded_project_dir", None),
            )
            if resolved is not None:
                return {
                    "source_path": str(resolved),
                    "rotation_deg": int(plan_ref.get("rotation_deg", 0) or 0),
                    "page_index": int(plan_ref.get("page_index", 0) or 0),
                    "settings": None,
                    "px_per_m": None,
                }
        return None

    def handle_ui_event(self, program, event):
        import time
        if getattr(self, "_add_floor_dialog", None) is not None:
            res = self._add_floor_dialog.handle_event(event)
            if res == "cancel":
                self._add_floor_dialog.kill()
                self._add_floor_dialog = None
            elif res == "new_file":
                self._add_floor_dialog.kill()
                self._add_floor_dialog = None
                path = _open_file_dialog("Select Floorplan", SUPPORTED_FLOORPLAN_FILETYPES)
                if path:
                    program.imported_image_path = path
                    program.pending_floor_clone = None
                    program.transition_to(CalibrationState())
            elif res == "same_file":
                self._add_floor_dialog.kill()
                self._add_floor_dialog = None
                clone_req = self._find_active_source_for_clone(program)
                if clone_req:
                    program.imported_image_path = None
                    program.pending_floor_clone = clone_req
                    program.transition_to(CalibrationState())
            return

        if getattr(self, "_delete_floor_dialog", None) is not None:
            res = self._delete_floor_dialog.handle_event(event)
            if res == "cancel":
                self._delete_floor_dialog.kill()
                self._delete_floor_dialog = None
            elif res == "delete":
                self._delete_floor_dialog.kill()
                self._delete_floor_dialog = None
                world = program.simulation.grid_world if program.simulation else None
                if world is not None and len(world.floors) > 1:
                    deleted_idx = world.active_floor_id
                    if _delete_floor_from_world(program, deleted_idx):
                        cal_floors = getattr(program, "calibrated_floors", None)
                        if cal_floors is not None and 0 <= deleted_idx < len(cal_floors):
                            cal_floors.pop(deleted_idx)
                        program.update_window_for_active_floor()
                        if getattr(program.view_options, "show_plan_overlay", False):
                            self._ensure_active_floor_overlay(program)
                        program.simulation.prepare()
                        program.plan_drone_routes()
                        self._rebuild_ui(program)
            return

        world = program.simulation.grid_world if program.simulation else None
        if event.ui_element == self._save_btn:
            _save_project(program)
        elif event.ui_element == self._menu_btn:
            cur_size = program.screen.get_size()
            if program.ui_manager:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            pass
        elif event.ui_element == self._nav_sim:
            program.simulation.prepare()
            program.plan_drone_routes()
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_wall:
            self.selected_tool = Tool.WALL
            self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_btn_eraser", None):
            self.selected_tool = Tool.ERASER
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
        elif event.ui_element == getattr(self, "_btn_clear_entities", None):
            world.clear_people_and_drones(world.active_floor_id, program.simulation)
            if program.drone_controller is not None:
                program.drone_controller.reset()
            program.plan_drone_routes()
            self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_btn_floor_height", None):
            self._show_height_dialog(program)
        elif event.ui_element == getattr(self, "_height_same_all_btn", None):
            program.same_height_for_all = not getattr(program, "same_height_for_all", True)
            if program.simulation and program.simulation.grid_world:
                program.simulation.grid_world.same_height_for_all = program.same_height_for_all
            state_txt = "ON" if program.same_height_for_all else "OFF"
            self._height_same_all_btn.set_text(f"Same height for all floors: {state_txt}")
            self._height_same_all_btn.change_object_id("#tool_btn_selected" if program.same_height_for_all else "#tool_btn")
        elif event.ui_element == getattr(self, "_height_btn_apply", None) or (
            event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == getattr(self, "_height_entry", None)
        ):
            if program.simulation and getattr(self, "_height_entry", None):
                try:
                    val_str = self._height_entry.get_text().strip().replace(",", ".")
                    val = float(val_str)
                    if val <= 0:
                        raise ValueError("Height must be positive")
                    val = max(1.0, min(20.0, val))
                    same = getattr(program, "same_height_for_all", True)
                    world.set_floor_height(world.active_floor_id, val, same_for_all=same)
                except Exception as e:
                    logger.warning("Invalid floor height input: %s", e)
            if getattr(self, "_height_window", None):
                self._height_window.kill()
                self._height_window = None
            self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_height_btn_cancel", None):
            if getattr(self, "_height_window", None):
                self._height_window.kill()
                self._height_window = None
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
        elif event.ui_element == getattr(self, "_btn_plan_overlay_off", None):
            if getattr(program.view_options, "show_plan_overlay", False):
                program.view_options.show_plan_overlay = False
                self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_btn_plan_overlay_on", None):
            if world and self._has_floor_plan_reference(world.active_floor, program):
                if not getattr(program.view_options, "show_plan_overlay", False):
                    program.view_options.show_plan_overlay = True
                    self._ensure_active_floor_overlay(program)
                    self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_btn_zoom_in", None):
            screen_w, screen_h = program.screen.get_size()
            center = ((screen_w - SIDEBAR_WIDTH) // 2, (screen_h - 48) // 2 + 48)
            self._zoom(program, 1.25, center)
        elif event.ui_element == getattr(self, "_btn_zoom_out", None):
            screen_w, screen_h = program.screen.get_size()
            center = ((screen_w - SIDEBAR_WIDTH) // 2, (screen_h - 48) // 2 + 48)
            self._zoom(program, 0.8, center)
        elif event.ui_element == self._btn_add_floor:
            clone_req = self._find_active_source_for_clone(program)
            self._add_floor_dialog = AddFloorChoiceDialog(
                program.ui_manager,
                program.screen.get_size(),
                has_source_file=bool(clone_req),
            )
        elif event.ui_element == getattr(self, "_btn_delete_floor", None):
            if world is not None and len(world.floors) > 1:
                self._delete_floor_dialog = DeleteFloorConfirmDialog(
                    program.ui_manager,
                    program.screen.get_size(),
                    world.active_floor_id + 1,
                )
        else:
            handled = False
            for cid, btn in self._stair_link_buttons.items():
                if event.ui_element == btn:
                    if world.pending_stair_endpoint == (world.active_floor_id, cid):
                        world.pending_stair_endpoint = None
                        world.stair_link_notice = None
                        world.stair_link_error = None
                    else:
                        world.pending_stair_endpoint = (world.active_floor_id, cid)
                        s_name = world.get_stair_name(world.active_floor_id, cid)
                        world.stair_link_notice = f"{s_name} selected. Switch floor and click target stairs!"
                        world.stair_link_notice_time = time.time()
                        world.stair_link_error = None
                    self._rebuild_ui(program)
                    handled = True
                    break
            if not handled:
                for cid, btn in self._stair_unlink_buttons.items():
                    if event.ui_element == btn:
                        s_name = world.get_stair_name(world.active_floor_id, cid)
                        world.disconnect_stair_cluster((world.active_floor_id, cid))
                        world.stair_link_notice = f"Unlinked {s_name} (Floor {world.active_floor_id + 1})"
                        world.stair_link_notice_time = time.time()
                        world.stair_link_error = None
                        program.plan_drone_routes()
                        self._rebuild_ui(program)
                        handled = True
                        break
            if not handled:
                for f_id, tab_btn in self._floor_tabs.items():
                    if event.ui_element == tab_btn:
                        if f_id != world.active_floor_id:
                            program.save_active_floor_camera()
                            world.set_active_floor(f_id)
                            program.update_window_for_active_floor()
                            self.on_floor_changed(program)
                        break

    def handle_key(self, program, key):
        if key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)
        import utils.config as config
        if key == pygame.K_LEFT:
            config.camera_offset_x += 24
            program.save_active_floor_camera()
        elif key == pygame.K_RIGHT:
            config.camera_offset_x -= 24
            program.save_active_floor_camera()
        elif key == pygame.K_UP:
            config.camera_offset_y += 24
            program.save_active_floor_camera()
        elif key == pygame.K_DOWN:
            config.camera_offset_y -= 24
            program.save_active_floor_camera()

    def handle_mouse_wheel(self, program, event):
        if event.y > 0:
            self._zoom(program, 1.2, pygame.mouse.get_pos())
        elif event.y < 0:
            self._zoom(program, 0.833, pygame.mouse.get_pos())

    def handle_mouse_up(self, program, event):
        if event.button == 2:
            self._is_panning = False
            program.save_active_floor_camera()
        elif event.button == 1 and self._line_start is not None:
            end_pt = self._line_end or self._line_start
            for cx, cy in _bresenham_cells(self._line_start[0], self._line_start[1], end_pt[0], end_pt[1]):
                place(program.simulation, cx, cy, Tool.WALL)
            self._line_start = None
            self._line_end = None

    def handle_mouse(self, program, event):
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return

        if event.button == 2:
            self._is_panning = True
            self._pan_start = event.pos
            return

        if event.button == 4:
            self._zoom(program, 1.2, event.pos)
            return
        elif event.button == 5:
            self._zoom(program, 0.833, event.pos)
            return

        world = program.simulation.grid_world
        if event.button == 3:
            if world.pending_stair_endpoint is not None:
                world.pending_stair_endpoint = None
                world.stair_link_notice = None
                world.stair_link_error = None
                self._rebuild_ui(program)
                return
            handle_map_editor(program.simulation, event.pos, 3, self.selected_tool)
            return

        if event.button == 1:
            grid_x, grid_y = pixel_to_grid(*event.pos)
            if (
                self.selected_tool == Tool.WALL
                and (pygame.key.get_mods() & pygame.KMOD_SHIFT)
                and world.map_grid.in_bounds(grid_x, grid_y)
            ):
                self._line_start = (grid_x, grid_y)
                self._line_end = (grid_x, grid_y)
                return

            if world.map_grid.in_bounds(grid_x, grid_y):
                endpoint = world.stair_cluster_at(world.active_floor_id, grid_x, grid_y)
                if endpoint is not None:
                    self._handle_stair_click(program, endpoint)
                    return
                elif world.pending_stair_endpoint is not None and self.selected_tool != Tool.STAIRS:
                    world.pending_stair_endpoint = None
                    world.stair_link_notice = None
                    world.stair_link_error = None
                    self._rebuild_ui(program)

            handle_map_editor(program.simulation, event.pos, 1, self.selected_tool)

    def handle_mouse_motion(self, program, event):
        if self._is_panning:
            dx = event.pos[0] - self._pan_start[0]
            dy = event.pos[1] - self._pan_start[1]
            import utils.config as config
            config.camera_offset_x += dx
            config.camera_offset_y += dy
            self._pan_start = event.pos
            return

        if self._line_start is not None and event.buttons[0]:
            world = program.simulation.grid_world
            gx, gy = pixel_to_grid(*event.pos)
            gx = max(0, min(world.map_grid.width - 1, gx))
            gy = max(0, min(world.map_grid.height - 1, gy))
            self._line_end = (gx, gy)
            return

        buttons = event.buttons
        if not buttons[0] and not buttons[2]:
            return
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        btn = 1 if buttons[0] else 3
        handle_map_editor(program.simulation, event.pos, btn, self.selected_tool)

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
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        import utils.config as config

        world = program.simulation.grid_world if program.simulation else None
        active_floor = world.active_floor if world else None
        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        zoom_pct = int(round((getattr(config, 'drawing_size', 8) / 8.0) * 100))
        self._draw_bottom_canvas_pills(
            screen,
            zoom_pct,
            has_ref=self._has_floor_plan_reference(active_floor, program),
        )

        if self._line_start is not None and self._line_end is not None:
            canvas_rect = pygame.Rect(0, TOP_BAR_HEIGHT, canvas_w, max(100, screen_h - TOP_BAR_HEIGHT))
            screen.set_clip(canvas_rect)
            d_size = getattr(config, 'drawing_size', 8)
            line_cells = _bresenham_cells(self._line_start[0], self._line_start[1], self._line_end[0], self._line_end[1])
            preview_surf = pygame.Surface((d_size, d_size), pygame.SRCALPHA)
            preview_surf.fill((59, 130, 246, 140))
            for lx, ly in line_cells:
                px, py = grid_to_pixel(lx, ly)
                screen.blit(preview_surf, (px, py))
                pygame.draw.rect(screen, (147, 197, 253), (px, py, d_size, d_size), 1)
            p1 = grid_to_pixel_center(*self._line_start)
            p2 = grid_to_pixel_center(*self._line_end)
            pygame.draw.line(screen, (245, 158, 11), p1, p2, 2)
            screen.set_clip(None)

        hint_text = (
            "Left click: draw • Shift+Drag: straight wall • Right click: erase • Scroll: zoom • Arrows: pan"
            if self.selected_tool == Tool.WALL
            else "Left click: draw • Right click: erase • Scroll: zoom • Arrows: pan"
        )
        hint_surf = self.tiny_font.render(hint_text, True, (150, 155, 165))
        hint_rect = pygame.Rect(canvas_w // 2 - hint_surf.get_width() // 2 - 10, 92, hint_surf.get_width() + 20, 28)
        pygame.draw.rect(screen, (24, 25, 28), hint_rect, border_radius=5)
        pygame.draw.rect(screen, (48, 52, 60), hint_rect, 1, border_radius=5)
        screen.blit(hint_surf, hint_surf.get_rect(center=hint_rect.center))

        world = program.simulation.grid_world if program.simulation else None
        if world and len(world.floors) > 1:
            active_floor_id = world.active_floor_id
            stair_clusters = world.get_stair_clusters(active_floor_id)
            pending = world.pending_stair_endpoint
            act_floor_id = world.active_floor_id
            stair_div_y = 438 if act_floor_id > 0 else 406
            stair_y = stair_div_y + 30
            tool_x = screen_w - SIDEBAR_WIDTH + 16

            if not stair_clusters:
                no_s = self.tiny_font.render("No stairs placed on this floor", True, (120, 125, 135))
                screen.blit(no_s, no_s.get_rect(left=tool_x, centery=stair_y + 13))
            else:
                for cid, c in stair_clusters:
                    s_name = world.get_stair_name(active_floor_id, cid)
                    target = getattr(c, "linked_to", None)
                    if target is not None:
                        other_fid, other_cid = target
                        other_name = world.get_stair_name(other_fid, other_cid)
                        lbl_txt = f"{s_name} <-> F{other_fid + 1} ({other_name})"
                        col = (16, 185, 129)
                    elif pending == (active_floor_id, cid):
                        lbl_txt = f"{s_name}: (Select target)"
                        col = (245, 158, 11)
                    else:
                        lbl_txt = f"{s_name}: (Unlinked)"
                        col = (170, 175, 185)
                    lbl_surf = self.tiny_font.render(lbl_txt, True, col)
                    screen.blit(lbl_surf, lbl_surf.get_rect(left=tool_x, centery=stair_y + 13))
                    stair_y += 32


class OverviewState(ProgramState):
    name = "overview"

    def enter(self, program):
        program.transition_to(EditorState())


class SimulationRunningState(ProgramState):
    name = "simulation_running"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = ("L: Toggle A* / DQN drones",)

    def __init__(self):
        super().__init__()
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_sim = None
        self._btn_hazard = None
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
        self._routing_dropdown = None
        self._speed_buttons = {}

    def enter(self, program):
        if program.simulation and not program.simulation.started:
            program.simulation.start()
        if not getattr(program, "drone_routes", None):
            program.plan_drone_routes()
        program.sim_accumulator = program.simulation.get_step_duration_s() if program.simulation else 0.5
        self._rebuild_ui(program)

    def handle_resize(self, program, *args):
        self._rebuild_ui(program)

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        self._speed_buttons = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        title_w = 90
        save_x = max(180, 48 + title_w + 24)
        save_w = 110
        menu_x = save_x + save_w + 8
        menu_w = 70

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(save_x, 9, save_w, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(menu_x, 9, menu_w, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 260
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 128
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "Simulate",
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
        btn_w_half = 122
        gap = 4
        opts = program.view_options

        self._btn_hazard = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 270, tool_w, 30),
            "Place Hazards (Pause first)",
            manager=program.ui_manager,
            object_id="#tool_btn",
            tool_tip_text="Pause simulation to place or edit hazards",
        )
        self._btn_hazard.disable()

        y = 338
        self._btn_toggle_clusters = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 26),
            "Clusters",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_clusters else "#tool_btn",
        )
        self._btn_toggle_connections = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 26),
            "Links",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_cluster_connections else "#tool_btn",
        )
        y = 368
        self._btn_toggle_routes = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 26),
            "Guide Routes",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_drone_routes else "#tool_btn",
        )
        self._btn_toggle_assignments = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 26),
            "Assign",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_assignments else "#tool_btn",
        )
        y = 398
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 26),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_grid else "#tool_btn",
        )
        self._btn_toggle_heatmap = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 26),
            "Heatmap",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if getattr(opts, "show_heatmap", False) else "#tool_btn",
        )

        y = 462
        current_algo = getattr(program, "routing_algorithm", "Dijkstra")
        algo_options = ["Dijkstra", "A*", "Bidirectional Dijkstra", "BFS"]
        self._routing_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=algo_options,
            starting_option=current_algo,
            relative_rect=pygame.Rect(tool_x, y, tool_w, 32),
            manager=program.ui_manager,
            object_id="#dropdown_routing",
        )

        y = 532
        self._btn_pause = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 32),
            "Pause",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        self._btn_reset = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 32),
            "Reset",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        y = 568
        self._btn_finish = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, 30),
            "Finish Simulation",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )

        y = 636
        speed_opts = [("1x", 1.0), ("2x", 2.0), ("4x", 4.0), ("8x", 8.0)]
        sp_w = (tool_w - 3 * gap) // 4
        cur_speed = getattr(program, "simulation_speed", 1.0)
        for idx, (lbl, val) in enumerate(speed_opts):
            sp_x = tool_x + idx * (sp_w + gap)
            is_active = (cur_speed == val)
            btn = pygame_gui.elements.UIButton(
                pygame.Rect(sp_x, y, sp_w, 28),
                lbl,
                manager=program.ui_manager,
                object_id="#tool_btn_selected" if is_active else "#tool_btn",
            )
            self._speed_buttons[val] = btn

        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        zoom_pill_x = canvas_w - 212 - 20
        zoom_pill_y = screen_h - 52
        self._btn_zoom_out = pygame_gui.elements.UIButton(
            pygame.Rect(zoom_pill_x + 4, zoom_pill_y + 4, 68, 30),
            "Zoom -",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        self._btn_zoom_in = pygame_gui.elements.UIButton(
            pygame.Rect(zoom_pill_x + 140, zoom_pill_y + 4, 68, 30),
            "Zoom +",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

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
            cur_size = program.screen.get_size()
            if program.ui_manager:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
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
        elif event.ui_element == getattr(self, "_btn_zoom_in", None):
            canvas_w = max(100, program.screen.get_width() - SIDEBAR_WIDTH)
            self._zoom(program, 1.25, (canvas_w // 2, program.screen.get_height() // 2))
        elif event.ui_element == getattr(self, "_btn_zoom_out", None):
            canvas_w = max(100, program.screen.get_width() - SIDEBAR_WIDTH)
            self._zoom(program, 0.8, (canvas_w // 2, program.screen.get_height() // 2))
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self._routing_dropdown:
            selected = event.text
            program.set_routing_algorithm(selected)
        else:
            for val, btn in self._speed_buttons.items():
                if event.ui_element == btn:
                    program.simulation_speed = val
                    self._rebuild_ui(program)
                    return
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
            program.transition_to(SimulationPausedState())
        elif key == pygame.K_h:
            program.view_options.show_assignments = not program.view_options.show_assignments
            self._rebuild_ui(program)
        elif key == pygame.K_x:
            program.transition_to(ResultsState())
        elif key == pygame.K_f:
            program.simulation_options.autonomous_drones = not program.simulation_options.autonomous_drones
        elif key == pygame.K_l:
            program.toggle_drone_policy()
        elif key == pygame.K_r:
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
        elif key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)
        import utils.config as config
        if key == pygame.K_LEFT:
            config.camera_offset_x += 24
            program.save_active_floor_camera()
        elif key == pygame.K_RIGHT:
            config.camera_offset_x -= 24
            program.save_active_floor_camera()
        elif key == pygame.K_UP:
            config.camera_offset_y += 24
            program.save_active_floor_camera()
        elif key == pygame.K_DOWN:
            config.camera_offset_y -= 24
            program.save_active_floor_camera()

    def _zoom(self, program, factor, mouse_pos):
        import utils.config as config
        old_size = getattr(config, 'drawing_size', 8)
        new_size = max(4, min(24, int(round(old_size * factor))))
        if new_size != old_size:
            mx, my = mouse_pos
            gx = (mx - getattr(config, 'camera_offset_x', 0)) / old_size
            gy = (my - getattr(config, 'camera_offset_y', 0)) / old_size
            config.drawing_size = new_size
            config.camera_offset_x = int(mx - gx * new_size)
            config.camera_offset_y = int(my - gy * new_size)
            self._rebuild_ui(program)

    def handle_mouse_wheel(self, program, event):
        if event.y > 0:
            self._zoom(program, 1.2, pygame.mouse.get_pos())
        elif event.y < 0:
            self._zoom(program, 0.833, pygame.mouse.get_pos())

    def handle_mouse_up(self, program, event):
        if event.button == 2:
            self._is_panning = False
            program.save_active_floor_camera()

    def handle_mouse(self, program, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            self._is_panning = True
            self._pan_start = event.pos

    def handle_mouse_motion(self, program, event):
        if getattr(self, "_is_panning", False) and getattr(self, "_pan_start", None) is not None:
            dx = event.pos[0] - self._pan_start[0]
            dy = event.pos[1] - self._pan_start[1]
            import utils.config as config
            config.camera_offset_x += dx
            config.camera_offset_y += dy
            self._pan_start = event.pos

    def on_floor_changed(self, program):
        self._rebuild_ui(program)

    def update(self, program):
        super().update(program)
        if program.simulation and program.simulation.started and len(program.simulation.evacuees) == 0:
            program.transition_to(ResultsState())
            return

        dt = getattr(program, "time_delta", 0.03)
        raw_speed = getattr(program, "simulation_speed", 1.0)
        if isinstance(raw_speed, str):
            try:
                speed = float(raw_speed.replace("x", ""))
            except (ValueError, TypeError):
                speed = 1.0
        else:
            speed = float(raw_speed) if raw_speed else 1.0
        step_duration = program.simulation.get_step_duration_s() if program.simulation else 0.5
        program.sim_accumulator = getattr(program, "sim_accumulator", 0.0) + dt * speed

        max_steps = 10
        steps_taken = 0
        while program.sim_accumulator >= step_duration and steps_taken < max_steps:
            program.sim_accumulator -= step_duration
            steps_taken += 1

            options = program.simulation_options
            if options.rl_evacuees and program.rl_evacuee_controller:
                evacuee_actions = program.rl_evacuee_controller.get_actions(program.simulation)
            else:
                evacuee_actions = get_actions(program.simulation.evacuees, options.action)

            if options.autonomous_drones and program.drone_controller:
                drone_actions = program.autonomous_drone_actions()
            else:
                drone_actions = get_actions(program.simulation.drones, options.action)

            program.simulation.step(evacuee_actions, drone_actions)
            if options.action is Action.USE_STAIRS:
                options.action = Action.STAY

            if program.simulation.started and len(program.simulation.evacuees) == 0:
                program.transition_to(ResultsState())
                break

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        import utils.config as config
        zoom_pill_x = canvas_w - 212 - 20
        zoom_pill_y = screen_h - 52
        zoom_pill_rect = pygame.Rect(zoom_pill_x, zoom_pill_y, 212, 38)
        pygame.draw.rect(screen, Color.CARD_BG.value, zoom_pill_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, zoom_pill_rect, 1, border_radius=7)
        zoom_pct = int(round((getattr(config, 'drawing_size', 8) / 8.0) * 100))
        zoom_txt = self.font.render(f"{zoom_pct}%", True, (240, 240, 240))
        screen.blit(
            zoom_txt,
            (zoom_pill_rect.centerx - zoom_txt.get_width() // 2, zoom_pill_rect.centery - zoom_txt.get_height() // 2),
        )

        hint_surf = self.tiny_font.render("Scroll: zoom • Arrows: pan", True, (150, 155, 165))
        hint_rect = pygame.Rect(canvas_w // 2 - hint_surf.get_width() // 2 - 10, 92, hint_surf.get_width() + 20, 28)
        pygame.draw.rect(screen, (24, 25, 28), hint_rect, border_radius=5)
        pygame.draw.rect(screen, (48, 52, 60), hint_rect, 1, border_radius=5)
        screen.blit(hint_surf, hint_surf.get_rect(center=hint_rect.center))


class SimulationPausedState(ProgramState):
    name = "simulation_paused"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = ()

    def __init__(self):
        super().__init__()
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
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
        self._routing_dropdown = None
        self._speed_buttons = {}

    def enter(self, program):
        self._rebuild_ui(program)

    def handle_resize(self, program, *args):
        self._rebuild_ui(program)

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def _rebuild_ui(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self._floor_tabs = {}
        self._speed_buttons = {}
        screen_w, screen_h = program.screen.get_size()
        sidebar_x = screen_w - SIDEBAR_WIDTH

        title_w = 90
        save_x = max(180, 48 + title_w + 24)
        save_w = 110
        menu_x = save_x + save_w + 8
        menu_w = 70

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(save_x, 9, save_w, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(menu_x, 9, menu_w, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 260
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 128
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "Simulate",
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
        btn_w_half = 122
        gap = 4
        opts = program.view_options

        self._btn_hazard = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, 270, tool_w, 30),
            "Place Hazards (M)",
            manager=program.ui_manager,
            object_id="#tool_btn",
        )

        y = 338
        self._btn_toggle_clusters = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 26),
            "Clusters",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_clusters else "#tool_btn",
        )
        self._btn_toggle_connections = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 26),
            "Links",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_cluster_connections else "#tool_btn",
        )
        y = 368
        self._btn_toggle_routes = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 26),
            "Guide Routes",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_drone_routes else "#tool_btn",
        )
        self._btn_toggle_assignments = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 26),
            "Assign",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_assignments else "#tool_btn",
        )
        y = 398
        self._btn_toggle_grid = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 26),
            "Grid",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if opts.show_grid else "#tool_btn",
        )
        self._btn_toggle_heatmap = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 26),
            "Heatmap",
            manager=program.ui_manager,
            object_id="#tool_btn_selected" if getattr(opts, "show_heatmap", False) else "#tool_btn",
        )

        y = 462
        current_algo = getattr(program, "routing_algorithm", "Dijkstra")
        algo_options = ["Dijkstra", "A*", "Bidirectional Dijkstra", "BFS"]
        self._routing_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=algo_options,
            starting_option=current_algo,
            relative_rect=pygame.Rect(tool_x, y, tool_w, 32),
            manager=program.ui_manager,
            object_id="#dropdown_routing",
        )

        y = 532
        resume_text = "Resume" if (program.simulation and program.simulation.started) else "Start"
        self._btn_resume = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, btn_w_half, 32),
            resume_text,
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        self._btn_reset = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x + btn_w_half + gap, y, btn_w_half, 32),
            "Reset",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        y = 568
        self._btn_finish = pygame_gui.elements.UIButton(
            pygame.Rect(tool_x, y, tool_w, 30),
            "Finish Simulation",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )

        y = 636
        speed_opts = [("1x", 1.0), ("2x", 2.0), ("4x", 4.0), ("8x", 8.0)]
        sp_w = (tool_w - 3 * gap) // 4
        cur_speed = getattr(program, "simulation_speed", 1.0)
        for idx, (lbl, val) in enumerate(speed_opts):
            sp_x = tool_x + idx * (sp_w + gap)
            is_active = (cur_speed == val)
            btn = pygame_gui.elements.UIButton(
                pygame.Rect(sp_x, y, sp_w, 28),
                lbl,
                manager=program.ui_manager,
                object_id="#tool_btn_selected" if is_active else "#tool_btn",
            )
            self._speed_buttons[val] = btn

        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        zoom_pill_x = canvas_w - 212 - 20
        zoom_pill_y = screen_h - 52
        self._btn_zoom_out = pygame_gui.elements.UIButton(
            pygame.Rect(zoom_pill_x + 4, zoom_pill_y + 4, 68, 30),
            "Zoom -",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )
        self._btn_zoom_in = pygame_gui.elements.UIButton(
            pygame.Rect(zoom_pill_x + 140, zoom_pill_y + 4, 68, 30),
            "Zoom +",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

    def handle_ui_event(self, program, event):
        if event.ui_element == self._save_btn:
            _save_project(program)
        elif event.ui_element == self._btn_resume:
            if program.simulation and not program.simulation.started:
                program.simulation.start()
            program.transition_to(SimulationRunningState())
        elif event.ui_element == self._btn_finish:
            program.transition_to(ResultsState())
        elif event.ui_element == self._menu_btn:
            cur_size = program.screen.get_size()
            if program.ui_manager:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element == self._btn_hazard:
            program.transition_to(SimulationEditorState())
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
            self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_btn_zoom_in", None):
            canvas_w = max(100, program.screen.get_width() - SIDEBAR_WIDTH)
            self._zoom(program, 1.25, (canvas_w // 2, program.screen.get_height() // 2))
        elif event.ui_element == getattr(self, "_btn_zoom_out", None):
            canvas_w = max(100, program.screen.get_width() - SIDEBAR_WIDTH)
            self._zoom(program, 0.8, (canvas_w // 2, program.screen.get_height() // 2))
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self._routing_dropdown:
            selected = event.text
            program.set_routing_algorithm(selected)
        else:
            for val, btn in self._speed_buttons.items():
                if event.ui_element == btn:
                    program.simulation_speed = val
                    self._rebuild_ui(program)
                    return
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
            if program.simulation and not program.simulation.started:
                program.simulation.start()
            program.transition_to(SimulationRunningState())
        elif key == pygame.K_m:
            program.transition_to(SimulationEditorState())
        elif key == pygame.K_h:
            program.view_options.show_assignments = not program.view_options.show_assignments
            self._rebuild_ui(program)
        elif key == pygame.K_r:
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
        elif key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)
        import utils.config as config
        if key == pygame.K_LEFT:
            config.camera_offset_x += 24
            program.save_active_floor_camera()
        elif key == pygame.K_RIGHT:
            config.camera_offset_x -= 24
            program.save_active_floor_camera()
        elif key == pygame.K_UP:
            config.camera_offset_y += 24
            program.save_active_floor_camera()
        elif key == pygame.K_DOWN:
            config.camera_offset_y -= 24
            program.save_active_floor_camera()

    def _zoom(self, program, factor, mouse_pos):
        import utils.config as config
        old_size = getattr(config, 'drawing_size', 8)
        new_size = max(4, min(24, int(round(old_size * factor))))
        if new_size != old_size:
            mx, my = mouse_pos
            gx = (mx - getattr(config, 'camera_offset_x', 0)) / old_size
            gy = (my - getattr(config, 'camera_offset_y', 0)) / old_size
            config.drawing_size = new_size
            config.camera_offset_x = int(mx - gx * new_size)
            config.camera_offset_y = int(my - gy * new_size)
            self._rebuild_ui(program)

    def handle_mouse_wheel(self, program, event):
        if event.y > 0:
            self._zoom(program, 1.2, pygame.mouse.get_pos())
        elif event.y < 0:
            self._zoom(program, 0.833, pygame.mouse.get_pos())

    def handle_mouse_up(self, program, event):
        if event.button == 2:
            self._is_panning = False
            program.save_active_floor_camera()

    def handle_mouse(self, program, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            self._is_panning = True
            self._pan_start = event.pos

    def handle_mouse_motion(self, program, event):
        if getattr(self, "_is_panning", False) and getattr(self, "_pan_start", None) is not None:
            dx = event.pos[0] - self._pan_start[0]
            dy = event.pos[1] - self._pan_start[1]
            import utils.config as config
            config.camera_offset_x += dx
            config.camera_offset_y += dy
            self._pan_start = event.pos

    def on_floor_changed(self, program):
        self._rebuild_ui(program)

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        import utils.config as config
        zoom_pill_x = canvas_w - 212 - 20
        zoom_pill_y = screen_h - 52
        zoom_pill_rect = pygame.Rect(zoom_pill_x, zoom_pill_y, 212, 38)
        pygame.draw.rect(screen, Color.CARD_BG.value, zoom_pill_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, zoom_pill_rect, 1, border_radius=7)
        zoom_pct = int(round((getattr(config, 'drawing_size', 8) / 8.0) * 100))
        zoom_txt = self.font.render(f"{zoom_pct}%", True, (240, 240, 240))
        screen.blit(
            zoom_txt,
            (zoom_pill_rect.centerx - zoom_txt.get_width() // 2, zoom_pill_rect.centery - zoom_txt.get_height() // 2),
        )

        hint_surf = self.tiny_font.render("Scroll: zoom • Arrows: pan", True, (150, 155, 165))
        hint_rect = pygame.Rect(canvas_w // 2 - hint_surf.get_width() // 2 - 10, 92, hint_surf.get_width() + 20, 28)
        pygame.draw.rect(screen, (24, 25, 28), hint_rect, border_radius=5)
        pygame.draw.rect(screen, (48, 52, 60), hint_rect, 1, border_radius=5)
        screen.blit(hint_surf, hint_surf.get_rect(center=hint_rect.center))


class SimulationEditorState(ProgramState):
    name = "simulation_editor"
    can_pan = True
    render_options_enabled = True
    floor_switch_enabled = True
    key_bindings = ()

    def __init__(self):
        super().__init__()
        self._floor_tabs = {}
        self._save_btn = None
        self._menu_btn = None
        self._nav_edit = None
        self._nav_sim = None
        self._btn_done = None
        self._btn_clear_hazards = None

    def enter(self, program):
        self._rebuild_ui(program)

    def on_floor_changed(self, program):
        self._rebuild_ui(program)

    def handle_resize(self, program, *args):
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

        title_w = 90
        save_x = max(180, 48 + title_w + 24)
        save_w = 110
        menu_x = save_x + save_w + 8
        menu_w = 70

        self._save_btn = pygame_gui.elements.UIButton(
            pygame.Rect(save_x, 9, save_w, 30),
            "Save (Ctrl+S)",
            manager=program.ui_manager,
            object_id="#save_button",
        )
        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(menu_x, 9, menu_w, 30),
            "Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )

        switcher_w = 260
        switcher_x = screen_w - switcher_w - 16
        tab_item_w = 128
        self._nav_edit = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2, 9, tab_item_w, 30),
            "Edit Map",
            manager=program.ui_manager,
            object_id="#nav_item_inactive",
        )
        self._nav_sim = pygame_gui.elements.UIButton(
            pygame.Rect(switcher_x + 2 + tab_item_w, 9, tab_item_w, 30),
            "Simulate",
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

        overlay_on = getattr(program.view_options, "show_plan_overlay", False)
        has_ref = self._has_floor_plan_reference(active_floor, program)
        self._build_bottom_canvas_controls(
            program.ui_manager,
            screen_w,
            screen_h,
            overlay_on and has_ref,
            has_ref,
            zoom_btn_id="#btn_secondary",
        )

    def _zoom(self, program, factor, mouse_pos):
        import utils.config as config
        old_size = getattr(config, 'drawing_size', 8)
        new_size = max(4, min(24, int(round(old_size * factor))))
        if new_size != old_size:
            mx, my = mouse_pos
            gx = (mx - getattr(config, 'camera_offset_x', 0)) / old_size
            gy = (my - getattr(config, 'camera_offset_y', 0)) / old_size
            config.drawing_size = new_size
            config.camera_offset_x = int(mx - gx * new_size)
            config.camera_offset_y = int(my - gy * new_size)
            program.save_active_floor_camera()
            self._rebuild_ui(program)

    def handle_mouse_wheel(self, program, event):
        if event.y > 0:
            self._zoom(program, 1.2, pygame.mouse.get_pos())
        elif event.y < 0:
            self._zoom(program, 0.833, pygame.mouse.get_pos())

    def handle_ui_event(self, program, event):
        if event.ui_element == self._save_btn:
            _save_project(program)
        elif event.ui_element == self._menu_btn:
            cur_size = program.screen.get_size()
            if program.ui_manager:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())
        elif event.ui_element == self._nav_edit:
            program.transition_to(EditorState())
        elif event.ui_element in (self._nav_sim, self._btn_done):
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_clear_hazards:
            program.simulation.grid_world.clear_temporary_walls()
            program.replan_drone_routes_after_hazard()
            self._rebuild_ui(program)
        elif event.ui_element == getattr(self, "_btn_zoom_in", None):
            canvas_w = max(100, program.screen.get_width() - SIDEBAR_WIDTH)
            self._zoom(program, 1.25, (canvas_w // 2, program.screen.get_height() // 2))
        elif event.ui_element == getattr(self, "_btn_zoom_out", None):
            canvas_w = max(100, program.screen.get_width() - SIDEBAR_WIDTH)
            self._zoom(program, 0.8, (canvas_w // 2, program.screen.get_height() // 2))
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
        if key in (pygame.K_m, pygame.K_RETURN, pygame.K_ESCAPE):
            program.transition_to(SimulationPausedState())
        elif key == pygame.K_s and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            _save_project(program)
        import utils.config as config
        if key == pygame.K_LEFT:
            config.camera_offset_x += 24
            program.save_active_floor_camera()
        elif key == pygame.K_RIGHT:
            config.camera_offset_x -= 24
            program.save_active_floor_camera()
        elif key == pygame.K_UP:
            config.camera_offset_y += 24
            program.save_active_floor_camera()
        elif key == pygame.K_DOWN:
            config.camera_offset_y -= 24
            program.save_active_floor_camera()

    def handle_mouse(self, program, event):
        if event.button == 2:
            self._is_panning = True
            self._pan_start = event.pos
            return
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        grid_x, grid_y = pixel_to_grid(*event.pos)
        world = program.simulation.grid_world
        if not world.map_grid.in_bounds(grid_x, grid_y):
            return

        if event.button == 1:
            current_cell = world.map_grid.get(grid_x, grid_y)
            if current_cell not in (CellType.WALL, CellType.EXIT):
                world.set_temporary_wall(world.active_floor_id, grid_x, grid_y, True)
                program.replan_drone_routes_after_hazard()
        elif event.button == 3:
            world.set_temporary_wall(world.active_floor_id, grid_x, grid_y, False)
            program.replan_drone_routes_after_hazard()

    def handle_mouse_up(self, program, event):
        if event.button == 2:
            self._is_panning = False
            program.save_active_floor_camera()

    def handle_mouse_motion(self, program, event):
        if getattr(self, "_is_panning", False) and getattr(self, "_pan_start", None) is not None:
            dx = event.pos[0] - self._pan_start[0]
            dy = event.pos[1] - self._pan_start[1]
            import utils.config as config
            config.camera_offset_x += dx
            config.camera_offset_y += dy
            self._pan_start = event.pos
            return
        buttons = event.buttons
        if not buttons[0] and not buttons[2]:
            return
        screen_w, screen_h = program.screen.get_size()
        if event.pos[0] >= screen_w - SIDEBAR_WIDTH or event.pos[1] < TOP_BAR_HEIGHT:
            return
        grid_x, grid_y = pixel_to_grid(*event.pos)
        world = program.simulation.grid_world
        if not world.map_grid.in_bounds(grid_x, grid_y):
            return

        if buttons[0]:
            current_cell = world.map_grid.get(grid_x, grid_y)
            if current_cell not in (CellType.WALL, CellType.EXIT):
                world.set_temporary_wall(world.active_floor_id, grid_x, grid_y, True)
                program.replan_drone_routes_after_hazard()
        elif buttons[2]:
            world.set_temporary_wall(world.active_floor_id, grid_x, grid_y, False)
            program.replan_drone_routes_after_hazard()

    def render(self, program):
        program.window.draw_simulation(program, self.key_bindings)
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        canvas_w = max(100, screen_w - SIDEBAR_WIDTH)
        import utils.config as config
        zoom_pill_x = canvas_w - 212 - 20
        zoom_pill_y = screen_h - 52
        zoom_pill_rect = pygame.Rect(zoom_pill_x, zoom_pill_y, 212, 38)
        pygame.draw.rect(screen, Color.CARD_BG.value, zoom_pill_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, zoom_pill_rect, 1, border_radius=7)
        zoom_pct = int(round((getattr(config, 'drawing_size', 8) / 8.0) * 100))
        zoom_txt = self.font.render(f"{zoom_pct}%", True, (240, 240, 240))
        screen.blit(
            zoom_txt,
            (zoom_pill_rect.centerx - zoom_txt.get_width() // 2, zoom_pill_rect.centery - zoom_txt.get_height() // 2),
        )

        hint_surf = self.tiny_font.render("Left click: hazard • Right click: erase • Scroll: zoom • Arrows: pan", True, (150, 155, 165))
        hint_rect = pygame.Rect(canvas_w // 2 - hint_surf.get_width() // 2 - 10, 92, hint_surf.get_width() + 20, 28)
        pygame.draw.rect(screen, (24, 25, 28), hint_rect, border_radius=5)
        pygame.draw.rect(screen, (48, 52, 60), hint_rect, 1, border_radius=5)
        screen.blit(hint_surf, hint_surf.get_rect(center=hint_rect.center))


class ResultsState(ProgramState):
    name = "results"
    can_pan = False
    render_options_enabled = False
    floor_switch_enabled = False
    key_bindings = ()

    def __init__(self):
        self._menu_btn = None
        self._btn_export_png = None
        self._btn_export_csv = None
        self._btn_back_sim = None
        self._btn_back_edit = None
        self._chart_surface = None
        self._chart_surface_size = None
        self._feedback_msg = ""
        self._feedback_timer = 0.0
        self.screen_size = (1280, 800)
        self.title_font = get_font(22, bold=True)
        self.font = get_font(15)
        self.small_font = get_font(13)
        self.tiny_font = get_font(11, bold=True)

    def enter(self, program):
        if program.ui_manager is None:
            return
        program.ui_manager.clear_and_reset()
        self.screen_size = program.screen.get_size()
        screen_w, screen_h = self.screen_size

        pad_x = 24 if screen_w >= 1150 else 16
        content_w = screen_w - 2 * pad_x

        ui_scale = min(1.35, max(1.0, screen_w / 1280.0, screen_h / 800.0))
        bottom_bar_h = int(76 * ui_scale)
        card_h = int(74 * ui_scale)
        bottom_y = screen_h - bottom_bar_h - 14

        btn_y = bottom_y + int(28 * ui_scale)
        btn_h = max(34, int(34 * ui_scale))

        exp_x = pad_x + 16
        self._btn_export_png = pygame_gui.elements.UIButton(
            pygame.Rect(exp_x, btn_y, int(165 * ui_scale), btn_h),
            "Export Report (PNG)",
            manager=program.ui_manager,
            object_id="#btn_primary",
        )
        self._btn_export_csv = pygame_gui.elements.UIButton(
            pygame.Rect(exp_x + int(176 * ui_scale), btn_y, int(145 * ui_scale), btn_h),
            "Export Data (CSV)",
            manager=program.ui_manager,
            object_id="#btn_secondary",
        )

        nav_right = pad_x + content_w - 16
        menu_w = int(100 * ui_scale)
        edit_w = int(95 * ui_scale)
        sim_w = int(150 * ui_scale)

        self._menu_btn = pygame_gui.elements.UIButton(
            pygame.Rect(nav_right - menu_w, btn_y, menu_w, btn_h),
            "Main Menu",
            manager=program.ui_manager,
            object_id="#menu_button",
        )
        self._btn_back_edit = pygame_gui.elements.UIButton(
            pygame.Rect(nav_right - menu_w - 10 - edit_w, btn_y, edit_w, btn_h),
            "Edit Map",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )
        self._btn_back_sim = pygame_gui.elements.UIButton(
            pygame.Rect(nav_right - menu_w - 10 - edit_w - 10 - sim_w, btn_y, sim_w, btn_h),
            "Back to Simulation",
            manager=program.ui_manager,
            object_id="#btn_playback",
        )

        has_multifloor = False
        if program.simulation and program.simulation.grid_world:
            has_multifloor = len(program.simulation.grid_world.floors) > 1

        strip_h = int(36 * ui_scale) if has_multifloor else 0
        charts_y = 60 + card_h + (strip_h + 10 if has_multifloor else 12)
        chart_h = max(240, bottom_y - charts_y - 12)

        if program.simulation and hasattr(program.simulation, "metrics"):
            self._chart_surface = program.simulation.metrics.generate_charts_surface(content_w, chart_h)
            self._chart_surface_size = (content_w, chart_h)
        else:
            self._chart_surface = None

    def exit(self, program):
        if program.ui_manager is not None:
            program.ui_manager.clear_and_reset()

    def handle_resize(self, program, *args):
        self.enter(program)

    def update(self, program):
        if self._feedback_timer > 0.0:
            self._feedback_timer -= 0.03
            if self._feedback_timer <= 0.0:
                self._feedback_msg = ""

    def render(self, program):
        screen = program.screen
        screen_w, screen_h = screen.get_size()
        if (screen_w, screen_h) != self.screen_size:
            self.enter(program)
            screen_w, screen_h = self.screen_size

        screen.fill((18, 18, 18))

        pygame.draw.rect(screen, Color.TOPNAV_BG.value, (0, 0, screen_w, 48))
        pygame.draw.line(screen, Color.DIVIDER.value, (0, 47), (screen_w, 47), 1)

        badge_rect = pygame.Rect(16, (48 - 28) // 2, 28, 28)
        pygame.draw.rect(screen, (255, 255, 255), badge_rect, border_radius=6)
        logo = getattr(program.window, "logo_surf", None)
        if logo:
            lx = badge_rect.centerx - logo.get_width() // 2
            ly = badge_rect.centery - logo.get_height() // 2
            screen.blit(logo, (lx, ly))
        else:
            pygame.draw.rect(screen, Color.ACCENT_BLUE.value, badge_rect, border_radius=6)

        title_text = "ARIADNE - Evacuation Summary"
        title_surf = self.title_font.render(title_text, True, Color.TEXT_PRIMARY.value)
        screen.blit(title_surf, title_surf.get_rect(left=badge_rect.right + 12, centery=24))

        cur_algo = getattr(program, "routing_algorithm", "Dijkstra")
        algo_label = f"Routing Algorithm: {cur_algo}"
        algo_surf = self.small_font.render(algo_label, True, (170, 170, 170))
        screen.blit(algo_surf, algo_surf.get_rect(right=screen_w - 24, centery=24))

        pad_x = 24 if screen_w >= 1150 else 16
        content_w = screen_w - 2 * pad_x

        if program.simulation and hasattr(program.simulation, "metrics"):
            kpis = program.simulation.metrics.get_summary_kpis(program.simulation)
        else:
            kpis = {
                "total_time_str": "00:00",
                "total_steps": 0,
                "evacuated_str": "0 / 0 (0%)",
                "avg_flow_str": "0.0 people/s",
                "efficiency_str": "100% Walking / 0% Waiting",
                "efficiency_val": "100% Walking",
                "efficiency_sub": "queueing vs movement ratio",
                "drone_impact_val": "No guide agents used",
                "drone_impact_sub": "guide agents inactive",
            }

        ui_scale = min(1.35, max(1.0, screen_w / 1280.0, screen_h / 800.0))
        card_gap = 12
        card_w = (content_w - 3 * card_gap) // 4
        card_h = int(74 * ui_scale)
        card_y = 60

        kpi_defs = [
            ("TOTAL EVACUATION TIME", kpis["total_time_str"], f"{kpis['total_steps']} simulation steps"),
            ("EVACUATED / FLOW RATE", kpis["evacuated_str"], f"avg {kpis['avg_flow_str']}"),
            ("MOVEMENT EFFICIENCY", kpis.get("efficiency_val", kpis["efficiency_str"]), kpis["efficiency_sub"]),
            ("GUIDE AGENT IMPACT", kpis["drone_impact_val"], kpis["drone_impact_sub"]),
        ]

        for i, (hdr, val, sub) in enumerate(kpi_defs):
            cx = pad_x + i * (card_w + card_gap)
            c_rect = pygame.Rect(cx, card_y, card_w, card_h)
            pygame.draw.rect(screen, Color.CARD_BG.value, c_rect, border_radius=7)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, c_rect, 1, border_radius=7)

            h_surf = self.tiny_font.render(hdr, True, (150, 150, 150))
            v_font = self.title_font
            v_surf = v_font.render(val, True, (245, 245, 245))
            if v_surf.get_width() > card_w - 24:
                for sz in (20, 18, 16, 14):
                    fallback_font = get_font(sz, bold=True)
                    v_surf = fallback_font.render(val, True, (245, 245, 245))
                    if v_surf.get_width() <= card_w - 24:
                        break
            s_surf = self.tiny_font.render(sub, True, (130, 130, 130))

            screen.blit(h_surf, (cx + 12, card_y + 8))
            screen.blit(v_surf, (cx + 12, card_y + 26))
            screen.blit(s_surf, (cx + 12, card_y + 52))

        has_multifloor = False
        if program.simulation and program.simulation.grid_world:
            has_multifloor = len(program.simulation.grid_world.floors) > 1

        strip_h = int(36 * ui_scale) if has_multifloor else 0
        charts_y = card_y + card_h + (strip_h + 10 if has_multifloor else 12)

        if has_multifloor and program.simulation:
            mf_y = card_y + card_h + 10
            strip_rect = pygame.Rect(pad_x, mf_y, content_w, int(36 * ui_scale))
            pygame.draw.rect(screen, Color.CARD_BG.value, strip_rect, border_radius=6)
            pygame.draw.rect(screen, Color.CARD_BORDER.value, strip_rect, 1, border_radius=6)

            title_part = self.small_font.render("Floor Emptying Times:", True, (56, 189, 248))
            screen.blit(title_part, (strip_rect.x + 14, strip_rect.y + 10))
            sub_part = self.tiny_font.render("(time when last person left)", True, (130, 130, 130))
            screen.blit(sub_part, (strip_rect.x + 14 + title_part.get_width() + 6, strip_rect.y + 12))

            fc_times = program.simulation.metrics.floor_clearance_times
            curr_tx = strip_rect.x + title_part.get_width() + sub_part.get_width() + 32
            floor_items = []
            for f_id in sorted(program.simulation.grid_world.floors.keys()):
                t_val = fc_times.get(f_id)
                init_evac = program.simulation.metrics.floor_initial_counts.get(f_id, 0)
                if init_evac == 0:
                    t_str = "- (No evacuees)"
                elif t_val is not None:
                    t_str = f"{int(t_val // 60):02d}:{int(t_val % 60):02d}"
                else:
                    t_str = "--:--"
                floor_items.append(f"Floor {f_id + 1}: {t_str}")

            joined_txt = "  •  ".join(floor_items)
            item_txt = self.small_font.render(joined_txt, True, (220, 225, 235))
            screen.blit(item_txt, (curr_tx, strip_rect.y + 10))

        bottom_bar_h = int(76 * ui_scale)
        bottom_y = screen_h - bottom_bar_h - 14
        chart_h = max(240, bottom_y - charts_y - 12)

        if self._chart_surface is None or self._chart_surface_size != (content_w, chart_h):
            if program.simulation and hasattr(program.simulation, "metrics"):
                self._chart_surface = program.simulation.metrics.generate_charts_surface(content_w, chart_h)
                self._chart_surface_size = (content_w, chart_h)

        if self._chart_surface is not None:
            screen.blit(self._chart_surface, (pad_x, charts_y))

        bottom_rect = pygame.Rect(pad_x, bottom_y, content_w, bottom_bar_h)
        pygame.draw.rect(screen, Color.CARD_BG.value, bottom_rect, border_radius=7)
        pygame.draw.rect(screen, Color.CARD_BORDER.value, bottom_rect, 1, border_radius=7)

        header_y = bottom_rect.y + int(8 * ui_scale)
        exp_lbl = self.tiny_font.render("EXPORT", True, (140, 155, 180))
        screen.blit(exp_lbl, (bottom_rect.x + 16, header_y))

        nav_lbl = self.tiny_font.render("NAVIGATION", True, (140, 155, 180))
        nav_lbl_x = pad_x + content_w - int(365 * ui_scale)
        screen.blit(nav_lbl, (nav_lbl_x, header_y))

        if self._feedback_msg:
            fb_txt = self.small_font.render(self._feedback_msg, True, (16, 185, 129))
            fb_x = bottom_rect.x + int(340 * ui_scale)
            fb_y = bottom_rect.y + int(34 * ui_scale)
            screen.blit(fb_txt, (fb_x, fb_y))

    def handle_ui_event(self, program, event):
        if event.ui_element == self._btn_export_png:
            if program.simulation and hasattr(program.simulation, "metrics"):
                path = _save_file_dialog(
                    "Export Evacuation Report (PNG)",
                    ".png",
                    [("PNG image", "*.png")],
                    initialfile="ariadne_report.png",
                )
                if path:
                    msg = program.simulation.metrics.export_png(program.screen, filepath=path)
                    self._feedback_msg = f"Report saved: {os.path.basename(msg)}"
                    self._feedback_timer = 5.0
        elif event.ui_element == self._btn_export_csv:
            if program.simulation and hasattr(program.simulation, "metrics"):
                path = _save_file_dialog(
                    "Export Evacuation Data (CSV)",
                    ".csv",
                    [("CSV file", "*.csv")],
                    initialfile="ariadne_data.csv",
                )
                if path:
                    msg = program.simulation.metrics.export_csv(filepath=path)
                    self._feedback_msg = f"Data saved: {os.path.basename(msg)}"
                    self._feedback_timer = 5.0
        elif event.ui_element == self._btn_back_sim:
            program.transition_to(SimulationPausedState())
        elif event.ui_element == self._btn_back_edit:
            program.transition_to(EditorState())
        elif event.ui_element == self._menu_btn:
            cur_size = program.screen.get_size()
            if program.ui_manager:
                program.ui_manager.set_window_resolution(cur_size)
            program.transition_to(MainMenuState())

    def handle_key(self, program, key):
        if key in (pygame.K_ESCAPE, pygame.K_p):
            program.transition_to(SimulationPausedState())
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            program.transition_to(EditorState())
