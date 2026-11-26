import pygame
import pygame_gui

from map.dxf_importer import apply_dxf_layer_preset


class AddFloorChoiceDialog:
    def __init__(self, ui_manager, screen_size, has_source_file=False):
        win_w, win_h = 340, 210
        sw, sh = screen_size
        win_x = max(20, (sw - win_w) // 2)
        win_y = max(20, (sh - win_h) // 2)

        self.window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=ui_manager,
            window_display_title="Add Floor",
            object_id="#scale_window",
        )

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, 10, win_w - 32, 24),
            text="Choose floorplan source for the new floor:",
            manager=ui_manager,
            container=self.window,
        )

        self.btn_new_file = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 42, win_w - 32, 34),
            text="New File...",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_primary",
        )

        self.btn_same_file = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 84, win_w - 32, 34),
            text="Same File (New Crop / Page)",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_secondary",
        )
        if not has_source_file:
            self.btn_same_file.disable()

        self.btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 128, win_w - 32, 30),
            text="Cancel",
            manager=ui_manager,
            container=self.window,
            object_id="#menu_button",
        )

    def handle_event(self, event):
        if event.type == pygame_gui.UI_WINDOW_CLOSE and event.ui_element == self.window:
            return "cancel"
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_new_file:
                return "new_file"
            if event.ui_element == self.btn_same_file:
                return "same_file"
            if event.ui_element == self.btn_cancel:
                return "cancel"
        return None

    def kill(self):
        if self.window is not None:
            self.window.kill()
            self.window = None


class DxfLayersDialog:
    def __init__(self, ui_manager, screen_size, dxf_layers):
        sw, sh = screen_size
        win_w = min(480, max(360, sw - 80))
        win_h = min(520, max(320, sh - 80))
        win_x = max(20, (sw - win_w) // 2)
        win_y = max(20, (sh - win_h) // 2)

        self._layers = dxf_layers
        self._selected = {l.name: bool(l.selected) for l in dxf_layers}
        self._row_buttons = {}

        self.window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=ui_manager,
            window_display_title="DXF Layers",
            object_id="#scale_window",
        )

        inner_w = win_w - 32
        gap = 6
        pw = (inner_w - 3 * gap) // 4
        self.btn_walls_only = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(12, 10, pw + 10, 28),
            text="Walls Only",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_secondary",
        )
        self.btn_walls_fixed = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(12 + pw + 10 + gap, 10, pw + 14, 28),
            text="Walls + Fixed",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_secondary",
        )
        rem_w = inner_w - (pw + 10 + gap + pw + 14 + gap)
        bw2 = (rem_w - gap) // 2
        x3 = 12 + pw + 10 + gap + pw + 14 + gap
        self.btn_all = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x3, 10, bw2, 28),
            text="All",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_secondary",
        )
        self.btn_none = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x3 + bw2 + gap, 10, bw2, 28),
            text="None",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_secondary",
        )

        scroll_h = win_h - 140
        self.scroll = pygame_gui.elements.UIScrollingContainer(
            relative_rect=pygame.Rect(12, 46, inner_w, scroll_h),
            manager=ui_manager,
            container=self.window,
            allow_scroll_x=False,
            allow_scroll_y=True,
        )

        row_h = 30
        row_gap = 4
        total_h = max(scroll_h - 4, len(self._layers) * (row_h + row_gap) + 8)
        content_w = inner_w - 26
        self.scroll.set_scrollable_area_dimensions((content_w, total_h))

        group_tag = {
            "walls": "wall",
            "fixed": "fixed",
            "ignore": "skip",
            "other": "other",
        }

        y = 4
        for layer in self._layers:
            is_on = self._selected[layer.name]
            chk = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(4, y, 32, 26),
                text="X" if is_on else "",
                manager=ui_manager,
                container=self.scroll,
                object_id="#tool_btn_selected" if is_on else "#tool_btn",
            )
            tag = group_tag.get(layer.group, "")
            label_txt = f"{layer.name}  ({layer.entity_count}, {tag})"
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(44, y, max(120, content_w - 48), 26),
                text=label_txt,
                manager=ui_manager,
                container=self.scroll,
            )
            self._row_buttons[chk] = layer.name
            y += row_h + row_gap

        bot_y = win_h - 84
        btn_w = (inner_w - 10) // 2
        self.btn_apply = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(12, bot_y, btn_w, 32),
            text="Apply",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_primary",
        )
        self.btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(12 + btn_w + 10, bot_y, btn_w, 32),
            text="Cancel",
            manager=ui_manager,
            container=self.window,
            object_id="#menu_button",
        )

    def _sync_checkbox_buttons(self):
        for btn, name in self._row_buttons.items():
            is_on = self._selected.get(name, False)
            btn.set_text("X" if is_on else "")
            btn.change_object_id("#tool_btn_selected" if is_on else "#tool_btn")

    def _apply_preset_local(self, preset):
        backup = [l.selected for l in self._layers]
        apply_dxf_layer_preset(self._layers, preset)
        for l, old in zip(self._layers, backup):
            self._selected[l.name] = l.selected
            l.selected = old
        self._sync_checkbox_buttons()

    def handle_event(self, event):
        if event.type == pygame_gui.UI_WINDOW_CLOSE and event.ui_element == self.window:
            return "cancel"
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            el = event.ui_element
            if el == self.btn_apply:
                for l in self._layers:
                    l.selected = self._selected.get(l.name, False)
                return "apply"
            if el == self.btn_cancel:
                return "cancel"
            if el == self.btn_walls_only:
                self._apply_preset_local("walls_only")
                return None
            if el == self.btn_walls_fixed:
                self._apply_preset_local("walls_fixed")
                return None
            if el == self.btn_all:
                self._apply_preset_local("all")
                return None
            if el == self.btn_none:
                self._apply_preset_local("none")
                return None
            if el in self._row_buttons:
                name = self._row_buttons[el]
                self._selected[name] = not self._selected.get(name, False)
                self._sync_checkbox_buttons()
                return None
        return None

    def kill(self):
        if self.window is not None:
            self.window.kill()
            self.window = None


class DeleteFloorConfirmDialog:
    def __init__(self, ui_manager, screen_size, floor_number):
        sw, sh = screen_size
        win_w, win_h = 380, 175
        win_x = max(20, (sw - win_w) // 2)
        win_y = max(20, (sh - win_h) // 2)

        self.window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=ui_manager,
            window_display_title="Delete Floor",
            object_id="#scale_window",
        )
        pygame_gui.elements.UITextBox(
            html_text=(
                f"Delete Floor {floor_number}? All walls, entities, and stair links "
                "on this floor will be removed."
            ),
            relative_rect=pygame.Rect(14, 10, win_w - 40, 68),
            manager=ui_manager,
            container=self.window,
        )
        btn_w = (win_w - 50) // 2
        self.btn_delete = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(14, 88, btn_w, 34),
            text="Delete",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_destructive",
        )
        self.btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(24 + btn_w, 88, btn_w, 34),
            text="Cancel",
            manager=ui_manager,
            container=self.window,
            object_id="#menu_button",
        )

    def handle_event(self, event):
        if event.type == pygame_gui.UI_WINDOW_CLOSE and event.ui_element == self.window:
            return "cancel"
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_delete:
                return "delete"
            if event.ui_element == self.btn_cancel:
                return "cancel"
        return None

    def kill(self):
        if self.window is not None:
            self.window.kill()
            self.window = None


class NoticeDialog:
    def __init__(self, ui_manager, screen_size, title, message):
        sw, sh = screen_size
        win_w, win_h = 400, 170
        win_x = max(20, (sw - win_w) // 2)
        win_y = max(20, (sh - win_h) // 2)

        self.window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect(win_x, win_y, win_w, win_h),
            manager=ui_manager,
            window_display_title=title,
            object_id="#scale_window",
        )
        pygame_gui.elements.UITextBox(
            html_text=message,
            relative_rect=pygame.Rect(14, 10, win_w - 40, 72),
            manager=ui_manager,
            container=self.window,
        )
        self.btn_ok = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((win_w - 130) // 2, 92, 110, 32),
            text="OK",
            manager=ui_manager,
            container=self.window,
            object_id="#btn_primary",
        )

    def handle_event(self, event):
        if event.type == pygame_gui.UI_WINDOW_CLOSE and event.ui_element == self.window:
            return "close"
        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == self.btn_ok:
            return "close"
        return None

    def kill(self):
        if self.window is not None:
            self.window.kill()
            self.window = None
