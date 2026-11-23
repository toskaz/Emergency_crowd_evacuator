import os
from pathlib import Path
import pygame

_FONT_CACHE = {}
_SANS_PATH = None
_SANS_BOLD_PATH = None
_MONO_PATH = None
_FONTS_INITIALIZED = False


def _init_font_paths():
    global _SANS_PATH, _SANS_BOLD_PATH, _MONO_PATH, _FONTS_INITIALIZED
    if _FONTS_INITIALIZED:
        return
    _FONTS_INITIALIZED = True

    project_root = Path(__file__).resolve().parent.parent
    assets_fonts = project_root / "assets" / "fonts"

    segoe_reg = assets_fonts / "SegoeUI-Regular.ttf"
    segoe_bold = assets_fonts / "SegoeUI-Bold.ttf"
    consolas_reg = assets_fonts / "Consolas-Regular.ttf"

    if segoe_reg.exists():
        _SANS_PATH = str(segoe_reg)
    else:
        for name in ("segoeui", "dejavusans", "arial"):
            matched = pygame.font.match_font(name)
            if matched and os.path.exists(matched):
                _SANS_PATH = matched
                break

    if segoe_bold.exists():
        _SANS_BOLD_PATH = str(segoe_bold)
    else:
        for name in ("segoeuib", "dejavusansbold", "arialbold"):
            matched = pygame.font.match_font(name)
            if matched and os.path.exists(matched):
                _SANS_BOLD_PATH = matched
                break
        if not _SANS_BOLD_PATH:
            _SANS_BOLD_PATH = _SANS_PATH

    if consolas_reg.exists():
        _MONO_PATH = str(consolas_reg)
    else:
        for name in ("consolas", "dejavusansmono", "couriernew"):
            matched = pygame.font.match_font(name)
            if matched and os.path.exists(matched):
                _MONO_PATH = matched
                break


def get_font(size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    _init_font_paths()

    key = (size, bold, mono)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    if mono:
        path = _MONO_PATH
    elif bold and _SANS_BOLD_PATH:
        path = _SANS_BOLD_PATH
    else:
        path = _SANS_PATH

    try:
        if path and os.path.exists(path):
            font = pygame.font.Font(path, size)
            if bold and path == _SANS_PATH:
                font.set_bold(True)
        else:
            font = pygame.font.Font(None, size)
            if bold:
                font.set_bold(True)
    except Exception:
        font = pygame.font.Font(None, size)
        if bold:
            font.set_bold(True)

    _FONT_CACHE[key] = font
    return font


def get_sans_path() -> str:
    _init_font_paths()
    return _SANS_PATH


def get_sans_bold_path() -> str:
    _init_font_paths()
    return _SANS_BOLD_PATH or _SANS_PATH


def get_mono_path() -> str:
    _init_font_paths()
    return _MONO_PATH


def init_ui_manager_fonts(ui_manager):
    if ui_manager is None:
        return
    _init_font_paths()
    sans_reg = get_sans_path()
    sans_bold = get_sans_bold_path()
    if sans_reg and os.path.exists(sans_reg):
        try:
            ui_manager.add_font_paths("segoe_ui", sans_reg, sans_bold)
        except Exception:
            pass
