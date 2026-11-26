from dataclasses import dataclass, field
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pygame

from map.dxf_importer import (
    UNIT_TO_METERS,
    DxfLayerInfo,
    apply_dxf_layer_preset,
    inspect_dxf,
    render_dxf,
)
from map.image_extractor import (
    ImageFilterSettings,
    preprocess_floorplan_array,
    sample_wall_color_cielab,
)
from map.pdf_importer import (
    inspect_pdf,
    render_scanned_pdf,
    render_vector_pdf,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_FLOORPLAN_FILETYPES = [
    ("All Supported Floorplans", "*.png *.jpg *.jpeg *.pdf *.dxf"),
    ("Images", "*.png *.jpg *.jpeg"),
    ("PDF Documents", "*.pdf"),
    ("AutoCAD DXF", "*.dxf"),
]

PDF_POINTS_PER_INCH = 72.0
METERS_PER_INCH = 0.0254


def normalize_plan_source_path(source_path: str | Path | None) -> str | None:
    if not source_path:
        return None
    raw = str(source_path).strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        return raw.replace("\\", "/")
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return raw.replace("\\", "/")


def resolve_plan_source_path(
    source_path: str | Path | None,
    project_json_dir: str | Path | None = None,
) -> Path | None:
    if not source_path:
        return None
    raw = str(source_path).strip()
    if not raw:
        return None
    norm = raw.replace("\\", "/")

    p_raw = Path(raw)
    if p_raw.is_file():
        return p_raw
    p_norm = Path(norm)
    if p_norm.is_file():
        return p_norm

    if not p_norm.is_absolute():
        for cand_str in (norm, norm.lstrip("/")):
            p_proj = PROJECT_ROOT / cand_str
            if p_proj.is_file():
                return p_proj
        if project_json_dir:
            p_json = Path(project_json_dir) / norm
            if p_json.is_file():
                return p_json

    fname = p_norm.name
    if fname:
        p_fp = PROJECT_ROOT / "assets" / "floorplans" / fname
        if p_fp.is_file():
            return p_fp
    return None


@dataclass
class ImportSettings:
    sensitivity: float = 0.40
    image_filters: ImageFilterSettings = field(default_factory=ImageFilterSettings)
    dxf_selected_layers: set[str] | None = None
    dxf_units: str | None = None
    pdf_min_thickness_pt: float = 0.55
    pdf_include_fixed: bool = False
    pdf_filled_walls_only: bool = False
    pdf_scale_denom: int | None = None


@dataclass
class LoadedFloorplan:
    source_path: str
    source_type: str
    page_index: int = 0
    page_count: int = 1
    rotation_deg: int = 0
    crop_rect: tuple[float, float, float, float] | None = None
    full_bounds: tuple[float, float, float, float] | None = None
    extraction_image: np.ndarray | None = None
    reference_image: np.ndarray | None = None
    base_extraction_image: np.ndarray | None = None
    base_reference_image: np.ndarray | None = None
    pixels_per_source_unit: float = 1.0
    dxf_inspected: dict | None = None
    dxf_layers: list[DxfLayerInfo] = field(default_factory=list)
    dxf_units: str = "m"
    pdf_scale_denom: int | None = None
    px_per_m: float | None = None
    has_native_scale: bool = False

    @property
    def width(self) -> int:
        if self.extraction_image is None:
            return 0
        return int(self.extraction_image.shape[1])

    @property
    def height(self) -> int:
        if self.extraction_image is None:
            return 0
        return int(self.extraction_image.shape[0])

    @property
    def source_label(self) -> str:
        return {
            "image": "Image",
            "dxf": "DXF CAD",
            "vector_pdf": "Vector PDF",
            "scanned_pdf": "Scanned PDF",
        }.get(self.source_type, "Floorplan")

    def to_reference_metadata(self) -> dict:
        return {
            "source_path": normalize_plan_source_path(self.source_path),
            "source_type": self.source_type,
            "page_index": int(self.page_index),
            "rotation_deg": int(self.rotation_deg) % 360,
            "crop_rect": list(self.crop_rect) if self.crop_rect is not None else None,
        }


def compute_pdf_px_per_m(pixels_per_pt: float, scale_denom: int) -> float:
    meters_per_pixel = (METERS_PER_INCH / PDF_POINTS_PER_INCH) * float(scale_denom) / max(1e-6, float(pixels_per_pt))
    return 1.0 / meters_per_pixel


def compute_dxf_px_per_m(pixels_per_unit: float, units: str) -> float:
    meters_per_unit = UNIT_TO_METERS.get(units, 1.0)
    return float(pixels_per_unit) / max(1e-9, meters_per_unit)


def compute_smart_block_size(
    img_w: int,
    img_h: int,
    px_per_m: float | None = None,
    target_cell_meters: float = 0.50,
    max_cells_long_side: int = 190,
) -> int:
    long_side = max(1, img_w, img_h)
    min_safe = max(8, int(math.ceil(long_side / float(max_cells_long_side))))
    if px_per_m is not None and px_per_m > 0:
        ideal = int(round(float(target_cell_meters) * px_per_m))
        return max(min_safe, min(140, ideal))
    return max(min_safe, 25)


def apply_rotation(img: np.ndarray | None, rotation_deg: int) -> np.ndarray | None:
    if img is None:
        return None
    rot = int(rotation_deg) % 360
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img.copy()


def _unrotate_norm_rect(u0: float, v0: float, u1: float, v1: float, rotation_deg: int):
    rot = int(rotation_deg) % 360
    if rot == 90:
        fu0, fu1 = v0, v1
        fv0, fv1 = 1.0 - u1, 1.0 - u0
    elif rot == 180:
        fu0, fu1 = 1.0 - u1, 1.0 - u0
        fv0, fv1 = 1.0 - v1, 1.0 - v0
    elif rot == 270:
        fu0, fu1 = 1.0 - v1, 1.0 - v0
        fv0, fv1 = u0, u1
    else:
        fu0, fu1, fv0, fv1 = u0, u1, v0, v1
    return (
        max(0.0, min(1.0, fu0)),
        max(0.0, min(1.0, fv0)),
        max(0.0, min(1.0, fu1)),
        max(0.0, min(1.0, fv1)),
    )


def _sync_layer_sizes(ext_img: np.ndarray, ref_img: np.ndarray):
    if ext_img.shape[:2] != ref_img.shape[:2]:
        h, w = ext_img.shape[:2]
        ref_img = cv2.resize(ref_img, (w, h), interpolation=cv2.INTER_AREA)
    return ext_img, ref_img


def numpy_bgr_to_pygame_surface(bgr_img: np.ndarray) -> pygame.Surface:
    if bgr_img.ndim == 2:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    try:
        return surf.convert()
    except pygame.error:
        return surf


def build_aligned_reference_surface(
    ref_bgr: np.ndarray | None,
    grid_width: int,
    grid_height: int,
    block_size: int | None = None,
) -> pygame.Surface | None:
    if ref_bgr is None or ref_bgr.size == 0:
        return None
    rh, rw = ref_bgr.shape[:2]
    bs = int(block_size) if block_size and block_size > 0 else max(1, math.ceil(rw / max(1, grid_width)))
    target_w = max(1, int(grid_width) * bs)
    target_h = max(1, int(grid_height) * bs)
    if rw == target_w and rh == target_h:
        aligned = ref_bgr
    else:
        aligned = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        copy_w = min(rw, target_w)
        copy_h = min(rh, target_h)
        if ref_bgr.ndim == 2:
            src = cv2.cvtColor(ref_bgr[:copy_h, :copy_w], cv2.COLOR_GRAY2BGR)
        else:
            src = ref_bgr[:copy_h, :copy_w]
        aligned[:copy_h, :copy_w] = src
    return numpy_bgr_to_pygame_surface(aligned)


def build_preprocessed_for_floorplan(loaded: LoadedFloorplan, settings: ImportSettings) -> dict:
    if loaded.source_type in ("dxf", "vector_pdf"):
        return preprocess_floorplan_array(loaded.extraction_image, is_vector_raster=True)

    img_settings = ImageFilterSettings(
        mode=settings.image_filters.mode,
        ignore_colors=settings.image_filters.ignore_colors,
        gray_blocks=settings.image_filters.gray_blocks,
        remove_thin_px=settings.image_filters.remove_thin_px,
        picked_rgb=settings.image_filters.picked_rgb,
        picked_lab=settings.image_filters.picked_lab,
        sensitivity=settings.sensitivity,
    )
    return preprocess_floorplan_array(
        loaded.extraction_image,
        filter_settings=img_settings,
        sensitivity=settings.sensitivity,
        is_vector_raster=False,
    )


def load_floorplan(
    path: str,
    page_index: int = 0,
    crop_rect: tuple[float, float, float, float] | None = None,
    rotation_deg: int = 0,
    settings: ImportSettings | None = None,
) -> LoadedFloorplan:
    if settings is None:
        settings = ImportSettings()
    ext = Path(path).suffix.lower()

    if ext == ".dxf":
        inspected = inspect_dxf(path)
        layers = inspected["layers"]
        if settings.dxf_selected_layers is not None:
            for l in layers:
                l.selected = l.name in settings.dxf_selected_layers
        selected = {l.name for l in layers if l.selected}
        units = settings.dxf_units or inspected["units"]
        ext_img, ref_img, used_bounds, px_per_unit = render_dxf(
            inspected, selected, clip_rect=crop_rect
        )
        ext_img, ref_img = _sync_layer_sizes(ext_img, ref_img)
        ext_rot = apply_rotation(ext_img, rotation_deg)
        ref_rot = apply_rotation(ref_img, rotation_deg)
        px_per_m = compute_dxf_px_per_m(px_per_unit, units)
        return LoadedFloorplan(
            source_path=path,
            source_type="dxf",
            page_index=0,
            page_count=1,
            rotation_deg=int(rotation_deg) % 360,
            crop_rect=crop_rect,
            full_bounds=inspected["full_bounds"],
            extraction_image=ext_rot,
            reference_image=ref_rot,
            pixels_per_source_unit=px_per_unit,
            dxf_inspected=inspected,
            dxf_layers=layers,
            dxf_units=units,
            px_per_m=px_per_m,
            has_native_scale=True,
        )

    if ext == ".pdf":
        info = inspect_pdf(path, page_index=page_index)
        p_idx = info["page_index"]
        p_cnt = info["page_count"]
        full_bounds = info["full_bounds"]
        if info["is_vector"]:
            ext_img, ref_img, used_bounds, px_per_pt = render_vector_pdf(
                path,
                page_index=p_idx,
                clip_rect=crop_rect,
                min_wall_thickness_pt=settings.pdf_min_thickness_pt,
                include_fixed_blocks=settings.pdf_include_fixed,
                filled_walls_only=settings.pdf_filled_walls_only,
            )
            ext_img, ref_img = _sync_layer_sizes(ext_img, ref_img)
            ext_rot = apply_rotation(ext_img, rotation_deg)
            ref_rot = apply_rotation(ref_img, rotation_deg)
            px_per_m = (
                compute_pdf_px_per_m(px_per_pt, settings.pdf_scale_denom)
                if settings.pdf_scale_denom
                else None
            )
            return LoadedFloorplan(
                source_path=path,
                source_type="vector_pdf",
                page_index=p_idx,
                page_count=p_cnt,
                rotation_deg=int(rotation_deg) % 360,
                crop_rect=crop_rect,
                full_bounds=full_bounds,
                extraction_image=ext_rot,
                reference_image=ref_rot,
                pixels_per_source_unit=px_per_pt,
                pdf_scale_denom=settings.pdf_scale_denom,
                px_per_m=px_per_m,
                has_native_scale=bool(settings.pdf_scale_denom),
            )
        else:
            bgr, _, zoom = render_scanned_pdf(path, page_index=p_idx)
            h, w = bgr.shape[:2]
            pix_bounds = (0.0, 0.0, float(w), float(h))
            sliced = _slice_pixel_rect(bgr, crop_rect)
            sliced_rot = apply_rotation(sliced, rotation_deg)
            return LoadedFloorplan(
                source_path=path,
                source_type="scanned_pdf",
                page_index=p_idx,
                page_count=p_cnt,
                rotation_deg=int(rotation_deg) % 360,
                crop_rect=crop_rect,
                full_bounds=pix_bounds,
                extraction_image=sliced_rot,
                reference_image=sliced_rot.copy(),
                base_extraction_image=bgr,
                base_reference_image=bgr.copy(),
                pixels_per_source_unit=zoom,
            )

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w = bgr.shape[:2]
    pix_bounds = (0.0, 0.0, float(w), float(h))
    sliced = _slice_pixel_rect(bgr, crop_rect)
    sliced_rot = apply_rotation(sliced, rotation_deg)
    return LoadedFloorplan(
        source_path=path,
        source_type="image",
        page_index=0,
        page_count=1,
        rotation_deg=int(rotation_deg) % 360,
        crop_rect=crop_rect,
        full_bounds=pix_bounds,
        extraction_image=sliced_rot,
        reference_image=sliced_rot.copy(),
        base_extraction_image=bgr,
        base_reference_image=bgr.copy(),
        pixels_per_source_unit=1.0,
    )


def _slice_pixel_rect(img: np.ndarray, crop_rect: tuple[float, float, float, float] | None) -> np.ndarray:
    if crop_rect is None:
        return img.copy()
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, int(round(crop_rect[0]))))
    y0 = max(0, min(h - 1, int(round(crop_rect[1]))))
    x1 = max(x0 + 1, min(w, int(round(crop_rect[2]))))
    y1 = max(y0 + 1, min(h, int(round(crop_rect[3]))))
    return img[y0:y1, x0:x1].copy()


def reprocess_floorplan(loaded: LoadedFloorplan, settings: ImportSettings) -> LoadedFloorplan:
    if loaded.source_type == "dxf":
        if loaded.dxf_inspected is None:
            loaded.dxf_inspected = inspect_dxf(loaded.source_path)
            loaded.dxf_layers = loaded.dxf_inspected["layers"]
        if settings.dxf_selected_layers is not None:
            for l in loaded.dxf_layers:
                l.selected = l.name in settings.dxf_selected_layers
        selected = {l.name for l in loaded.dxf_layers if l.selected}
        if settings.dxf_units:
            loaded.dxf_units = settings.dxf_units
        ext_img, ref_img, _, px_per_unit = render_dxf(
            loaded.dxf_inspected, selected, clip_rect=loaded.crop_rect
        )
        ext_img, ref_img = _sync_layer_sizes(ext_img, ref_img)
        loaded.extraction_image = apply_rotation(ext_img, loaded.rotation_deg)
        loaded.reference_image = apply_rotation(ref_img, loaded.rotation_deg)
        loaded.pixels_per_source_unit = px_per_unit
        if loaded.has_native_scale or loaded.px_per_m is None:
            loaded.px_per_m = compute_dxf_px_per_m(px_per_unit, loaded.dxf_units)
            loaded.has_native_scale = True
        return loaded

    if loaded.source_type in ("vector_pdf",):
        ext_img, ref_img, _, px_per_pt = render_vector_pdf(
            loaded.source_path,
            page_index=loaded.page_index,
            clip_rect=loaded.crop_rect,
            min_wall_thickness_pt=settings.pdf_min_thickness_pt,
            include_fixed_blocks=settings.pdf_include_fixed,
            filled_walls_only=settings.pdf_filled_walls_only,
        )
        ext_img, ref_img = _sync_layer_sizes(ext_img, ref_img)
        loaded.extraction_image = apply_rotation(ext_img, loaded.rotation_deg)
        loaded.reference_image = apply_rotation(ref_img, loaded.rotation_deg)
        loaded.pixels_per_source_unit = px_per_pt
        loaded.pdf_scale_denom = settings.pdf_scale_denom
        if settings.pdf_scale_denom:
            loaded.px_per_m = compute_pdf_px_per_m(px_per_pt, settings.pdf_scale_denom)
            loaded.has_native_scale = True
        return loaded

    base_ext = loaded.base_extraction_image
    base_ref = loaded.base_reference_image
    if base_ext is not None and base_ref is not None:
        s_ext = _slice_pixel_rect(base_ext, loaded.crop_rect)
        s_ref = _slice_pixel_rect(base_ref, loaded.crop_rect)
        s_ext, s_ref = _sync_layer_sizes(s_ext, s_ref)
        loaded.extraction_image = apply_rotation(s_ext, loaded.rotation_deg)
        loaded.reference_image = apply_rotation(s_ref, loaded.rotation_deg)
    return loaded


def rotate_loaded_floorplan(loaded: LoadedFloorplan, delta_deg: int) -> LoadedFloorplan:
    step = int(delta_deg) % 360
    loaded.rotation_deg = (loaded.rotation_deg + step) % 360
    loaded.extraction_image = apply_rotation(loaded.extraction_image, step)
    loaded.reference_image = apply_rotation(loaded.reference_image, step)
    return loaded


def _reprocess_preserving_scale(loaded: LoadedFloorplan, settings: ImportSettings) -> LoadedFloorplan:
    old_px_per_unit = loaded.pixels_per_source_unit
    reprocess_floorplan(loaded, settings)
    if (
        loaded.source_type in ("dxf", "vector_pdf")
        and not loaded.has_native_scale
        and loaded.px_per_m is not None
        and old_px_per_unit > 0
    ):
        loaded.px_per_m *= loaded.pixels_per_source_unit / old_px_per_unit
    return loaded


def crop_loaded_floorplan(
    loaded: LoadedFloorplan,
    display_rect_px: tuple[float, float, float, float],
    settings: ImportSettings,
) -> LoadedFloorplan:
    cur_w = max(1, loaded.width)
    cur_h = max(1, loaded.height)

    rx0, ry0, rx1, ry1 = display_rect_px
    x0 = max(0.0, min(float(cur_w), min(rx0, rx1)))
    x1 = max(0.0, min(float(cur_w), max(rx0, rx1)))
    y0 = max(0.0, min(float(cur_h), min(ry0, ry1)))
    y1 = max(0.0, min(float(cur_h), max(ry0, ry1)))

    if (x1 - x0) < 10 or (y1 - y0) < 10:
        return loaded

    u0, v0, u1, v1 = x0 / cur_w, y0 / cur_h, x1 / cur_w, y1 / cur_h
    fu0, fv0, fu1, fv1 = _unrotate_norm_rect(u0, v0, u1, v1, loaded.rotation_deg)

    active_bounds = loaded.crop_rect if loaded.crop_rect is not None else loaded.full_bounds
    bx0, by0, bx1, by1 = active_bounds
    bw = bx1 - bx0
    bh = by1 - by0

    if loaded.source_type == "dxf":
        new_crop = (
            bx0 + fu0 * bw,
            by1 - fv1 * bh,
            bx0 + fu1 * bw,
            by1 - fv0 * bh,
        )
    else:
        new_crop = (
            bx0 + fu0 * bw,
            by0 + fv0 * bh,
            bx0 + fu1 * bw,
            by0 + fv1 * bh,
        )

    loaded.crop_rect = new_crop
    return _reprocess_preserving_scale(loaded, settings)


def reset_crop_floorplan(loaded: LoadedFloorplan, settings: ImportSettings) -> LoadedFloorplan:
    loaded.crop_rect = None
    return _reprocess_preserving_scale(loaded, settings)


def restore_reference_image(
    metadata: dict | None,
    project_json_dir: str | Path | None = None,
) -> np.ndarray | None:
    if not metadata or not isinstance(metadata, dict):
        return None
    src = metadata.get("source_path")
    resolved = resolve_plan_source_path(src, project_json_dir=project_json_dir)
    if resolved is None:
        return None
    try:
        page_index = int(metadata.get("page_index", 0) or 0)
        rotation_deg = int(metadata.get("rotation_deg", 0) or 0)
        raw_crop = metadata.get("crop_rect")
        crop_rect = tuple(float(v) for v in raw_crop) if raw_crop and len(raw_crop) == 4 else None
        loaded = load_floorplan(
            str(resolved),
            page_index=page_index,
            crop_rect=crop_rect,
            rotation_deg=rotation_deg,
        )
        return loaded.reference_image
    except Exception as e:
        logger.warning("Could not restore reference overlay from %s: %s", src, e)
        return None


def sample_wall_at_loaded_coords(loaded: LoadedFloorplan, px: int, py: int):
    return sample_wall_color_cielab(loaded.extraction_image, px, py)


__all__ = [
    "ImportSettings",
    "LoadedFloorplan",
    "SUPPORTED_FLOORPLAN_FILETYPES",
    "apply_dxf_layer_preset",
    "build_aligned_reference_surface",
    "build_preprocessed_for_floorplan",
    "compute_dxf_px_per_m",
    "compute_pdf_px_per_m",
    "compute_smart_block_size",
    "crop_loaded_floorplan",
    "load_floorplan",
    "normalize_plan_source_path",
    "numpy_bgr_to_pygame_surface",
    "reprocess_floorplan",
    "reset_crop_floorplan",
    "resolve_plan_source_path",
    "restore_reference_image",
    "rotate_loaded_floorplan",
    "sample_wall_at_loaded_coords",
]
