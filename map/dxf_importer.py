from dataclasses import dataclass
import math

import cv2
import numpy as np

try:
    import ezdxf
    from ezdxf import path as ezpath
except ImportError:
    ezdxf = None
    ezpath = None


IGNORED_ENTITY_TYPES = frozenset(
    {"TEXT", "MTEXT", "DIMENSION", "LEADER", "MLEADER", "VIEWPORT"}
)

PRIORITY_EXCLUDE_PATTERNS = (
    "kolor",
    "color",
    "tabelk",
    "ramk",
    "opis",
    "podpis",
)

IGNORE_PATTERNS = (
    "door",
    "drzwi",
    "dim",
    "wym",
    "wymiar",
    "text",
    "opis",
    "anno",
    "podpis",
    "elec",
    "elektr",
    "sanit",
    "plumb",
    "hvac",
    "inst",
    "axis",
    "osie",
    "grid",
    "siatk",
    "ttlb",
    "ramk",
    "tab",
    "tabelk",
    "nplt",
    "defpoints",
    "vport",
    "evac",
    "ewak",
    "fire",
    "ppoz",
    "poż",
    "poz",
    "arrow",
    "strzal",
    "strzał",
    "pomocnicz",
    "granic",
    "areas",
    "wyzej",
    "wyżej",
    "widok",
    "zieleń",
    "zielen",
    "trawa",
    "teren",
    "staffage",
    "mapa",
    "linie zabudowy",
    "hatch kolor",
)

WALL_PATTERNS = (
    "wall",
    "scian",
    "ścian",
    "mur",
    "przekr",
    "struc",
    "col",
    "slup",
    "słup",
    "obrys",
    "shaft",
    "szacht",
    "elev",
    "lift",
    "winda",
    "void",
    "komin",
    "core",
    "istn",
    "wind",
    "okn",
    "glaz",
)

FIXED_PATTERNS = (
    "case",
    "fixd",
    "fixed",
    "mill",
    "cab",
    "count",
    "desk",
    "shelf",
    "regal",
    "regał",
    "eqpm",
    "furn",
    "mebl",
    "umeblow",
    "seat",
    "bench",
    "stair",
    "schod",
)

REFERENCE_SKIP_PATTERNS = (
    "vport",
    "defpoints",
    "hatch kolor",
    "zieleń",
    "zielen",
    "trawa",
    "tabelk",
    "ramk",
    "nplt",
)

REFERENCE_DETAIL_PATTERNS = (
    "widok",
    "drzwi",
    "door",
    "schod",
    "stair",
    "okn",
    "wind",
)

REFERENCE_FURNITURE_PATTERNS = (
    "mebl",
    "umeblow",
    "furn",
    "eqpm",
    "regal",
    "regał",
)

INSUNITS_MAP = {
    1: ("in", 0.0254),
    2: ("ft", 0.3048),
    4: ("mm", 0.001),
    5: ("cm", 0.01),
    6: ("m", 1.0),
}

UNIT_TO_METERS = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "in": 0.0254,
    "ft": 0.3048,
}


@dataclass
class DxfLayerInfo:
    name: str
    group: str
    selected: bool
    entity_count: int = 0


def ensure_ezdxf_available():
    if ezdxf is None:
        raise ImportError(
            "DXF support requires 'ezdxf'. Install it with: pip install ezdxf"
        )


def categorize_dxf_layer(layer_name: str) -> str:
    low = (layer_name or "").lower()
    if any(p in low for p in PRIORITY_EXCLUDE_PATTERNS):
        return "ignore"
    if any(p in low for p in IGNORE_PATTERNS):
        return "ignore"
    if any(p in low for p in WALL_PATTERNS):
        return "walls"
    if any(p in low for p in FIXED_PATTERNS):
        return "fixed"
    return "other"


def _is_reference_skipped(layer_name: str) -> bool:
    low = (layer_name or "").lower()
    return any(p in low for p in REFERENCE_SKIP_PATTERNS)


def _is_reference_detail(layer_name: str) -> bool:
    low = (layer_name or "").lower()
    return any(p in low for p in REFERENCE_DETAIL_PATTERNS)


def _is_reference_furniture(layer_name: str) -> bool:
    low = (layer_name or "").lower()
    return any(p in low for p in REFERENCE_FURNITURE_PATTERNS)


def _effective_layer(entity, parent_layer: str | None) -> str:
    layer = getattr(entity.dxf, "layer", "0") or "0"
    if layer == "0" and parent_layer:
        return parent_layer
    return layer


def _iter_expanded_entities(entities, parent_layer=None, depth=0, max_depth=8):
    for entity in entities:
        etype = entity.dxftype()
        if etype in IGNORED_ENTITY_TYPES:
            continue
        eff_layer = _effective_layer(entity, parent_layer)
        if etype == "INSERT":
            if depth >= max_depth:
                continue
            try:
                virtuals = list(entity.virtual_entities())
            except Exception:
                continue
            yield from _iter_expanded_entities(
                virtuals, parent_layer=eff_layer, depth=depth + 1, max_depth=max_depth
            )
        else:
            yield eff_layer, entity


def _sample_arc_points(cx, cy, r, start_deg, end_deg, steps=24):
    s = math.radians(start_deg)
    e = math.radians(end_deg)
    if e <= s:
        e += 2.0 * math.pi
    pts = []
    for i in range(steps + 1):
        ang = s + (e - s) * (i / steps)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def _extract_entity_primitives(entity):
    etype = entity.dxftype()
    polylines = []
    polygons = []

    try:
        if etype == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            polylines.append([(float(s.x), float(s.y)), (float(e.x), float(e.y))])

        elif etype == "LWPOLYLINE":
            with entity.points("xy") as pts:
                coords = [(float(p[0]), float(p[1])) for p in pts]
            if len(coords) >= 2:
                if entity.closed:
                    coords.append(coords[0])
                polylines.append(coords)

        elif etype == "POLYLINE":
            coords = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in entity.vertices]
            if len(coords) >= 2:
                if getattr(entity, "is_closed", False):
                    coords.append(coords[0])
                polylines.append(coords)

        elif etype == "CIRCLE":
            c = entity.dxf.center
            r = float(entity.dxf.radius)
            if r > 0:
                pts = _sample_arc_points(float(c.x), float(c.y), r, 0.0, 360.0, steps=36)
                polylines.append(pts)

        elif etype == "ARC":
            c = entity.dxf.center
            r = float(entity.dxf.radius)
            if r > 0:
                pts = _sample_arc_points(
                    float(c.x),
                    float(c.y),
                    r,
                    float(entity.dxf.start_angle),
                    float(entity.dxf.end_angle),
                    steps=24,
                )
                polylines.append(pts)

        elif etype in ("ELLIPSE", "SPLINE") and ezpath is not None:
            p = ezpath.make_path(entity)
            coords = [(float(v.x), float(v.y)) for v in p.flattening(0.05)]
            if len(coords) >= 2:
                polylines.append(coords)

        elif etype == "HATCH":
            is_solid = bool(getattr(entity.dxf, "solid_fill", 0))
            pattern = str(getattr(entity.dxf, "pattern_name", "")).upper()
            has_paths = len(getattr(entity, "paths", ())) > 0
            if (is_solid or pattern == "SOLID" or has_paths) and ezpath is not None:
                try:
                    for p in ezpath.from_hatch(entity):
                        subs = list(p.sub_paths()) if hasattr(p, "sub_paths") else [p]
                        for sub in (subs or [p]):
                            coords = [(float(v.x), float(v.y)) for v in sub.flattening(0.05)]
                            if len(coords) >= 3:
                                polygons.append(coords)
                except Exception:
                    pass
    except Exception:
        pass

    return polylines, polygons


def _collect_dxf_geometry(doc):
    msp = doc.modelspace()
    layer_counts = {}
    items = []

    for eff_layer, entity in _iter_expanded_entities(msp):
        polylines, polygons = _extract_entity_primitives(entity)
        if not polylines and not polygons:
            continue
        layer_counts[eff_layer] = layer_counts.get(eff_layer, 0) + 1
        items.append((eff_layer, polylines, polygons))

    layers = []
    for name in sorted(layer_counts.keys(), key=lambda s: s.lower()):
        grp = categorize_dxf_layer(name)
        layers.append(
            DxfLayerInfo(
                name=name,
                group=grp,
                selected=(grp == "walls"),
                entity_count=layer_counts[name],
            )
        )

    if layers and not any(l.selected for l in layers):
        non_ignored = [l for l in layers if l.group != "ignore"]
        targets = non_ignored if non_ignored else layers
        for l in targets:
            l.selected = True

    return layers, items


def _item_bounds(polylines, polygons):
    xs = []
    ys = []
    for poly in polylines:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    for poly in polygons:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, x1, y1, 0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _filter_axis_outliers(entries, axis_idx):
    n = len(entries)
    if n < 8:
        return entries

    centers = np.array([e[4 + axis_idx] for e in entries], dtype=np.float64)
    min_vals = np.array([e[axis_idx] for e in entries], dtype=np.float64)
    max_vals = np.array([e[2 + axis_idx] for e in entries], dtype=np.float64)

    full_min = float(np.min(min_vals))
    full_max = float(np.max(max_vals))
    full_span = full_max - full_min
    if full_span <= 1e-6:
        return entries

    keep_mask = np.ones(n, dtype=bool)

    for _ in range(3):
        idx_active = np.where(keep_mask)[0]
        if len(idx_active) < 8:
            break
        cur_c = centers[idx_active]
        cur_span = float(np.max(max_vals[idx_active]) - np.min(min_vals[idx_active]))
        if cur_span <= 1e-6:
            break
        order = np.argsort(cur_c)
        sorted_c = cur_c[order]
        diffs = np.diff(sorted_c)
        if len(diffs) == 0:
            break
        max_gap_pos = int(np.argmax(diffs))
        max_gap = float(diffs[max_gap_pos])
        if max_gap <= 0.35 * cur_span:
            break
        left_count = max_gap_pos + 1
        right_count = len(idx_active) - left_count
        if left_count >= 0.88 * len(idx_active) and right_count < 0.12 * len(idx_active):
            drop_idx = idx_active[order[max_gap_pos + 1 :]]
            keep_mask[drop_idx] = False
        elif right_count >= 0.88 * len(idx_active) and left_count < 0.12 * len(idx_active):
            drop_idx = idx_active[order[: max_gap_pos + 1]]
            keep_mask[drop_idx] = False
        else:
            break

    idx_active = np.where(keep_mask)[0]
    if len(idx_active) >= 10:
        cur_c = centers[idx_active]
        cur_span = float(np.max(max_vals[idx_active]) - np.min(min_vals[idx_active]))
        p2, p98 = np.percentile(cur_c, [2.0, 98.0])
        core_span = float(p98 - p2)
        if core_span > 1e-6 and cur_span > 1.8 * core_span:
            margin = 0.15 * core_span
            lo = p2 - margin
            hi = p98 + margin
            inside = (cur_c >= lo) & (cur_c <= hi)
            if int(np.sum(inside)) >= 0.85 * len(idx_active):
                keep_mask[idx_active[~inside]] = False

    kept = [e for e, k in zip(entries, keep_mask) if k]
    return kept if kept else entries


def _compute_bounds(items, layer_filter=None, filter_outliers=False):
    entries = []
    for layer_name, polylines, polygons in items:
        if layer_filter is not None and layer_name not in layer_filter:
            continue
        ib = _item_bounds(polylines, polygons)
        if ib is not None:
            entries.append(ib)
    if not entries:
        return None

    if filter_outliers and len(entries) >= 8:
        entries_x = _filter_axis_outliers(entries, 0)
        entries_xy = _filter_axis_outliers(entries_x, 1)
        if entries_xy:
            entries = entries_xy

    xmin = min(e[0] for e in entries)
    ymin = min(e[1] for e in entries)
    xmax = max(e[2] for e in entries)
    ymax = max(e[3] for e in entries)
    return (xmin, ymin, xmax, ymax)


def _detect_dxf_units(doc, full_bounds) -> str:
    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    if insunits in INSUNITS_MAP:
        u = INSUNITS_MAP[insunits][0]
        if u in UNIT_TO_METERS:
            return u
    if full_bounds is not None:
        span = max(full_bounds[2] - full_bounds[0], full_bounds[3] - full_bounds[1])
        if span > 5000:
            return "mm"
        if span > 500:
            return "cm"
    return "m"


def inspect_dxf(path: str):
    ensure_ezdxf_available()
    doc = ezdxf.readfile(path)
    layers, items = _collect_dxf_geometry(doc)

    selected_set = {l.name for l in layers if l.selected}
    non_skipped = {l.name for l in layers if not _is_reference_skipped(l.name)}
    full_bounds = _compute_bounds(items, selected_set or None, filter_outliers=True)
    if full_bounds is None:
        full_bounds = _compute_bounds(items, non_skipped or None, filter_outliers=True)
    if full_bounds is None:
        full_bounds = (0.0, 0.0, 100.0, 100.0)

    dx = max(1e-6, full_bounds[2] - full_bounds[0])
    dy = max(1e-6, full_bounds[3] - full_bounds[1])
    pad_x = dx * 0.03
    pad_y = dy * 0.03
    padded_bounds = (
        full_bounds[0] - pad_x,
        full_bounds[1] - pad_y,
        full_bounds[2] + pad_x,
        full_bounds[3] + pad_y,
    )

    units = _detect_dxf_units(doc, full_bounds)
    return {
        "doc": doc,
        "layers": layers,
        "items": items,
        "full_bounds": padded_bounds,
        "units": units,
    }


def apply_dxf_layer_preset(layers: list[DxfLayerInfo], preset: str):
    if preset == "walls_only":
        for l in layers:
            l.selected = (l.group == "walls")
        if layers and not any(l.selected for l in layers):
            non_ignored = [l for l in layers if l.group != "ignore"]
            for l in (non_ignored or layers):
                l.selected = True
    elif preset == "walls_fixed":
        for l in layers:
            l.selected = l.group in ("walls", "fixed")
        if layers and not any(l.selected for l in layers):
            non_ignored = [l for l in layers if l.group != "ignore"]
            for l in (non_ignored or layers):
                l.selected = True
    elif preset == "all":
        for l in layers:
            l.selected = True
    elif preset == "none":
        for l in layers:
            l.selected = False


def render_dxf(
    inspected: dict,
    selected_layers: set[str],
    clip_rect: tuple[float, float, float, float] | None = None,
    max_size: int = 2000,
):
    items = inspected["items"]
    bounds = clip_rect if clip_rect is not None else inspected["full_bounds"]
    xmin, ymin, xmax, ymax = bounds
    world_w = max(1e-6, xmax - xmin)
    world_h = max(1e-6, ymax - ymin)

    scale = max_size / max(world_w, world_h)
    canvas_w = max(64, int(round(world_w * scale)))
    canvas_h = max(64, int(round(world_h * scale)))

    pixels_per_unit = canvas_w / world_w

    extraction = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    reference = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    full_bounds = inspected["full_bounds"]
    full_area = max(
        1e-6,
        (full_bounds[2] - full_bounds[0]) * (full_bounds[3] - full_bounds[1]),
    )

    def _to_px(coords):
        arr = np.empty((len(coords), 2), dtype=np.int32)
        for i, (wx, wy) in enumerate(coords):
            px = int(round((wx - xmin) * pixels_per_unit))
            py = int(round((ymax - wy) * pixels_per_unit))
            arr[i, 0] = px
            arr[i, 1] = py
        return arr

    def _poly_bbox_area(coords):
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _intersects_view(coords):
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return not (max(xs) < xmin or min(xs) > xmax or max(ys) < ymin or min(ys) > ymax)

    detail_thickness = 2 if max(canvas_w, canvas_h) >= 1200 else 1
    for layer_name, polylines, polygons in items:
        if layer_name in selected_layers:
            continue
        if _is_reference_skipped(layer_name):
            continue
        grp = categorize_dxf_layer(layer_name)
        is_detail = _is_reference_detail(layer_name)
        is_furn = _is_reference_furniture(layer_name)
        if grp == "ignore" and not (is_detail or is_furn):
            continue

        if is_detail:
            ref_color = (105, 105, 115)
            ref_thick = detail_thickness
        elif is_furn:
            ref_color = (150, 150, 150)
            ref_thick = 1
        else:
            ref_color = (170, 170, 170)
            ref_thick = 1

        for poly in polygons:
            if not _intersects_view(poly) or _poly_bbox_area(poly) > 0.60 * full_area:
                continue
            pts = _to_px(poly)
            cv2.fillPoly(reference, [pts], ref_color)
        for poly in polylines:
            if not _intersects_view(poly):
                continue
            pts = _to_px(poly)
            cv2.polylines(reference, [pts], False, ref_color, ref_thick, lineType=cv2.LINE_AA)

    line_thickness = 3 if max(canvas_w, canvas_h) >= 1400 else 2
    for layer_name, polylines, polygons in items:
        if layer_name not in selected_layers:
            continue

        for poly in polygons:
            if not _intersects_view(poly):
                continue
            if _poly_bbox_area(poly) > 0.60 * full_area:
                continue
            pts = _to_px(poly)
            cv2.fillPoly(extraction, [pts], (0, 0, 0))
            cv2.fillPoly(reference, [pts], (40, 40, 40))

        for poly in polylines:
            if not _intersects_view(poly):
                continue
            pts = _to_px(poly)
            cv2.polylines(extraction, [pts], False, (0, 0, 0), line_thickness, lineType=cv2.LINE_8)
            cv2.polylines(reference, [pts], False, (40, 40, 40), max(2, line_thickness - 1), lineType=cv2.LINE_AA)

    return extraction, reference, bounds, pixels_per_unit
