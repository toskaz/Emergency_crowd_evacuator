import colorsys
import math

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


def ensure_pymupdf_available():
    if fitz is None:
        raise ImportError(
            "PDF support requires 'PyMuPDF'. Install it with: pip install pymupdf"
        )


def _is_dashed(dashes) -> bool:
    if not dashes:
        return False
    s = str(dashes).strip()
    if s.startswith("[]"):
        return False
    if "[" in s and "]" in s:
        inner = s[s.find("[") + 1 : s.find("]")].strip()
        return len(inner) > 0
    return True


def _color_metrics(rgb):
    if rgb is None or len(rgb) < 3:
        return 0.0, 0.0, 0.0
    r = max(0.0, min(1.0, float(rgb[0])))
    g = max(0.0, min(1.0, float(rgb[1])))
    b = max(0.0, min(1.0, float(rgb[2])))
    _, s, v = colorsys.rgb_to_hsv(r, g, b)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return s, v, lum


def _is_saturated_color(rgb) -> bool:
    s, v, _ = _color_metrics(rgb)
    return s > 0.25 and v > 0.45


def _eval_cubic_bezier(p0, p1, p2, p3, steps=10):
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = (
            (mt ** 3) * p0.x
            + 3 * (mt ** 2) * t * p1.x
            + 3 * mt * (t ** 2) * p2.x
            + (t ** 3) * p3.x
        )
        y = (
            (mt ** 3) * p0.y
            + 3 * (mt ** 2) * t * p1.y
            + 3 * mt * (t ** 2) * p2.y
            + (t ** 3) * p3.y
        )
        pts.append((x, y))
    return pts


def _pixmap_to_bgr(pix) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def inspect_pdf(path: str, page_index: int = 0) -> dict:
    ensure_pymupdf_available()
    doc = fitz.open(path)
    try:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError("PDF file contains no pages")
        p_idx = max(0, min(page_count - 1, int(page_index)))
        page = doc[p_idx]
        page_rect = page.rect
        page_area = max(1e-6, page_rect.width * page_rect.height)

        drawings = page.get_drawings()
        non_frame_count = 0
        for d in drawings:
            r = d.get("rect")
            if r is not None:
                area = max(0.0, r.width * r.height)
                if area >= 0.90 * page_area:
                    continue
            non_frame_count += 1

        is_vector = non_frame_count > 20
        return {
            "page_count": page_count,
            "page_index": p_idx,
            "is_vector": is_vector,
            "non_frame_drawings": non_frame_count,
            "full_bounds": (
                float(page_rect.x0),
                float(page_rect.y0),
                float(page_rect.x1),
                float(page_rect.y1),
            ),
        }
    finally:
        doc.close()


def render_scanned_pdf(
    path: str,
    page_index: int = 0,
    target_max_px: int = 2000,
):
    ensure_pymupdf_available()
    doc = fitz.open(path)
    try:
        p_idx = max(0, min(len(doc) - 1, int(page_index)))
        page = doc[p_idx]
        page_rect = page.rect
        long_pt = max(1e-6, max(page_rect.width, page_rect.height))
        dpi_scale = 200.0 / 72.0
        max_scale = min(2200.0, float(target_max_px)) / long_pt
        zoom = min(dpi_scale, max_scale)
        zoom = max(0.5, zoom)

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        bgr = _pixmap_to_bgr(pix)
        bounds = (
            float(page_rect.x0),
            float(page_rect.y0),
            float(page_rect.x1),
            float(page_rect.y1),
        )
        return bgr, bounds, float(zoom)
    finally:
        doc.close()


def _is_short_thin_axis_segment(d, stroke_w: float) -> bool:
    d_type = d.get("type", "")
    if "f" in d_type or stroke_w >= 0.6:
        return False
    items = d.get("items") or []
    if len(items) != 1 or items[0][0] != "l":
        return False
    p1, p2 = items[0][1], items[0][2]
    return math.hypot(p2.x - p1.x, p2.y - p1.y) < 8.0


def _strip_small_vector_text_glyphs(extraction: np.ndarray, pixels_per_pt: float) -> None:
    gray = extraction[:, :, 0]
    fg = (gray < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels <= 1:
        return

    max_glyph_px = 18.0 * max(0.5, float(pixels_per_pt))
    remove_lut = np.zeros(n_labels, dtype=bool)
    for comp_id in range(1, n_labels):
        w_bb = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h_bb = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        max_dim = max(w_bb, h_bb)
        if area < 6:
            remove_lut[comp_id] = True
            continue
        if max_dim <= max_glyph_px:
            fill_ratio = area / float(max(1, w_bb * h_bb))
            if fill_ratio < 0.45:
                remove_lut[comp_id] = True

    if np.any(remove_lut):
        extraction[remove_lut[labels]] = 255


def render_vector_pdf(
    path: str,
    page_index: int = 0,
    clip_rect: tuple[float, float, float, float] | None = None,
    min_wall_thickness_pt: float = 0.55,
    ignore_door_arcs: bool = True,
    ignore_dashed_lines: bool = True,
    ignore_colored_markings: bool = True,
    include_fixed_blocks: bool = False,
    filled_walls_only: bool = False,
    target_max_px: int = 2000,
):
    ensure_pymupdf_available()
    doc = fitz.open(path)
    try:
        p_idx = max(0, min(len(doc) - 1, int(page_index)))
        page = doc[p_idx]
        page_rect = page.rect
        page_area = max(1e-6, page_rect.width * page_rect.height)

        if clip_rect is not None:
            rect = fitz.Rect(*clip_rect) & page_rect
            if rect.is_empty or rect.width < 1.0 or rect.height < 1.0:
                rect = page_rect
        else:
            rect = page_rect

        long_pt = max(1e-6, max(rect.width, rect.height))
        zoom = float(target_max_px) / long_pt
        mat = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        reference = _pixmap_to_bgr(pix)
        canvas_h, canvas_w = reference.shape[:2]
        extraction = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        scale_x = canvas_w / max(1e-6, rect.width)
        scale_y = canvas_h / max(1e-6, rect.height)
        pixels_per_pt = (scale_x + scale_y) * 0.5

        def _pt_to_px(x, y):
            return (
                int(round((x - rect.x0) * scale_x)),
                int(round((y - rect.y0) * scale_y)),
            )

        drawings = page.get_drawings()
        for d in drawings:
            d_rect = d.get("rect")
            if d_rect is not None:
                pad = max(1.0, float(d.get("width") or 0.0))
                if (
                    d_rect.x1 + pad < rect.x0
                    or d_rect.x0 - pad > rect.x1
                    or d_rect.y1 + pad < rect.y0
                    or d_rect.y0 - pad > rect.y1
                ):
                    continue
                d_area = max(0.0, d_rect.width * d_rect.height)
            else:
                d_area = 0.0

            if ignore_dashed_lines and _is_dashed(d.get("dashes")):
                continue

            d_type = d.get("type", "")
            items = d.get("items") or []
            if not items:
                continue

            fill_rgb = d.get("fill")
            if ("f" in d_type) and fill_rgb is not None:
                max_bg_ratio = 0.60 if filled_walls_only else 0.80
                if d_area < max_bg_ratio * page_area:
                    if not (filled_walls_only and d_area < 12.0):
                        if not (ignore_colored_markings and _is_saturated_color(fill_rgb)):
                            f_sat, _, f_lum = _color_metrics(fill_rgb)
                            is_dark_wall_fill = f_lum < 0.35
                            is_fixed_gray_fill = (
                                (not filled_walls_only)
                                and include_fixed_blocks
                                and f_sat < 0.12
                                and 0.35 <= f_lum <= 0.80
                                and d_area < 0.15 * page_area
                            )
                            if is_dark_wall_fill or is_fixed_gray_fill:
                                _draw_pdf_fill(extraction, items, _pt_to_px)

            if filled_walls_only:
                continue

            stroke_rgb = d.get("color")
            stroke_w = float(d.get("width") or 0.0)
            if ("s" in d_type) and stroke_rgb is not None:
                if ignore_dashed_lines and _is_short_thin_axis_segment(d, stroke_w):
                    continue
                if stroke_w >= float(min_wall_thickness_pt):
                    if not (ignore_colored_markings and _is_saturated_color(stroke_rgb)):
                        _, _, s_lum = _color_metrics(stroke_rgb)
                        if s_lum < 0.72:
                            px_thick = max(1, min(24, int(round(stroke_w * pixels_per_pt))))
                            _draw_pdf_stroke(
                                extraction,
                                items,
                                _pt_to_px,
                                px_thick,
                                ignore_door_arcs=ignore_door_arcs,
                            )

        _strip_small_vector_text_glyphs(extraction, pixels_per_pt)

        used_bounds = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        return extraction, reference, used_bounds, float(pixels_per_pt)
    finally:
        doc.close()


def _draw_pdf_fill(canvas: np.ndarray, items: list, pt_to_px):
    polys = []
    cur = []

    def _flush():
        nonlocal cur
        if len(cur) >= 3:
            polys.append(np.array([pt_to_px(x, y) for x, y in cur], dtype=np.int32))
        cur = []

    def _append_pt(pt):
        px, py = float(pt[0]), float(pt[1])
        if not cur:
            cur.append((px, py))
        else:
            lx, ly = cur[-1]
            if abs(px - lx) > 1e-3 or abs(py - ly) > 1e-3:
                cur.append((px, py))

    for item in items:
        op = item[0]
        if op == "l":
            p1, p2 = item[1], item[2]
            if cur and (abs(p1.x - cur[-1][0]) > 1e-2 or abs(p1.y - cur[-1][1]) > 1e-2):
                _flush()
            _append_pt((p1.x, p1.y))
            _append_pt((p2.x, p2.y))
        elif op == "c":
            p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
            if cur and (abs(p1.x - cur[-1][0]) > 1e-2 or abs(p1.y - cur[-1][1]) > 1e-2):
                _flush()
            _append_pt((p1.x, p1.y))
            for bx, by in _eval_cubic_bezier(p1, p2, p3, p4, steps=8):
                _append_pt((bx, by))
        elif op == "re":
            _flush()
            r = item[1]
            polys.append(
                np.array(
                    [
                        pt_to_px(r.x0, r.y0),
                        pt_to_px(r.x1, r.y0),
                        pt_to_px(r.x1, r.y1),
                        pt_to_px(r.x0, r.y1),
                    ],
                    dtype=np.int32,
                )
            )
        elif op == "qu":
            _flush()
            q = item[1]
            polys.append(
                np.array(
                    [
                        pt_to_px(q.ul.x, q.ul.y),
                        pt_to_px(q.ur.x, q.ur.y),
                        pt_to_px(q.lr.x, q.lr.y),
                        pt_to_px(q.ll.x, q.ll.y),
                    ],
                    dtype=np.int32,
                )
            )

    _flush()
    if polys:
        cv2.fillPoly(canvas, polys, (0, 0, 0))


def _draw_pdf_stroke(
    canvas: np.ndarray,
    items: list,
    pt_to_px,
    thickness: int,
    ignore_door_arcs: bool = True,
):
    for item in items:
        op = item[0]
        if op == "l":
            p1, p2 = item[1], item[2]
            cv2.line(
                canvas,
                pt_to_px(p1.x, p1.y),
                pt_to_px(p2.x, p2.y),
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_8,
            )
        elif op == "c":
            if ignore_door_arcs:
                continue
            p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
            pts = [pt_to_px(p1.x, p1.y)] + [
                pt_to_px(bx, by) for bx, by in _eval_cubic_bezier(p1, p2, p3, p4, steps=10)
            ]
            cv2.polylines(
                canvas,
                [np.array(pts, dtype=np.int32)],
                False,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_8,
            )
        elif op == "re":
            r = item[1]
            pts = np.array(
                [
                    pt_to_px(r.x0, r.y0),
                    pt_to_px(r.x1, r.y0),
                    pt_to_px(r.x1, r.y1),
                    pt_to_px(r.x0, r.y1),
                ],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], True, (0, 0, 0), thickness, lineType=cv2.LINE_8)
        elif op == "qu":
            q = item[1]
            pts = np.array(
                [
                    pt_to_px(q.ul.x, q.ul.y),
                    pt_to_px(q.ur.x, q.ur.y),
                    pt_to_px(q.lr.x, q.lr.y),
                    pt_to_px(q.ll.x, q.ll.y),
                ],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], True, (0, 0, 0), thickness, lineType=cv2.LINE_8)
