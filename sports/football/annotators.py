from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from .config import FootballFieldConfig


def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int,
) -> Tuple[int, int]:
    """Scale field point (yards) to pixel space and apply padding."""
    return (
        int(round(float(point[0]) * scale + padding)),
        int(round(float(point[1]) * scale + padding)),
    )


def draw_field(
    config: FootballFieldConfig,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 4,
    line_color: sv.Color = sv.Color.WHITE,
    background_color: sv.Color = sv.Color.from_hex("#2E7D32"),  # green-ish
    draw_stripes: bool = True,
    draw_midfield: bool = True,
    draw_endzone_lines: bool = True,
    draw_debug_number_stripes: bool = True,
) -> np.ndarray:
    """
    Render an NFL field canvas (upgraded styling, same API).

    - Dimensions: 120 x 53.333 yards (includes endzones)
    - Adds: filled endzones, cleaner anti-aliased lines, stronger 10-yard lines,
      subtle 5-yard stripes, and yard numbers at top/bottom (like common tracking plots).
    - Keeps your debug "number stripes" (y_top / y_bottom) when enabled.
    """
    # ----------------------------
    # CANVAS
    # ----------------------------
    field_h_px = int(round(config.field_width * scale))
    field_w_px = int(round(config.field_length * scale))

    img = np.zeros((field_h_px + 2 * padding, field_w_px + 2 * padding, 3), dtype=np.uint8)
    img[:, :] = background_color.as_bgr()

    # ----------------------------
    # STYLE (internal; no signature changes)
    # ----------------------------
    turf_bgr = background_color.as_bgr()
    white_bgr = line_color.as_bgr()

    # Closer to the Plotly example vibe (still respects your background_color param)
    stripe_bgr = (220, 220, 220)     # subtle vertical stripes
    ten_line_bgr = white_bgr         # 10-yard lines
    five_line_bgr = stripe_bgr       # 5-yard lines (subtle)

    # Default endzone fills (you can later wire these to team colors elsewhere)
    endzone_left_bgr = sv.Color.from_hex("#1E3A8A").as_bgr()   # deep blue
    endzone_right_bgr = sv.Color.from_hex("#7C2D12").as_bgr()  # deep red

    def aa_line(p0: Tuple[int, int], p1: Tuple[int, int], bgr: Tuple[int, int, int], thickness: int) -> None:
        cv2.line(img, p0, p1, bgr, thickness, lineType=cv2.LINE_AA)

    def aa_rect(tl: Tuple[int, int], br: Tuple[int, int], bgr: Tuple[int, int, int], thickness: int) -> None:
        cv2.rectangle(img, tl, br, bgr, thickness, lineType=cv2.LINE_AA)

    def fill_rect(x0: float, y0: float, x1: float, y1: float, bgr: Tuple[int, int, int]) -> None:
        tl_px = _to_pixel((x0, y0), scale, padding)
        br_px = _to_pixel((x1, y1), scale, padding)
        cv2.rectangle(img, tl_px, br_px, bgr, thickness=-1, lineType=cv2.LINE_AA)

    def put_centered_text(
        text: str,
        center_xy: Tuple[int, int],
        font_scale: float,
        color_bgr: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(center_xy[0] - tw / 2)
        y = int(center_xy[1] + th / 2)
        cv2.putText(img, text, (x, y), font, font_scale, color_bgr, thickness, lineType=cv2.LINE_AA)

    # ----------------------------
    # ENDZONES (filled)
    # ----------------------------
    # left endzone: x in [0, 10]
    fill_rect(0.0, 0.0, 10.0, config.field_width, endzone_left_bgr)
    # right endzone: x in [110, 120]
    fill_rect(config.field_length - 10.0, 0.0, config.field_length, config.field_width, endzone_right_bgr)

    # Repaint play area turf (between goal lines) so endzones stay colored even if background_color changes
    fill_rect(10.0, 0.0, config.field_length - 10.0, config.field_width, turf_bgr)

    # ----------------------------
    # BOUNDARY
    # ----------------------------
    tl = _to_pixel((0.0, 0.0), scale, padding)
    br = _to_pixel((config.field_length, config.field_width), scale, padding)
    aa_rect(tl, br, white_bgr, line_thickness)

    # ----------------------------
    # GOAL LINES + MIDFIELD
    # ----------------------------
    if draw_endzone_lines:
        for x in (10.0, config.field_length - 10.0):
            p0 = _to_pixel((x, 0.0), scale, padding)
            p1 = _to_pixel((x, config.field_width), scale, padding)
            aa_line(p0, p1, white_bgr, line_thickness)

    if draw_midfield:
        x = config.field_length / 2.0
        p0 = _to_pixel((x, 0.0), scale, padding)
        p1 = _to_pixel((x, config.field_width), scale, padding)
        aa_line(p0, p1, white_bgr, max(1, line_thickness - 1))

    # ----------------------------
    # YARD STRIPES / LINES (visual)
    # ----------------------------
    if draw_stripes:
        # - subtle line every 5 yards
        # - stronger line every 10 yards
        # keep them under the boundary so the boundary stays crisp
        for x in np.arange(0, config.field_length + 1e-6, 5.0):
            is_10 = (abs(x % 10.0) < 1e-6)

            # Skip drawing on top of the endlines (x=0 and x=120) to avoid double-thick corners
            if abs(x - 0.0) < 1e-6 or abs(x - config.field_length) < 1e-6:
                continue

            thickness = 1
            color = five_line_bgr
            if is_10:
                thickness = max(1, line_thickness - 2)
                color = ten_line_bgr

            p0 = _to_pixel((float(x), 0.0), scale, padding)
            p1 = _to_pixel((float(x), config.field_width), scale, padding)
            aa_line(p0, p1, color, thickness)

    # ----------------------------
    # YARD NUMBERS (top/bottom) - tied to your existing debug flag
    # ----------------------------
    # This mirrors your plotly numbers placement (x=20..110 step 10; y=5 and y=width-5)
    if draw_debug_number_stripes:
        xs = list(range(20, 111, 10))
        # plotly example creates: [10,20,30,40,50,40,30,20,10]
        labels = ["10", "20", "30", "40", "50", "40", "30", "20", "10"]

        y_bot = 5.0
        y_top = config.field_width - 5.0

        # scale text with canvas; keep it readable across scales
        font_scale = max(0.6, scale / 18.0)
        text_thickness = max(2, line_thickness // 2)

        for x, lab in zip(xs, labels):
            c_bot = _to_pixel((float(x), float(y_bot)), scale, padding)
            c_top = _to_pixel((float(x), float(y_top)), scale, padding)
            put_centered_text(lab, c_bot, font_scale, white_bgr, text_thickness)
            put_centered_text(lab, c_top, font_scale, white_bgr, text_thickness)

        # keep your original "number stripes" too (yb/yt guide lines)
        yb = config.y_bottom
        yt = config.y_top
        p0 = _to_pixel((0.0, yb), scale, padding)
        p1 = _to_pixel((config.field_length, yb), scale, padding)
        aa_line(p0, p1, (180, 180, 180), 1)

        p0 = _to_pixel((0.0, yt), scale, padding)
        p1 = _to_pixel((config.field_length, yt), scale, padding)
        aa_line(p0, p1, (180, 180, 180), 1)

    return img


def draw_points_on_field(
    config: FootballFieldConfig,
    xy: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    fill_color: Optional[sv.Color] = sv.Color.BLACK,
    text_color: sv.Color = sv.Color.WHITE,
    edge_color: Optional[sv.Color] = sv.Color.WHITE,
    size: int = 10,
    edge_thickness: Optional[int] = None,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 4,
    field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw points on the field (same behavior as draw_points_on_court).
    """
    if field is None:
        field = draw_field(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    if xy is None or np.size(xy) == 0:
        return field

    pts = np.atleast_2d(xy)
    n = pts.shape[0]

    labels = labels if labels is not None else [None] * n
    if len(labels) < n:
        labels = list(labels) + [None] * (n - len(labels))

    stroke = edge_thickness if edge_thickness is not None else max(2, line_thickness // 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, size / 18.0)
    font_thickness = max(1, size // 6)

    out = field.copy()

    for i in range(n):
        cx, cy = _to_pixel((float(pts[i, 0]), float(pts[i, 1])), scale=scale, padding=padding)

        # Fill
        if fill_color is not None:
            cv2.circle(out, (cx, cy), size, fill_color.as_bgr(), -1, lineType=cv2.LINE_AA)

        # Edge
        if edge_color is not None and stroke > 0:
            cv2.circle(out, (cx, cy), size, edge_color.as_bgr(), stroke, lineType=cv2.LINE_AA)

        # Label
        label = labels[i]
        if label is not None and str(label) != "":
            text = str(label)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            tx = int(cx - tw / 2)
            ty = int(cy + th / 2)
            cv2.putText(out, text, (tx, ty), font, font_scale, text_color.as_bgr(),
                        font_thickness, lineType=cv2.LINE_AA)

    return out


def draw_paths_on_field(
    config: FootballFieldConfig,
    paths: List[np.ndarray],
    color: Optional[sv.Color] = sv.Color.BLACK,
    thickness: Optional[int] = None,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 4,
    field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw polylines in field coordinates. NaN rows split segments.
    Mirrors draw_paths_on_court.
    """
    if field is None:
        field = draw_field(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    if not paths or color is None:
        return field

    out = field.copy()
    stroke = thickness if thickness is not None else line_thickness
    bgr = color.as_bgr()

    def to_segments(pts: np.ndarray) -> List[np.ndarray]:
        pts = np.atleast_2d(pts).astype(float)
        segments: List[np.ndarray] = []
        cur: List[np.ndarray] = []
        for p in pts:
            if np.isnan(p).any():
                if len(cur) > 0:
                    segments.append(np.asarray(cur, dtype=float))
                    cur = []
            else:
                cur.append(p)
        if len(cur) > 0:
            segments.append(np.asarray(cur, dtype=float))
        return segments

    for path in paths:
        if path is None or np.size(path) == 0:
            continue

        for seg in to_segments(path):
            if seg.shape[0] >= 2:
                poly = np.array(
                    [[_to_pixel((float(x), float(y)), scale, padding) for x, y in seg]],
                    dtype=np.int32,
                )
                cv2.polylines(out, poly, isClosed=False, color=bgr,
                              thickness=stroke, lineType=cv2.LINE_AA)
            elif seg.shape[0] == 1:
                cx, cy = _to_pixel((float(seg[0, 0]), float(seg[0, 1])), scale, padding)
                cv2.circle(out, (cx, cy), max(1, stroke // 2), bgr, -1, lineType=cv2.LINE_AA)

    return out