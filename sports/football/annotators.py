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
    background_color: sv.Color = sv.Color.from_hex("#2E7D32"),
    draw_stripes: bool = True,
    draw_midfield: bool = True,
    draw_endzone_lines: bool = True,
    draw_debug_number_stripes: bool = True,
) -> np.ndarray:
    # --- CANVAS SETUP ---
    field_h_px = int(round(config.field_width * scale))
    field_w_px = int(round(config.field_length * scale))
    img = np.zeros((field_h_px + 2 * padding, field_w_px + 2 * padding, 3), dtype=np.uint8)
    img[:, :] = background_color.as_bgr()
    white_bgr = line_color.as_bgr()

    # Geometry Helpers
    def aa_line(p0, p1, bgr, thickness):
        cv2.line(img, p0, p1, bgr, thickness, lineType=cv2.LINE_AA)

    def fill_rect(x0, y0, x1, y1, bgr):
        tl_px = _to_pixel((x0, y0), scale, padding)
        br_px = _to_pixel((x1, y1), scale, padding)
        cv2.rectangle(img, tl_px, br_px, bgr, thickness=-1, lineType=cv2.LINE_AA)

    # Rotated text helper with FIX for IndexError (Clipping)
    def put_rotated_text(text, center_xy, angle, font_scale, color_bgr, thickness):
        font = cv2.FONT_HERSHEY_DUPLEX 
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 1. Create a large enough buffer so rotation doesn't cut off corners
        buffer = 200 
        text_img = np.zeros((th + buffer, tw + buffer, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (buffer//2, th + buffer//2), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
        
        # 2. Rotate the text
        M = cv2.getRotationMatrix2D(((tw + buffer) / 2, (th + buffer) / 2), angle, 1)
        rotated_text = cv2.warpAffine(text_img, M, (tw + buffer, th + buffer))
        
        # 3. Calculate target coordinates on the main field image
        h, w = rotated_text.shape[:2]
        x1, y1 = int(center_xy[0] - w/2), int(center_xy[1] - h/2)
        x2, y2 = x1 + w, y1 + h

        # 4. CLIPPING LOGIC: Prevents "IndexError: size of axis is 0"
        img_h, img_w = img.shape[:2]
        
        overlap_x1 = max(0, x1)
        overlap_y1 = max(0, y1)
        overlap_x2 = min(img_w, x2)
        overlap_y2 = min(img_h, y2)

        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            source_x1 = overlap_x1 - x1
            source_y1 = overlap_y1 - y1
            source_x2 = source_x1 + (overlap_x2 - overlap_x1)
            source_y2 = source_y1 + (overlap_y2 - overlap_y1)

            cropped_text = rotated_text[source_y1:source_y2, source_x1:source_x2]
            target_area = img[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
            
            mask = np.any(cropped_text > 0, axis=-1)
            target_area[mask] = cropped_text[mask]

    # --- ENDZONES ---
    # Left Endzone (Blue) - Face field
    fill_rect(0.0, 0.0, 10.0, config.field_width, sv.Color.from_hex("#1E3A8A").as_bgr())
    put_rotated_text("ENDZONE", _to_pixel((5.0, config.field_width/2), scale, padding), 
                     90, 1.5, white_bgr, 3)

    # Right Endzone (Red) - Face field
    fill_rect(110.0, 0.0, 120.0, config.field_width, sv.Color.from_hex("#7C2D12").as_bgr())
    put_rotated_text("ENDZONE", _to_pixel((115.0, config.field_width/2), scale, padding), 
                     -90, 1.5, white_bgr, 3)

    # --- FIELD BOUNDARY & YARD LINES ---
    cv2.rectangle(img, _to_pixel((0,0), scale, padding), _to_pixel((120, 53.333), scale, padding), 
                  white_bgr, line_thickness, cv2.LINE_AA)
    
    if draw_stripes:
        for x in range(10, 111, 5):
            thick = line_thickness - 2 if x % 10 == 0 else 1
            aa_line(_to_pixel((float(x), 0.0), scale, padding), 
                    _to_pixel((float(x), config.field_width), scale, padding), white_bgr, thick)

    # NFL Hashmarks
    y_hashes = [23.5, 53.333 - 23.5]
    for x in range(11, 110):
        for y in y_hashes:
            aa_line(_to_pixel((float(x), y - 0.5), scale, padding), 
                    _to_pixel((float(x), y + 0.5), scale, padding), white_bgr, 1)

    # --- YARD NUMBERS (Horizontal) ---
    if draw_debug_number_stripes:
        xs = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = ["10", "20", "30", "40", "50", "40", "30", "20", "10"]
        
        num_scale = 1.75 # Original readable scale
        num_thick = 2

        for x, lab in zip(xs, labels):
            c_bot = _to_pixel((float(x), config.y_bottom), scale, padding)
            c_top = _to_pixel((float(x), config.y_top), scale, padding)
            
            put_rotated_text(lab, c_bot, 0, num_scale, white_bgr, num_thick)
            put_rotated_text(lab, c_top, 0, num_scale, white_bgr, num_thick)

    return img

def draw_los_dotted(
    field_img: np.ndarray,
    config: FootballFieldConfig,
    los_x_world: float,
    color: sv.Color = sv.Color.BLUE,
    scale: float = 20,
    padding: int = 50,
    thickness: int = 2,
    dash_px: int = 10,
    gap_px: int = 8,
) -> np.ndarray:
    out = field_img.copy()
    bgr = color.as_bgr()

    # x position in pixels (same x for all y)
    x_px, _ = _to_pixel((float(los_x_world), 0.0), scale=scale, padding=padding)

    # y range of field in pixels
    _, y0 = _to_pixel((0.0, 0.0), scale=scale, padding=padding)
    _, y1 = _to_pixel((0.0, float(config.field_width)), scale=scale, padding=padding)

    y = int(min(y0, y1))
    y_end = int(max(y0, y1))

    while y < y_end:
        y2 = min(y + int(dash_px), y_end)
        cv2.line(out, (x_px, y), (x_px, y2), bgr, int(thickness), lineType=cv2.LINE_AA)
        y = y2 + int(gap_px)

    return out

def draw_receiver_mph_sep_labels(
    field: np.ndarray,
    config: FootballFieldConfig,
    recv_xy: np.ndarray,
    recv_tids: np.ndarray,
    speed_mph_by_tid: dict[int, float],
    sep_yd_by_tid: dict[int, float],
    scale: float = 20,
    padding: int = 50,
    y0_px: int = 16,        # starting offset below dot
    line_gap_px: int = 16,  # spacing between mph and sep lines
    text_color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
):
    out = field.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55

    recv_xy = np.atleast_2d(recv_xy)
    recv_tids = np.asarray(recv_tids, dtype=int)

    for (x, y), tid in zip(recv_xy, recv_tids):
        tid = int(tid)
        mph = float(speed_mph_by_tid.get(tid, 0.0))
        sep = sep_yd_by_tid.get(tid, None)

        line1 = f"{mph:.1f} mph"
        line2 = f"sep {sep:.1f} yd" if sep is not None else "sep --"

        cx, cy = _to_pixel((float(x), float(y)), scale=scale, padding=padding)

        # Draw line 1 (mph)
        (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        org1 = (int(cx - tw1 / 2), int(cy + y0_px + th1))
        cv2.putText(out, line1, org1, font, font_scale, text_color.as_bgr(), thickness, cv2.LINE_AA)

        # Draw line 2 (sep) slightly below
        (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        org2 = (int(cx - tw2 / 2), int(cy + y0_px + line_gap_px + th2))
        cv2.putText(out, line2, org2, font, font_scale, text_color.as_bgr(), thickness, cv2.LINE_AA)

    return out

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
    if field is None:
        field = draw_field(config=config, scale=scale, padding=padding, line_thickness=line_thickness)

    if xy is None or np.size(xy) == 0:
        return field

    pts = np.atleast_2d(xy)
    n = pts.shape[0]
    labels = labels if labels is not None else [None] * n
    
    stroke = edge_thickness if edge_thickness is not None else max(2, line_thickness // 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.35, size / 18.0)
    font_thickness = max(1, size // 6)

    out = field.copy()
    for i in range(n):
        cx, cy = _to_pixel((float(pts[i, 0]), float(pts[i, 1])), scale=scale, padding=padding)
        if fill_color is not None:
            cv2.circle(out, (cx, cy), size, fill_color.as_bgr(), -1, lineType=cv2.LINE_AA)
        if edge_color is not None and stroke > 0:
            cv2.circle(out, (cx, cy), size, edge_color.as_bgr(), stroke, lineType=cv2.LINE_AA)
        
        label = labels[i]
        if label is not None and str(label) != "":
            (tw, th), _ = cv2.getTextSize(str(label), font, font_scale, font_thickness)
            cv2.putText(out, str(label), (int(cx - tw / 2), int(cy + th / 2)), font, font_scale, 
                        text_color.as_bgr(), font_thickness, lineType=cv2.LINE_AA)
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
    if field is None:
        field = draw_field(config=config, scale=scale, padding=padding, line_thickness=line_thickness)

    if not paths or color is None:
        return field

    out = field.copy()
    stroke = thickness if thickness is not None else line_thickness
    bgr = color.as_bgr()

    for path in paths:
        if path is None or np.size(path) == 0:
            continue
        
        pts = np.atleast_2d(path).astype(float)
        segments: List[np.ndarray] = []
        cur: List[np.ndarray] = []
        for p in pts:
            if np.isnan(p).any():
                if cur: segments.append(np.asarray(cur)); cur = []
            else: cur.append(p)
        if cur: segments.append(np.asarray(cur))

        for seg in segments:
            if seg.shape[0] >= 2:
                poly = np.array([[_to_pixel((float(x), float(y)), scale, padding) for x, y in seg]], dtype=np.int32)
                cv2.polylines(out, poly, isClosed=False, color=bgr, thickness=stroke, lineType=cv2.LINE_AA)
            elif seg.shape[0] == 1:
                cx, cy = _to_pixel((float(seg[0, 0]), float(seg[0, 1])), scale, padding)
                cv2.circle(out, (cx, cy), max(1, stroke // 2), bgr, -1, lineType=cv2.LINE_AA)
    return out