from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from .config import FootballFieldConfig


def _get_name(d: Dict, key_a: str, key_b: str) -> Optional[str]:
    """Return d[key_a] if present else d[key_b]."""
    if key_a in d and d[key_a] is not None:
        return str(d[key_a])
    v = d.get(key_b)
    return str(v) if v is not None else None


def _kps_to_dict(kps: List[Dict]) -> Dict[str, Dict]:
    """
    Keypoints come back like:
      {'x':..., 'y':..., 'confidence':..., 'class_name':'tip'}
    We index them by name ('tip','0','num').
    """
    out: Dict[str, Dict] = {}
    for kp in kps:
        name = _get_name(kp, "class_name", "class")
        if name is None:
            continue
        out[name] = kp
    return out


def _dist(a: Dict, b: Dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _yard_from_class_name(name: str) -> int:
    """
    Accepts class names like:
      - "num_10", "num_20", ...
    Returns the integer yard value.
    """
    m = re.search(r"num_(\d+)", str(name))
    if not m:
        raise ValueError(f"Bad class_name: {name}")
    return int(m.group(1))


def classify_variant(kpd: Dict[str, Dict], w: float, h: float) -> Optional[str]:
    """
    Variant classification rules (your heuristic):

    TOP vs BOTTOM:
      Compare which is closer to tip (num vs 0)
        - if num closer -> TOP
        - if 0 closer   -> BOTTOM

    LEFT vs RIGHT:
      Compare tip.x to centroid of (num, 0)
        - tip right -> RIGHT
        - tip left  -> LEFT

    Returns:
      "TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"
      or None if ambiguous.
    """
    diag = max(math.hypot(w, h), 1e-6)

    d_tip_num = _dist(kpd["tip"], kpd["num"])
    d_tip_0 = _dist(kpd["tip"], kpd["0"])

    # reject if too ambiguous
    if abs(d_tip_num - d_tip_0) < 0.02 * diag:
        return None

    tb = "TOP" if d_tip_num < d_tip_0 else "BOTTOM"

    cx = (float(kpd["num"]["x"]) + float(kpd["0"]["x"])) / 2.0
    dx = float(kpd["tip"]["x"]) - cx

    if abs(dx) < 0.02 * diag:
        return None

    lr = "RIGHT" if dx > 0 else "LEFT"
    return f"{tb}_{lr}"


@dataclass(frozen=True)
class Landmark:
    """
    One correspondence point for homography:
      src_xy: pixel location in the frame (image space)
      dst_xy: known location on the field (world/field space)
    """
    yard: int
    variant: str
    src_xy: Tuple[float, float]
    dst_xy: Tuple[float, float]
    det_conf: float


def build_landmarks_from_predictions(
    predictions: List[Dict],
    cfg: FootballFieldConfig,
    det_conf_min: float = 0.4,
    kp_conf_min: float = 0.25,
    fifty_no_lr: bool = True,
) -> List[Landmark]:
    """
    Convert number-model predictions into homography landmarks.

    Notes:
      - Expects `predictions` to be a list of dicts, where each dict contains:
          - confidence (float)
          - class_name / class (e.g. "num_20")
          - width, height
          - keypoints: list of dicts with names ('tip','0','num') and confidences
      - Uses 'num' keypoint as src_xy (image anchor), matching your current pipeline.
    """
    out: List[Landmark] = []

    for det in predictions:
        if not isinstance(det, dict):
            # defensive: skip anything unexpected (tuples/objects/etc.)
            continue

        det_conf = float(det.get("confidence", 0.0))
        if det_conf < det_conf_min:
            continue

        det_name = _get_name(det, "class_name", "class")
        if det_name is None:
            continue

        try:
            yard = _yard_from_class_name(det_name)
        except Exception:
            continue

        kpd = _kps_to_dict(det.get("keypoints", []))
        if not all(k in kpd for k in ("tip", "0", "num")):
            continue

        if any(float(kpd[k].get("confidence", 0.0)) < kp_conf_min for k in ("tip", "0", "num")):
            continue

        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))
        if w <= 0.0 or h <= 0.0:
            continue

        # Source point in image: use "num" keypoint (your current convention)
        src = (float(kpd["num"]["x"]), float(kpd["num"]["y"]))

        if yard == 50 and fifty_no_lr:
            # only TOP/BOTTOM for 50 (no LR)
            diag = max(math.hypot(w, h), 1e-6)
            d_tip_num = _dist(kpd["tip"], kpd["num"])
            d_tip_0 = _dist(kpd["tip"], kpd["0"])
            if abs(d_tip_num - d_tip_0) < 0.02 * diag:
                continue

            tb = "TOP" if d_tip_num < d_tip_0 else "BOTTOM"
            dst = cfg.target_xy(yard, tb)
            out.append(Landmark(yard=yard, variant=tb, src_xy=src, dst_xy=dst, det_conf=det_conf))
            continue

        variant = classify_variant(kpd, w, h)
        if variant is None:
            continue

        dst = cfg.target_xy(yard, variant)
        out.append(Landmark(yard=yard, variant=variant, src_xy=src, dst_xy=dst, det_conf=det_conf))

    return out


def filter_landmarks(landmarks: List[Landmark], max_per_key: int = 1) -> List[Landmark]:
    """
    Keep up to `max_per_key` landmarks per (yard, variant),
    preferring higher det_conf.
    """
    landmarks_sorted = sorted(landmarks, key=lambda lm: lm.det_conf, reverse=True)
    used: Dict[Tuple[int, str], int] = {}
    kept: List[Landmark] = []

    for lm in landmarks_sorted:
        key = (lm.yard, lm.variant)
        if used.get(key, 0) >= max_per_key:
            continue
        kept.append(lm)
        used[key] = used.get(key, 0) + 1

    return kept


def landmarks_to_arrays(landmarks: List[Landmark]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert landmarks into (src_pts, dst_pts) arrays for homography.
    """
    src = np.asarray([lm.src_xy for lm in landmarks], dtype=np.float32)
    dst = np.asarray([lm.dst_xy for lm in landmarks], dtype=np.float32)
    return src, dst