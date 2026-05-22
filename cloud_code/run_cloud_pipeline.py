#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal cloud pipeline runner:
- load a received JSON batch
- decode vehicle crop from base64
- apply lightweight preprocessing
- use ROI from JSON for fine plate extraction
- optional classification with Ultralytics classification models
- optional OCR with Tesseract
- save a result JSON usable as a pipeline output
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ocr_tn_inference import TunisiaOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simplified cloud pipeline on received JSON batches.")
    parser.add_argument("--input-json", required=True, help="Path to a payload batch JSON.")
    parser.add_argument("--output-json", required=True, help="Path to the pipeline result JSON.")
    parser.add_argument("--classifier-model", default=None, help="Optional Ultralytics classification model.")
    parser.add_argument(
        "--classifier-min-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence required to trust predicted plate type. Below this threshold, prediction is ignored.",
    )
    parser.add_argument(
        "--gate-ocr-by-classifier",
        action="store_true",
        help="Run OCR only if predicted plate type is not in --classifier-ignore-labels.",
    )
    parser.add_argument(
        "--classifier-ignore-labels",
        default="others",
        help="Comma-separated predicted labels for which OCR must be skipped.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "tesseract", "tesseract_digits_latin", "tunisia_custom", "paddleocr"],
        default="none",
        help="OCR engine to run after classification.",
    )
    parser.add_argument(
        "--ocr-fallback-engine",
        choices=["none", "tesseract", "tesseract_digits_latin", "tunisia_custom", "paddleocr"],
        default="none",
        help="Optional OCR fallback if primary OCR returns empty text.",
    )
    parser.add_argument("--ocr-fallback-lang", default="eng", help="Language code for fallback OCR.")
    parser.add_argument("--ocr-fallback-model", default=None, help="Optional model path for fallback OCR.")
    parser.add_argument("--ocr-lang", default="eng", help="Language code for OCR.")
    parser.add_argument("--ocr-model", default=None, help="Optional future OCR model path.")
    parser.add_argument(
        "--ocr-allowed-labels",
        default="standard vehicle,temporary registration,ministerial vehicle",
        help="Comma-separated predicted labels allowed for OCR. Others are skipped.",
    )
    parser.add_argument(
        "--fine-detector",
        choices=["roi", "pose", "detect"],
        default="roi",
        help="Fine plate detection source: ROI from JSON or YOLO-Pose model.",
    )
    parser.add_argument("--pose-model", default=None, help="YOLO-Pose model path used when --fine-detector pose.")
    parser.add_argument("--detector-model", default=None, help="YOLO/RT-DETR detect model path used when --fine-detector detect.")
    parser.add_argument("--save-debug-dir", default=None, help="Optional directory for decoded/debug images.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of records to process.")
    parser.add_argument("--ocr-min-width", type=int, default=160, help="Upscale OCR input to at least this width.")
    parser.add_argument("--ocr-min-height", type=int, default=48, help="Upscale OCR input to at least this height.")
    parser.add_argument("--plate-box-expand-x", type=float, default=1.45, help="Expand plate crop width scale (>=1.0).")
    parser.add_argument("--plate-box-expand-y", type=float, default=1.20, help="Expand plate crop height scale (>=1.0).")
    parser.add_argument("--detect-min-confidence", type=float, default=0.35, help="Min confidence to accept detect box.")
    parser.add_argument("--detect-min-width", type=int, default=20, help="Min detected plate width in pixels.")
    parser.add_argument("--detect-min-height", type=int, default=8, help="Min detected plate height in pixels.")
    parser.add_argument("--detect-max-area-ratio", type=float, default=0.20, help="Reject detect box if it covers too much of vehicle crop.")
    parser.add_argument("--detect-max-height-ratio", type=float, default=0.45, help="Reject detect box if it is too tall relative to vehicle crop.")
    parser.add_argument("--detect-min-aspect-ratio", type=float, default=1.2, help="Reject detect box if too square/tall for a plate.")
    parser.add_argument("--detect-max-aspect-ratio", type=float, default=12.0, help="Reject detect box if too flat/extreme.")
    parser.add_argument(
        "--stn-quad-expand",
        type=float,
        default=1.20,
        help="Expand factor applied to pose quadrilateral before perspective correction (>=1.0).",
    )
    return parser.parse_args()


def decode_base64_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Impossible de decoder image_data")
    return image


def parse_label_list(raw: str) -> List[str]:
    return [x.strip().lower() for x in str(raw).split(",") if x.strip()]


def preprocess_vehicle_crop(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image_bgr, 7, 35, 35)


def expand_xyxy(box: List[int], width: int, height: int, scale_x: float, scale_y: float) -> List[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if scale_x <= 1.0 and scale_y <= 1.0:
        return [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1) * scale_x)
    bh = max(1.0, (y2 - y1) * scale_y)
    nx1 = max(0, int(round(cx - bw / 2.0)))
    ny1 = max(0, int(round(cy - bh / 2.0)))
    nx2 = min(width, int(round(cx + bw / 2.0)))
    ny2 = min(height, int(round(cy + bh / 2.0)))
    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)
    return [nx1, ny1, nx2, ny2]


def build_plate_patch(
    image_bgr: np.ndarray,
    roi: Dict[str, Any],
    box_expand_x: float = 1.0,
    box_expand_y: float = 1.0,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    vehicle_box = roi.get("vehicle_box") or [0, 0, w, h]
    plate_box = roi.get("plate_box")
    if plate_box is None:
        return image_bgr

    vx1, vy1, _, _ = vehicle_box
    px1, py1, px2, py2 = plate_box
    local_box = [px1 - vx1, py1 - vy1, px2 - vx1, py2 - vy1]
    rx1, ry1, rx2, ry2 = expand_xyxy(local_box, w, h, max(1.0, box_expand_x), max(1.0, box_expand_y))
    patch = image_bgr[ry1:ry2, rx1:rx2]
    return patch if patch.size else image_bgr


def order_quad_points(quad: List[List[int]]) -> np.ndarray:
    pts = np.array(quad, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def expand_quad_points(pts: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    if scale <= 1.0:
        return pts
    center = pts.mean(axis=0, keepdims=True)
    expanded = center + (pts - center) * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, height - 1))
    return expanded.astype(np.float32)


def perspective_correct_plate(
    vehicle_crop: np.ndarray,
    roi: Dict[str, Any],
    quad_expand: float = 1.0,
    box_expand_x: float = 1.0,
    box_expand_y: float = 1.0,
) -> np.ndarray:
    quad = roi.get("plate_quad")
    plate_box = roi.get("plate_box")
    vehicle_box = roi.get("vehicle_box")
    if not quad or not plate_box or not vehicle_box:
        return build_plate_patch(vehicle_crop, roi, box_expand_x=box_expand_x, box_expand_y=box_expand_y)

    vx1, vy1, _, _ = vehicle_box
    local_quad = [[x - vx1, y - vy1] for x, y in quad]
    src = order_quad_points(local_quad)
    vh, vw = vehicle_crop.shape[:2]
    src = expand_quad_points(src, quad_expand, vw, vh)

    width_a = np.linalg.norm(src[2] - src[3])
    width_b = np.linalg.norm(src[1] - src[0])
    height_a = np.linalg.norm(src[1] - src[2])
    height_b = np.linalg.norm(src[0] - src[3])
    max_w = max(1, int(round(max(width_a, width_b))))
    max_h = max(1, int(round(max(height_a, height_b))))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(vehicle_crop, matrix, (max_w, max_h))
    return warped if warped.size else build_plate_patch(vehicle_crop, roi, box_expand_x=box_expand_x, box_expand_y=box_expand_y)


def to_local_box(global_box: Optional[List[int]], global_vehicle_box: Optional[List[int]], width: int, height: int) -> Optional[List[int]]:
    if not global_box or not global_vehicle_box or len(global_box) != 4 or len(global_vehicle_box) != 4:
        return None
    vx1, vy1, _, _ = [int(round(v)) for v in global_vehicle_box]
    x1, y1, x2, y2 = [int(round(v)) for v in global_box]
    local = [x1 - vx1, y1 - vy1, x2 - vx1, y2 - vy1]
    lx1, ly1, lx2, ly2 = local
    lx1 = max(0, min(width - 1, lx1))
    ly1 = max(0, min(height - 1, ly1))
    lx2 = max(lx1 + 1, min(width, lx2))
    ly2 = max(ly1 + 1, min(height, ly2))
    return [lx1, ly1, lx2, ly2]


def box_area(box: Optional[List[int]]) -> float:
    if not box:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def box_iou(a: Optional[List[int]], b: Optional[List[int]]) -> float:
    if not a or not b:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    den = box_area(a) + box_area(b) - inter
    return float(inter / den) if den > 0 else 0.0


def box_union(a: List[int], b: List[int], width: int, height: int) -> List[int]:
    u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    u[0] = max(0, min(width - 1, u[0]))
    u[1] = max(0, min(height - 1, u[1]))
    u[2] = max(u[0] + 1, min(width, u[2]))
    u[3] = max(u[1] + 1, min(height, u[3]))
    return u


class OptionalPoseDetector:
    def __init__(self, model_path: Optional[str]) -> None:
        self.model = None
        if model_path:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

    def predict_quad_and_box(self, image_bgr: np.ndarray) -> Tuple[Optional[List[List[int]]], Optional[List[int]], Optional[float]]:
        if self.model is None:
            return None, None, None
        result = self.model.predict(source=image_bgr, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None, None, None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        keypoints = result.keypoints.xy.cpu().numpy()  # [n, k, 2]
        if keypoints.shape[0] == 0:
            return None, None, None

        best_idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0
        quad = keypoints[best_idx].tolist()
        quad_int = [[int(round(pt[0])), int(round(pt[1]))] for pt in quad]

        box = boxes_xyxy[best_idx].tolist()
        bbox = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
        conf = float(confs[best_idx]) if confs is not None and len(confs) else None
        return quad_int, bbox, conf


class OptionalBoxDetector:
    def __init__(self, model_path: Optional[str]) -> None:
        self.model = None
        if model_path:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

    def predict_box_and_label(
        self, image_bgr: np.ndarray
    ) -> Tuple[Optional[List[int]], Optional[float], Optional[int], Optional[str]]:
        if self.model is None:
            return None, None, None, None
        result = self.model.predict(source=image_bgr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None, None
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
        names = self.model.names if hasattr(self.model, "names") else {}
        best_idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0
        box = boxes_xyxy[best_idx].tolist()
        bbox = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
        conf = float(confs[best_idx]) if confs is not None and len(confs) else None
        cls_id = int(classes[best_idx]) if classes is not None and len(classes) else None
        cls_name = names.get(cls_id, str(cls_id)) if cls_id is not None else None
        return bbox, conf, cls_id, cls_name

    def predict_box_for_labels(
        self, image_bgr: np.ndarray, allowed_labels: List[str]
    ) -> Tuple[Optional[List[int]], Optional[float], Optional[int], Optional[str]]:
        if self.model is None:
            return None, None, None, None
        result = self.model.predict(source=image_bgr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None, None
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
        names = self.model.names if hasattr(self.model, "names") else {}
        allowed = {str(x).strip().lower() for x in allowed_labels if str(x).strip()}
        best_idx = None
        best_conf = -1.0
        for idx, box in enumerate(boxes_xyxy):
            cls_id = int(classes[idx]) if classes is not None and len(classes) > idx else None
            cls_name = names.get(cls_id, str(cls_id)) if cls_id is not None else None
            if cls_name is None or str(cls_name).strip().lower() not in allowed:
                continue
            conf = float(confs[idx]) if confs is not None and len(confs) > idx else 0.0
            if conf > best_conf:
                best_conf = conf
                best_idx = idx
        if best_idx is None:
            return None, None, None, None
        box = boxes_xyxy[best_idx].tolist()
        bbox = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
        conf = float(confs[best_idx]) if confs is not None and len(confs) > best_idx else None
        cls_id = int(classes[best_idx]) if classes is not None and len(classes) > best_idx else None
        cls_name = names.get(cls_id, str(cls_id)) if cls_id is not None else None
        return bbox, conf, cls_id, cls_name

    def predict_box(self, image_bgr: np.ndarray) -> Tuple[Optional[List[int]], Optional[float]]:
        bbox, conf, _, _ = self.predict_box_and_label(image_bgr)
        return bbox, conf


class OptionalSegQuadDetector:
    def __init__(self, model_path: Optional[str]) -> None:
        self.model = None
        self.model_path = model_path
        self.loaded = False
        if model_path:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.loaded = True

    def predict_quad_and_box(self, image_bgr: np.ndarray) -> Tuple[Optional[List[List[int]]], Optional[List[int]], Optional[float]]:
        if self.model is None:
            return None, None, None
        result = self.model.predict(source=image_bgr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        masks_xy = result.masks.xy if result.masks is not None and result.masks.xy is not None else []
        best_idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0

        box = boxes_xyxy[best_idx].tolist()
        bbox = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
        conf = float(confs[best_idx]) if confs is not None and len(confs) else None

        quad = None
        if best_idx < len(masks_xy):
            poly = np.array(masks_xy[best_idx], dtype=np.float32)
            if poly.size:
                rect = cv2.minAreaRect(poly)
                quad = cv2.boxPoints(rect)

        if quad is None:
            x1, y1, x2, y2 = bbox
            quad = np.array([[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]], dtype=np.float32)

        h, w = image_bgr.shape[:2]
        quad[:, 0] = np.clip(quad[:, 0], 0, max(0, w - 1))
        quad[:, 1] = np.clip(quad[:, 1], 0, max(0, h - 1))
        quad_int = order_quad_points(quad.tolist()).astype(int).tolist()
        return quad_int, bbox, conf

    def predict_quad_box_and_label(
        self, image_bgr: np.ndarray
    ) -> Tuple[Optional[List[List[int]]], Optional[List[int]], Optional[float], Optional[int], Optional[str]]:
        if self.model is None:
            return None, None, None, None, None
        result = self.model.predict(source=image_bgr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None, None, None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
        names = self.model.names if hasattr(self.model, "names") else {}
        masks_xy = result.masks.xy if result.masks is not None and result.masks.xy is not None else []
        best_idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0

        box = boxes_xyxy[best_idx].tolist()
        bbox = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
        conf = float(confs[best_idx]) if confs is not None and len(confs) else None
        cls_id = int(classes[best_idx]) if classes is not None and len(classes) else None
        cls_name = names.get(cls_id, str(cls_id)) if cls_id is not None else None

        quad = None
        if best_idx < len(masks_xy):
            poly = np.array(masks_xy[best_idx], dtype=np.float32)
            if poly.size:
                rect = cv2.minAreaRect(poly)
                quad = cv2.boxPoints(rect)

        if quad is None:
            x1, y1, x2, y2 = bbox
            quad = np.array([[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]], dtype=np.float32)

        h, w = image_bgr.shape[:2]
        quad[:, 0] = np.clip(quad[:, 0], 0, max(0, w - 1))
        quad[:, 1] = np.clip(quad[:, 1], 0, max(0, h - 1))
        quad_int = order_quad_points(quad.tolist()).astype(int).tolist()
        return quad_int, bbox, conf, cls_id, cls_name


def preprocess_plate_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def build_paddle_ocr_variants(image_bgr: np.ndarray, plate_type: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Build multiple OCR-ready variants of the same plate image.
    This improves robustness for blur/rain/contrast edge cases.
    """
    variants: List[Tuple[str, np.ndarray]] = []
    base = image_bgr
    variants.append(("base", base))

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    variants.append(("gray_eq", cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)))

    # Unsharp mask.
    blur = cv2.GaussianBlur(base, (0, 0), 1.0)
    sharp = cv2.addWeighted(base, 1.6, blur, -0.6, 0)
    variants.append(("sharp", sharp))

    # Mild denoise + contrast.
    den = cv2.bilateralFilter(base, 5, 35, 35)
    variants.append(("denoise", den))

    # Adaptive threshold mapped back to 3 channels.
    ad = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants.append(("adaptive_bin", cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR)))

    # Brightness / contrast / sharpness variants.
    bright = cv2.convertScaleAbs(base, alpha=1.20, beta=18)
    dark = cv2.convertScaleAbs(base, alpha=0.95, beta=-10)
    variants.append(("bright", bright))
    variants.append(("dark", dark))
    sharp_bright = cv2.addWeighted(bright, 1.5, cv2.GaussianBlur(bright, (0, 0), 1.2), -0.5, 0)
    variants.append(("bright_sharp", sharp_bright))

    ptype = str(plate_type or "").lower()
    is_ministerial_like = any(k in ptype for k in ("ministerial", "ministere", "administrative"))
    if is_ministerial_like:
        # Red text on white background: emphasize red tones.
        b, g, r = cv2.split(base)
        red_boost = cv2.merge(
            [
                cv2.convertScaleAbs(b, alpha=0.7, beta=0),
                cv2.convertScaleAbs(g, alpha=0.7, beta=0),
                cv2.convertScaleAbs(r, alpha=1.4, beta=15),
            ]
        )
        variants.append(("ministerial_red_boost", red_boost))

        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
        m2 = cv2.inRange(hsv, (165, 50, 40), (180, 255, 255))
        red_mask = cv2.bitwise_or(m1, m2)
        red_text = cv2.bitwise_and(base, base, mask=red_mask)
        variants.append(("ministerial_red_mask", red_text))
    else:
        # White text on dark plate: inverted view may recover missing characters.
        variants.append(("invert", 255 - base))

    return variants


def upscale_for_ocr(image: np.ndarray, min_w: int, min_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    sx = max(1.0, float(min_w) / max(1, w))
    sy = max(1.0, float(min_h) / max(1, h))
    scale = max(sx, sy)
    if scale <= 1.0:
        return image
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)


def normalize_plate_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    # Normalize Arabic-Indic digits to ASCII for stable regex/rules.
    digit_map = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    src = str(text).translate(digit_map)
    normalized = "".join(ch for ch in src.upper() if ch.isalnum())
    return normalized or None


def only_digits(text: Optional[str]) -> str:
    if not text:
        return ""
    digit_map = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    src = str(text).translate(digit_map)
    return "".join(ch for ch in src if ch.isdigit())


def has_tmp_marker(text: Optional[str]) -> bool:
    if not text:
        return False
    s = str(text).upper()
    return ("RS" in s) or (AR_TMP in str(text)) or bool(re.search(r"[\u0646\u062a]{1,2}", str(text)))


def crop_xratio(img: np.ndarray, x0: float, x1: float) -> np.ndarray:
    h, w = img.shape[:2]
    xa = max(0, min(w - 1, int(round(w * x0))))
    xb = max(xa + 1, min(w, int(round(w * x1))))
    return img[:, xa:xb]


def add_structured_zone_candidates(
    plate_img: np.ndarray,
    plate_type_label: str,
    ocr_engine: "OptionalOCR",
    base_candidates: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    candidates = list(base_candidates or [])
    p = str(plate_type_label or "").lower()
    if ocr_engine.engine != "paddleocr":
        return candidates

    def ocr_zone(img: np.ndarray, ptype: str) -> Tuple[Optional[str], Optional[float]]:
        res = ocr_engine.predict(img, plate_type=ptype)
        return res.get("text"), res.get("confidence")

    try:
        if any(k in p for k in ("standard", "vehicule", "vehicle")):
            z_left = crop_xratio(plate_img, 0.00, 0.42)
            z_mid = crop_xratio(plate_img, 0.28, 0.72)
            z_right = crop_xratio(plate_img, 0.56, 1.00)

            t_left, c_left = ocr_zone(z_left, "standard vehicle")
            t_mid, c_mid = ocr_zone(z_mid, "standard vehicle")
            t_right, c_right = ocr_zone(z_right, "standard vehicle")

            left_digits = only_digits(t_left)[:3]
            right_digits = only_digits(t_right)[-4:]
            mid_ok = ("TN" in (normalize_plate_text(t_mid) or "")) or (AR_TUNIS in str(t_mid or "")) or bool(ARABIC_RANGE_RE.search(str(t_mid or "")))
            if left_digits and right_digits:
                synth = f"{left_digits} {AR_TUNIS if mid_ok else AR_TUNIS} {right_digits}"
                confs = [x for x in [c_left, c_mid, c_right] if isinstance(x, (int, float))]
                conf = round(float(sum(confs) / len(confs)), 4) if confs else 0.0
                candidates.append({"text": synth, "confidence": conf, "variant": "zone3", "mode": "structured"})

        elif any(k in p for k in ("temporary", "temporaire", "registration")):
            z_num_l = crop_xratio(plate_img, 0.00, 0.72)
            z_num_r = crop_xratio(plate_img, 0.28, 1.00)
            z_marker = crop_xratio(plate_img, 0.64, 1.00)

            t_num_l, c_num_l = ocr_zone(z_num_l, "temporary registration")
            t_num_r, c_num_r = ocr_zone(z_num_r, "temporary registration")
            t_marker, c_marker = ocr_zone(z_marker, "temporary registration")

            d_l = only_digits(t_num_l)
            d_r = only_digits(t_num_r)
            d = d_l if len(d_l) >= len(d_r) else d_r
            mk = "RS" if has_tmp_marker(t_marker) or has_tmp_marker(t_num_l) or has_tmp_marker(t_num_r) else AR_TMP
            if d:
                confs = [x for x in [c_num_l, c_num_r, c_marker] if isinstance(x, (int, float))]
                conf = round(float(sum(confs) / len(confs)), 4) if confs else 0.0
                candidates.append({"text": f"{d} {mk}", "confidence": conf, "variant": "zone_tmp", "mode": "structured"})
                candidates.append({"text": f"{mk} {d}", "confidence": conf, "variant": "zone_tmp", "mode": "structured"})

        elif any(k in p for k in ("ministerial", "ministere", "administrative")):
            # Keep ministerial on global OCR + color pre-processing only.
            # Zone synthesis can overfit and hurt this class on current data.
            pass
    except Exception:
        return candidates

    return candidates


def add_digit_en_zone_candidates(
    plate_img: np.ndarray,
    plate_type_label: str,
    ocr_digits_engine: Optional["OptionalOCR"],
    base_candidates: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    candidates = list(base_candidates or [])
    if ocr_digits_engine is None or ocr_digits_engine.engine != "paddleocr":
        return candidates

    p = str(plate_type_label or "").lower()

    def zone_digits(x0: float, x1: float) -> Tuple[str, Optional[float]]:
        z = crop_xratio(plate_img, x0, x1)
        pred = ocr_digits_engine.predict(z, plate_type="digits")
        d = only_digits(pred.get("text"))
        return d, pred.get("confidence")

    try:
        if any(k in p for k in ("standard", "vehicule", "vehicle")):
            l1, c1 = zone_digits(0.00, 0.45)
            r1, c2 = zone_digits(0.55, 1.00)
            all_d, c3 = zone_digits(0.00, 1.00)
            left = l1[:3] if l1 else ""
            right = r1[-4:] if r1 else ""
            if (not right) and all_d and len(all_d) >= 4:
                right = all_d[-4:]
            if (not left) and all_d:
                left = all_d[: min(3, max(1, len(all_d) - len(right) if right else 3))]
            if left and right:
                confs = [x for x in [c1, c2, c3] if isinstance(x, (int, float))]
                conf = round(float(sum(confs) / len(confs)), 4) if confs else 0.0
                candidates.append({"text": f"{left} {AR_TUNIS} {right}", "confidence": conf, "variant": "digit_en_zone", "mode": "structured"})

        elif any(k in p for k in ("temporary", "temporaire", "registration")):
            d_all, c1 = zone_digits(0.00, 1.00)
            d_l, c2 = zone_digits(0.00, 0.72)
            d_r, c3 = zone_digits(0.28, 1.00)
            d = max([d_all, d_l, d_r], key=lambda x: len(x or ""))
            if d:
                confs = [x for x in [c1, c2, c3] if isinstance(x, (int, float))]
                conf = round(float(sum(confs) / len(confs)), 4) if confs else 0.0
                candidates.append({"text": f"{d} {AR_TMP}", "confidence": conf, "variant": "digit_en_zone", "mode": "structured"})
                candidates.append({"text": f"{d} RS", "confidence": conf, "variant": "digit_en_zone", "mode": "structured"})

        elif any(k in p for k in ("ministerial", "ministere", "administrative")):
            l1, c1 = zone_digits(0.00, 0.34)
            r1, c2 = zone_digits(0.26, 1.00)
            all_d, c3 = zone_digits(0.00, 1.00)
            left = l1[:2] if l1 else (all_d[:2] if len(all_d) >= 2 else "")
            right = r1 if len(r1) >= 3 else (all_d[2:] if len(all_d) >= 5 else "")
            if left and len(right) >= 3:
                confs = [x for x in [c1, c2, c3] if isinstance(x, (int, float))]
                conf = round(float(sum(confs) / len(confs)), 4) if confs else 0.0
                candidates.append({"text": f"{left}-{right}", "confidence": conf, "variant": "digit_en_zone", "mode": "structured"})
    except Exception:
        return candidates

    return candidates


ARABIC_RANGE_RE = re.compile(r"[\u0600-\u06FF]+")
RE_STD_TN = re.compile(r"^\d{1,3}TN\d{1,4}$")
RE_TMP_RS = re.compile(r"^\d{1,8}RS$")
RE_TMP_RS_BIDIR = re.compile(r"^(?:\d{1,8}RS|RS\d{1,8})$")
RE_MINISTRY = re.compile(r"^\d{2}(?:[- ]?\d{3,6})$")

AR_TUNIS = "\u062a\u0648\u0646\u0633"
AR_TMP = "\u0646\u062a"


def synthesize_standard_candidates(raw_texts: List[str]) -> List[str]:
    """
    Build synthetic candidates by combining best left and right numeric groups
    seen across OCR hypotheses for standard Tunisian plates.
    """
    pairs: List[Tuple[str, str]] = []
    for txt in raw_texts:
        toks = txt.replace("-", " ").split()
        nums = ["".join(ch for ch in t if ch.isdigit()) for t in toks]
        nums = [n for n in nums if n]
        if len(nums) >= 2:
            l = nums[0][:3]
            r = nums[-1][:4]
            if l and r:
                pairs.append((l, r))
        elif len(nums) == 1:
            n = nums[0]
            if len(n) >= 4:
                l = n[: min(3, max(1, len(n) - 4))]
                r = n[-4:]
                if l and r:
                    pairs.append((l, r))
    pairs = [(l, r) for l, r in pairs if 1 <= len(l) <= 3 and 1 <= len(r) <= 4]
    if not pairs:
        return []

    # Keep only observed (left,right) couples to avoid synthetic hallucinations.
    uniq_pairs = sorted(set(pairs), key=lambda p: (-len(p[1]), -len(p[0]), p[0], p[1]))
    out: List[str] = []
    for l, r in uniq_pairs[:8]:
        out.append(f"{l} {AR_TUNIS} {r}")
    return out


def postprocess_tunisian_ocr(raw_text: Optional[str], plate_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not raw_text:
        return None, None

    cleaned = " ".join(str(raw_text).strip().split())
    normalized_default = normalize_plate_text(cleaned)
    plate_type_l = str(plate_type or "").lower()
    is_standard_like = any(k in plate_type_l for k in ("standard", "vehicule", "vehicle"))
    is_temporary_like = any(k in plate_type_l for k in ("temporary", "temporaire", "registration"))
    is_ministerial_like = any(k in plate_type_l for k in ("ministerial", "ministere", "administrative"))

    if is_ministerial_like:
        # Ministerial plate should be numeric only (allow visual separator "-" or space).
        cleaned = ARABIC_RANGE_RE.sub(" ", cleaned)
        cleaned = " ".join(cleaned.split())
        groups = re.findall(r"\d+", cleaned)
        if len(groups) >= 2:
            left = groups[0][:2]
            right = "".join(groups[1:])
            if len(left) == 2 and len(right) >= 3:
                return f"{left}-{right}", f"{left}{right}"
        digits = "".join(re.findall(r"\d", cleaned))
        if len(digits) >= 5:
            return f"{digits[:2]}-{digits[2:]}", digits
        return cleaned, normalized_default

    if is_temporary_like:
        digits = only_digits(cleaned)
        text_u = cleaned.upper()
        has_rs_ascii = bool(re.search(r"R\s*[-_ ]?\s*S", text_u))
        has_ar_tmp = (AR_TMP in cleaned) or bool(re.search(r"[\u0646\u062a]{1,2}", cleaned))
        marker_first = bool(re.match(r"^\s*(?:RS|[\u0646\u062a]{1,2})\b", cleaned, flags=re.IGNORECASE))
        if digits and not has_rs_ascii and not has_ar_tmp:
            return f"{digits} {AR_TMP}", f"{digits}RS"
        if digits and (has_rs_ascii or has_ar_tmp):
            if marker_first:
                return f"{AR_TMP} {digits}", f"{digits}RS"
            return cleaned, f"{digits}RS"
        return cleaned, normalized_default

    if not is_standard_like:
        return cleaned, normalized_default

    tokens = cleaned.replace("-", " ").split()
    if not tokens:
        return cleaned, normalized_default

    num_tokens = [(idx, "".join(ch for ch in tok if ch.isdigit())) for idx, tok in enumerate(tokens)]
    num_tokens = [(i, n) for i, n in num_tokens if n]
    if len(num_tokens) < 2:
        return cleaned, normalized_default

    left_idx, left_num = num_tokens[0]
    right_idx, right_num = num_tokens[-1]
    if left_idx >= right_idx:
        return cleaned, normalized_default

    middle = " ".join(tokens[left_idx + 1 : right_idx]).strip()
    middle_ascii = "".join(ch for ch in middle if ch.isascii() and ch.isalnum()).upper()
    middle_has_arabic = bool(ARABIC_RANGE_RE.search(middle))
    weak_middle = (
        middle == ""
        or middle_ascii in {"Y", "V", "TN", "TNNS", "TUNIS", "TUNISIE", "TUN"}
        or len(middle_ascii) <= 2
        or (middle_has_arabic and AR_TUNIS not in middle)
    )
    if weak_middle:
        return f"{left_num} {AR_TUNIS} {right_num}", f"{left_num}TN{right_num}"
    return cleaned, normalized_default


def choose_best_ocr_hypothesis(
    raw_text: Optional[str],
    candidates: Optional[List[Dict[str, Any]]],
    plate_type: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    pool: List[Tuple[Optional[str], Optional[float]]] = []
    if raw_text:
        pool.append((raw_text, None))
    for cand in candidates or []:
        txt = cand.get("text")
        conf = cand.get("confidence")
        if txt:
            pool.append((str(txt), float(conf) if conf is not None else None))
    if not pool:
        return None, None, None

    # Deduplicate by raw text, keep best confidence.
    dedup: Dict[str, Optional[float]] = {}
    for txt, conf in pool:
        if not txt:
            continue
        key = txt.strip()
        if not key:
            continue
        if key not in dedup or ((conf is not None) and (dedup[key] is None or conf > dedup[key])):
            dedup[key] = conf

    plate_type_l = str(plate_type or "").lower()
    is_standard_like = any(k in plate_type_l for k in ("standard", "vehicule", "vehicle"))
    is_temporary_like = any(k in plate_type_l for k in ("temporary", "temporaire", "registration"))
    is_ministerial_like = any(k in plate_type_l for k in ("ministerial", "ministere", "administrative"))
    if is_standard_like:
        for synth in synthesize_standard_candidates(list(dedup.keys())):
            dedup.setdefault(synth, 0.0)

    observed_digit_strings: List[str] = []
    for t in dedup.keys():
        d = only_digits(t)
        if d:
            observed_digit_strings.append(d)

    best_key = None
    best_score: Tuple[float, float] = (-1e9, -1e9)
    best_out: Tuple[Optional[str], Optional[str]] = (None, None)
    for txt, conf in dedup.items():
        out_text, norm = postprocess_tunisian_ocr(txt, plate_type)
        n = norm or ""
        conf_v = float(conf) if conf is not None else 0.0
        digits_n = only_digits(n)

        # Score against all known Tunisian templates, then add a small prior for predicted type.
        score_std = 0.0
        if RE_STD_TN.fullmatch(n):
            score_std += 100.0
        else:
            has_tn = "TN" in n
            groups = re.findall(r"\d+", n)
            if len(groups) >= 2:
                left, right = groups[0], groups[-1]
                score_std += 38.0
                if 1 <= len(left) <= 3:
                    score_std += 10.0
                if 1 <= len(right) <= 4:
                    score_std += 15.0
                if len(right) == 4:
                    score_std += 25.0
                elif len(right) == 3:
                    score_std += 10.0
                else:
                    score_std += float(min(len(right), 4) * 2)
                if has_tn:
                    score_std += 20.0
            elif re.fullmatch(r"^\d{3,7}$", n or ""):
                score_std += 8.0

        score_tmp = 0.0
        if RE_TMP_RS.fullmatch(n):
            score_tmp += 100.0
        elif RE_TMP_RS_BIDIR.fullmatch(n):
            score_tmp += 70.0
        elif re.fullmatch(r"^\d{3,8}$", n or ""):
            score_tmp += 20.0

        score_min = 0.0
        if RE_MINISTRY.fullmatch(n):
            score_min += 100.0
        else:
            groups = re.findall(r"\d+", n)
            if len(groups) >= 2 and len(groups[0]) == 2 and len(groups[-1]) >= 3:
                score_min += 70.0
            elif re.fullmatch(r"^\d{5,8}$", n or ""):
                score_min += 35.0

        template_score = max(score_std, score_tmp, score_min, 5.0 if n else 0.0)
        if is_standard_like:
            template_score += score_std * 0.08
        elif is_temporary_like:
            template_score += score_tmp * 0.08
            # Conservative guard against over-generated temporary numbers.
            if digits_n and len(digits_n) > 7:
                template_score -= 80.0
            if digits_n and len(digits_n) < 3:
                template_score -= 40.0
            if digits_n and observed_digit_strings:
                support = 0
                for od in observed_digit_strings:
                    if not od:
                        continue
                    if digits_n in od or od in digits_n:
                        support += 1
                    elif abs(len(od) - len(digits_n)) <= 1:
                        # Loose overlap ratio.
                        common = sum(1 for a, b in zip(od, digits_n) if a == b)
                        if common >= max(2, min(len(od), len(digits_n)) - 2):
                            support += 1
                if support == 0:
                    template_score -= 35.0
                elif support == 1:
                    template_score -= 8.0
        elif is_ministerial_like:
            template_score += score_min * 0.08

        completeness = float(min(len(n), 16))
        score = (template_score + completeness + (0.75 * conf_v), conf_v)
        if score > best_score:
            best_score = score
            best_key = txt
            best_out = (out_text, norm)

    return best_out[0], best_out[1], best_key


class OptionalClassifier:
    def __init__(self, model_path: Optional[str]) -> None:
        self.model = None
        if model_path:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            return {"label": None, "confidence": None}

        result = self.model.predict(source=image_bgr, verbose=False)[0]
        names = self.model.names if hasattr(self.model, "names") else {}

        # Classification model path: use top-1 probs.
        probs = result.probs
        if probs is not None:
            top_idx = int(probs.top1)
            return {"label": names.get(top_idx, str(top_idx)), "confidence": float(probs.top1conf)}

        # Detect model path: use highest-confidence detected class on the patch.
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
            clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
            if clss is not None and len(clss) > 0:
                best_idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0
                cls_idx = int(clss[best_idx])
                conf = float(confs[best_idx]) if confs is not None and len(confs) else None
                return {"label": names.get(cls_idx, str(cls_idx)), "confidence": conf}

        return {"label": None, "confidence": None}


class OptionalOCR:
    def __init__(self, engine: str, lang: str, model_path: Optional[str]) -> None:
        self.engine = engine
        self.lang = lang
        self.available = False
        self.module = None
        self.paddle_ocr = None
        self.custom_model = TunisiaOCR(model_path=model_path) if engine == "tunisia_custom" else None

        if engine in {"tesseract", "tesseract_digits_latin"}:
            try:
                import pytesseract

                self.module = pytesseract
                self.available = True
            except Exception:
                self.available = False
        elif engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR

                # PaddleOCR constructor differs across versions (v2/v3).
                # Try several signatures from oldest to newest and keep the first that works.
                self.paddle_ocr = None
                init_errors: List[str] = []
                constructor_candidates = [
                    {"use_angle_cls": True, "lang": self.lang, "show_log": False, "use_gpu": False},
                    {"use_angle_cls": True, "lang": self.lang, "use_gpu": False},
                    {"use_textline_orientation": True, "lang": self.lang, "use_gpu": False},
                    {"use_textline_orientation": True, "lang": self.lang},
                    {"lang": self.lang},
                ]
                for kwargs in constructor_candidates:
                    try:
                        self.paddle_ocr = PaddleOCR(**kwargs)
                        break
                    except Exception as err:
                        init_errors.append(f"{type(err).__name__}: {err}")
                self.available = self.paddle_ocr is not None
                if not self.available and init_errors:
                    print(f"[WARN] PaddleOCR init failed: {' | '.join(init_errors[:2])}")
            except Exception:
                self.available = False

    def predict(self, image: np.ndarray, plate_type: Optional[str] = None) -> Dict[str, Any]:
        if self.engine == "none":
            return {"engine": "none", "available": True, "text": None, "confidence": None}
        if self.engine == "tunisia_custom":
            pred = self.custom_model.predict(image) if self.custom_model else None
            return {
                "engine": "tunisia_custom",
                "available": bool(pred and pred.available),
                "text": pred.text if pred else None,
                "confidence": pred.confidence if pred else None,
                "model_name": pred.model_name if pred else "tunisia_ocr_custom",
            }
        if self.engine == "paddleocr":
            if not self.available or self.paddle_ocr is None:
                return {"engine": "paddleocr", "available": False, "text": None, "confidence": None}

            def extract_from_paddle_output(result_obj: Any) -> Tuple[List[str], List[float]]:
                out_texts: List[str] = []
                out_confs: List[float] = []
                if not isinstance(result_obj, list) or not result_obj:
                    return out_texts, out_confs

                # Case A (det=False): [[('TEXT', conf)]]
                if (
                    isinstance(result_obj[0], list)
                    and result_obj[0]
                    and isinstance(result_obj[0][0], (list, tuple))
                    and len(result_obj[0][0]) >= 1
                    and isinstance(result_obj[0][0][0], str)
                ):
                    rec = result_obj[0][0]
                    rec_text = str(rec[0]).strip()
                    rec_conf = float(rec[1]) if len(rec) > 1 else None
                    if rec_text:
                        out_texts.append(rec_text)
                        if rec_conf is not None:
                            out_confs.append(rec_conf)
                    return out_texts, out_confs

                # Case B (det=True): [[ [box], ('TEXT', conf) ], ...]
                lines = result_obj[0] if isinstance(result_obj[0], list) else result_obj
                for line in lines:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    pred = line[1]
                    if not isinstance(pred, (list, tuple)) or len(pred) < 1:
                        continue
                    text = str(pred[0]).strip()
                    conf = float(pred[1]) if len(pred) > 1 else None
                    if text:
                        out_texts.append(text)
                        if conf is not None:
                            out_confs.append(conf)
                return out_texts, out_confs

            cands: List[Tuple[Optional[str], Optional[float], str, str]] = []
            variants = build_paddle_ocr_variants(
                image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                plate_type=plate_type,
            )

            for var_name, var_img in variants:
                # Candidate A: recognition-only pass.
                try:
                    direct_rec = self.paddle_ocr.ocr(var_img, det=False, rec=True, cls=False)
                    texts, confidences = extract_from_paddle_output(direct_rec)
                    cands.append(
                        (
                            " ".join(texts).strip() or None,
                            round(sum(confidences) / len(confidences), 4) if confidences else None,
                            var_name,
                            "rec_only",
                        )
                    )
                except Exception:
                    pass

                # Candidate B: full det+rec pass.
                try:
                    det_rec = self.paddle_ocr.ocr(var_img, cls=True)
                    texts, confidences = extract_from_paddle_output(det_rec)
                    cands.append(
                        (
                            " ".join(texts).strip() or None,
                            round(sum(confidences) / len(confidences), 4) if confidences else None,
                            var_name,
                            "det_rec",
                        )
                    )
                except Exception:
                    pass

            # Choose best candidate by completeness first (normalized length), then confidence.
            def cand_score(item: Tuple[Optional[str], Optional[float], str, str]) -> Tuple[int, float]:
                txt, conf, _, _ = item
                norm = normalize_plate_text(txt) or ""
                return (len(norm), float(conf) if conf is not None else 0.0)

            cands = [c for c in cands if c[0]]
            if not cands:
                return {"engine": "paddleocr", "available": False, "text": None, "confidence": None, "candidates": []}
            best_text, best_conf, best_variant, best_mode = max(cands, key=cand_score)
            cand_objs = [{"text": t, "confidence": c, "variant": v, "mode": m} for t, c, v, m in cands]

            return {
                "engine": "paddleocr",
                "available": True,
                "text": best_text,
                "confidence": best_conf,
                "variant": best_variant,
                "mode": best_mode,
                "candidates": cand_objs,
            }
        if not self.available or self.module is None:
            return {"engine": self.engine, "available": False, "text": None, "confidence": None}

        tesseract_config = "--psm 7 --oem 3"
        if self.engine == "tesseract_digits_latin":
            # Strict non-Arabic recognition: digits + latin uppercase only.
            tesseract_config += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        data = self.module.image_to_data(
            image,
            lang=self.lang,
            config=tesseract_config,
            output_type=self.module.Output.DICT,
        )

        texts: List[str] = []
        confidences: List[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = (text or "").strip()
            try:
                conf_value = float(conf)
            except Exception:
                conf_value = -1.0
            if text and conf_value >= 0:
                texts.append(text)
                confidences.append(conf_value)

        raw_text = " ".join(texts).strip() or None
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
        return {
            "engine": self.engine,
            "available": True,
            "text": raw_text,
            "confidence": avg_conf,
        }


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    output_json = Path(args.output_json).resolve()
    debug_dir = Path(args.save_debug_dir).resolve() if args.save_debug_dir else None
    classifier = OptionalClassifier(args.classifier_model)
    pose_detector = OptionalPoseDetector(args.pose_model if args.fine_detector == "pose" else None)
    box_detector = OptionalBoxDetector(args.detector_model if args.fine_detector == "detect" else None)
    # Initialize OCR after detector models to avoid potential DLL load conflicts on Windows.
    ocr_engine = OptionalOCR(args.ocr_engine, args.ocr_lang, args.ocr_model)
    ocr_digits_engine = OptionalOCR("paddleocr", "en", None) if args.ocr_engine == "paddleocr" else None
    fallback_engine = None
    if args.ocr_fallback_engine != "none" and args.ocr_fallback_engine != args.ocr_engine:
        fallback_engine = OptionalOCR(args.ocr_fallback_engine, args.ocr_fallback_lang, args.ocr_fallback_model)

    records = json.loads(input_json.read_text(encoding="utf-8"))
    if args.limit > 0:
        records = records[: args.limit]
    ignore_labels = set(parse_label_list(args.classifier_ignore_labels))
    allowed_labels = set(parse_label_list(args.ocr_allowed_labels))

    outputs: List[Dict[str, Any]] = []
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records, start=1):
        vehicle_crop = decode_base64_image(record["image_data"])
        preprocessed = preprocess_vehicle_crop(vehicle_crop)

        roi_input = dict(record.get("roi", {}))
        roi_runtime = dict(roi_input)
        fine_detection = {
            "source": "roi_from_json",
            "plate_box": roi_runtime.get("plate_box"),
            "plate_quad": roi_runtime.get("plate_quad"),
            "confidence": None,
        }

        if args.fine_detector == "pose":
            quad, box, conf = pose_detector.predict_quad_and_box(preprocessed)
            if box is not None:
                h, w = preprocessed.shape[:2]
                roi_runtime["vehicle_box"] = [0, 0, w, h]
                roi_runtime["plate_box"] = box
                roi_runtime["plate_quad"] = quad
                fine_detection = {
                    "source": "pose_model",
                    "plate_box": box,
                    "plate_quad": quad,
                    "confidence": conf,
                }
        elif args.fine_detector == "detect":
            box, conf = box_detector.predict_box(preprocessed)
            if box is not None:
                bw = max(0, int(box[2]) - int(box[0]))
                bh = max(0, int(box[3]) - int(box[1]))
                conf_ok = (conf is not None and conf >= args.detect_min_confidence)
                size_ok = (bw >= args.detect_min_width and bh >= args.detect_min_height)
                h, w = preprocessed.shape[:2]
                area_ratio = float((bw * bh) / max(1.0, float(w * h)))
                height_ratio = float(bh / max(1.0, float(h)))
                aspect_ratio = float(bw / max(1.0, float(bh)))
                geom_ok = (
                    area_ratio <= args.detect_max_area_ratio
                    and height_ratio <= args.detect_max_height_ratio
                    and aspect_ratio >= args.detect_min_aspect_ratio
                    and aspect_ratio <= args.detect_max_aspect_ratio
                )
                if conf_ok and size_ok and geom_ok:
                    roi_runtime["vehicle_box"] = [0, 0, w, h]
                    # Merge detect box with input ROI local box when overlapping and detect box is too tight.
                    input_local_box = to_local_box(roi_input.get("plate_box"), roi_input.get("vehicle_box"), w, h)
                    merged_box = box
                    if input_local_box is not None:
                        iou = box_iou(box, input_local_box)
                        det_area = box_area(box)
                        in_area = box_area(input_local_box)
                        if iou >= 0.05 and det_area < 0.85 * in_area:
                            merged_box = box_union(box, input_local_box, w, h)
                    roi_runtime["plate_box"] = merged_box
                    roi_runtime["plate_quad"] = None
                    fine_detection = {
                        "source": "detect_model",
                        "plate_box": merged_box,
                        "plate_quad": None,
                        "confidence": conf,
                        "accepted": True,
                    }
                else:
                    fine_detection = {
                        "source": "roi_from_json",
                        "plate_box": roi_runtime.get("plate_box"),
                        "plate_quad": roi_runtime.get("plate_quad"),
                        "confidence": None,
                        "accepted": False,
                        "rejected_detect_box": box,
                        "rejected_detect_confidence": conf,
                        "rejected_reason": {
                            "conf_ok": conf_ok,
                            "size_ok": size_ok,
                            "geom_ok": geom_ok,
                            "width": bw,
                            "height": bh,
                            "area_ratio": round(area_ratio, 4),
                            "height_ratio": round(height_ratio, 4),
                            "aspect_ratio": round(aspect_ratio, 4),
                        },
                    }

        plate_patch = build_plate_patch(
            preprocessed,
            roi_runtime,
            box_expand_x=max(1.0, args.plate_box_expand_x),
            box_expand_y=max(1.0, args.plate_box_expand_y),
        )
        rectified_plate = perspective_correct_plate(
            preprocessed,
            roi_runtime,
            quad_expand=max(1.0, args.stn_quad_expand),
            box_expand_x=max(1.0, args.plate_box_expand_x),
            box_expand_y=max(1.0, args.plate_box_expand_y),
        )
        if args.ocr_engine in {"paddleocr", "tunisia_custom"}:
            # PaddleOCR/TrOCR generally perform better on non-binarized plate crops.
            ocr_input = rectified_plate
        else:
            ocr_input = preprocess_plate_for_ocr(rectified_plate)
        ocr_input = upscale_for_ocr(ocr_input, args.ocr_min_width, args.ocr_min_height)

        classification = classifier.predict(plate_patch)
        raw_label = classification.get("label")
        raw_conf = classification.get("confidence")
        low_conf = bool(raw_conf is None or float(raw_conf) < float(args.classifier_min_confidence))
        if low_conf:
            classification["label_raw"] = raw_label
            classification["confidence_raw"] = raw_conf
            classification["label"] = None
            classification["filtered_low_confidence"] = True
            classification["filter_threshold"] = float(args.classifier_min_confidence)

        pred_label = str(classification.get("label") or "").strip().lower()
        plate_type_for_post = classification.get("label") or (record.get("object") or {}).get("plate_type")
        plate_type_for_post_l = str(plate_type_for_post or "").strip().lower()
        skip_ocr_by_others = bool(pred_label == "others" or plate_type_for_post_l == "others")
        skip_ocr_by_ignore = bool(args.gate_ocr_by_classifier and pred_label and pred_label in ignore_labels)
        skip_ocr_by_allowed = bool(allowed_labels and (not plate_type_for_post_l or plate_type_for_post_l not in allowed_labels))
        skip_ocr = bool(skip_ocr_by_others or skip_ocr_by_ignore or skip_ocr_by_allowed)

        if skip_ocr:
            if skip_ocr_by_others:
                reason = "ocr_skipped_for_others"
            elif skip_ocr_by_ignore:
                reason = f"classifier_predicted_{pred_label}"
            else:
                reason = f"ocr_not_allowed_for_{plate_type_for_post_l or 'unknown'}"
            ocr_result = {
                "engine": args.ocr_engine,
                "available": True,
                "text": None,
                "confidence": None,
                "skipped": True,
                "skip_reason": reason,
            }
        else:
            ocr_result = ocr_engine.predict(ocr_input, plate_type=plate_type_for_post_l)
            if args.ocr_engine == "paddleocr":
                ocr_result["candidates"] = add_digit_en_zone_candidates(
                    plate_img=ocr_input if ocr_input.ndim == 3 else cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR),
                    plate_type_label=plate_type_for_post_l,
                    ocr_digits_engine=ocr_digits_engine,
                    base_candidates=ocr_result.get("candidates"),
                )
                ocr_result["candidates"] = add_structured_zone_candidates(
                    plate_img=ocr_input if ocr_input.ndim == 3 else cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR),
                    plate_type_label=plate_type_for_post_l,
                    ocr_engine=ocr_engine,
                    base_candidates=ocr_result.get("candidates"),
                )
        if fallback_engine is not None:
            primary_text = normalize_plate_text(ocr_result.get("text"))
            if not primary_text:
                if args.ocr_fallback_engine in {"paddleocr", "tunisia_custom"}:
                    fallback_input = rectified_plate
                else:
                    fallback_input = preprocess_plate_for_ocr(rectified_plate)
                fallback_result = fallback_engine.predict(fallback_input, plate_type=plate_type_for_post_l)
                fallback_text = normalize_plate_text(fallback_result.get("text"))
                if fallback_text:
                    ocr_result = fallback_result
        ocr_text_display, normalized_text, selected_raw = choose_best_ocr_hypothesis(
            ocr_result.get("text"),
            ocr_result.get("candidates"),
            plate_type_for_post,
        )

        if debug_dir:
            cv2.imwrite(str(debug_dir / f"{index:04d}_vehicle.jpg"), preprocessed)
            cv2.imwrite(str(debug_dir / f"{index:04d}_plate_patch.jpg"), plate_patch)
            cv2.imwrite(str(debug_dir / f"{index:04d}_plate_rectified.jpg"), rectified_plate)
            cv2.imwrite(str(debug_dir / f"{index:04d}_plate_ocr_input.jpg"), ocr_input)
            if args.ocr_engine == "paddleocr" and ocr_result.get("variant"):
                try:
                    variant_name = str(ocr_result.get("variant"))
                    vlist = build_paddle_ocr_variants(
                        ocr_input if ocr_input.ndim == 3 else cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR),
                        plate_type=str(plate_type_for_post_l or ""),
                    )
                    selected_variant = None
                    for vn, vi in vlist:
                        if vn == variant_name:
                            selected_variant = vi
                            break
                    if selected_variant is not None:
                        cv2.imwrite(str(debug_dir / f"{index:04d}_plate_ocr_input_best_variant.jpg"), selected_variant)
                except Exception:
                    pass

        outputs.append(
            {
                "meta": record["meta"],
                "object_input": record["object"],
                "roi": roi_runtime,
                "pipeline": {
                    "preprocessing": "bilateral_filter",
                    "fine_detection": fine_detection,
                    "classification": classification,
                    "perspective_correction": {
                        "ready": bool(roi_runtime.get("plate_quad")),
                        "applied": bool(roi_runtime.get("plate_quad")),
                    },
                    "ocr_transformer": {
                        "engine": ocr_result.get("engine"),
                        "available": ocr_result.get("available"),
                        "text": ocr_text_display,
                        "confidence": ocr_result.get("confidence"),
                        "variant": ocr_result.get("variant"),
                        "mode": ocr_result.get("mode"),
                        "selected_raw_text": selected_raw,
                        "candidates": ocr_result.get("candidates"),
                        "skipped": ocr_result.get("skipped", False),
                        "skip_reason": ocr_result.get("skip_reason"),
                    },
                    "post_processing": {
                        "normalized_text": normalized_text,
                    },
                },
                "final_result": {
                    "vehicle_class": record["object"].get("class"),
                    "plate_type_input": record["object"].get("plate_type"),
                    "plate_type_predicted": classification.get("label"),
                    "plate_type_confidence": classification.get("confidence"),
                    "ocr_text": ocr_text_display,
                    "ocr_confidence": ocr_result.get("confidence"),
                    "normalized_text": normalized_text,
                },
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(outputs)} records processed -> {output_json}")


if __name__ == "__main__":
    main()
