#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import torch
from ultralytics import YOLO


PLATE_NAMES = {
    0: "Ministerial Vehicle",
    1: "Standard Vehicle",
    2: "Temporary registration",
    3: "Others",
}

VEHICLE_CLASSES = [2, 5, 7]


def parse_args() -> argparse.Namespace:
    base = Path("/home/pi/ai_camera")
    p = argparse.ArgumentParser(description="Replay a folder of images through the Raspberry Pi local pipeline.")
    p.add_argument("--input-dir", required=True, help="Folder containing test images.")
    p.add_argument("--output-dir", default=str(base / "output"), help="Folder where result.json and crops are written.")
    p.add_argument("--vehicle-model", default=str(base / "models" / "yolo11n-seg.pt"))
    p.add_argument("--plate-model", default=str(base / "models" / "best.pt"))
    p.add_argument("--vehicle-task", default="segment", help="Ultralytics task for the vehicle model.")
    p.add_argument("--plate-task", default="detect", help="Ultralytics task for the plate model.")
    p.add_argument("--camera-id", default="CAM_576")
    p.add_argument("--vehicle-conf", type=float, default=0.25)
    p.add_argument("--plate-conf", type=float, default=0.30)
    p.add_argument("--loop", action="store_true", help="Continuously replay the image folder.")
    p.add_argument("--sleep", type=float, default=0.0, help="Optional delay in seconds between images.")
    p.add_argument("--clear-output", action="store_true", help="Delete old crops before starting.")
    p.add_argument("--fp16", action="store_true", help="Run inference in float16 when supported.")
    return p.parse_args()


def image_paths(input_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def infer_task(model_path: str, fallback: str) -> str:
    name = Path(model_path).name.lower()
    if "seg" in name:
        return "segment"
    if "pose" in name:
        return "pose"
    if "cls" in name or "class" in name:
        return "classify"
    return fallback


def describe_backend_dtype(model: YOLO) -> str:
    backend = getattr(model, "model", None)
    try:
        return str(next(backend.parameters()).dtype)
    except Exception:
        return f"backend={type(backend).__name__}"


def resolve_name(names, cls_id: int, fallback: str) -> str:
    try:
        idx = int(cls_id)
    except Exception:
        return fallback
    if isinstance(names, dict):
        if idx in names:
            return str(names[idx])
        key = str(idx)
        if key in names:
            return str(names[key])
        return fallback
    if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
        return str(names[idx])
    return fallback


def save_json(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=4, ensure_ascii=False), encoding="utf-8")


def clamp_box(box: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def process_image(
    image_path: Path,
    tracking_start: int,
    model_v: YOLO,
    model_p: YOLO,
    crops_dir: Path,
    camera_id: str,
    vehicle_conf: float,
    plate_conf: float,
    fp16: bool,
) -> List[dict]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return []

    h, w = frame.shape[:2]
    v_res = model_v(frame, conf=vehicle_conf, classes=VEHICLE_CLASSES, verbose=False, half=fp16)[0]
    p_res = model_p(frame, conf=plate_conf, verbose=False, half=fp16)[0]

    vehicles = v_res.boxes.xyxy.cpu().numpy() if v_res.boxes is not None else []
    v_classes = v_res.boxes.cls.cpu().numpy().astype(int) if v_res.boxes is not None else []
    plates = p_res.boxes.xyxy.cpu().numpy() if p_res.boxes is not None else []
    p_classes = p_res.boxes.cls.cpu().numpy().astype(int) if p_res.boxes is not None else []
    p_confs = p_res.boxes.conf.cpu().numpy() if p_res.boxes is not None else []

    rows: List[dict] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, v_box in enumerate(vehicles):
        x1, y1, x2, y2 = clamp_box([int(v) for v in v_box], w, h)
        if not (x2 > x1 and y2 > y1):
            continue

        best_plate = None
        best_conf = 0.0
        for j, p_box in enumerate(plates):
            px1, py1, px2, py2 = [int(v) for v in p_box]
            if px1 > x1 and py1 > y1 and px2 < x2 and py2 < y2:
                conf = float(p_confs[j])
                if conf > best_conf:
                    best_conf = conf
                    best_plate = (px1, py1, px2, py2, j)

        crop = frame[y1:y2, x1:x2]
        tracking_id = tracking_start + idx
        image_name = f"vehicule_{tracking_id}.jpg"
        crop_path = crops_dir / image_name
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)

        vehicle_type = resolve_name(model_v.names, v_classes[idx], "vehicle")
        plate_type = "Non detectee"
        plate_bbox = None

        if best_plate:
            px1, py1, px2, py2, plate_idx = best_plate
            plate_type = resolve_name(model_p.names, p_classes[plate_idx], "Unknown")
            plate_bbox = [px1, py1, px2, py2]

        rows.append(
            {
                "camera_id": camera_id,
                "tracking_id": tracking_id,
                "timestamp": timestamp,
                "vehicle_type": vehicle_type,
                "plate_type": plate_type,
                "best_score": round(best_conf, 6),
                "speed_kmh": 0.0,
                "status": "ReplayImage",
                "coords": {
                    "vehicle_bbox": [x1, y1, x2, y2],
                    "plate_bbox": plate_bbox,
                },
                "image_file": image_name,
                "source_image": image_path.name,
            }
        )

    return rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_output and crops_dir.exists():
        shutil.rmtree(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    files = image_paths(input_dir)
    if not files:
        print(f"No images found in {input_dir}")
        return 1

    print("[replay] loading models...")
    vehicle_task = infer_task(args.vehicle_model, args.vehicle_task)
    plate_task = infer_task(args.plate_model, args.plate_task)
    model_v = YOLO(args.vehicle_model, task=vehicle_task)
    model_p = YOLO(args.plate_model, task=plate_task)
    dtype_v = describe_backend_dtype(model_v)
    dtype_p = describe_backend_dtype(model_p)
    if args.fp16:
        print(f"[replay] fp16 requested vehicle_dtype={dtype_v} plate_dtype={dtype_p}")
        if not torch.cuda.is_available():
            print("[replay] warning: running FP16 on CPU; if Ultralytics/PyTorch backend rejects it, keep FP32 as the valid benchmark baseline.")
    else:
        print(f"[replay] fp16 disabled vehicle_dtype={dtype_v} plate_dtype={dtype_p}")
    print(f"[replay] vehicle_task={vehicle_task} plate_task={plate_task}")
    print(f"[replay] images={len(files)} loop={args.loop}")

    tracking_id = 0
    all_rows: List[dict] = []
    try:
        while True:
            all_rows = []
            for image_path in files:
                rows = process_image(
                    image_path=image_path,
                    tracking_start=tracking_id,
                    model_v=model_v,
                    model_p=model_p,
                    crops_dir=crops_dir,
                    camera_id=args.camera_id,
                    vehicle_conf=args.vehicle_conf,
                    plate_conf=args.plate_conf,
                    fp16=args.fp16,
                )
                tracking_id += max(1, len(rows))
                all_rows.extend(rows)
                save_json(all_rows, output_dir / "result.json")
                save_json(all_rows, output_dir / "resultats_live.json")
                print(f"[replay] {image_path.name}: records={len(rows)} total={len(all_rows)}")
                if args.sleep > 0:
                    time.sleep(args.sleep)

            save_json(all_rows, output_dir / "resultats_finaux.json")
            if not args.loop:
                break
    except KeyboardInterrupt:
        save_json(all_rows, output_dir / "result.json")
        save_json(all_rows, output_dir / "resultats_live.json")
        save_json(all_rows, output_dir / "resultats_finaux.json")
        print("[replay] stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
