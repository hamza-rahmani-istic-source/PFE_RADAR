#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert embedded JSON detections to cloud payload format expected by run_cloud_pipeline.py.

Input format example (per record):
{
  "timestamp": "2026-03-25 18:11:44",
  "id": 1,
  "type_vehicule": "car",
  "sens": "Normal",
  "vitesse": 13.7,
  "bbox_vehicule": [x1, y1, x2, y2],
  "bbox_plaque": [x1, y1, x2, y2]
}
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert embedded detections JSON to cloud payload JSON.")
    parser.add_argument("--input-json", required=True, help="Embedded detections JSON path.")
    parser.add_argument(
        "--frame-image",
        required=True,
        help="Full frame image path used to crop vehicle region for each detection.",
    )
    parser.add_argument("--output-json", required=True, help="Output cloud payload JSON path.")
    parser.add_argument("--dataset-name", default="embedded_received")
    parser.add_argument("--coord-mode", choices=["4", "8"], default="4")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--limit", type=int, default=0, help="Optional max records to convert.")
    return parser.parse_args()


def clip_box(box: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def encode_crop_base64(image_bgr, box: List[int], jpeg_quality: int) -> str:
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return base64.b64encode(buf).decode("utf-8")


def parse_timestamp(value: Optional[str]) -> str:
    if not value:
        return "2026-03-25T00:00:00Z"
    ts = value.strip().replace(" ", "T")
    if ts.endswith("Z") or "+" in ts:
        return ts
    return f"{ts}Z"


def build_payload_record(
    item: Dict[str, Any], frame_bgr, dataset_name: str, coord_mode: str, jpeg_quality: int, frame_id: int
) -> Dict[str, Any]:
    h, w = frame_bgr.shape[:2]
    vehicle_box = clip_box(item.get("bbox_vehicule") or [0, 0, w, h], w, h)
    plate_box = clip_box(item.get("bbox_plaque") or vehicle_box, w, h)

    image_data = encode_crop_base64(frame_bgr, vehicle_box, jpeg_quality)
    tracking_id = int(item.get("id", frame_id))
    vehicle_class = str(item.get("type_vehicule", "car")).strip().lower() or "car"

    return {
        "meta": {
            "dataset_name": dataset_name,
            "coord_mode": coord_mode,
            "frame_id": frame_id,
            "timestamp": parse_timestamp(item.get("timestamp")),
        },
        "object": {
            "tracking_id": tracking_id,
            "class": vehicle_class,
            # Plate type is intentionally not provided by embedded side.
            # Cloud infers it from the seg8 model.
            "plate_type": None,
            "plate_conf": 1.0,
            "best_score": float(item.get("vitesse", 0.0)),
        },
        "roi": {
            "vehicle_box": vehicle_box,
            "plate_box": plate_box,
            "plate_quad": None,
        },
        "image_data": image_data,
        "embedded_meta": {
            "sens": item.get("sens"),
            "vitesse": item.get("vitesse"),
        },
    }


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    frame_image = Path(args.frame_image).resolve()
    output_json = Path(args.output_json).resolve()

    rows = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("input JSON must be a list of detections.")

    frame_bgr = cv2.imread(str(frame_image))
    if frame_bgr is None:
        raise RuntimeError(f"Cannot read frame image: {frame_image}")

    if args.limit > 0:
        rows = rows[: args.limit]

    payload = [
        build_payload_record(
            item=row,
            frame_bgr=frame_bgr,
            dataset_name=args.dataset_name,
            coord_mode=args.coord_mode,
            jpeg_quality=args.jpeg_quality,
            frame_id=idx + 1,
        )
        for idx, row in enumerate(rows)
    ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(payload)} records converted -> {output_json}")


if __name__ == "__main__":
    main()
