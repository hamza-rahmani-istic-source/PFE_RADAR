#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
light_cloud_sender.py

Sender séparé et léger pour Raspberry:
- lit périodiquement un JSON local (format result.json)
- envoie UNIQUEMENT les nouveaux tracking_id
- fréquence configurable (par défaut 5 secondes)
- faible charge CPU/RAM

Usage:
python3 /home/pi/ai_camera/light_cloud_sender.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import requests
from PIL import Image


DEFAULT_CLOUD_URL = "https://radar-cloud-api-641438482543.europe-west1.run.app/v1/ingest/payload"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight sender for new Raspberry detections")
    p.add_argument("--input-json", default="/home/pi/ai_camera/output/result.json")
    p.add_argument("--images-dir", default="/home/pi/ai_camera/output/crops")
    p.add_argument("--cloud-url", default=DEFAULT_CLOUD_URL)
    p.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds")
    p.add_argument("--timeout", type=float, default=12.0, help="HTTP timeout seconds")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--state-file", default="/home/pi/ai_camera/output/sent_ids_state.json")
    p.add_argument("--vehicle-model", default="/app/yolo11n.pt")
    p.add_argument("--seg-model", default="/app/best_8coordonnees.pt")
    p.add_argument("--pose-model", default="/app/yolo11n-pose.pt")
    p.add_argument("--ocr-model", default="/app/best")
    return p.parse_args()


def to_iso(ts: str) -> str:
    raw = (ts or "").strip()
    if not raw:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = raw.replace(" ", "T")
    if raw.endswith("Z") or "+" in raw:
        return raw
    return raw + "Z"


def clip_box(box: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def global_to_local(plate_bbox: List[int], vehicle_bbox: List[int], w: int, h: int) -> List[int]:
    pvx1, pvy1, pvx2, pvy2 = [int(v) for v in plate_bbox]
    vx1, vy1, _, _ = [int(v) for v in vehicle_bbox]
    local = [pvx1 - vx1, pvy1 - vy1, pvx2 - vx1, pvy2 - vy1]
    return clip_box(local, w, h)


def encode_jpeg_b64(img_bgr, jpeg_quality: int) -> str:
    # Keep OpenCV images in BGR for local processing, but always encode the
    # payload from an RGB buffer so the Cloud receives canonical colors.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buffer = io.BytesIO()
    Image.fromarray(img_rgb).save(buffer, format="JPEG", quality=int(jpeg_quality))
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_state(state_file: Path) -> Set[str]:
    if not state_file.exists():
        return set()
    try:
        rows = json.loads(state_file.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            return set(str(x) for x in rows)
    except Exception:
        pass
    return set()


def save_state(state_file: Path, sent_keys: Set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(sorted(sent_keys), ensure_ascii=False, indent=2), encoding="utf-8")


def unique_key(row: Dict[str, Any]) -> str:
    # Keep one send per saved crop file for a tracking id.
    return f"{row.get('tracking_id')}|{row.get('image_file')}"


def build_record(row: Dict[str, Any], img_bgr, jpeg_quality: int) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    coords = row.get("coords") or {}
    vehicle_bbox_global = coords.get("vehicle_bbox") or [0, 0, w, h]
    plate_bbox_global = coords.get("plate_bbox") or vehicle_bbox_global

    # image_file is expected to be the vehicle crop.
    vehicle_box_local = [0, 0, w, h]
    plate_box_local = global_to_local(plate_bbox_global, vehicle_bbox_global, w, h)

    return {
        "meta": {
            "dataset_name": "raspberry_live",
            "coord_mode": "4",
            "frame_id": int(row.get("tracking_id", 0)) + 1,
            "timestamp": to_iso(str(row.get("timestamp", ""))),
        },
        "object": {
            "tracking_id": int(row.get("tracking_id", 0)),
            "class": str(row.get("vehicle_type", "car")).strip().lower() or "car",
            "plate_type": None,
            "plate_conf": 1.0,
            "best_score": float(row.get("best_score", 0.0)),
        },
        "roi": {
            "vehicle_box": vehicle_box_local,
            "plate_box": plate_box_local,
            "plate_quad": None,
        },
        "image_data": encode_jpeg_b64(img_bgr, jpeg_quality=jpeg_quality),
        "embedded_meta": {
            "speed_kmh": float(row.get("speed_kmh", 0.0)),
            "status": row.get("status"),
            "camera_id": row.get("camera_id"),
        },
    }


def send_records(cloud_url: str, timeout: float, records: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[bool, str]:
    payload = {
        "records": records,
        "options": {
            "vehicle_model": args.vehicle_model,
            "fine_detector": "seg8",
            "seg_model": args.seg_model,
            "pose_model": args.pose_model,
            "ocr_engine": "tunisia_custom",
            "ocr_model": args.ocr_model,
        },
        "publish_events": True,
    }

    try:
        resp = requests.post(cloud_url, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}: {resp.text[:220]}"
        body = resp.json()
        return True, f"trace_id={body.get('trace_id')} count={body.get('count')}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    images_dir = Path(args.images_dir)
    state_file = Path(args.state_file)

    sent_keys = load_state(state_file)
    print(f"[sender] start interval={args.interval}s, sent_keys={len(sent_keys)}")

    while True:
        rows = load_rows(input_json)
        if rows:
            new_rows = [r for r in rows if unique_key(r) not in sent_keys]
            if new_rows:
                records = []
                keys = []
                for r in new_rows:
                    image_name = r.get("image_file")
                    if not image_name:
                        continue
                    image_path = images_dir / str(image_name)
                    if not image_path.exists():
                        continue
                    img = cv2.imread(str(image_path))
                    if img is None:
                        continue
                    records.append(build_record(r, img, args.jpeg_quality))
                    keys.append(unique_key(r))

                if records:
                    ok, msg = send_records(args.cloud_url, args.timeout, records, args)
                    if ok:
                        for k in keys:
                            sent_keys.add(k)
                        save_state(state_file, sent_keys)
                        print(f"[sender] sent {len(records)} new records -> {msg}")
                    else:
                        print(f"[sender] send failed: {msg}")
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

