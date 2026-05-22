# -*- coding: utf-8 -*-
import argparse
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls
import cv2
import json
import os
import time
from datetime import datetime
import torch

# ============================================================
# CONFIGURATION RASPBERRY (PROD SIMPLIFIED)
# ============================================================
BASE_DIR = "/home/pi/ai_camera"
MODEL_VEHICLES = os.path.join(BASE_DIR, "models", "yolo11n-seg.pt")
MODEL_PLATES = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

CAMERA_ID = "CAM_576"
CAM_W = 2304
CAM_H = 1296
INF_W = 1280
INF_H = 720
PROCESS_EVERY_N_FRAMES = 1

PLATE_NAMES = {
    0: "Ministerial Vehicle",
    1: "Standard Vehicle",
    2: "Temporary registration",
    3: "Others",
}

VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck
JSON_SAVE_INTERVAL = 5
PLATE_CONF = 0.30
VEHICLE_CONF = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description="Run Raspberry Pi live inference pipeline.")
    parser.add_argument("--fp16", action="store_true", help="Run inference in float16 when supported.")
    return parser.parse_args()


def is_inside(p_box, v_box):
    return (
        p_box[0] > v_box[0]
        and p_box[1] > v_box[1]
        and p_box[2] < v_box[2]
        and p_box[3] < v_box[3]
    )


def scale_box(box, sx, sy):
    x1, y1, x2, y2 = box
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def save_json(best_shots, filename):
    rows = []
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for vid, shot in best_shots.items():
        rows.append(
            {
                "camera_id": CAMERA_ID,
                "tracking_id": int(vid),
                "timestamp": now_ts,
                "vehicle_type": shot.get("v_type", "car"),
                "plate_type": shot.get("p_type", "Others"),
                "best_score": round(float(shot.get("conf", 0.0)), 6),
                "speed_kmh": 0.0,
                "status": "Normal",
                "coords": {
                    "vehicle_bbox": [int(v) for v in shot.get("v_box", [0, 0, 1, 1])],
                    "plate_bbox": [int(v) for v in shot.get("p_box", [0, 0, 1, 1])],
                },
                "image_file": f"vehicule_{int(vid)}.jpg",
            }
        )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def run(fp16=False):
    print("[final_pi] loading models...")
    model_v = YOLO(MODEL_VEHICLES)
    model_p = YOLO(MODEL_PLATES)
    if fp16:
        print(
            f"[final_pi] fp16 requested vehicle_dtype={next(model_v.model.parameters()).dtype} "
            f"plate_dtype={next(model_p.model.parameters()).dtype}"
        )
        if not torch.cuda.is_available():
            print("[final_pi] warning: running FP16 on CPU; if Ultralytics/PyTorch backend rejects it, keep FP32 as the valid benchmark baseline.")
    else:
        print(
            f"[final_pi] fp16 disabled vehicle_dtype={next(model_v.model.parameters()).dtype} "
            f"plate_dtype={next(model_p.model.parameters()).dtype}"
        )

    print("[final_pi] initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (CAM_W, CAM_H), "format": "RGB888"})
    picam2.configure(config)
    picam2.set_controls(
        {
            "AfMode": controls.AfModeEnum.Continuous,
            "AeEnable": True,
            "AwbEnable": True,
            "Sharpness": 1.5,
            "Contrast": 1.1,
        }
    )
    picam2.start()
    time.sleep(2)

    best_shots = {}
    frame_idx = 0
    last_json_save = time.time()

    sx = CAM_W / INF_W
    sy = CAM_H / INF_H

    print("[final_pi] running...")

    try:
        while True:
            frame_bgr = picam2.capture_array()

            frame_idx += 1
            if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                continue

            infer_frame = cv2.resize(frame_bgr, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)

            v_results = model_v.track(
                infer_frame,
                persist=True,
                verbose=False,
                classes=VEHICLE_CLASSES,
                conf=VEHICLE_CONF,
                half=fp16,
            )[0]

            p_results = model_p.predict(infer_frame, conf=PLATE_CONF, verbose=False, half=fp16)[0]
            plates_small = p_results.boxes.xyxy.cpu().numpy() if p_results.boxes is not None else []

            if v_results.boxes is not None and v_results.boxes.id is not None:
                boxes_small = v_results.boxes.xyxy.cpu().numpy()
                ids = v_results.boxes.id.cpu().numpy().astype(int)
                clss = v_results.boxes.cls.cpu().numpy().astype(int)

                for i, box_small in enumerate(boxes_small):
                    v_id = int(ids[i])
                    v_box = clamp_box(scale_box(box_small, sx, sy), CAM_W, CAM_H)
                    x1, y1, x2, y2 = v_box

                    for p_idx, p_box_small in enumerate(plates_small):
                        p_box = clamp_box(scale_box(p_box_small, sx, sy), CAM_W, CAM_H)
                        px1, py1, px2, py2 = p_box

                        if is_inside(p_box, v_box):
                            conf = float(p_results.boxes.conf[p_idx].item())

                            # BEST SHOT: keep highest-confidence shot per tracking_id
                            if v_id not in best_shots or conf > best_shots[v_id]["conf"]:
                                v_crop = frame_bgr[y1:y2, x1:x2].copy()
                                cv2.rectangle(v_crop, (px1 - x1, py1 - y1), (px2 - x1, py2 - y1), (0, 255, 0), 2)

                                p_cls_id = int(p_results.boxes.cls[p_idx].item())
                                p_name = PLATE_NAMES.get(p_cls_id, "Others")

                                best_shots[v_id] = {
                                    "conf": conf,
                                    "img": v_crop,
                                    "v_type": model_v.names[clss[i]],
                                    "p_type": p_name,
                                    "v_box": [x1, y1, x2, y2],
                                    "p_box": [px1, py1, px2, py2],
                                }

                                crop_path = os.path.join(CROPS_DIR, f"vehicule_{v_id}.jpg")
                                cv2.imwrite(crop_path, v_crop)

            now = time.time()
            if now - last_json_save >= JSON_SAVE_INTERVAL:
                save_json(best_shots, os.path.join(OUTPUT_DIR, "result.json"))
                save_json(best_shots, os.path.join(OUTPUT_DIR, "resultats_live.json"))
                last_json_save = now

    finally:
        save_json(best_shots, os.path.join(OUTPUT_DIR, "result.json"))
        save_json(best_shots, os.path.join(OUTPUT_DIR, "resultats_finaux.json"))
        picam2.stop()


if __name__ == "__main__":
    args = parse_args()
    run(fp16=args.fp16)
