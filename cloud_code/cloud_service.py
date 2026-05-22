#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from run_cloud_pipeline import (
    OptionalBoxDetector,
    OptionalClassifier,
    OptionalOCR,
    OptionalPoseDetector,
    OptionalSegQuadDetector,
    build_plate_patch,
    decode_base64_image,
    normalize_plate_text,
    perspective_correct_plate,
    preprocess_plate_for_ocr,
    preprocess_vehicle_crop,
    upscale_for_ocr,
)


class CloudPipelineService:
    def __init__(
        self,
        classifier_model: Optional[str],
        vehicle_model: Optional[str],
        ocr_engine: str,
        ocr_lang: str = "eng",
        ocr_model: Optional[str] = None,
        enable_vehicle_detection: bool = False,
        vehicle_detector_model: Optional[str] = None,
        fine_detector: str = "roi",
        pose_model: Optional[str] = None,
        detector_model: Optional[str] = None,
        seg_model: Optional[str] = None,
    ) -> None:
        self.vehicle_classifier = OptionalClassifier(vehicle_model)
        self.enable_vehicle_detection = bool(enable_vehicle_detection)
        self.vehicle_detector_path = vehicle_detector_model
        self.vehicle_detector = OptionalBoxDetector(vehicle_detector_model if self.enable_vehicle_detection else None)
        self.classifier = OptionalClassifier(classifier_model)
        self.ocr = OptionalOCR(ocr_engine, ocr_lang, ocr_model)
        self.fine_detector = str(fine_detector or "roi").strip().lower()
        self.seg_model_path = seg_model
        self.pose_detector = OptionalPoseDetector(pose_model if self.fine_detector == "pose" else None)
        self.box_detector = OptionalBoxDetector(detector_model if self.fine_detector == "detect" else None)
        self.seg_detector = OptionalSegQuadDetector(seg_model if self.fine_detector == "seg8" else None)

    def _detect_vehicle_crop(self, image_bgr) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "enabled": self.enable_vehicle_detection,
            "applied": False,
            "source": "object_input",
            "label": None,
            "confidence": None,
            "global_box": None,
            "model_path": self.vehicle_detector_path,
        }
        if not self.enable_vehicle_detection or self.vehicle_detector.model is None:
            return result
        box, conf, _, cls_name = self.vehicle_detector.predict_box_for_labels(
            image_bgr, ["car", "truck", "bus", "motorcycle", "motorbike", "bicycle"]
        )
        if box is None:
            return result
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return result
        result.update(
            {
                "applied": True,
                "source": "vehicle_detector_model",
                "label": cls_name,
                "confidence": conf,
                "global_box": box,
            }
        )
        result["crop"] = crop
        return result

    def _process_one(self, record: Dict[str, Any], index: int, debug_dir: Optional[Path]) -> Dict[str, Any]:
        input_image = decode_base64_image(record["image_data"])
        vehicle_detection = self._detect_vehicle_crop(input_image)
        vehicle_crop = vehicle_detection.pop("crop", None) if vehicle_detection.get("applied") else input_image
        preprocessed = preprocess_vehicle_crop(vehicle_crop)
        if vehicle_detection.get("applied"):
            vh, vw = preprocessed.shape[:2]
            roi_runtime = {
                "vehicle_box": [0, 0, vw, vh],
                "plate_box": None,
                "plate_quad": None,
            }
        else:
            roi_runtime = dict(record.get("roi", {}))
        fine_detection = {
            "source": "roi_from_json",
            "plate_box": roi_runtime.get("plate_box"),
            "plate_quad": roi_runtime.get("plate_quad"),
            "confidence": None,
        }

        if self.fine_detector == "pose":
            quad, box, conf = self.pose_detector.predict_quad_and_box(preprocessed)
            if box is not None:
                h, w = preprocessed.shape[:2]
                roi_runtime["vehicle_box"] = [0, 0, w, h]
                roi_runtime["plate_box"] = box
                roi_runtime["plate_quad"] = quad
                fine_detection = {"source": "pose_model", "plate_box": box, "plate_quad": quad, "confidence": conf}
        elif self.fine_detector == "detect":
            box, conf = self.box_detector.predict_box(preprocessed)
            if box is not None:
                h, w = preprocessed.shape[:2]
                roi_runtime["vehicle_box"] = [0, 0, w, h]
                roi_runtime["plate_box"] = box
                roi_runtime["plate_quad"] = None
                fine_detection = {"source": "detect_model", "plate_box": box, "plate_quad": None, "confidence": conf}
        elif self.fine_detector == "seg8":
            quad, box, conf, cls_id, cls_name = self.seg_detector.predict_quad_box_and_label(preprocessed)
            if box is not None:
                h, w = preprocessed.shape[:2]
                roi_runtime["vehicle_box"] = [0, 0, w, h]
                roi_runtime["plate_box"] = box
                roi_runtime["plate_quad"] = quad
                fine_detection = {
                    "source": "seg8_model",
                    "plate_box": box,
                    "plate_quad": quad,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "model_path": self.seg_model_path,
                    "model_loaded": bool(getattr(self.seg_detector, "loaded", False)),
                }

        plate_patch = build_plate_patch(preprocessed, roi_runtime)
        rectified_plate = perspective_correct_plate(preprocessed, roi_runtime)

        # Option2 strict flow:
        # 1) seg8 gives type + initial 8coords
        # 2) run YOLO pose on plate crop (if pose model provided) to refine quad
        # 3) STN perspective correction from pose-quad
        pose_refinement = {
            "applied": False,
            "plate_box": None,
            "plate_quad": None,
            "confidence": None,
        }
        if self.pose_detector and self.pose_detector.model is not None and plate_patch is not None and plate_patch.size:
            pose_quad, pose_box, pose_conf = self.pose_detector.predict_quad_and_box(plate_patch)
            if pose_box is not None and pose_quad is not None:
                ph, pw = plate_patch.shape[:2]
                pose_roi = {
                    "vehicle_box": [0, 0, pw, ph],
                    "plate_box": pose_box,
                    "plate_quad": pose_quad,
                }
                refined = perspective_correct_plate(plate_patch, pose_roi)
                if refined is not None and refined.size:
                    rectified_plate = refined
                    pose_refinement = {
                        "applied": True,
                        "plate_box": pose_box,
                        "plate_quad": pose_quad,
                        "confidence": pose_conf,
                    }

        # TrOCR input: non-binarized + upscaled
        if self.ocr.engine in {"tunisia_custom", "paddleocr"}:
            ocr_input = rectified_plate
        else:
            ocr_input = preprocess_plate_for_ocr(rectified_plate)
        ocr_input = upscale_for_ocr(ocr_input, 160, 48)

        # Classification for plate type must come from seg8 detector model (best_8coordonnees.pt).
        classification = {"label": None, "confidence": None}
        if fine_detection.get("source") == "seg8_model":
            classification = {
                "label": fine_detection.get("class_name"),
                "confidence": fine_detection.get("confidence"),
                "source": "seg8_model",
                "class_id": fine_detection.get("class_id"),
            }
        ocr_result = self.ocr.predict(ocr_input)
        normalized_text = normalize_plate_text(ocr_result.get("text"))
        vehicle_pred = self.vehicle_classifier.predict(preprocessed)
        vehicle_class = vehicle_pred.get("label") or vehicle_detection.get("label") or record.get("object", {}).get("class")
        vehicle_conf = vehicle_pred.get("confidence") if vehicle_pred.get("label") else vehicle_detection.get("confidence")
        plate_type_predicted = classification.get("label") or fine_detection.get("class_name")
        plate_type_confidence = classification.get("confidence")
        if plate_type_confidence is None and fine_detection.get("source") == "seg8_model":
            plate_type_confidence = fine_detection.get("confidence")

        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{index:04d}_vehicle.jpg"), preprocessed)
            cv2.imwrite(str(debug_dir / f"{index:04d}_plate_patch.jpg"), plate_patch)
            cv2.imwrite(str(debug_dir / f"{index:04d}_plate_rectified.jpg"), rectified_plate)
            cv2.imwrite(str(debug_dir / f"{index:04d}_plate_ocr_input.jpg"), ocr_input)

        result = {
            "meta": record.get("meta", {}),
            "object_input": record.get("object", {}),
            "roi": roi_runtime,
            "embedded_meta": record.get("embedded_meta", {}),
            "pipeline": {
                "preprocessing": "bilateral_filter",
                "fine_detection": fine_detection,
                "classification": classification,
                "vehicle_detection": {
                    "label": vehicle_class,
                    "confidence": vehicle_conf,
                    "source": (
                        "vehicle_model"
                        if vehicle_pred.get("label")
                        else vehicle_detection.get("source") or "object_input"
                    ),
                    "enabled": vehicle_detection.get("enabled", False),
                    "applied": vehicle_detection.get("applied", False),
                    "global_box": vehicle_detection.get("global_box"),
                    "model_path": vehicle_detection.get("model_path"),
                },
                "option2_pose_refinement": pose_refinement,
                "perspective_correction": {
                    "ready": bool(roi_runtime.get("plate_quad")),
                    "applied": bool(roi_runtime.get("plate_quad")),
                },
                "ocr_transformer": {
                    "engine": ocr_result.get("engine"),
                    "available": ocr_result.get("available"),
                    "text": ocr_result.get("text"),
                    "confidence": ocr_result.get("confidence"),
                },
                "post_processing": {
                    "normalized_text": normalized_text,
                },
            },
            "final_result": {
                "vehicle_class": vehicle_class,
                "vehicle_confidence": vehicle_conf,
                "plate_type_input": record.get("object", {}).get("plate_type"),
                "plate_type_predicted": plate_type_predicted,
                "plate_type_confidence": plate_type_confidence,
                "ocr_text": ocr_result.get("text"),
                "ocr_confidence": ocr_result.get("confidence"),
                "normalized_text": normalized_text,
            },
            "option2_output": {
                "fine_detector": self.fine_detector,
                "plate_box": fine_detection.get("plate_box"),
                "plate_quad": fine_detection.get("plate_quad"),
                "plate_confidence": fine_detection.get("confidence"),
                "plate_type_predicted": plate_type_predicted,
                "ocr_text": ocr_result.get("text"),
                "ocr_confidence": ocr_result.get("confidence"),
            },
        }
        return result

    def process_records(
        self,
        records: List[Dict[str, Any]],
        debug_dir: Optional[Path] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        rows = records[:limit] if limit > 0 else records
        outputs: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            outputs.append(self._process_one(row, index, debug_dir))
        return outputs
