#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference wrapper for the Tunisia-specific OCR model.

Supports a trained Hugging Face VisionEncoderDecoder checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class OCRPrediction:
    text: Optional[str]
    confidence: Optional[float]
    model_name: str
    available: bool


class TunisiaOCR:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path).resolve() if model_path else None
        self.model_ref = model_path
        self.model_name = "tunisia_ocr_custom"
        self.available = False
        self.processor = None
        self.model = None

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            # 1) Preferred: local fine-tuned checkpoint directory.
            if self.model_path and self.model_path.exists():
                ref = str(self.model_path)
                self.model_name = self.model_path.name or self.model_name
            # 2) Fallback: HF model id passed as string (e.g. microsoft/trocr-small-printed).
            elif model_path:
                ref = model_path
                self.model_name = model_path
            # 3) Last fallback: default base TrOCR model.
            else:
                ref = "microsoft/trocr-small-printed"
                self.model_name = ref

            self.processor = TrOCRProcessor.from_pretrained(ref)
            self.model = VisionEncoderDecoderModel.from_pretrained(ref)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False

    def preprocess(self, plate_bgr: np.ndarray) -> np.ndarray:
        if len(plate_bgr.shape) == 2:
            gray = plate_bgr
        else:
            gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        # Keep more visual information for TrOCR (avoid hard binarization).
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)
        return gray

    def predict(self, plate_bgr: np.ndarray, plate_type: Optional[str] = None) -> OCRPrediction:
        _ = plate_type
        processed = self.preprocess(plate_bgr)
        if not self.available or self.processor is None or self.model is None:
            return OCRPrediction(
                text=None,
                confidence=None,
                model_name=self.model_name,
                available=False,
            )

        import torch
        from PIL import Image

        image = Image.fromarray(processed).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=32,
                num_beams=1,
            )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        normalized = "".join(ch for ch in text.upper() if ch.isalnum()) or None
        return OCRPrediction(
            text=normalized,
            confidence=None,
            model_name=self.model_name,
            available=True,
        )
