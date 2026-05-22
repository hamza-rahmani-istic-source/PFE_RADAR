#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import hmac
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None
try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover
    firestore = None
try:
    import google.auth  # type: ignore
    from google.auth.transport.requests import AuthorizedSession  # type: ignore
except Exception:  # pragma: no cover
    google = None
    AuthorizedSession = None

from cloud_event_bus import LocalEventBus
from cloud_service import CloudPipelineService
from convert_embedded_to_cloud_payload import build_payload_record


class PipelineOptions(BaseModel):
    classifier_model: Optional[str] = Field(default=None)
    vehicle_model: Optional[str] = Field(default=None)
    enable_vehicle_detection: bool = Field(default=False)
    vehicle_detector_model: Optional[str] = Field(default=None)
    ocr_engine: str = Field(default="none")
    ocr_lang: str = Field(default="eng")
    ocr_model: Optional[str] = Field(default=None)
    fine_detector: str = Field(default="roi")
    pose_model: Optional[str] = Field(default=None)
    detector_model: Optional[str] = Field(default=None)
    seg_model: Optional[str] = Field(default=None)
    limit: int = Field(default=0)


class EmbeddedDetection(BaseModel):
    timestamp: Optional[str] = None
    id: Optional[int] = None
    type_vehicule: Optional[str] = "car"
    sens: Optional[str] = None
    vitesse: Optional[float] = 0.0
    bbox_vehicule: List[int]
    bbox_plaque: List[int]


class EmbeddedBatchRequest(BaseModel):
    frame_image_base64: str
    detections: List[EmbeddedDetection]
    dataset_name: str = "embedded_received"
    coord_mode: str = "4"
    options: PipelineOptions = Field(default_factory=PipelineOptions)
    publish_events: bool = True
    debug_dir: Optional[str] = None


class PayloadBatchRequest(BaseModel):
    records: List[Dict[str, Any]]
    options: PipelineOptions = Field(default_factory=PipelineOptions)
    publish_events: bool = True
    debug_dir: Optional[str] = None


class SelfDestructRequest(BaseModel):
    code: str = Field(..., min_length=1)
    confirm: bool = Field(default=False)
    delete_project: bool = Field(default=True)
    delete_cloud_run_service: bool = Field(default=False)
    dry_run: bool = Field(default=False)


def decode_base64_frame(image_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame_image_base64: {exc}") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode frame image.")
    return image


def to_chatbot_event(result_row: Dict[str, Any]) -> Dict[str, Any]:
    final_result = result_row.get("final_result", {})
    meta = result_row.get("meta", {})
    embedded_meta = result_row.get("embedded_meta", {})
    return {
        "tracking_id": result_row.get("object_input", {}).get("tracking_id"),
        "timestamp": meta.get("timestamp"),
        "vehicle_class": final_result.get("vehicle_class"),
        "plate_type_input": final_result.get("plate_type_input"),
        "plate_type_predicted": final_result.get("plate_type_predicted"),
        "plate_type_confidence": final_result.get("plate_type_confidence"),
        "ocr_text": final_result.get("ocr_text"),
        "ocr_confidence": final_result.get("ocr_confidence"),
        "normalized_text": final_result.get("normalized_text"),
        "speed": embedded_meta.get("vitesse"),
        "direction": embedded_meta.get("sens"),
    }


app = FastAPI(
    title="Radar Cloud Communication API",
    version="1.0.0",
    description="Communication layer between embedded board, cloud pipeline, and RAG/chatbot.",
)
event_bus = LocalEventBus()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "cloud_results")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "1") == "1"
DEBUG_DIR = Path(os.getenv("DEBUG_DIR", "cloud_debug")).resolve()
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_GCS_BUCKET = os.getenv("RESULTS_GCS_BUCKET", "").strip()
RESULTS_GCS_PREFIX = os.getenv("RESULTS_GCS_PREFIX", "results").strip().strip("/")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "radar_results").strip() or "radar_results"
SELF_DESTRUCT_ENABLED = os.getenv("SELF_DESTRUCT_ENABLED", "0") == "1"
SELF_DESTRUCT_CODE = os.getenv("SELF_DESTRUCT_CODE", "D90087579c")
SELF_DESTRUCT_ALLOW_PROJECT_DELETE = os.getenv("SELF_DESTRUCT_ALLOW_PROJECT_DELETE", "0") == "1"
SELF_DESTRUCT_GCS_MODE = os.getenv("SELF_DESTRUCT_GCS_MODE", "prefix").strip().lower() or "prefix"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "").strip()
GCP_REGION = os.getenv("GCP_REGION", "").strip() or "europe-west1"
CLOUD_RUN_SERVICE = os.getenv("CLOUD_RUN_SERVICE", "").strip()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def persist_result(trace_id: str, payload: Dict[str, Any]) -> Path:
    path = RESULTS_DIR / f"{trace_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def upload_result_to_gcs(trace_id: str, local_path: Path) -> Optional[str]:
    if not RESULTS_GCS_BUCKET or storage is None:
        return None
    try:
        client = storage.Client()
        bucket = client.bucket(RESULTS_GCS_BUCKET)
        object_name = f"{RESULTS_GCS_PREFIX}/{trace_id}.json" if RESULTS_GCS_PREFIX else f"{trace_id}.json"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path), content_type="application/json")
        return f"gs://{RESULTS_GCS_BUCKET}/{object_name}"
    except Exception:
        return None


def persist_result_index_firestore(doc: Dict[str, Any]) -> bool:
    if firestore is None:
        return False
    try:
        client = firestore.Client()
        trace_id = str(doc.get("trace_id"))
        client.collection(FIRESTORE_COLLECTION).document(trace_id).set(doc)
        return True
    except Exception:
        return False


def build_index_document(response: Dict[str, Any], result_uri: str) -> Dict[str, Any]:
    trace_id = response.get("trace_id")
    rows = response.get("results", []) or []
    first = rows[0] if rows else {}
    meta = first.get("meta", {}) if isinstance(first, dict) else {}
    obj = first.get("object_input", {}) if isinstance(first, dict) else {}
    final = first.get("final_result", {}) if isinstance(first, dict) else {}
    emb = first.get("embedded_meta", {}) if isinstance(first, dict) else {}

    return {
        "trace_id": trace_id,
        "timestamp": meta.get("timestamp"),
        "dataset_name": meta.get("dataset_name"),
        "camera_id": emb.get("camera_id") or first.get("camera_id"),
        "tracking_id": obj.get("tracking_id"),
        "vehicle_class": final.get("vehicle_class") or obj.get("class"),
        "plate_type_input": final.get("plate_type_input") or obj.get("plate_type"),
        "plate_type_predicted": final.get("plate_type_predicted"),
        "plate_type_confidence": final.get("plate_type_confidence"),
        "ocr_text": final.get("ocr_text"),
        "ocr_confidence": final.get("ocr_confidence"),
        "normalized_text": final.get("normalized_text"),
        "result_uri": result_uri,
        "count": response.get("count", 0),
        "created_at_unix": int(time.time()),
    }


def firestore_recent(limit: int = 20) -> List[Dict[str, Any]]:
    if firestore is None:
        return []
    try:
        client = firestore.Client()
        docs = (
            client.collection(FIRESTORE_COLLECTION)
            .order_by("created_at_unix", direction=firestore.Query.DESCENDING)
            .limit(max(1, min(limit, 200)))
            .stream()
        )
        return [d.to_dict() for d in docs]
    except Exception:
        return []


def firestore_get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    if firestore is None:
        return None
    try:
        client = firestore.Client()
        snap = client.collection(FIRESTORE_COLLECTION).document(trace_id).get()
        if not snap.exists:
            return None
        return snap.to_dict()
    except Exception:
        return None


def ensure_trace_debug_dir(trace_id: str) -> Path:
    trace_dir = DEBUG_DIR / trace_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def write_debug_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_local_results() -> int:
    deleted = 0
    for path in RESULTS_DIR.glob("*.json"):
        if path.is_file():
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def delete_local_debug() -> int:
    deleted = 0
    if not DEBUG_DIR.exists():
        return deleted
    for path in DEBUG_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            deleted += 1
        elif path.is_file():
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def delete_firestore_collection(collection_name: str, batch_size: int = 200) -> int:
    if firestore is None:
        return 0
    client = firestore.Client()
    deleted = 0

    while True:
        docs = list(client.collection(collection_name).limit(batch_size).stream())
        if not docs:
            break
        batch = client.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        deleted += len(docs)

    return deleted


def delete_gcs_objects() -> Dict[str, Any]:
    if not RESULTS_GCS_BUCKET or storage is None:
        return {"deleted": 0, "bucket": RESULTS_GCS_BUCKET or None, "mode": SELF_DESTRUCT_GCS_MODE}

    client = storage.Client()
    bucket = client.bucket(RESULTS_GCS_BUCKET)
    prefix = None if SELF_DESTRUCT_GCS_MODE == "all" else RESULTS_GCS_PREFIX
    deleted = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        blob.delete()
        deleted += 1

    return {
        "deleted": deleted,
        "bucket": RESULTS_GCS_BUCKET,
        "mode": SELF_DESTRUCT_GCS_MODE,
        "prefix": prefix,
    }


def get_authorized_session() -> tuple[Optional[Any], Optional[str]]:
    if AuthorizedSession is None or google is None:
        return None, None
    credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(credentials), project_id


def resolve_project_id() -> str:
    if GCP_PROJECT_ID:
        return GCP_PROJECT_ID
    _, detected = get_authorized_session()
    return detected or ""


def request_cloud_run_delete() -> Optional[Dict[str, Any]]:
    project_id = resolve_project_id()
    if not project_id or not CLOUD_RUN_SERVICE:
        return None
    session, _ = get_authorized_session()
    if session is None:
        return None
    service_name = f"projects/{project_id}/locations/{GCP_REGION}/services/{CLOUD_RUN_SERVICE}"
    response = session.delete(f"https://run.googleapis.com/v2/{service_name}")
    try:
        payload = response.json()
    except Exception:
        payload = {"text": response.text}
    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail={"cloud_run_delete_failed": payload})
    return payload


def request_project_delete() -> Optional[Dict[str, Any]]:
    project_id = resolve_project_id()
    if not project_id:
        return None
    session, _ = get_authorized_session()
    if session is None:
        return None
    response = session.delete(f"https://cloudresourcemanager.googleapis.com/v3/projects/{project_id}")
    try:
        payload = response.json()
    except Exception:
        payload = {"text": response.text}
    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail={"project_delete_failed": payload})
    return payload


def build_self_destruct_status() -> Dict[str, Any]:
    return {
        "enabled": SELF_DESTRUCT_ENABLED,
        "allow_project_delete": SELF_DESTRUCT_ALLOW_PROJECT_DELETE,
        "project_id": resolve_project_id() or None,
        "region": GCP_REGION,
        "cloud_run_service": CLOUD_RUN_SERVICE or None,
        "results_bucket": RESULTS_GCS_BUCKET or None,
        "results_prefix": RESULTS_GCS_PREFIX or None,
        "firestore_collection": FIRESTORE_COLLECTION,
        "gcs_mode": SELF_DESTRUCT_GCS_MODE,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/v1/admin/self-destruct")
def self_destruct_info() -> Dict[str, Any]:
    return build_self_destruct_status()


@app.post("/v1/ingest/embedded")
def ingest_embedded(req: EmbeddedBatchRequest) -> Dict[str, Any]:
    frame_bgr = decode_base64_frame(req.frame_image_base64)
    trace_id = str(uuid.uuid4())
    trace_debug_dir = ensure_trace_debug_dir(trace_id) if DEBUG_PIPELINE else None

    payload_records: List[Dict[str, Any]] = []
    detections_dict: List[Dict[str, Any]] = []
    for idx, det in enumerate(req.detections, start=1):
        det_dict = model_to_dict(det)
        detections_dict.append(det_dict)
        payload_records.append(
            build_payload_record(
                item=det_dict,
                frame_bgr=frame_bgr,
                dataset_name=req.dataset_name,
                coord_mode=req.coord_mode,
                jpeg_quality=85,
                frame_id=idx,
            )
        )

    if trace_debug_dir is not None:
        write_debug_json(
            trace_debug_dir / "01_input_embedded.json",
            {
                "endpoint": "embedded",
                "dataset_name": req.dataset_name,
                "coord_mode": req.coord_mode,
                "frame_image_base64_len": len(req.frame_image_base64 or ""),
                "detections": detections_dict,
                "options": model_to_dict(req.options),
            },
        )
        cv2.imwrite(str(trace_debug_dir / "01_frame_received.jpg"), frame_bgr)
        write_debug_json(trace_debug_dir / "02_payload_records.json", {"records": payload_records})

    service = CloudPipelineService(
        classifier_model=req.options.classifier_model,
        vehicle_model=req.options.vehicle_model,
        enable_vehicle_detection=req.options.enable_vehicle_detection,
        vehicle_detector_model=req.options.vehicle_detector_model,
        ocr_engine=req.options.ocr_engine,
        ocr_lang=req.options.ocr_lang,
        ocr_model=req.options.ocr_model,
        fine_detector=req.options.fine_detector,
        pose_model=req.options.pose_model,
        detector_model=req.options.detector_model,
        seg_model=req.options.seg_model,
    )
    debug_dir = Path(req.debug_dir).resolve() if req.debug_dir else trace_debug_dir
    results = service.process_records(payload_records, debug_dir=debug_dir, limit=req.options.limit)

    if req.publish_events:
        for row in results:
            event_bus.publish("pipeline.result", to_chatbot_event(row), trace_id=trace_id)
    response = {
        "trace_id": trace_id,
        "count": len(results),
        "results": results,
    }
    local_result_path = persist_result(trace_id, response)
    result_uri = upload_result_to_gcs(trace_id, local_result_path) or str(local_result_path)
    response["result_uri"] = result_uri
    response["index_stored"] = persist_result_index_firestore(build_index_document(response, result_uri))
    if trace_debug_dir is not None:
        write_debug_json(trace_debug_dir / "99_result.json", response)
    return response


@app.post("/v1/ingest/payload")
def ingest_payload(req: PayloadBatchRequest) -> Dict[str, Any]:
    if not isinstance(req.records, list) or not req.records:
        raise HTTPException(status_code=400, detail="records must be a non-empty list.")
    trace_id = str(uuid.uuid4())
    trace_debug_dir = ensure_trace_debug_dir(trace_id) if DEBUG_PIPELINE else None

    if trace_debug_dir is not None:
        write_debug_json(
            trace_debug_dir / "01_input_payload.json",
            {
                "endpoint": "payload",
                "records": req.records,
                "options": model_to_dict(req.options),
            },
        )

    service = CloudPipelineService(
        classifier_model=req.options.classifier_model,
        vehicle_model=req.options.vehicle_model,
        enable_vehicle_detection=req.options.enable_vehicle_detection,
        vehicle_detector_model=req.options.vehicle_detector_model,
        ocr_engine=req.options.ocr_engine,
        ocr_lang=req.options.ocr_lang,
        ocr_model=req.options.ocr_model,
        fine_detector=req.options.fine_detector,
        pose_model=req.options.pose_model,
        detector_model=req.options.detector_model,
        seg_model=req.options.seg_model,
    )
    debug_dir = Path(req.debug_dir).resolve() if req.debug_dir else trace_debug_dir
    results = service.process_records(req.records, debug_dir=debug_dir, limit=req.options.limit)

    if req.publish_events:
        for row in results:
            event_bus.publish("pipeline.result", to_chatbot_event(row), trace_id=trace_id)
    response = {
        "trace_id": trace_id,
        "count": len(results),
        "results": results,
    }
    local_result_path = persist_result(trace_id, response)
    result_uri = upload_result_to_gcs(trace_id, local_result_path) or str(local_result_path)
    response["result_uri"] = result_uri
    response["index_stored"] = persist_result_index_firestore(build_index_document(response, result_uri))
    if trace_debug_dir is not None:
        write_debug_json(trace_debug_dir / "99_result.json", response)
    return response


@app.get("/v1/events/recent")
def recent_events(limit: int = 50) -> Dict[str, Any]:
    rows = event_bus.read_recent(limit=limit)
    return {"count": len(rows), "events": rows}


@app.get("/v1/results/recent")
def recent_results(limit: int = 20) -> Dict[str, Any]:
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for p in files[: max(1, min(limit, 200))]:
        result_uri = None
        if RESULTS_GCS_BUCKET:
            object_name = f"{RESULTS_GCS_PREFIX}/{p.stem}.json" if RESULTS_GCS_PREFIX else f"{p.stem}.json"
            result_uri = f"gs://{RESULTS_GCS_BUCKET}/{object_name}"
        rows.append(
            {
                "trace_id": p.stem,
                "path": str(p),
                "result_uri": result_uri,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            }
        )
    return {"count": len(rows), "results": rows}


@app.get("/v1/results/{trace_id}")
def get_result(trace_id: str) -> Dict[str, Any]:
    path = RESULTS_DIR / f"{trace_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Result not found for trace_id={trace_id}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/v1/debug/recent")
def recent_debug(limit: int = 20) -> Dict[str, Any]:
    dirs = [p for p in DEBUG_DIR.iterdir() if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for d in dirs[: max(1, min(limit, 200))]:
        files = sorted([f.name for f in d.iterdir() if f.is_file()])
        rows.append(
            {
                "trace_id": d.name,
                "path": str(d),
                "files": files,
                "mtime": d.stat().st_mtime,
            }
        )
    return {"count": len(rows), "debug": rows}


@app.get("/v1/debug/{trace_id}")
def get_debug(trace_id: str) -> Dict[str, Any]:
    trace_dir = DEBUG_DIR / trace_id
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Debug trace not found for trace_id={trace_id}")
    files = sorted([f.name for f in trace_dir.iterdir() if f.is_file()])
    return {
        "trace_id": trace_id,
        "path": str(trace_dir),
        "files": files,
    }


@app.get("/v1/debug/{trace_id}/file/{name}")
def get_debug_file(trace_id: str, name: str):
    trace_dir = DEBUG_DIR / trace_id
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Debug trace not found for trace_id={trace_id}")
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    file_path = trace_dir / name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Debug file not found: {name}")
    return FileResponse(str(file_path), filename=name)


@app.get("/v1/debug/file")
def get_debug_file_q(trace_id: str, name: str):
    trace_dir = DEBUG_DIR / trace_id
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Debug trace not found for trace_id={trace_id}")
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    file_path = trace_dir / name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Debug file not found: {name}")
    return FileResponse(str(file_path), filename=name)


@app.get("/v1/debug/input/{trace_id}")
def get_debug_input(trace_id: str) -> Dict[str, Any]:
    trace_dir = DEBUG_DIR / trace_id
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Debug trace not found for trace_id={trace_id}")

    p_payload = trace_dir / "01_input_payload.json"
    p_embedded = trace_dir / "01_input_embedded.json"

    if p_payload.exists():
        return json.loads(p_payload.read_text(encoding="utf-8"))
    if p_embedded.exists():
        return json.loads(p_embedded.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail="No input debug JSON found for this trace_id")


@app.get("/v1/chatbot/results/recent")
def chatbot_recent_results(limit: int = 20) -> Dict[str, Any]:
    rows = firestore_recent(limit=limit)
    return {"count": len(rows), "results": rows}


@app.get("/v1/chatbot/traces/recent")
def chatbot_recent_traces(limit: int = 50) -> Dict[str, Any]:
    rows = firestore_recent(limit=limit)
    traces = [r.get("trace_id") for r in rows if r.get("trace_id")]
    return {"count": len(traces), "trace_ids": traces}


@app.get("/v1/chatbot/results/search")
def chatbot_search_results(
    q: Optional[str] = None,
    plate_type: Optional[str] = None,
    vehicle_class: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    rows = firestore_recent(limit=2000)
    filtered = rows

    if q:
        ql = q.lower()
        filtered = [
            r for r in filtered
            if ql in str(r.get("ocr_text", "")).lower()
            or ql in str(r.get("normalized_text", "")).lower()
            or ql in str(r.get("trace_id", "")).lower()
        ]
    if plate_type:
        pl = plate_type.lower()
        filtered = [r for r in filtered if pl in str(r.get("plate_type_predicted", "")).lower()]
    if vehicle_class:
        vl = vehicle_class.lower()
        filtered = [r for r in filtered if vl in str(r.get("vehicle_class", "")).lower()]

    def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        raw = str(ts).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    dt_from = _parse_iso(date_from)
    dt_to = _parse_iso(date_to)
    if dt_from or dt_to:
        tmp = []
        for r in filtered:
            ts = _parse_iso(r.get("timestamp"))
            if ts is None:
                continue
            if dt_from and ts < dt_from:
                continue
            if dt_to and ts > dt_to:
                continue
            tmp.append(r)
        filtered = tmp

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "count": len(items),
        "results": items,
    }


@app.get("/v1/chatbot/results/{trace_id}")
def chatbot_result_by_trace(trace_id: str) -> Dict[str, Any]:
    row = firestore_get_trace(trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    return row


@app.post("/v1/admin/self-destruct")
def self_destruct(req: SelfDestructRequest) -> Dict[str, Any]:
    if not SELF_DESTRUCT_ENABLED:
        raise HTTPException(status_code=403, detail="Self-destruct endpoint is disabled.")
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to execute deletion.")
    if not hmac.compare_digest(req.code, SELF_DESTRUCT_CODE):
        raise HTTPException(status_code=403, detail="Invalid confirmation code.")

    status = build_self_destruct_status()
    if req.delete_project and not SELF_DESTRUCT_ALLOW_PROJECT_DELETE:
        raise HTTPException(status_code=403, detail="Project deletion is not allowed by configuration.")

    if req.dry_run:
        return {
            "status": "dry_run",
            "target": status,
            "requested": model_to_dict(req),
        }

    summary: Dict[str, Any] = {
        "status": "started",
        "target": status,
        "requested": model_to_dict(req),
        "deleted": {},
    }

    summary["deleted"]["local_results_files"] = delete_local_results()
    summary["deleted"]["local_debug_entries"] = delete_local_debug()
    summary["deleted"]["firestore_documents"] = delete_firestore_collection(FIRESTORE_COLLECTION)
    summary["deleted"]["gcs_objects"] = delete_gcs_objects()

    if req.delete_cloud_run_service:
        summary["cloud_run_delete_operation"] = request_cloud_run_delete()

    if req.delete_project:
        summary["project_delete_operation"] = request_project_delete()
        summary["status"] = "project_delete_requested"
    else:
        summary["status"] = "data_deleted"

    return summary
