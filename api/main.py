"""
api/main.py - FastAPI REST layer for the Cell Image Analysis Platform
Run with: uvicorn api.main:app --reload --port 8000

Wraps the same detection, database, and report-generation logic in
utils/ that app/app.py (the Streamlit UI) uses. Both entry points
share one SQLite DB (data/cell_analysis.db) and one model loader.
"""

import os
import io
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from utils.cell_detector import analyze_image, load_model
from utils.database import (
    init_db,
    save_session,
    save_detections,
    get_all_sessions,
    get_session_detections,
)
from utils.report_generator import generate_report

app = FastAPI(
    title="BioVision AI - Cell Analysis API",
    description="REST API for automated cell detection, counting, and classification "
                 "in microscope images.",
    version="1.0.0",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt"
)

model = None


@app.on_event("startup")
def startup():
    """Initialize the DB and load the detection model once, at process start."""
    global model
    init_db()
    model = load_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)


def _session_to_dict(session) -> dict:
    return {
        "id": session.id,
        "session_name": session.session_name,
        "image_filename": session.image_filename,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "total_cells": session.total_cells,
        "healthy_cells": session.healthy_cells,
        "abnormal_cells": session.abnormal_cells,
        "confidence_avg": session.confidence_avg,
        "model_used": session.model_used,
        "notes": session.notes,
    }


@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), session_name: str = "", notes: str = ""):
    """Upload an image, run detection, save the session to history, return the result."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    annotated, detections, summary = analyze_image(
        image, model=model, conf_threshold=0.3, use_opencv_fallback=(model is None)
    )

    session_id = save_session({
        "session_name": session_name or file.filename,
        "image_filename": file.filename,
        "total_cells": summary["total_cells"],
        "healthy_cells": summary["healthy_cells"],
        "abnormal_cells": summary["abnormal_cells"],
        "confidence_avg": summary["confidence_avg"],
        "model_used": summary["detection_method"],
        "notes": notes,
    })

    save_detections(session_id, [{
        "cell_class": d.class_name,
        "confidence": d.confidence,
        "bbox_x1": d.bbox[0], "bbox_y1": d.bbox[1],
        "bbox_x2": d.bbox[2], "bbox_y2": d.bbox[3],
        "area": d.area,
    } for d in detections])

    return {
        "session_id": session_id,
        "summary": summary,
        "detections": [{
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox": d.bbox,
            "is_abnormal": d.is_abnormal,
            "area": d.area,
        } for d in detections],
    }


@app.get("/history")
def history():
    """List all past analysis sessions, most recent first."""
    sessions = get_all_sessions()
    return [_session_to_dict(s) for s in sessions]


@app.get("/history/{session_id}")
def history_detail(session_id: int):
    """Fetch one past session plus its individual cell detections."""
    sessions = get_all_sessions()
    session = next((s for s in sessions if s.id == session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    detections = get_session_detections(session_id)
    return {
        **_session_to_dict(session),
        "detections": [{
            "cell_class": d.cell_class,
            "confidence": d.confidence,
            "bbox": [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2],
            "area": d.area,
        } for d in detections],
    }


@app.get("/report/{session_id}")
def report(session_id: int):
    """
    Regenerate and download the PDF report for a session.

    Note: the annotated image itself isn't stored in the DB, only
    detection metadata - so this re-renders a report from the saved
    summary/detections rather than re-running inference on the
    original image (which isn't persisted either).
    """
    sessions = get_all_sessions()
    session = next((s for s in sessions if s.id == session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    detections = get_session_detections(session_id)
    class_counts = {}
    for d in detections:
        class_counts[d.cell_class] = class_counts.get(d.cell_class, 0) + 1

    summary = {
        "total_cells": session.total_cells,
        "healthy_cells": session.healthy_cells,
        "abnormal_cells": session.abnormal_cells,
        "class_counts": class_counts,
        "confidence_avg": session.confidence_avg,
        "detection_method": session.model_used,
    }

    # Blank canvas placeholder: the original/annotated image isn't
    # persisted in the DB (see note above), so the PDF's image
    # section is a blank frame. Persist annotated images to disk in
    # /analyze if you need this section to show real image content.
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)

    pdf_bytes = generate_report(
        image_filename=session.image_filename,
        summary=summary,
        annotated_image=blank,
        session_name=session.session_name,
        notes=session.notes or "",
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="cell_report_session_{session_id}.pdf"'
        },
    )
