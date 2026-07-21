"""
FastAPI layer for BioVision-AI — matched to the real interface used in app/app.py:

    utils.cell_detector  -> analyze_image(image, model, conf_threshold, use_opencv_fallback)
                              returns (annotated_img, detections, summary)
    utils.database        -> init_db(), save_session(dict) -> session_id,
                              save_detections(session_id, list[dict]), get_all_sessions()
    utils.report_generator -> generate_report(image_filename, summary, annotated_image,
                                               session_name, notes) -> pdf bytes

PLACE THIS FILE AT: api/main.py

RUN WITH (from the repo root, same venv you use for streamlit):
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs
"""

import os
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cell_detector import analyze_image, load_model
from utils.database import init_db, save_session, save_detections, get_all_sessions
from utils.report_generator import generate_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

app = FastAPI(
    title="BioVision-AI API",
    description="REST API for microscopic cell detection, history, and reporting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- init on startup, same as app.py does at import time ---
init_db()
model = load_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)


# ---------- Response models ----------

class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    is_abnormal: bool
    bbox: List[float]
    area: float


class AnalyzeResponse(BaseModel):
    session_id: int
    session_name: str
    image_filename: str
    total_cells: int
    healthy_cells: int
    abnormal_cells: int
    confidence_avg: float
    class_counts: dict
    detection_method: str
    detections: List[DetectionOut]


class HistoryItem(BaseModel):
    id: int
    session_name: str
    image_filename: str
    created_at: str
    total_cells: int
    healthy_cells: int
    abnormal_cells: int
    confidence_avg: float
    model_used: str


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "model_loaded": model is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    session_name: Optional[str] = None,
    notes: Optional[str] = "",
    conf_threshold: float = 0.3,
    use_opencv_fallback: bool = False,
):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    raw = await file.read()
    file_bytes = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        annotated_img, detections, summary = analyze_image(
            image,
            model=None if use_opencv_fallback else model,
            conf_threshold=conf_threshold,
            use_opencv_fallback=use_opencv_fallback or (model is None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    final_session_name = session_name or f"Session {datetime.now().strftime('%Y%m%d_%H%M%S')}"

    session_id = save_session({
        "session_name": final_session_name,
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

    # Cache the PDF now, since generating it later would need the annotated image
    # again, and history doesn't currently store that image.
    pdf_bytes = generate_report(
        image_filename=file.filename,
        summary=summary,
        annotated_image=annotated_img,
        session_name=final_session_name,
        notes=notes,
    )
    with open(os.path.join(REPORT_DIR, f"{session_id}.pdf"), "wb") as f:
        f.write(pdf_bytes)

    return {
        "session_id": session_id,
        "session_name": final_session_name,
        "image_filename": file.filename,
        "total_cells": summary["total_cells"],
        "healthy_cells": summary["healthy_cells"],
        "abnormal_cells": summary["abnormal_cells"],
        "confidence_avg": summary["confidence_avg"],
        "class_counts": summary.get("class_counts", {}),
        "detection_method": summary["detection_method"],
        "detections": [{
            "class_name": d.class_name,
            "confidence": d.confidence,
            "is_abnormal": d.is_abnormal,
            "bbox": list(d.bbox),
            "area": d.area,
        } for d in detections],
    }


@app.get("/history", response_model=List[HistoryItem])
def history():
    sessions = get_all_sessions()
    return [{
        "id": s.id,
        "session_name": s.session_name,
        "image_filename": s.image_filename,
        "created_at": s.created_at.isoformat(),
        "total_cells": s.total_cells,
        "healthy_cells": s.healthy_cells,
        "abnormal_cells": s.abnormal_cells,
        "confidence_avg": s.confidence_avg,
        "model_used": s.model_used,
    } for s in sessions]


@app.get("/history/{session_id}", response_model=HistoryItem)
def history_item(session_id: int):
    sessions = get_all_sessions()
    match = next((s for s in sessions if s.id == session_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": match.id,
        "session_name": match.session_name,
        "image_filename": match.image_filename,
        "created_at": match.created_at.isoformat(),
        "total_cells": match.total_cells,
        "healthy_cells": match.healthy_cells,
        "abnormal_cells": match.abnormal_cells,
        "confidence_avg": match.confidence_avg,
        "model_used": match.model_used,
    }


@app.get("/report/{session_id}")
def report(session_id: int):
    """
    Serves the PDF cached at analyze-time. If it's missing (e.g. session
    created before this endpoint existed), there's no way to regenerate it
    without the original image, so this returns 404.
    """
    path = os.path.join(REPORT_DIR, f"{session_id}.pdf")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found for this session")
    return FileResponse(path, media_type="application/pdf", filename=f"report_{session_id}.pdf")