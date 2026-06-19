"""
database.py - SQLite database setup using SQLAlchemy
Stores analysis sessions, results, and cell counts
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cell_analysis.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(255), nullable=False)
    image_filename = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_cells = Column(Integer, default=0)
    healthy_cells = Column(Integer, default=0)
    abnormal_cells = Column(Integer, default=0)
    confidence_avg = Column(Float, default=0.0)
    model_used = Column(String(100), default="YOLOv8")
    notes = Column(Text, nullable=True)


class CellDetection(Base):
    __tablename__ = "cell_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    cell_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    area = Column(Float)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_session(session_data: dict) -> int:
    """Save an analysis session and return its ID."""
    db = SessionLocal()
    try:
        session = AnalysisSession(**session_data)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session.id
    finally:
        db.close()


def save_detections(session_id: int, detections: list):
    """Save individual cell detections for a session."""
    db = SessionLocal()
    try:
        for det in detections:
            d = CellDetection(session_id=session_id, **det)
            db.add(d)
        db.commit()
    finally:
        db.close()


def get_all_sessions():
    """Retrieve all analysis sessions."""
    db = SessionLocal()
    try:
        return db.query(AnalysisSession).order_by(AnalysisSession.created_at.desc()).all()
    finally:
        db.close()


def get_session_detections(session_id: int):
    """Retrieve all detections for a session."""
    db = SessionLocal()
    try:
        return db.query(CellDetection).filter(CellDetection.session_id == session_id).all()
    finally:
        db.close()
