"""
app.py - Main Streamlit UI for Cell Image Analysis Platform
Run with: streamlit run app/app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import io
from datetime import datetime

# Add parent dir to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cell_detector import analyze_image, load_model
from utils.database import init_db, save_session, save_detections, get_all_sessions
from utils.report_generator import generate_report

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cell Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 4px 0;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }

    .healthy-badge {
        background: #2e7d32; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;
    }
    .abnormal-badge {
        background: #c62828; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;
    }
    .stAlert { border-radius: 8px; }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #1e3c72; border-bottom: 2px solid #2a5298;
        padding-bottom: 4px; margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
init_db()

@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
    return load_model(model_path if os.path.exists(model_path) else None)

model = get_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/microscope.png", width=80)
    st.title("🔬 Cell Analyzer")
    st.markdown("---")

    st.markdown("### Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
                                help="Lower = detect more (but more false positives)")
    use_fallback = st.checkbox(
        "Use OpenCV fallback",
        value=(model is None),
        disabled=(model is None),
        help="Use classic computer vision instead of YOLO"
    )

    st.markdown("---")
    model_status = "✅ Fine-tuned model loaded" if (model is not None and os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
    )) else "⚠️ Using base YOLOv8 (not fine-tuned)"
    st.info(model_status)

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🔬 Analyze Image", "📊 History", "ℹ️ How to Use"])

# ─────────────────────────────────────────────
# ANALYZE IMAGE PAGE
# ─────────────────────────────────────────────
if page == "🔬 Analyze Image":
    st.title("Cell Image Analysis")
    st.markdown("Upload a microscope/cell image to detect, count, and classify cells automatically.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<p class="section-header">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Supported formats: JPG, PNG, BMP, TIFF",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
        )
        session_name = st.text_input("Session Name (optional)", placeholder="e.g. Sample A - 2024-01-15")
        notes = st.text_area("Analyst Notes (optional)", height=80)

        if uploaded_file:
            st.image(uploaded_file, caption="Original Image", use_container_width=True)

    with col2:
        if uploaded_file:
            st.markdown('<p class="section-header">Analysis Results</p>', unsafe_allow_html=True)

            with st.spinner("Analyzing cells..."):
                # Load image
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Run analysis
                annotated_img, detections, summary = analyze_image(
                    image,
                    model=model if not use_fallback else None,
                    conf_threshold=conf_threshold,
                    use_opencv_fallback=use_fallback
                )

            # Show annotated image
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Cells (Green=Healthy, Red=Abnormal)", use_container_width=True)

    # ── Results section (full width) ──
    if uploaded_file and 'summary' in locals():
        st.markdown("---")
        st.markdown("### Detection Summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <h2>{summary['total_cells']}</h2>
                <p>Total Cells</p></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <h2>{summary['healthy_cells']}</h2>
                <p>Healthy Cells</p></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card" style="background: linear-gradient(135deg, #7b1fa2, #ab47bc);">
                <h2>{summary['abnormal_cells']}</h2>
                <p>Abnormal Cells</p></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card" style="background: linear-gradient(135deg, #00695c, #26a69a);">
                <h2>{summary['confidence_avg']:.0%}</h2>
                <p>Avg Confidence</p></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        chart_col, table_col = st.columns([1, 1])

        with chart_col:
            class_counts = summary.get("class_counts", {})
            if class_counts:
                fig = px.pie(
                    names=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    title="Cell Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        with table_col:
            if detections:
                df = pd.DataFrame([{
                    "Class": d.class_name,
                    "Confidence": f"{d.confidence:.2%}",
                    "Status": "Abnormal" if d.is_abnormal else "Healthy",
                    "Area (px²)": f"{d.area:.0f}"
                } for d in detections])
                st.dataframe(df, use_container_width=True, height=280)

        st.markdown("---")
        action_col1, action_col2 = st.columns([1, 1])

        with action_col1:
            if st.button("💾 Save to History", type="primary", use_container_width=True):
                session_data = {
                    "session_name": session_name or f"Session {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "image_filename": uploaded_file.name,
                    "total_cells": summary["total_cells"],
                    "healthy_cells": summary["healthy_cells"],
                    "abnormal_cells": summary["abnormal_cells"],
                    "confidence_avg": summary["confidence_avg"],
                    "model_used": summary["detection_method"],
                    "notes": notes
                }
                session_id = save_session(session_data)

                det_data = [{
                    "cell_class": d.class_name,
                    "confidence": d.confidence,
                    "bbox_x1": d.bbox[0], "bbox_y1": d.bbox[1],
                    "bbox_x2": d.bbox[2], "bbox_y2": d.bbox[3],
                    "area": d.area
                } for d in detections]
                save_detections(session_id, det_data)
                st.success(f"✅ Saved as session #{session_id}")

        with action_col2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    pdf_bytes = generate_report(
                        image_filename=uploaded_file.name,
                        summary=summary,
                        annotated_image=annotated_img,
                        session_name=session_name,
                        notes=notes
                    )
                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=pdf_bytes,
                    file_name=f"cell_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# ─────────────────────────────────────────────
# HISTORY PAGE
# ─────────────────────────────────────────────
elif page == "📊 History":
    st.title("Analysis History")
    sessions = get_all_sessions()

    if not sessions:
        st.info("No analysis sessions saved yet. Run an analysis and click 'Save to History'.")
    else:
        df = pd.DataFrame([{
            "ID": s.id,
            "Session": s.session_name,
            "Image": s.image_filename,
            "Date": s.created_at.strftime("%Y-%m-%d %H:%M"),
            "Total": s.total_cells,
            "Healthy": s.healthy_cells,
            "Abnormal": s.abnormal_cells,
            "Avg Conf": f"{s.confidence_avg:.1%}",
            "Model": s.model_used
        } for s in sessions])

        st.dataframe(df, use_container_width=True)

        # Trend chart
        if len(sessions) > 1:
            st.markdown("### Trends Over Time")
            fig = go.Figure()
            dates = [s.created_at for s in sessions]
            fig.add_trace(go.Scatter(x=dates, y=[s.total_cells for s in sessions], name="Total Cells", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=dates, y=[s.healthy_cells for s in sessions], name="Healthy", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=dates, y=[s.abnormal_cells for s in sessions], name="Abnormal", mode="lines+markers"))
            fig.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# HOW TO USE PAGE
# ─────────────────────────────────────────────
elif page == "ℹ️ How to Use":
    st.title("How to Use")

    st.markdown("""
    ## Quick Start

    ### 1. Upload an Image
    - Go to **Analyze Image** tab
    - Upload any microscope or cell image (JPG, PNG, TIFF, etc.)

    ### 2. Adjust Settings
    - Use the sidebar to set confidence threshold
    - Lower threshold = more detections (but more noise)
    - Enable **OpenCV Fallback** if YOLO model isn't trained yet

    ### 3. View Results
    - Annotated image shows detected cells
    - Green boxes = Healthy | Red boxes = Abnormal
    - See counts, confidence, and class distribution

    ### 4. Save & Export
    - Click **Save to History** to store results in SQLite
    - Click **Generate PDF Report** for a downloadable report

    ---

    ## Training Your Own Model

    ```bash
    # Step 1: Setup Kaggle API (one time)
    pip install kaggle
    # Place your kaggle.json in ~/.kaggle/

    # Step 2: Download dataset
    python models/download_dataset.py

    # Step 3: Train
    python models/train.py --data data/blood_cells/data.yaml --epochs 50

    # Step 4: Copy best weights
    cp runs/detect/cell_detector/weights/best.pt models/best.pt

    # Step 5: Restart the app
    streamlit run app/app.py
    ```

    ---

    ## Dataset
    - **Source**: Kaggle - Blood Cell Detection Dataset (BCCD)
    - **URL**: https://www.kaggle.com/datasets/drakeluo/blood-cell-detection-data-set
    - **Classes**: RBC (Red Blood Cells), WBC (White Blood Cells), Platelets
    - **Images**: ~364 annotated images in YOLO format
    """)
