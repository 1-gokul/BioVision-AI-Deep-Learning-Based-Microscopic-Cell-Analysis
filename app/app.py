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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cell_detector import analyze_image, load_model
from utils.database import init_db, save_session, save_detections, get_all_sessions
from utils.report_generator import generate_report

st.set_page_config(
    page_title="Cell Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stAlert { border-radius: 8px; }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #1e3c72; border-bottom: 2px solid #2a5298;
        padding-bottom: 4px; margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

init_db()

@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
    return load_model(model_path if os.path.exists(model_path) else None)

model = get_model()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/microscope.png", width=80)
    st.title("🔬 Cell Analyzer")
    st.markdown("---")
    st.markdown("### Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
                                help="Lower = detect more (but more false positives)")
    use_fallback = False
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Demo", "🔬 Analyze Image", "📊 History", "ℹ️ How to Use"])


# ─────────────────────────────────────────────
# DEMO PAGE
# ─────────────────────────────────────────────
if page == "🏠 Demo":
    st.title("🔬 BioVision AI — Cell Analysis Platform")
    st.markdown("Automated cell detection and classification from microscope images using YOLOv8.")
    st.markdown("---")

    st.markdown("### Upload Your Own Image")
    st.markdown("JPG, PNG, TIFF — any blood smear or microscope image")

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
        label_visibility="collapsed"
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_up, detections_up, summary_up = analyze_image(
            image, model=model, conf_threshold=conf_threshold,
            use_opencv_fallback=(model is None)
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(uploaded, caption="Original", use_container_width=True)
        with c2:
            st.image(cv2.cvtColor(annotated_up, cv2.COLOR_BGR2RGB),
                     caption="Detected Cells", use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cells", summary_up["total_cells"])
        m2.metric("Healthy", summary_up["healthy_cells"])
        m3.metric("Abnormal", summary_up["abnormal_cells"])
        m4.metric("Avg Confidence", f"{summary_up['confidence_avg']:.0%}")

        st.markdown("<br>", unsafe_allow_html=True)

        if summary_up.get("class_counts"):
            ch1, ch2 = st.columns(2)
            with ch1:
                fig = px.pie(
                    names=list(summary_up["class_counts"].keys()),
                    values=list(summary_up["class_counts"].values()),
                    title="Cell Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=260, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            with ch2:
                if detections_up:
                    df = pd.DataFrame([{
                        "Class": d.class_name,
                        "Confidence": f"{d.confidence:.2%}",
                        "Status": "Abnormal" if d.is_abnormal else "Healthy",
                    } for d in detections_up])
                    st.dataframe(df, use_container_width=True, height=260)

        st.markdown("<br>", unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("💾 Save to History", type="primary", use_container_width=True):
                sid = save_session({
                    "session_name": uploaded.name,
                    "image_filename": uploaded.name,
                    "total_cells": summary_up["total_cells"],
                    "healthy_cells": summary_up["healthy_cells"],
                    "abnormal_cells": summary_up["abnormal_cells"],
                    "confidence_avg": summary_up["confidence_avg"],
                    "model_used": summary_up["detection_method"],
                    "notes": ""
                })
                save_detections(sid, [{
                    "cell_class": d.class_name,
                    "confidence": d.confidence,
                    "bbox_x1": d.bbox[0], "bbox_y1": d.bbox[1],
                    "bbox_x2": d.bbox[2], "bbox_y2": d.bbox[3],
                    "area": d.area
                } for d in detections_up])
                st.success(f"✅ Saved as session #{sid}")

        with btn2:
            pdf_bytes = generate_report(
                image_filename=uploaded.name,
                summary=summary_up,
                annotated_image=annotated_up,
                session_name=uploaded.name,
                notes=""
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"cell_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown("---")

    st.markdown("### Or Explore Pre-Analyzed Samples")
    st.caption("No upload needed — see how the model performs on real blood smear images")

    SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
    sample_files = []
    if os.path.exists(SAMPLES_DIR):
        sample_files = [f for f in os.listdir(SAMPLES_DIR)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not sample_files:
        st.info("Add JPG images to the samples/ folder to show pre-analyzed results here.")
    else:
        @st.cache_data
        def get_sample_results(filename):
            path = os.path.join(SAMPLES_DIR, filename)
            img = cv2.imread(path)
            ann, dets, summ = analyze_image(
                img, model=model, conf_threshold=0.3,
                use_opencv_fallback=(model is None)
            )
            pdf = generate_report(
                image_filename=filename,
                summary=summ,
                annotated_image=ann,
                session_name=f"Sample - {filename}",
                notes="Pre-analyzed sample image"
            )
            return ann, dets, summ, pdf

        selected = st.radio(
            "Pick a sample:",
            sample_files,
            horizontal=True,
            label_visibility="collapsed"
        )
        ann, dets, summ, sample_pdf = get_sample_results(selected)

        s1, s2 = st.columns(2)
        with s1:
            st.image(os.path.join(SAMPLES_DIR, selected),
                     caption="Original", use_container_width=True)
        with s2:
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                     caption="Detected", use_container_width=True)

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Cells", summ["total_cells"])
        sm2.metric("Healthy", summ["healthy_cells"])
        sm3.metric("Abnormal", summ["abnormal_cells"])
        sm4.metric("Avg Confidence", f"{summ['confidence_avg']:.0%}")

        if summ.get("class_counts"):
            sc1, sc2 = st.columns(2)
            with sc1:
                fig = px.pie(
                    names=list(summ["class_counts"].keys()),
                    values=list(summ["class_counts"].values()),
                    title="Cell Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=260, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            with sc2:
                if dets:
                    df_s = pd.DataFrame([{
                        "Class": d.class_name,
                        "Confidence": f"{d.confidence:.2%}",
                        "Status": "Abnormal" if d.is_abnormal else "Healthy",
                    } for d in dets])
                    st.dataframe(df_s, use_container_width=True, height=260)

        st.markdown("<br>", unsafe_allow_html=True)

        st.download_button(
            label="📄 Download Sample Report",
            data=sample_pdf,
            file_name=f"sample_report_{selected}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ─────────────────────────────────────────────
# ANALYZE IMAGE PAGE
# ─────────────────────────────────────────────
elif page == "🔬 Analyze Image":
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
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                annotated_img, detections, summary = analyze_image(
                    image,
                    model=model if not use_fallback else None,
                    conf_threshold=conf_threshold,
                    use_opencv_fallback=use_fallback
                )

            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Cells (Green=Healthy, Red=Abnormal)", use_container_width=True)

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
            pdf_bytes = generate_report(
                image_filename=uploaded_file.name,
                summary=summary,
                annotated_image=annotated_img,
                session_name=session_name,
                notes=notes
            )
            st.download_button(
                label="📄 Download PDF Report",
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
    - Go to **Demo** or **Analyze Image** tab
    - Upload any microscope or cell image (JPG, PNG, TIFF)

    ### 2. View Results
    - Annotated image shows detected cells
    - Green boxes = Healthy | Red boxes = Abnormal
    - See counts, confidence, and class distribution

    ### 3. Save & Export
    - Click **Save to History** to store results in SQLite
    - Click **Download PDF Report** for a downloadable report

    ---

    ## Dataset
    - **Source**: Kaggle - Blood Cell Detection Dataset (BCCD)
    - **URL**: https://www.kaggle.com/datasets/drakeluo/blood-cell-detection-data-set
    - **Classes**: RBC, WBC, Platelets
    """)