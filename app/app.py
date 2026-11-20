# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import shutil
import urllib.request
import ctypes

# Step 1: Streamlit Cloud Linux hotfix for Debian Bookworm glib/gthread (no-op on Windows/macOS)
if sys.platform.startswith("linux"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(current_dir, "libs")
    os.makedirs(libs_dir, exist_ok=True)

    gthread_path = os.path.join(libs_dir, "libgthread-2.0.so.0")
    glib_path = os.path.join(libs_dir, "libglib-2.0.so.0")
    sentinel_path = os.path.join(libs_dir, ".bookworm_extracted")

    if not os.path.exists(sentinel_path):
        deb_url = "http://ftp.debian.org/debian/pool/main/g/glib2.0/libglib2.0-0_2.74.6-2+deb12u9_amd64.deb"
        deb_path = "/tmp/libglib_bookworm.deb"
        extract_dir = "/tmp/libglib_bookworm_extracted"
        try:
            urllib.request.urlretrieve(deb_url, deb_path)
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.run(["dpkg-deb", "-x", deb_path, extract_dir], check=True)
            src_dir = os.path.join(extract_dir, "usr", "lib", "x86_64-linux-gnu")
            for filename in os.listdir(src_dir):
                if "libgthread-2.0.so" in filename or "libglib-2.0.so" in filename:
                    shutil.copy2(os.path.join(src_dir, filename), os.path.join(libs_dir, filename))
            with open(sentinel_path, "w") as sf:
                sf.write("extracted")
        except Exception as e:
            print(f"[HOTFIX] Skipping Bookworm lib extraction: {e}")

    try:
        if os.path.exists(glib_path):
            ctypes.CDLL(glib_path, mode=ctypes.RTLD_GLOBAL)
        if os.path.exists(gthread_path):
            ctypes.CDLL(gthread_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"[HOTFIX] Skipping ctypes preload: {e}")

# Step 2: Set headless Matplotlib backend before pyplot import
import matplotlib
matplotlib.use("Agg")

# Step 3: Self-healing cv2 import
try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
    sys.modules.pop("cv2", None)
    import cv2

import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import streamlit as st

# Append project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.models.hybrid_model import HybridDRPredictor
from src.preprocess import apply_ben_graham_preprocessing
from app.report_generator import generate_pdf_report
from app.knowledge_graph import (
    DR_KNOWLEDGE_BASE,
    compute_systemic_risk_profile,
    evaluate_clinical_safeguards,
    render_knowledge_graph_svg,
)
from app.rag_engine import (
    CLINICAL_GUIDELINE_CORPUS,
    REFERENCE_RETINAL_CASES,
    RetinalVisualVectorDB,
    ClinicalGuidelineRAGEngine,
    render_rag_visual_evidence_card,
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="RetinaRAG Clinical Workstation | Diabetic Retinopathy Screening & PredictRx",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clean_html(html_str: str) -> str:
    return " ".join(line.strip() for line in html_str.split("\n") if line.strip())


def local_css(file_name: str):
    css_path = Path(__file__).resolve().parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")


def st_image_compatible(image, caption=None):
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def adjust_image(image: Image.Image, brightness: float, contrast: float, sharpness: float) -> Image.Image:
    img = ImageEnhance.Brightness(image).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def reset_sliders():
    st.session_state.fundus_brightness = 1.0
    st.session_state.fundus_contrast = 1.0
    st.session_state.fundus_sharpness = 1.0


# ----------------- RETINAL FUNDUS OOD VALIDATION SAFEGUARD -----------------
def validate_retinal_fundus(image: Image.Image):
    """
    Out-of-Distribution (OOD) & Image Quality Safeguard for Retinal Fundus Photographs.
    """
    img_rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(img_rgb, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    img_std = float(np.std(gray))
    if img_std < 12.0:
        return (
            False,
            "Image contrast is too flat to be a clinical retinal fundus photograph.",
            {"std": img_std},
        )

    fg_mask = gray > 15
    fg_ratio = float(np.mean(fg_mask))
    if fg_ratio < 0.15:
        return (
            False,
            "Image is predominantly dark/underexposed — retinal field not detected.",
            {"std": img_std, "fg_ratio": fg_ratio},
        )

    fg_pixels = resized[fg_mask]
    r_mean = float(np.mean(fg_pixels[:, 0]))
    g_mean = float(np.mean(fg_pixels[:, 1]))
    b_mean = float(np.mean(fg_pixels[:, 2]))

    channel_diff = float(
        np.mean(np.abs(fg_pixels[:, 0].astype(np.float32) - fg_pixels[:, 1].astype(np.float32)))
        + np.mean(np.abs(fg_pixels[:, 0].astype(np.float32) - fg_pixels[:, 2].astype(np.float32)))
    )
    if channel_diff < 8.0:
        return (
            False,
            "Grayscale / monochrome image detected. Please upload a color RGB retinal fundus photograph.",
            {"std": img_std, "channel_diff": channel_diff},
        )

    if r_mean < b_mean * 1.05:
        return (
            False,
            "Color profile does not match a retinal fundus photograph (expected warm red/orange retinal spectrum).",
            {"r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean},
        )

    lap_var = float(cv2.Laplacian(resized[:, :, 1], cv2.CV_64F).var())
    if lap_var < 4.0:
        return (
            False,
            "Image appears severely blurred or out of focus. Please recapture a sharp fundus scan.",
            {"lap_var": lap_var},
        )

    return (
        True,
        "Valid color retinal fundus photograph verified.",
        {
            "std": round(img_std, 1),
            "fg_ratio": round(fg_ratio * 100, 1),
            "lap_var": round(lap_var, 1),
            "r_b_ratio": round(r_mean / (b_mean + 1e-5), 2),
        },
    )


# ----------------- CACHED MODEL & VECTOR DB LOADERS -----------------
@st.cache_resource
def load_trained_model(backbone: str = "efficientnet_b0"):
    weights_name = f"best_dr_model_{backbone}.pth"
    weights_path = base_dir / "models" / weights_name

    loaded = False
    if weights_path.exists():
        model = HybridDRPredictor(backbone=backbone, num_classes=5, pretrained=False)
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
            loaded = True
        except Exception as e:
            print(f"[WARN] Could not load {weights_path}: {e}")
            model = HybridDRPredictor(backbone=backbone, num_classes=5, pretrained=True)
    else:
        model = HybridDRPredictor(backbone=backbone, num_classes=5, pretrained=True)

    model.eval()
    return model, loaded


@st.cache_resource
def load_rag_and_vector_dbs():
    visual_vdb = RetinalVisualVectorDB(embedding_dim=256)
    clinical_rag = ClinicalGuidelineRAGEngine()
    return visual_vdb, clinical_rag


visual_vdb, clinical_rag = load_rag_and_vector_dbs()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------- SESSION STATE -----------------
if "session_total_scans" not in st.session_state:
    st.session_state.session_total_scans = 0
if "diagnoses_history" not in st.session_state:
    st.session_state.diagnoses_history = []
if "session_nodr" not in st.session_state:
    st.session_state.session_nodr = 0
if "session_referable" not in st.session_state:
    st.session_state.session_referable = 0
if "session_vtdr" not in st.session_state:
    st.session_state.session_vtdr = 0
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_rag_state" not in st.session_state:
    st.session_state.last_rag_state = None


# ----------------- SIDEBAR (CLEAN CLINICAL WORKSTATION SPEC) -----------------
with st.sidebar:
    st.markdown(
        clean_html(f"""
            <div style="padding: 0.25rem 0 1rem 0; border-bottom: 1px solid #1f2937; margin-bottom: 1rem;">
                <div class="sidebar-label" style="margin-bottom:0.2rem; color:#60a5fa;">HEALTHSPHERE AI • PREDICTRX</div>
                <div style="font-size:1.02rem; font-weight:700; color:#f9fafb; letter-spacing:-0.01em;">RetinaRAG Workstation</div>
                <div style="font-size:0.76rem; color:#9ca3af; margin-top:2px;">AI Builders Congress — Competition 1</div>
            </div>

            <div class="sidebar-section">
                <span class="sidebar-label">System Architecture</span>
                <div class="sidebar-title">Hybrid CNN-CBAM + Dual RAG</div>
                <p class="sidebar-text">
                    256-D retinal feature embeddings coupled with local Vector Database retrieval ({len(REFERENCE_RETINAL_CASES)} reference prototypes &amp; {len(CLINICAL_GUIDELINE_CORPUS)} clinical guideline documents).
                </p>
            </div>

            <div class="sidebar-section">
                <span class="sidebar-label">ICDR Severity Scale</span>
                <div style="font-size:0.78rem; line-height:1.75; color:#d1d5db;">
                    <div style="display:flex; justify-content:space-between;"><span><b>Grade 0</b> No DR</span><span style="color:#34d399;">Routine</span></div>
                    <div style="display:flex; justify-content:space-between;"><span><b>Grade 1</b> Mild NPDR</span><span style="color:#60a5fa;">6–12 mo</span></div>
                    <div style="display:flex; justify-content:space-between;"><span><b>Grade 2</b> Moderate NPDR</span><span style="color:#fbbf24;">Referable</span></div>
                    <div style="display:flex; justify-content:space-between;"><span><b>Grade 3</b> Severe NPDR</span><span style="color:#fb923c;">Urgent VTDR</span></div>
                    <div style="display:flex; justify-content:space-between;"><span><b>Grade 4</b> Proliferative DR</span><span style="color:#f87171;">Immediate</span></div>
                </div>
            </div>

            <div class="sidebar-section">
                <span class="sidebar-label">Documented Safeguards</span>
                <div class="sidebar-text" style="line-height:1.65;">
                    • Pre-inference Fundus OOD &amp; blur check<br>
                    • Low-confidence (&lt;70%) HITL escalation<br>
                    • Inline clinical citations ([ICDR], [ADA])<br>
                    • Zero-PHI local in-memory execution
                </div>
            </div>
        """),
        unsafe_allow_html=True,
    )


# ----------------- TAB 2: RAG & KNOWLEDGE GRAPH EXPLORER -----------------
def render_rag_and_knowledge_graph_tab():
    st.markdown(
        clean_html("""
            <div style="margin-bottom: 1.25rem;">
                <h3 style="color: #f9fafb; font-weight: 600; font-size: 1.15rem; margin-bottom: 0.25rem;">Clinical Vector Database, RAG Q&amp;A &amp; Knowledge Graph</h3>
                <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">
                    Query the local ophthalmology &amp; pharmacovigilance Vector Database directly, inspect indexed medical evidence chunks, and review the Retinal-Systemic Knowledge Graph.
                </p>
            </div>
        """),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        sel_col1, sel_col2, sel_col3, sel_col4 = st.columns([1.4, 1, 1, 1])
        with sel_col1:
            preview_grade = st.selectbox(
                "ICDR Severity Stage Context",
                options=[0, 1, 2, 3, 4],
                format_func=lambda g: f"Grade {g}: {DR_KNOWLEDGE_BASE[g]['name']}",
                index=2,
                key="kg_preview_grade",
            )
        with sel_col2:
            sim_hba1c = st.number_input("Simulated HbA1c (%)", min_value=4.0, max_value=16.0, value=8.5, step=0.1, key="kg_hba1c")
        with sel_col3:
            sim_years = st.number_input("Diabetes Duration (yrs)", min_value=0, max_value=50, value=10, step=1, key="kg_years")
        with sel_col4:
            sim_sbp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=220, value=145, step=2, key="kg_sbp")

        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            custom_q = st.text_input(
                "Clinical, Triage, or Medication Safeguard Query",
                value="What medication safeguards apply when lowering HbA1c rapidly or treating macular edema risk?",
                key="rag_explorer_query",
            )
        with q_col2:
            top_k_docs = st.selectbox("Top-K Vector Chunks", [2, 3, 4, 5], index=1, key="rag_explorer_topk")

        qa_res = clinical_rag.answer_clinician_question(
            question=custom_q,
            grade=preview_grade,
            grade_name=DR_KNOWLEDGE_BASE[preview_grade]["name"],
            hba1c=sim_hba1c,
            systolic_bp=sim_sbp,
        )
        st.markdown(
            clean_html(f"""
                <div class="sub-card" style="border-left: 3px solid #6b7280; margin-top: 0.4rem;">
                    <div style="font-size:0.74rem; font-weight:600; color:#b4bac6; margin-bottom:4px;">{qa_res['mode']}</div>
                    <div style="font-size:0.86rem; color:#e5e8ee; line-height:1.55;">{qa_res['answer']}</div>
                </div>
            """),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='font-size:0.86rem; font-weight:600; color:#d1d5db; margin: 1rem 0 0.5rem 0;'>Retrieved Vector Database Guideline Chunks</div>", unsafe_allow_html=True)
    for chunk in qa_res["chunks"][:top_k_docs]:
        st.markdown(
            clean_html(f"""
                <div class="report-card" style="margin-top:0.5rem; padding:0.9rem 1.1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="color:#f3f4f6; font-size:0.86rem;">[{chunk['doc_id']}] {chunk['title']}</strong>
                        <span class="xai-badge">Cosine: {chunk['relevance_score']:.3f}</span>
                    </div>
                    <div style="color:#9ca3af; font-size:0.75rem; margin:3px 0 6px 0;">Source: {chunk['source']} • Category: {chunk['category']}</div>
                    <div style="color:#d1d5db; font-size:0.82rem; line-height:1.5;">{chunk['content']}</div>
                </div>
            """),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='font-size:0.86rem; font-weight:600; color:#d1d5db; margin: 1.25rem 0 0.5rem 0;'>PredictRx Clinical Retinal-Systemic Knowledge Graph</div>", unsafe_allow_html=True)
    sim_risk = compute_systemic_risk_profile(preview_grade, sim_hba1c, sim_years, sim_sbp)
    st.markdown(clean_html(render_knowledge_graph_svg(preview_grade, sim_risk)), unsafe_allow_html=True)


# ----------------- TAB 3: PERFORMANCE & BENCHMARK HUB -----------------
def render_performance_hub():
    st.markdown(
        clean_html("""
            <div style="margin-bottom: 1.25rem;">
                <h3 style="color: #f9fafb; font-weight: 600; font-size: 1.15rem; margin-bottom: 0.25rem;">APTOS 2019 Hybrid CNN-CBAM Architecture Specifications</h3>
                <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">
                    Comparison of the three supported Hybrid CNN + CBAM Attention backbones across parameter footprint, 256-D embedding dimension, and clinical screening deployment tiers.
                </p>
            </div>
        """),
        unsafe_allow_html=True,
    )

    ckpt_status = {
        "DenseNet121": (base_dir / "models" / "best_dr_model_densenet121.pth").exists(),
        "ResNet50": (base_dir / "models" / "best_dr_model_resnet50.pth").exists(),
        "EfficientNet-B0": (base_dir / "models" / "best_dr_model_efficientnet_b0.pth").exists(),
    }
    if sum(ckpt_status.values()) == 0:
        st.info(
            "No trained APTOS 2019 `.pth` checkpoints are currently present in `models/`. "
            "Run `notebooks/02_hybrid_dr_*.ipynb` or `python src/train.py` on APTOS 2019 to save trained weights."
        )

    st.markdown(
        clean_html(f"""
            <div class="history-table-container">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Hybrid Backbone</th>
                            <th>Attention Module</th>
                            <th>Parameters</th>
                            <th>Vector DB Embedding</th>
                            <th>Selection Metric</th>
                            <th>Checkpoint Status</th>
                            <th>Clinical Deployment Tier</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>EfficientNet-B0</strong></td>
                            <td>CBAM (Channel + 7x7 Spatial)</td>
                            <td>~4.0M</td>
                            <td><code>256-D L2-Norm</code></td>
                            <td>Quadratic Weighted Kappa (QWK)</td>
                            <td><span class="xai-badge">{'Loaded' if ckpt_status['EfficientNet-B0'] else 'ImageNet Preview'}</span></td>
                            <td>Primary Care &amp; Rural Tele-Ophthalmology</td>
                        </tr>
                        <tr>
                            <td><strong>DenseNet121</strong></td>
                            <td>CBAM (Channel + 7x7 Spatial)</td>
                            <td>~7.0M</td>
                            <td><code>256-D L2-Norm</code></td>
                            <td>Quadratic Weighted Kappa (QWK)</td>
                            <td><span class="xai-badge">{'Loaded' if ckpt_status['DenseNet121'] else 'ImageNet Preview'}</span></td>
                            <td>Hospital Ophthalmology Triage Desk</td>
                        </tr>
                        <tr>
                            <td><strong>ResNet50</strong></td>
                            <td>CBAM (Channel + 7x7 Spatial)</td>
                            <td>~23.5M</td>
                            <td><code>256-D L2-Norm</code></td>
                            <td>Quadratic Weighted Kappa (QWK)</td>
                            <td><span class="xai-badge">{'Loaded' if ckpt_status['ResNet50'] else 'ImageNet Preview'}</span></td>
                            <td>Secondary Confirmatory Audit</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div style="margin-top: 2rem; margin-bottom: 1.25rem;">
                <h3 style="color: #f9fafb; font-weight: 600; font-size: 1.15rem; margin-bottom: 0.25rem;">Empirical Hold-Out Validation Benchmarks (APTOS 2019 Blindness Detection)</h3>
                <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">
                    Measured on the stratified 366-image hold-out validation set across 5-stage ICDR Diabetic Retinopathy severity grading with balanced class weighting.
                </p>
            </div>
            <div class="history-table-container">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Hold-Out Val QWK</th>
                            <th>Hold-Out Val Accuracy</th>
                            <th>Macro F1-Score</th>
                            <th>Best Epoch</th>
                            <th>Convergence</th>
                            <th>Benchmark Ranking</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Hybrid EfficientNet-B0 + CBAM</strong></td>
                            <td><strong style="color: #34d399;">0.8766</strong></td>
                            <td>80.87%</td>
                            <td>0.6880</td>
                            <td>16 / 20</td>
                            <td>Epoch 16 (0.8766 QWK)</td>
                            <td><span class="xai-badge" style="background: rgba(16,185,129,0.15); color: #34d399; border-color: rgba(16,185,129,0.3);">🏆 Top Performer</span></td>
                        </tr>
                        <tr>
                            <td><strong>Hybrid DenseNet121 + CBAM</strong></td>
                            <td><strong style="color: #60a5fa;">0.8727</strong></td>
                            <td>78.14%</td>
                            <td>0.6281</td>
                            <td>12 / 20</td>
                            <td>Early Stop (Epoch 17)</td>
                            <td><span class="xai-badge" style="background: rgba(59,130,246,0.15); color: #60a5fa; border-color: rgba(59,130,246,0.3);">🥈 Runner-Up</span></td>
                        </tr>
                        <tr>
                            <td><strong>Hybrid ResNet50 + CBAM</strong></td>
                            <td><strong style="color: #9ca3af;">0.8588</strong></td>
                            <td>74.86%</td>
                            <td>0.5814</td>
                            <td>10 / 20</td>
                            <td>Early Stop (Epoch 15)</td>
                            <td><span class="xai-badge" style="background: rgba(156,163,175,0.15); color: #d1d5db; border-color: rgba(156,163,175,0.3);">🥉 Verified Baseline</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        """),
        unsafe_allow_html=True,
    )


# ----------------- TOP WORKSTATION MASTHEAD (CLEAN HUMAN-DESIGNED HEADER) -----------------
st.markdown(
    clean_html(f"""
        <div class="workstation-header">
            <div class="workstation-brand">
                <div class="brand-mark">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                        <circle cx="12" cy="12" r="3"/>
                    </svg>
                </div>
                <div>
                    <div class="workstation-eyebrow">HealthSphere AI • PredictRx Clinical Workstation</div>
                    <div class="workstation-title">Diabetic Retinopathy Screening, Severity &amp; RAG Triage</div>
                    <div class="workstation-subtitle">Hybrid CNN-CBAM 5-Stage ICDR Grading • 256-D Vector Case Retrieval • Guideline-Grounded Clinical RAG</div>
                </div>
            </div>
            <div class="workstation-pills">
                <span class="status-pill"><span class="status-dot"></span>{clinical_rag.backend_name}</span>
                <span class="status-pill">APTOS 2019 • 5-Stage ICDR</span>
                <span class="status-pill">Zero-PHI Session</span>
            </div>
        </div>

        <div class="kpi-strip">
            <div class="kpi-box">
                <span class="kpi-box-label">Session Scans Screened</span>
                <div class="kpi-box-row">
                    <span class="kpi-box-value">{st.session_state.session_total_scans}</span>
                    <span class="kpi-box-sub">Local In-Memory</span>
                </div>
            </div>
            <div class="kpi-box">
                <span class="kpi-box-label">Non-Referable (Grade 0–1)</span>
                <div class="kpi-box-row">
                    <span class="kpi-box-value" style="color:#34d399;">{st.session_state.session_nodr}</span>
                    <span class="kpi-box-sub">Routine Follow-Up</span>
                </div>
            </div>
            <div class="kpi-box">
                <span class="kpi-box-label">Referable DR (Grade ≥ 2)</span>
                <div class="kpi-box-row">
                    <span class="kpi-box-value" style="color:#fbbf24;">{st.session_state.session_referable}</span>
                    <span class="kpi-box-sub">Ophthalmology Referral</span>
                </div>
            </div>
            <div class="kpi-box">
                <span class="kpi-box-label">Urgent VTDR (Grade ≥ 3)</span>
                <div class="kpi-box-row">
                    <span class="kpi-box-value" style="color:#f87171;">{st.session_state.session_vtdr}</span>
                    <span class="kpi-box-sub">Priority Escalation</span>
                </div>
            </div>
        </div>
    """),
    unsafe_allow_html=True,
)

# ----------------- VECTOR EVIDENCE HTML WIDGET -----------------
def build_vector_evidence_html(similar_cases: List[Dict], retrieved_chunks: List[Dict]) -> str:
    """
    Renders a crisp, 100% vector HTML/CSS evidence card for the Vector DB Similarity
    & RAG Profile, guaranteeing razor-sharp typography and zero blurriness on all screens.
    """
    cases_rows = ""
    for c in similar_cases[:3]:
        sim = c.get("similarity_pct", 0.0)
        is_match = c.get("grade_match", False)
        bar_color = "#10b981" if is_match else "#94a3b8"
        match_tag = (
            '<span style="background:rgba(16,185,129,0.15); color:#34d399; font-size:0.68rem; '
            'padding:2px 6px; border-radius:4px; font-weight:600; border:1px solid rgba(16,185,129,0.28); margin-left:6px;">'
            'Match</span>'
            if is_match
            else ""
        )
        cases_rows += f"""
        <div style="margin-bottom: 0.65rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.83rem; margin-bottom:4px;">
                <span>
                    <strong style="color:#f3f5f8; font-family:monospace; font-size:0.84rem;">{c.get('case_id', 'REF')}</strong>
                    <span style="color:#9ca3af; font-size:0.80rem; margin-left:5px;">• {c.get('grade_name', 'DR')}</span>
                    {match_tag}
                </span>
                <span style="color:{bar_color}; font-weight:700; font-size:0.84rem;">{sim:.1f}%</span>
            </div>
            <div style="background:#262930; height:7px; border-radius:4px; overflow:hidden;">
                <div style="background:{bar_color}; width:{min(100.0, max(0.0, sim))}%; height:100%; border-radius:4px;"></div>
            </div>
        </div>
        """

    def _clean_doc_title(doc_id: str, cat: str) -> str:
        return (
            cat.replace("Medication Risk Safeguard", "Medication Safeguard")
            .replace("Cardiovascular & Systemic Risk", "CV & Systemic Risk")
            .replace("Clinical Practice Guidelines", "Clinical Practice Guidelines")
        )

    chunks_rows = ""
    for d in retrieved_chunks[:3]:
        score = d.get("relevance_score", 0.0) * 100.0
        doc_id = d.get("doc_id", "REF")
        clean_title = _clean_doc_title(doc_id, d.get("category", "Clinical Evidence"))
        chunks_rows += f"""
        <div style="margin-bottom: 0.65rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.83rem; margin-bottom:4px;">
                <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:85%;">
                    <span style="background:#1e293b; color:#60a5fa; border:1px solid rgba(96,165,250,0.3); font-size:0.72rem; padding:2px 6px; border-radius:4px; font-weight:700; font-family:monospace;">[{doc_id}]</span>
                    <span style="color:#e2e8f0; font-weight:500; margin-left:6px; font-size:0.82rem;">{clean_title}</span>
                </span>
                <span style="color:#60a5fa; font-weight:700; font-size:0.84rem;">{score:.0f}%</span>
            </div>
            <div style="background:#262930; height:7px; border-radius:4px; overflow:hidden;">
                <div style="background:#3b82f6; width:{min(100.0, max(0.0, score))}%; height:100%; border-radius:4px;"></div>
            </div>
        </div>
        """

    return f"""
    <div style="background:#181a1e; border:1px solid #2d3037; border-radius:10px; padding:1.1rem 1.25rem; min-height:100%; display:flex; flex-direction:column; justify-content:space-between; box-sizing:border-box;">
        <div style="margin-bottom: 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; border-bottom:1px solid #282b31; padding-bottom:0.45rem;">
                <span style="font-size:0.85rem; font-weight:700; color:#f8fafc; letter-spacing:0.01em;">
                    Visual Vector DB: Top-3 Nearest Cases
                </span>
                <span style="font-size:0.72rem; color:#94a3b8; font-weight:500;">256-D L2 Embedding</span>
            </div>
            {cases_rows}
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; border-bottom:1px solid #282b31; padding-bottom:0.45rem;">
                <span style="font-size:0.85rem; font-weight:700; color:#f8fafc; letter-spacing:0.01em;">
                    Clinical Guideline Vector DB: Top Chunks
                </span>
                <span style="font-size:0.72rem; color:#94a3b8; font-weight:500;">ICDR &amp; PredictRx</span>
            </div>
            {chunks_rows}
        </div>
    </div>
    """

# ----------------- TABS -----------------
tab_diag, tab_kg, tab_perf = st.tabs([
    "Clinical Screening & RAG Triage",
    "Vector DB Explorer & Knowledge Graph",
    "Architecture & Benchmarks",
])

with tab_diag:
    upload_col, display_col = st.columns([1.05, 1.95], gap="medium")

    with upload_col:
        patient_name = ""
        patient_age = ""
        patient_id = ""
        patient_gender = "Male"
        eye_side = "Right Eye (OD)"
        hba1c_val = 0.0
        diabetes_years = 0
        systolic_bp = 0
        patient_symptoms = ""
        rag_mode = "Dual Vector DB RAG (Case Similarity + Clinical Guidelines)"
        top_k_chunks = 3
        gemini_api_key = ""
        selected_backbone = "efficientnet_b0"
        use_ben_graham = True
        backbone_choice = "EfficientNet-B0 (Hybrid CBAM)"
        analyze_button = False

        # --- STEP 01: FUNDUS UPLOAD (NATIVE BORDERED CONTAINER) ---
        with st.container(border=True):
            st.markdown(
                clean_html("""
                    <div class="panel-header">
                        <div class="panel-header-left">
                            <span class="step-index">01</span>
                            <h4 class="panel-title">Retinal Fundus Image Input</h4>
                        </div>
                        <span class="xai-badge">OOD Gate</span>
                    </div>
                """),
                unsafe_allow_html=True,
            )

            input_mode = st.radio(
                "Input Source",
                ["File Upload (PNG / JPG)", "Camera Capture"],
                horizontal=True,
                label_visibility="collapsed",
                key="input_mode_radio",
            )

            if input_mode.startswith("File"):
                uploaded_file = st.file_uploader(
                    "Select a color retinal fundus photograph",
                    type=["png", "jpg", "jpeg"],
                    key=f"file_uploader_{st.session_state.uploader_key}",
                )
            else:
                uploaded_file = st.camera_input(
                    "Capture retinal fundus image",
                    key=f"camera_input_{st.session_state.uploader_key}",
                )

            is_valid = True
            err_msg = ""
            quality_metrics = {}
            image = None
            display_image = None

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                is_valid, err_msg, quality_metrics = validate_retinal_fundus(image)
                if not is_valid:
                    st.error(f"Input Safeguard Blocked: {err_msg}")
                else:
                    with st.expander("Image Preprocessing & Display Controls", expanded=False):
                        use_ben_graham = st.checkbox(
                            "Apply Ben Graham Illumination Normalization",
                            value=True,
                            help="Crops dark borders and normalizes fundus illumination prior to CNN-CBAM inference.",
                        )
                        if "fundus_brightness" not in st.session_state:
                            st.session_state.fundus_brightness = 1.0
                        if "fundus_contrast" not in st.session_state:
                            st.session_state.fundus_contrast = 1.0
                        if "fundus_sharpness" not in st.session_state:
                            st.session_state.fundus_sharpness = 1.0

                        brightness = st.slider("Brightness", 0.5, 2.0, key="fundus_brightness", step=0.05)
                        contrast = st.slider("Contrast", 0.5, 2.0, key="fundus_contrast", step=0.05)
                        sharpness = st.slider("Sharpness", 0.0, 3.0, key="fundus_sharpness", step=0.05)
                        st.button("Reset Display Sliders", on_click=reset_sliders, use_container_width=True, type="secondary")

                    display_image = adjust_image(image, brightness, contrast, sharpness)
                    st_image_compatible(display_image, caption="Validated Fundus Scan")
                    st.markdown(
                        clean_html(f"""
                            <div style="background:#0f1420; border:1px solid #1f2937; border-left:3px solid #10b981; border-radius:6px; padding:0.5rem 0.75rem; font-size:0.76rem; color:#d1d5db;">
                                <b>OOD Quality Check Passed</b> • Focus Var: <code>{quality_metrics.get('lap_var', '-')}</code> • R/B Ratio: <code>{quality_metrics.get('r_b_ratio', '-')}</code>
                            </div>
                        """),
                        unsafe_allow_html=True,
                    )

        # --- STEP 02: PATIENT & SYSTEMIC RISK METADATA ---
        with st.container(border=True):
            st.markdown(
                clean_html("""
                    <div class="panel-header">
                        <div class="panel-header-left">
                            <span class="step-index">02</span>
                            <h4 class="panel-title">Patient &amp; Systemic Risk Profile</h4>
                        </div>
                        <span class="xai-badge">Optional</span>
                    </div>
                """),
                unsafe_allow_html=True,
            )

            p1, p2 = st.columns(2)
            with p1:
                patient_name = st.text_input("Patient Identifier / Name", placeholder="Anonymous", disabled=(uploaded_file is None or not is_valid))
                patient_age = st.text_input("Age (Years)", placeholder="54", disabled=(uploaded_file is None or not is_valid))
                eye_side = st.selectbox("Laterality", ["Right Eye (OD)", "Left Eye (OS)", "Bilateral (OU)"], disabled=(uploaded_file is None or not is_valid))
            with p2:
                patient_id = st.text_input("Case Reference ID", placeholder="DR-2041", disabled=(uploaded_file is None or not is_valid))
                patient_gender = st.selectbox("Sex", ["Male", "Female", "Other"], disabled=(uploaded_file is None or not is_valid))
                hba1c_val = st.number_input("HbA1c (%)", min_value=0.0, max_value=18.0, value=7.5, step=0.1, help="Screening default: 7.5% (normal < 5.7%, high risk >= 8.5%)", disabled=(uploaded_file is None or not is_valid))

            r1, r2 = st.columns(2)
            with r1:
                diabetes_years = st.number_input("Diabetes Duration (Yrs)", min_value=0, max_value=60, value=5, step=1, help="Duration of diagnosed diabetes in years", disabled=(uploaded_file is None or not is_valid))
            with r2:
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=240, value=130, step=5, help="Systolic blood pressure in mmHg (normal < 120, stage 2 >= 140)", disabled=(uploaded_file is None or not is_valid))

            patient_symptoms = st.text_area(
                "Clinical Notes & Current Medications (Indexed for RAG Query)",
                placeholder="e.g. Blurred vision, insulin/pioglitazone therapy, hypertension...",
                height=64,
                disabled=(uploaded_file is None or not is_valid),
            )

        # --- STEP 03: HYBRID BACKBONE & VECTOR DB RAG CONFIGURATION ---
        with st.container(border=True):
            st.markdown(
                clean_html("""
                    <div class="panel-header">
                        <div class="panel-header-left">
                            <span class="step-index">03</span>
                            <h4 class="panel-title">Model &amp; RAG Execution</h4>
                        </div>
                        <span class="xai-badge">256-D Vector DB</span>
                    </div>
                """),
                unsafe_allow_html=True,
            )

            backbone_choice = st.selectbox(
                "Hybrid CNN-CBAM Backbone",
                [
                    "EfficientNet-B0 (Hybrid CBAM)",
                    "DenseNet121 (Hybrid CBAM)",
                    "ResNet50 (Hybrid CBAM)",
                ],
                index=0,
                key="model_backbone_select",
                disabled=(uploaded_file is None or not is_valid),
            )
            backbone_map = {
                "EfficientNet-B0 (Hybrid CBAM)": "efficientnet_b0",
                "DenseNet121 (Hybrid CBAM)": "densenet121",
                "ResNet50 (Hybrid CBAM)": "resnet50",
            }
            selected_backbone = backbone_map[backbone_choice]

            rag_mode = st.selectbox(
                "RAG Retrieval Mode",
                [
                    "Dual Vector DB RAG (Case Similarity + Clinical Guidelines)",
                    "Clinical Guideline & Medication Safeguard RAG",
                    "Visual Case-Based Vector Retrieval (Top-3 Reference Cases)",
                ],
                key="rag_mode_select",
                disabled=(uploaded_file is None or not is_valid),
            )

            with st.expander("Advanced Retrieval & LLM Parameters", expanded=False):
                top_k_chunks = st.slider("Top-K Guideline Chunks", min_value=2, max_value=5, value=3)
                gemini_api_key = st.text_input(
                    "Optional Gemini API Key (Blank = Local Grounded Synthesizer)",
                    type="password",
                )

            model_status_placeholder = st.empty()

            btn_col1, btn_col2 = st.columns([1.4, 0.6])
            with btn_col1:
                analyze_button = st.button(
                    "Run Screening & RAG",
                    use_container_width=True,
                    type="primary",
                    disabled=(uploaded_file is None or not is_valid),
                )
            with btn_col2:
                if st.button("Reset", use_container_width=True, type="secondary"):
                    st.session_state.uploader_key += 1
                    st.session_state.last_rag_state = None
                    st.rerun()

    # Load selected model
    model, model_loaded = load_trained_model(selected_backbone)
    if uploaded_file is not None and is_valid and model_status_placeholder is not None:
        if model_loaded:
            model_status_placeholder.caption(f"Checkpoint ready: `best_dr_model_{selected_backbone}.pth`")
        else:
            model_status_placeholder.caption(
                f"Mode: ImageNet architecture preview (`best_dr_model_{selected_backbone}.pth` not yet trained)"
            )

    # ----------------- RIGHT COLUMN: DIAGNOSTIC, VECTOR DB & RAG OUTPUT -----------------
    with display_col:
        if uploaded_file is not None and is_valid and analyze_button:
            with st.spinner("Running CNN-CBAM inference, querying Vector Databases & synthesizing RAG report..."):
                processed_for_model = (
                    apply_ben_graham_preprocessing(image, img_size=224)
                    if use_ben_graham
                    else image.resize((224, 224))
                )
                input_tensor = inference_transform(processed_for_model).unsqueeze(0)

                with torch.no_grad():
                    outputs, emb_256d = model.forward_with_embedding(input_tensor)
                    probs = torch.softmax(outputs, dim=1).squeeze().tolist()
                    grade_idx = int(torch.argmax(outputs, dim=1).item())
                    confidence = probs[grade_idx] * 100.0
                    emb_np = emb_256d.squeeze(0).cpu().numpy()

                grade_info = DR_KNOWLEDGE_BASE[grade_idx]
                prediction_name = grade_info["name"]

                similar_cases = visual_vdb.query_similar_cases(
                    embedding_256d=emb_np,
                    predicted_grade=grade_idx,
                    probs=probs,
                    top_k=3,
                )

                rag_result = clinical_rag.generate_clinical_explanation(
                    grade=grade_idx,
                    grade_name=prediction_name,
                    confidence=confidence,
                    similar_cases=similar_cases,
                    hba1c=hba1c_val if hba1c_val > 0 else None,
                    diabetes_years=diabetes_years if diabetes_years > 0 else None,
                    systolic_bp=systolic_bp if systolic_bp > 0 else None,
                    symptoms=patient_symptoms,
                    top_k=top_k_chunks,
                    gemini_api_key=gemini_api_key,
                )

                rag_evidence_img = render_rag_visual_evidence_card(
                    similar_cases=similar_cases,
                    retrieved_chunks=rag_result["retrieved_chunks"],
                    predicted_grade=grade_idx,
                    grade_name=prediction_name,
                )

                risk_profile = compute_systemic_risk_profile(
                    grade=grade_idx,
                    hba1c=hba1c_val if hba1c_val > 0 else None,
                    diabetes_years=diabetes_years if diabetes_years > 0 else None,
                    systolic_bp=systolic_bp if systolic_bp > 0 else None,
                )
                safeguards = evaluate_clinical_safeguards(
                    grade=grade_idx,
                    confidence=confidence,
                    model_weights_loaded=model_loaded,
                )

                st.session_state.session_total_scans += 1
                if grade_idx <= 1:
                    st.session_state.session_nodr += 1
                if grade_idx >= 2:
                    st.session_state.session_referable += 1
                if grade_idx >= 3:
                    st.session_state.session_vtdr += 1

                st.session_state.diagnoses_history.append({
                    "timestamp": time.time(),
                    "patient_id": patient_id or "N/A",
                    "patient_name": patient_name or "Anonymous",
                    "eye_side": eye_side,
                    "grade_idx": grade_idx,
                    "diagnosis": f"Grade {grade_idx}: {prediction_name}",
                    "triage": grade_info["triage_priority"],
                    "confidence": f"{confidence:.2f}%",
                    "model": backbone_choice,
                    "xai_method": f"RAG ({', '.join(rag_result['citations'][:2])})",
                })

                st.session_state.last_rag_state = {
                    "grade_idx": grade_idx,
                    "prediction_name": prediction_name,
                    "confidence": confidence,
                    "probs": probs,
                    "grade_info": grade_info,
                    "similar_cases": similar_cases,
                    "rag_result": rag_result,
                    "rag_evidence_img": rag_evidence_img,
                    "risk_profile": risk_profile,
                    "safeguards": safeguards,
                    "display_image": display_image,
                    "raw_image": image,
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "patient_age": patient_age,
                    "patient_gender": patient_gender,
                    "patient_symptoms": patient_symptoms,
                    "eye_side": eye_side,
                    "hba1c_val": hba1c_val,
                    "systolic_bp": systolic_bp,
                    "backbone_choice": backbone_choice,
                    "rag_mode": rag_mode,
                    "gemini_api_key": gemini_api_key,
                }

        active = st.session_state.last_rag_state if (uploaded_file is not None and is_valid) else None
        if active is not None:
            grade_idx = active["grade_idx"]
            prediction_name = active["prediction_name"]
            confidence = active["confidence"]
            probs = active["probs"]
            grade_info = active["grade_info"]
            similar_cases = active["similar_cases"]
            rag_result = active["rag_result"]
            rag_evidence_img = active["rag_evidence_img"]
            risk_profile = active["risk_profile"]
            safeguards = active["safeguards"]

            # 1. Visual Comparison Row inside bordered container
            with st.container(border=True):
                st.markdown(
                    clean_html(f"""
                        <div class="panel-header">
                            <div class="panel-header-left">
                                <span class="step-index">REPORT</span>
                                <h4 class="panel-title">Severity Assessment &amp; Vector DB Evidence</h4>
                            </div>
                            <span class="xai-badge">{rag_result['generator_mode']}</span>
                        </div>
                    """),
                    unsafe_allow_html=True,
                )
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st_image_compatible(active["display_image"], caption="Input Retinal Fundus Scan")
                with img_c2:
                    st.markdown(
                        clean_html(build_vector_evidence_html(similar_cases, rag_result["retrieved_chunks"])),
                        unsafe_allow_html=True,
                    )

                with st.expander("View Matplotlib Publication Figure (Exportable)", expanded=False):
                    st_image_compatible(rag_evidence_img, caption="Vector DB Similarity & RAG Evidence Profile (High-DPI Matplotlib)")

            # 2. Severity, RAG Explanation & 5-Class Probabilities Card
            prob_bars_html = ""
            for g in range(5):
                g_info = DR_KNOWLEDGE_BASE[g]
                p_val = probs[g] * 100.0
                prob_bars_html += f"""
                <div style="margin-bottom: 0.5rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;">
                        <span style="color:#d4d8e0; font-weight:500;">Grade {g}: {g_info['name']}</span>
                        <span style="color:{g_info['color']}; font-weight:600;">{p_val:.2f}%</span>
                    </div>
                    <div style="background:#2b2e35; height:6px; border-radius:3px; overflow:hidden;">
                        <div style="background:{g_info['color']}; width:{p_val}%; height:100%; border-radius:3px;"></div>
                    </div>
                </div>
                """

            card_border = grade_info["color"]
            citations_badges = " ".join(
                f"<span class='xai-badge'>[{cid}]</span>" for cid in rag_result["citations"]
            )
            st.markdown(
                clean_html(f"""
                    <div class="report-card" style="border-left: 4px solid {card_border};">
                        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1.25rem;">
                            <div>
                                <div class="sidebar-label">Predicted ICDR Severity</div>
                                <div style="font-size:1.3rem; font-weight:700; color:{card_border}; margin-top:2px;">
                                    Grade {grade_idx}: {prediction_name}
                                </div>
                                <div style="font-size:0.84rem; color:#e5e7eb; margin-top:6px;">
                                    Confidence: <code style="color:#f9fafb;">{confidence:.2f}%</code> • <strong>{grade_info['triage_priority']}</strong>
                                </div>
                                <div style="font-size:0.8rem; color:#93c5fd; margin-top:4px;">
                                    Recommended Window: {grade_info['referral_window']}
                                </div>
                                <div style="margin-top:10px; display:flex; gap:4px; flex-wrap:wrap;">{citations_badges}</div>
                            </div>
                            <div>
                                <div class="sidebar-label">Grounded RAG Clinical Synthesis</div>
                                <div style="font-size:0.82rem; line-height:1.55; color:#d1d5db; margin-top:4px;">
                                    {rag_result['explanation_html']}
                                </div>
                            </div>
                        </div>

                        <div class="sub-card" style="margin-top:1rem;">
                            <div class="sidebar-label" style="margin-bottom:0.55rem;">5-Class ICDR Posterior Probability Distribution</div>
                            {prob_bars_html}
                        </div>

                        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.75rem; margin-top:0.85rem;">
                            <div class="sub-card">
                                <div class="sidebar-label" style="margin-bottom:2px;">Nearest Vector Case</div>
                                <div style="font-size:0.86rem; font-weight:600; color:#f9fafb;">{similar_cases[0]['case_id']} ({similar_cases[0]['similarity_pct']}%)</div>
                            </div>
                            <div class="sub-card">
                                <div class="sidebar-label" style="margin-bottom:2px;">Referable DR Status</div>
                                <div style="font-size:0.86rem; font-weight:600; color:{card_border};">{'Urgent VTDR' if grade_info['vtdr'] else ('Referable DR' if grade_info['referable_dr'] else 'Non-Referable')}</div>
                            </div>
                            <div class="sub-card">
                                <div class="sidebar-label" style="margin-bottom:2px;">PredictRx Systemic Risk</div>
                                <div style="font-size:0.86rem; font-weight:600; color:{risk_profile['risk_color']};">{risk_profile['risk_score']}/100 ({risk_profile['risk_tier']})</div>
                            </div>
                        </div>
                    </div>
                """),
                unsafe_allow_html=True,
            )

            # 3. Visual Case-Based Vector DB Top-3 Matches & Retrieved Guideline Chunks
            with st.expander("Visual Vector DB — Top-3 Matched Verified Reference Cases (256-D Embedding)", expanded=True):
                case_cols = st.columns(3)
                for idx, sc in enumerate(similar_cases):
                    with case_cols[idx]:
                        st.markdown(
                            clean_html(f"""
                                <div class="sub-card" style="border-top:2px solid {'#10b981' if sc['grade_match'] else '#3b82f6'};">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <strong style="color:#f9fafb; font-size:0.84rem;">#{sc['rank']} {sc['case_id']}</strong>
                                        <span class="xai-badge">{sc['similarity_pct']}%</span>
                                    </div>
                                    <div style="color:#93c5fd; font-size:0.78rem; font-weight:600; margin-top:4px;">Grade {sc['grade']}: {sc['grade_name']}</div>
                                    <div style="color:#d1d5db; font-size:0.76rem; margin-top:5px;"><b>Lesions:</b> {sc['lesion_summary']}</div>
                                    <div style="color:#9ca3af; font-size:0.74rem; margin-top:4px;"><b>Outcome:</b> {sc['clinical_outcome']}</div>
                                </div>
                            """),
                            unsafe_allow_html=True,
                        )

            with st.expander("Clinical Guideline Vector DB — Retrieved Evidence Chunks & Citations", expanded=False):
                for chunk in rag_result["retrieved_chunks"]:
                    st.markdown(
                        clean_html(f"""
                            <div class="sub-card" style="margin-top:0.45rem; border-left:3px solid #3b82f6;">
                                <div style="display:flex; justify-content:space-between;">
                                    <strong style="color:#f3f4f6; font-size:0.84rem;">[{chunk['doc_id']}] {chunk['title']}</strong>
                                    <span class="xai-badge">Score: {chunk['relevance_score']:.3f}</span>
                                </div>
                                <div style="color:#9ca3af; font-size:0.74rem; margin:2px 0 5px 0;">Source: {chunk['source']}</div>
                                <div style="color:#d1d5db; font-size:0.8rem; line-height:1.5;">{chunk['content']}</div>
                            </div>
                        """),
                        unsafe_allow_html=True,
                    )

            # 4. Interactive Clinician RAG Follow-Up Q&A
            with st.container(border=True):
                st.markdown("<div class='sidebar-label' style='color:#93c5fd;'>Interactive Clinical RAG Q&amp;A (Patient-Contextualized)</div>", unsafe_allow_html=True)
                patient_q = st.text_input(
                    "Ask about medication interactions, referral urgency, or cardiovascular/renal follow-up:",
                    placeholder="e.g. Which medications require caution for this patient and what systemic tests should we order?",
                    key="patient_followup_rag_q",
                )
                if patient_q.strip():
                    ans = clinical_rag.answer_clinician_question(
                        question=patient_q,
                        grade=grade_idx,
                        grade_name=prediction_name,
                        hba1c=active["hba1c_val"] if active["hba1c_val"] > 0 else None,
                        systolic_bp=active["systolic_bp"] if active["systolic_bp"] > 0 else None,
                        gemini_api_key=active["gemini_api_key"],
                    )
                    st.markdown(
                        clean_html(f"""
                            <div class="sub-card" style="border-left:3px solid #10b981; margin-top:0.35rem;">
                                <div style="font-size:0.82rem; color:#e5e7eb; line-height:1.55;">{ans['answer']}</div>
                            </div>
                        """),
                        unsafe_allow_html=True,
                    )

            # 5. Documented Safeguards Panel
            sg_color = safeguards["status_color"]
            sg_items_html = "".join(f"<li>{a}</li>" for a in safeguards["alerts"])
            st.markdown(
                clean_html(f"""
                    <div class="report-card" style="border-left: 4px solid {sg_color};">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span class="sidebar-label" style="margin:0;">PredictRx Documented Safeguards &amp; HITL Gate</span>
                            <span class="xai-badge" style="color:{sg_color} !important;">{safeguards['status_label']}</span>
                        </div>
                        <ul style="color:#d1d5db; font-size:0.81rem; margin:0; padding-left:1.15rem; line-height:1.55;">
                            {sg_items_html}
                        </ul>
                    </div>
                """),
                unsafe_allow_html=True,
            )

            # 6. Clinical Knowledge Graph Reasoning Pathway
            st.markdown("<div style='font-size:0.84rem; font-weight:600; color:#d1d5db; margin: 1rem 0 0.45rem 0;'>PredictRx Retinal-Systemic Knowledge Graph</div>", unsafe_allow_html=True)
            st.markdown(clean_html(render_knowledge_graph_svg(grade_idx, risk_profile)), unsafe_allow_html=True)

            # 7. In-Memory Clinical PDF Report Download
            pdf_evidence_card = render_rag_visual_evidence_card(
                similar_cases=similar_cases,
                retrieved_chunks=rag_result["retrieved_chunks"],
                predicted_grade=grade_idx,
                grade_name=prediction_name,
                light_theme=True,
            )

            pdf_data = generate_pdf_report(
                patient_name=active["patient_name"],
                patient_id=active["patient_id"],
                patient_age=active["patient_age"],
                patient_gender=active["patient_gender"],
                symptoms=active["patient_symptoms"],
                prediction=prediction_name,
                confidence=confidence,
                insights=rag_result["explanation_html"],
                original_image=active["raw_image"],
                xai_overlay=pdf_evidence_card,
                xai_method_name="Dual Vector DB & Clinical RAG",
                grade_idx=grade_idx,
                eye_side=active["eye_side"],
                triage_level=grade_info["triage_level"],
                referral_window=grade_info["referral_window"],
                risk_score=risk_profile["risk_score"],
                risk_tier=risk_profile["risk_tier"],
                safeguard_notes=safeguards["alerts"],
            )

            st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Download Clinical DR RAG & Triage PDF Report",
                data=pdf_data,
                file_name=f"DR_RAG_Triage_Report_{active['patient_id'] or 'Patient'}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        else:
            # Clean, Human-Designed Workstation Empty State (No Giant Glowing Emojis)
            st.markdown(
                clean_html("""
                    <div class="empty-workstation">
                        <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #1f2937; padding-bottom:0.9rem; margin-bottom:1.1rem;">
                            <div>
                                <div class="sidebar-label" style="margin-bottom:2px; color:#60a5fa;">WORKSTATION IDLE • READY FOR INPUT</div>
                                <h3 style="margin:0; font-size:1.1rem; font-weight:600; color:#f9fafb;">No Active Retinal Screening Report</h3>
                            </div>
                            <span class="status-pill"><span class="status-dot"></span>Engine Online</span>
                        </div>
                        <p style="font-size:0.85rem; color:#9ca3af; line-height:1.6; margin-top:0;">
                            Upload a color retinal fundus photograph in <b>Panel 01</b> on the left, enter optional systemic risk metrics (HbA1c, blood pressure, duration) in <b>Panel 02</b>, and execute <b>Run Screening &amp; RAG</b> in <b>Panel 03</b>.
                        </p>

                        <div class="spec-grid">
                            <div class="spec-item">
                                <div class="spec-item-tag">01 • Input Safeguard</div>
                                <h5>Retinal Fundus OOD &amp; Focus Gate</h5>
                                <p>Verifies RGB retinal chrominance, Laplacian focus sharpness, and exposure before allowing model inference.</p>
                            </div>
                            <div class="spec-item">
                                <div class="spec-item-tag">02 • Predictive ML</div>
                                <h5>Hybrid CNN-CBAM + 256-D Embeddings</h5>
                                <p>Estimates 5-stage ICDR severity and extracts L2-normalized 256-D feature vectors across 3 backbones.</p>
                            </div>
                            <div class="spec-item">
                                <div class="spec-item-tag">03 • Vector DB &amp; RAG</div>
                                <h5>Case Retrieval &amp; Guideline Citations</h5>
                                <p>Matches Top-3 verified reference scans and synthesizes ICDR, AAO, ADA, and PredictRx medication safeguards.</p>
                            </div>
                            <div class="spec-item">
                                <div class="spec-item-tag">04 • Data Sovereignty</div>
                                <h5>Zero-PHI In-Memory PDF Reporting</h5>
                                <p>Compiles print-ready ophthalmology triage certificates in RAM without storing patient images externally.</p>
                            </div>
                        </div>
                    </div>
                """),
                unsafe_allow_html=True,
            )

    # Session Triage History Table
    if st.session_state.diagnoses_history:
        st.markdown("<div style='font-size:0.9rem; font-weight:600; color:#e5e7eb; margin: 1.75rem 0 0.4rem 0;'>Local Session Screening Log (In-Memory)</div>", unsafe_allow_html=True)

        user_tz_str = getattr(getattr(st, "context", None), "timezone", None) or "UTC"
        try:
            user_tz = ZoneInfo(user_tz_str)
        except Exception:
            user_tz = ZoneInfo("UTC")

        rows_html = ""
        for item in reversed(st.session_state.diagnoses_history):
            dt_str = datetime.fromtimestamp(item["timestamp"], tz=user_tz).strftime("%d-%b-%Y %H:%M")
            badge_cls = f"history-badge-grade{item['grade_idx']}"
            rows_html += clean_html(f"""
                <tr>
                    <td><code>{dt_str}</code></td>
                    <td><code>{item['patient_id']}</code></td>
                    <td>{item['patient_name']}</td>
                    <td>{item['eye_side']}</td>
                    <td><span class="history-badge {badge_cls}">{item['diagnosis']}</span></td>
                    <td>{item['triage']}</td>
                    <td><code>{item['confidence']}</code></td>
                    <td>{item['model']}</td>
                    <td><span class="xai-badge">{item['xai_method']}</span></td>
                </tr>
            """)

        st.markdown(
            clean_html(f"""
                <div class="history-table-container">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Case ID</th>
                                <th>Patient</th>
                                <th>Eye</th>
                                <th>ICDR Grade</th>
                                <th>Triage Priority</th>
                                <th>Confidence</th>
                                <th>Backbone</th>
                                <th>RAG Citations</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            """),
            unsafe_allow_html=True,
        )

with tab_kg:
    render_rag_and_knowledge_graph_tab()

with tab_perf:
    render_performance_hub()

# ----------------- FOOTER -----------------
st.markdown(
    clean_html("""
        <div class="dashboard-footer">
            <div><strong style="color:#d1d5db;">RetinaRAG Clinical Workstation</strong> • HealthSphere AI (PredictRx)</div>
            <div>Decision-support screening prototype. Verify all severity grades and RAG citations with a certified ophthalmologist.</div>
            <div><code>Zero-PHI Local Execution</code></div>
        </div>
    """),
    unsafe_allow_html=True,
)
