# -*- coding: utf-8 -*-
"""
Hugging Face Spaces Workstation Entrypoint — RetinaRAG Clinical Workstation
Hybrid Explainable AI Framework for Automated Diabetic Retinopathy Detection,
5-Stage ICDR Severity Assessment, Dual Vector DB RAG & PredictRx Safeguards.
Designed with the exact dark-graphite clinical workstation UI of Streamlit.
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr  # type: ignore[import-untyped]

# Hugging Face ZeroGPU decorator (graceful fallback when running locally)
try:
    import spaces
except ImportError:
    class spaces:  # type: ignore
        @staticmethod
        def GPU(fn):
            return fn

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_model import HybridDRPredictor
from src.preprocess import apply_ben_graham_preprocessing
from app.rag_engine import (
    CLINICAL_GUIDELINE_CORPUS,
    REFERENCE_RETINAL_CASES,
    RetinalVisualVectorDB,
    ClinicalGuidelineRAGEngine,
    render_rag_visual_evidence_card,
)
from app.knowledge_graph import (
    DR_KNOWLEDGE_BASE,
    compute_systemic_risk_profile,
    evaluate_clinical_safeguards,
    render_knowledge_graph_svg,
)
from app.report_generator import generate_pdf_report

# ----------------- CACHED RESOURCES -----------------
VISUAL_VDB = RetinalVisualVectorDB(embedding_dim=256)
CLINICAL_RAG = ClinicalGuidelineRAGEngine()
MODEL_CACHE = {}

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BACKBONE_MAP = {
    "EfficientNet-B0 (Hybrid CBAM)": "efficientnet_b0",
    "DenseNet121 (Hybrid CBAM)": "densenet121",
    "ResNet50 (Hybrid CBAM)": "resnet50",
}


def get_model(backbone_key: str):
    if backbone_key in MODEL_CACHE:
        return MODEL_CACHE[backbone_key]

    weights_path = ROOT_DIR / "models" / f"best_dr_model_{backbone_key}.pth"
    loaded = False
    if weights_path.exists():
        model = HybridDRPredictor(backbone=backbone_key, num_classes=5, pretrained=False)
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
            loaded = True
        except Exception:
            model = HybridDRPredictor(backbone=backbone_key, num_classes=5, pretrained=True)
    else:
        model = HybridDRPredictor(backbone=backbone_key, num_classes=5, pretrained=True)

    model.eval()
    MODEL_CACHE[backbone_key] = (model, loaded)
    return model, loaded


def validate_retinal_fundus(image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    img_rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(img_rgb, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    img_std = float(np.std(gray))
    if img_std < 12.0:
        return False, "Image contrast is too flat to be a clinical retinal fundus photograph.", {}

    fg_mask = gray > 15
    if float(np.mean(fg_mask)) < 0.15:
        return False, "Image is predominantly dark/underexposed — retinal field not detected.", {}

    fg_pixels = resized[fg_mask]
    r_mean = float(np.mean(fg_pixels[:, 0]))
    b_mean = float(np.mean(fg_pixels[:, 2]))

    channel_diff = float(
        np.mean(np.abs(fg_pixels[:, 0].astype(np.float32) - fg_pixels[:, 1].astype(np.float32)))
        + np.mean(np.abs(fg_pixels[:, 0].astype(np.float32) - fg_pixels[:, 2].astype(np.float32)))
    )
    if channel_diff < 8.0:
        return False, "Grayscale / monochrome image detected. Please upload a color RGB retinal fundus photograph.", {}

    if r_mean < b_mean * 1.05:
        return False, "Color profile does not match a retinal fundus photograph (expected warm red/orange retinal spectrum).", {}

    lap_var = float(cv2.Laplacian(resized[:, :, 1], cv2.CV_64F).var())
    if lap_var < 4.0:
        return False, "Image appears severely blurred or out of focus. Please upload a sharper fundus scan.", {}

    rb_ratio = float(r_mean / (b_mean + 1e-5))
    metrics = {
        "lap_var": round(lap_var, 1),
        "r_b_ratio": round(rb_ratio, 2),
    }
    return True, f"Valid color retinal fundus verified (Focus Var: {lap_var:.1f}, R/B Ratio: {rb_ratio:.2f}).", metrics


def build_vector_evidence_html(similar_cases: List[Dict], retrieved_chunks: List[Dict]) -> str:
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


# ----------------- CUSTOM WORKSTATION CSS -----------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

body, .gradio-container, .gradio-container * {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.gradio-container {
    background-color: #141517 !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header & Panels */
.dr-header-card {
    background: #181a1e;
    border: 1px solid #23262d;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.kpi-card {
    background: #181a1e;
    border: 1px solid #23262d;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
}

.panel-box {
    background: #181a1e;
    border: 1px solid #23262d;
    border-radius: 12px;
    padding: 1.15rem;
    margin-bottom: 0.85rem;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #23262d;
    padding-bottom: 0.65rem;
    margin-bottom: 0.85rem;
}

.step-badge {
    background: #23262d;
    color: #e5e7eb;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: monospace;
}

.xai-pill {
    background: #1e293b;
    color: #60a5fa;
    border: 1px solid rgba(96,165,250,0.3);
    font-size: 0.72rem;
    padding: 3px 8px;
    border-radius: 999px;
    font-weight: 600;
}

.sub-card {
    background: #141517;
    border: 1px solid #282b31;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
}

/* Empty Workstation */
.empty-box {
    background: #181a1e;
    border: 1px solid #23262d;
    border-radius: 12px;
    padding: 2.2rem 2rem;
    text-align: left;
}
"""


def render_kpi_bar(scans=0, nodr=0, referable=0, vtdr=0) -> str:
    return f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div style="font-size:0.73rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Session Scans Screened</div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:4px;">
                <span style="font-size:1.6rem; font-weight:700; color:#f9fafb;">{scans}</span>
                <span style="font-size:0.75rem; color:#60a5fa;">Local In-Memory</span>
            </div>
        </div>
        <div class="kpi-card">
            <div style="font-size:0.73rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Non-Referable (Grade 0–1)</div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:4px;">
                <span style="font-size:1.6rem; font-weight:700; color:#10b981;">{nodr}</span>
                <span style="font-size:0.75rem; color:#10b981;">Routine Follow-Up</span>
            </div>
        </div>
        <div class="kpi-card">
            <div style="font-size:0.73rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Referable DR (Grade ≥ 2)</div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:4px;">
                <span style="font-size:1.6rem; font-weight:700; color:#f59e0b;">{referable}</span>
                <span style="font-size:0.75rem; color:#f59e0b;">Ophthalmology Referral</span>
            </div>
        </div>
        <div class="kpi-card">
            <div style="font-size:0.73rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Urgent VTDR (Grade ≥ 3)</div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:4px;">
                <span style="font-size:1.6rem; font-weight:700; color:#ef4444;">{vtdr}</span>
                <span style="font-size:0.75rem; color:#ef4444;">Priority Escalation</span>
            </div>
        </div>
    </div>
    """


EMPTY_STATE_HTML = """
<div class="empty-box">
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #23262d; padding-bottom:0.75rem; margin-bottom:1rem;">
        <div>
            <div style="font-size:0.74rem; color:#60a5fa; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">WORKSTATION IDLE • READY FOR INPUT</div>
            <h3 style="margin:2px 0 0 0; color:#f9fafb; font-size:1.15rem; font-weight:600;">No Active Retinal Screening Report</h3>
        </div>
        <span class="xai-pill" style="border-color:#10b981; color:#10b981;">Engine Online</span>
    </div>
    <p style="color:#9ca3af; font-size:0.86rem; line-height:1.6; margin:0 0 1rem 0;">
        Upload a color retinal fundus photograph on the left, optionally enter patient HbA1c and systolic blood pressure, then click <b>Run Screening &amp; RAG</b> to compute 5-stage ICDR grading, retrieve Vector DB reference cases, synthesize citation-grounded clinical guidance, and generate a verified PDF triage report.
    </p>
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:0.75rem;">
        <div class="sub-card">
            <div style="font-size:0.8rem; font-weight:700; color:#f9fafb;">1. 256-D Vector Retrieval</div>
            <div style="font-size:0.76rem; color:#9ca3af; margin-top:3px;">Matches penultimate CBAM embeddings against verified reference cases.</div>
        </div>
        <div class="sub-card">
            <div style="font-size:0.8rem; font-weight:700; color:#f9fafb;">2. Grounded Clinical RAG</div>
            <div style="font-size:0.76rem; color:#9ca3af; margin-top:3px;">Synthesizes authoritative ICDR and medication safeguard guidance.</div>
        </div>
        <div class="sub-card">
            <div style="font-size:0.8rem; font-weight:700; color:#f9fafb;">3. PredictRx Safeguards</div>
            <div style="font-size:0.76rem; color:#9ca3af; margin-top:3px;">Automated OOD verification, VTDR flags, and Zero-PHI data sovereignty.</div>
        </div>
    </div>
</div>
"""

# Global session scan counters
SESSION_STATS = {"scans": 0, "nodr": 0, "referable": 0, "vtdr": 0}


# ----------------- MAIN PIPELINE RUNNER -----------------
@spaces.GPU
def run_screening_pipeline(
    fundus_img,
    use_bg,
    backbone_label,
    top_k_chunks,
    patient_name,
    patient_id,
    patient_age,
    patient_gender,
    eye_side,
    hba1c_val,
    diabetes_years,
    systolic_bp,
    patient_symptoms,
    gemini_key,
):
    if fundus_img is None:
        return (
            "<div style='color:#ef4444; padding:10px; background:#1e1416; border-radius:6px;'>Please upload a color retinal fundus photograph to execute screening.</div>",
            None,
            "",
            "",
            "",
            "",
            "",
            None,
            render_kpi_bar(SESSION_STATS["scans"], SESSION_STATS["nodr"], SESSION_STATS["referable"], SESSION_STATS["vtdr"]),
        )

    image = fundus_img.convert("RGB")
    is_valid, ood_msg, quality_metrics = validate_retinal_fundus(image)
    if not is_valid:
        err_box = f"""
        <div class="panel-box" style="border-left: 4px solid #ef4444;">
            <div style="color:#ef4444; font-weight:700; font-size:0.95rem;">OOD Input Safeguard Blocked Scan</div>
            <p style="color:#d1d5db; font-size:0.86rem; margin-top:6px;">{ood_msg}</p>
        </div>
        """
        return err_box, None, "", "", "", "", "", None, render_kpi_bar(SESSION_STATS["scans"], SESSION_STATS["nodr"], SESSION_STATS["referable"], SESSION_STATS["vtdr"])

    backbone_key = BACKBONE_MAP.get(backbone_label, "efficientnet_b0")
    model, model_loaded = get_model(backbone_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    processed = apply_ben_graham_preprocessing(image, img_size=224) if use_bg else image.resize((224, 224))
    input_tensor = INFERENCE_TRANSFORM(processed).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, emb_256d = model.forward_with_embedding(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()
        grade_idx = int(torch.argmax(outputs, dim=1).item())
        confidence = probs[grade_idx] * 100.0
        emb_np = emb_256d.squeeze(0).cpu().numpy()

    grade_info = DR_KNOWLEDGE_BASE[grade_idx]
    prediction_name = grade_info["name"]

    similar_cases = VISUAL_VDB.query_similar_cases(
        embedding_256d=emb_np,
        predicted_grade=grade_idx,
        probs=probs,
        top_k=3,
    )
    rag_result = CLINICAL_RAG.generate_clinical_explanation(
        grade=grade_idx,
        grade_name=prediction_name,
        confidence=confidence,
        similar_cases=similar_cases,
        hba1c=hba1c_val if hba1c_val and hba1c_val > 0 else None,
        diabetes_years=diabetes_years if diabetes_years and diabetes_years > 0 else None,
        systolic_bp=systolic_bp if systolic_bp and systolic_bp > 0 else None,
        symptoms=patient_symptoms or "",
        top_k=int(top_k_chunks),
        gemini_api_key=gemini_key or None,
    )

    risk_profile = compute_systemic_risk_profile(
        grade=grade_idx,
        hba1c=hba1c_val if hba1c_val and hba1c_val > 0 else None,
        diabetes_years=diabetes_years if diabetes_years and diabetes_years > 0 else None,
        systolic_bp=systolic_bp if systolic_bp and systolic_bp > 0 else None,
    )
    safeguards = evaluate_clinical_safeguards(
        grade=grade_idx,
        confidence=confidence,
        model_weights_loaded=model_loaded,
    )

    # Update Session Counters
    SESSION_STATS["scans"] += 1
    if grade_idx <= 1:
        SESSION_STATS["nodr"] += 1
    if grade_idx >= 2:
        SESSION_STATS["referable"] += 1
    if grade_idx >= 3:
        SESSION_STATS["vtdr"] += 1

    # 1. Evidence Card HTML
    evidence_card_html = build_vector_evidence_html(similar_cases, rag_result["retrieved_chunks"])

    # 2. Probability Bars HTML
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
    citations_badges = " ".join(f"<span class='xai-pill'>[{cid}]</span>" for cid in rag_result["citations"])

    # 3. Severity & Grounded RAG Report Card HTML
    report_card_html = f"""
    <div class="panel-box" style="border-left: 4px solid {card_border};">
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem;">
            <div>
                <div style="font-size:0.74rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Predicted ICDR Severity</div>
                <div style="font-size:1.35rem; font-weight:700; color:{card_border}; margin-top:2px;">
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
                <div style="font-size:0.74rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Grounded RAG Clinical Synthesis</div>
                <div style="font-size:0.82rem; line-height:1.55; color:#d1d5db; margin-top:4px;">
                    {rag_result['explanation_html']}
                </div>
            </div>
        </div>

        <div class="sub-card" style="margin-top:1rem;">
            <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase; font-weight:600; margin-bottom:0.55rem;">5-Class ICDR Posterior Probability Distribution</div>
            {prob_bars_html}
        </div>

        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.75rem; margin-top:0.85rem;">
            <div class="sub-card">
                <div style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase;">Nearest Vector Case</div>
                <div style="font-size:0.88rem; font-weight:600; color:#f9fafb; margin-top:2px;">{similar_cases[0]['case_id']} ({similar_cases[0]['similarity_pct']}%)</div>
            </div>
            <div class="sub-card">
                <div style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase;">Referable DR Status</div>
                <div style="font-size:0.88rem; font-weight:600; color:{card_border}; margin-top:2px;">{'Urgent VTDR' if grade_info['vtdr'] else ('Referable DR' if grade_info['referable_dr'] else 'Non-Referable')}</div>
            </div>
            <div class="sub-card">
                <div style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase;">PredictRx Systemic Risk</div>
                <div style="font-size:0.88rem; font-weight:600; color:{risk_profile['risk_color']}; margin-top:2px;">{risk_profile['risk_score']}/100 ({risk_profile['risk_tier']})</div>
            </div>
        </div>
    </div>
    """

    # 4. Top-3 Visual Cases HTML
    cases_cards = ""
    for sc in similar_cases:
        cases_cards += f"""
        <div class="sub-card" style="border-top:2px solid {'#10b981' if sc['grade_match'] else '#3b82f6'};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong style="color:#f9fafb; font-size:0.84rem;">#{sc['rank']} {sc['case_id']}</strong>
                <span class="xai-pill">{sc['similarity_pct']}%</span>
            </div>
            <div style="color:#93c5fd; font-size:0.78rem; font-weight:600; margin-top:4px;">Grade {sc['grade']}: {sc['grade_name']}</div>
            <div style="color:#d1d5db; font-size:0.76rem; margin-top:5px;"><b>Lesions:</b> {sc['lesion_summary']}</div>
            <div style="color:#9ca3af; font-size:0.74rem; margin-top:4px;"><b>Outcome:</b> {sc['clinical_outcome']}</div>
        </div>
        """
    top3_cases_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:0.75rem;">
        {cases_cards}
    </div>
    """

    # 5. Retrieved Guideline Chunks HTML
    chunks_cards = ""
    for chunk in rag_result["retrieved_chunks"]:
        chunks_cards += f"""
        <div class="sub-card" style="border-left:3px solid #3b82f6;">
            <div style="display:flex; justify-content:space-between;">
                <strong style="color:#f3f4f6; font-size:0.84rem;">[{chunk['doc_id']}] {chunk['title']}</strong>
                <span class="xai-pill">Score: {chunk['relevance_score']:.3f}</span>
            </div>
            <div style="color:#9ca3af; font-size:0.74rem; margin:2px 0 5px 0;">Source: {chunk['source']}</div>
            <div style="color:#d1d5db; font-size:0.8rem; line-height:1.5;">{chunk['content']}</div>
        </div>
        """
    guidelines_html = chunks_cards

    # 6. Safeguards HTML
    sg_color = safeguards["status_color"]
    sg_items_html = "".join(f"<li>{a}</li>" for a in safeguards["alerts"])
    safeguards_html = f"""
    <div class="panel-box" style="border-left: 4px solid {sg_color};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
            <span style="font-size:0.75rem; color:#9ca3af; font-weight:700; text-transform:uppercase;">PredictRx Documented Safeguards &amp; HITL Gate</span>
            <span class="xai-pill" style="color:{sg_color} !important;">{safeguards['status_label']}</span>
        </div>
        <ul style="color:#d1d5db; font-size:0.81rem; margin:0; padding-left:1.15rem; line-height:1.55;">
            {sg_items_html}
        </ul>
    </div>
    """

    # 7. Knowledge Graph SVG
    kg_svg = render_knowledge_graph_svg(grade_idx, risk_profile)

    # 8. Verified PDF Generation
    try:
        pdf_evidence_card = render_rag_visual_evidence_card(
            similar_cases=similar_cases,
            retrieved_chunks=rag_result["retrieved_chunks"],
            predicted_grade=grade_idx,
            grade_name=prediction_name,
            light_theme=True,
        )
    except Exception:
        pdf_evidence_card = None

    pdf_bytes = generate_pdf_report(
        patient_name=patient_name or "Anonymous",
        patient_id=patient_id or "N/A",
        patient_age=patient_age or "N/A",
        patient_gender=patient_gender or "Male",
        symptoms=patient_symptoms or "",
        prediction=prediction_name,
        confidence=confidence,
        insights=rag_result["explanation_html"],
        original_image=image,
        xai_overlay=pdf_evidence_card,
        xai_method_name="Dual Vector DB & Clinical RAG",
        grade_idx=grade_idx,
        eye_side=eye_side or "Right Eye (OD)",
        triage_level=grade_info["triage_level"],
        referral_window=grade_info["referral_window"],
        risk_score=risk_profile["risk_score"],
        risk_tier=risk_profile["risk_tier"],
        safeguard_notes=safeguards["alerts"],
    )
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="DR_RAG_Report_")
    tmp_pdf.write(pdf_bytes)
    tmp_pdf.close()

    updated_kpis = render_kpi_bar(
        SESSION_STATS["scans"], SESSION_STATS["nodr"], SESSION_STATS["referable"], SESSION_STATS["vtdr"]
    )

    return (
        report_card_html,
        evidence_card_html,
        top3_cases_html,
        guidelines_html,
        safeguards_html,
        kg_svg,
        tmp_pdf.name,
        updated_kpis,
    )


def run_standalone_qa(question, grade_label, hba1c, diabetes_years, systolic_bp, gemini_key):
    grade_idx = int(grade_label.split(":")[0].replace("Grade", "").strip())
    grade_name = DR_KNOWLEDGE_BASE[grade_idx]["name"]

    qa = CLINICAL_RAG.answer_clinician_question(
        question=question or "What medication safeguards apply?",
        grade=grade_idx,
        grade_name=grade_name,
        hba1c=hba1c if hba1c and hba1c > 0 else None,
        systolic_bp=systolic_bp if systolic_bp and systolic_bp > 0 else None,
        gemini_api_key=gemini_key or None,
    )

    chunks_html = "".join(
        f"""
        <div class="sub-card" style="border-left:3px solid #60a5fa; margin-top:8px;">
            <div style="display:flex; justify-content:space-between;">
                <strong style="color:#f3f5f8; font-size:0.84rem;">[{c['doc_id']}] {c['title']}</strong>
                <span class="xai-pill">Cosine: {c['relevance_score']:.3f}</span>
            </div>
            <div style="color:#9ca3af; font-size:0.75rem; margin:3px 0;">Source: {c['source']} • {c['category']}</div>
            <div style="color:#d4d8e0; font-size:0.81rem; line-height:1.5;">{c['content']}</div>
        </div>
        """
        for c in qa["chunks"]
    )
    answer_html = f"""
    <div class="panel-box">
        <div style="font-size:0.75rem; color:#60a5fa; font-weight:700; text-transform:uppercase; margin-bottom:6px;">{qa['mode']}</div>
        <div style="font-size:0.88rem; color:#f3f4f6; line-height:1.6;">{qa['answer']}</div>
        <div style="margin-top:14px; font-size:0.76rem; color:#9ca3af; text-transform:uppercase; font-weight:700;">Retrieved Guideline Evidence Chunks</div>
        {chunks_html}
    </div>
    """
    risk_profile = compute_systemic_risk_profile(grade_idx, hba1c, diabetes_years, systolic_bp)
    return answer_html, render_knowledge_graph_svg(grade_idx, risk_profile)


# ----------------- BUILD GRADIO INTERFACE -----------------
with gr.Blocks(title="RetinaRAG Clinical Workstation | Diabetic Retinopathy Screening & PredictRx", css=CUSTOM_CSS) as demo:
    # 1. Top Clinical Header
    gr.HTML(f"""
    <div class="dr-header-card">
        <div>
            <div style="font-size:0.74rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
                HealthSphere AI • PredictRx Clinical Workstation
            </div>
            <div style="font-size:1.32rem; font-weight:700; color:#f9fafb; margin-top:2px;">
                Diabetic Retinopathy Screening, Severity &amp; RAG Triage
            </div>
            <div style="font-size:0.82rem; color:#9ca3af; margin-top:2px;">
                Hybrid CNN-CBAM 5-Stage ICDR Grading • 256-D Vector Case Retrieval • Guideline-Grounded Clinical RAG
            </div>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <span class="xai-pill" style="border-color:#10b981; color:#10b981;">● NumPy Exact Cosine Vector Store</span>
            <span class="xai-pill">APTOS 2019 • 5-Stage ICDR</span>
            <span class="xai-pill">Zero-PHI Session</span>
        </div>
    </div>
    """)

    # 2. Top KPI Metrics Bar
    kpi_bar_html = gr.HTML(render_kpi_bar(0, 0, 0, 0))

    # 3. Main Navigation Tabs
    with gr.Tabs():
        with gr.Tab("Clinical Screening & RAG Triage"):
            with gr.Row():
                # --- LEFT COLUMN: INPUTS & CONTROLS ---
                with gr.Column(scale=1):
                    # Panel 01: Retinal Fundus Input
                    with gr.Group():
                        gr.HTML("""
                        <div class="panel-header">
                            <div>
                                <span class="step-badge">01</span>
                                <strong style="color:#f9fafb; margin-left:6px; font-size:0.92rem;">Retinal Fundus Image Input</strong>
                            </div>
                            <span class="xai-pill">OOD Gate</span>
                        </div>
                        """)
                        fundus_in = gr.Image(type="pil", label="Select a color retinal fundus photograph", height=230)
                        use_bg_cb = gr.Checkbox(value=True, label="Apply Ben Graham Retinal Illumination Normalization")

                    # Panel 02: Patient & Systemic Risk Profile
                    with gr.Group():
                        gr.HTML("""
                        <div class="panel-header" style="margin-top:0.8rem;">
                            <div>
                                <span class="step-badge">02</span>
                                <strong style="color:#f9fafb; margin-left:6px; font-size:0.92rem;">Patient &amp; Systemic Risk Profile</strong>
                            </div>
                            <span class="xai-pill">Optional</span>
                        </div>
                        """)
                        with gr.Row():
                            p_name_in = gr.Textbox(label="Patient Identifier / Name", placeholder="Anonymous")
                            p_id_in = gr.Textbox(label="Case Reference ID", placeholder="DR-2041")
                        with gr.Row():
                            p_age_in = gr.Textbox(label="Age (Years)", placeholder="54")
                            p_gender_in = gr.Dropdown(["Male", "Female", "Other"], value="Male", label="Sex")
                        p_eye_in = gr.Dropdown(["Right Eye (OD)", "Left Eye (OS)", "Bilateral (OU)"], value="Right Eye (OD)", label="Laterality")
                        with gr.Row():
                            p_hba1c_in = gr.Number(value=7.5, label="HbA1c (%)")
                            p_years_in = gr.Number(value=5, label="Diabetes Duration (Yrs)")
                            p_sbp_in = gr.Number(value=130, label="Systolic BP (mmHg)")
                        p_symptoms_in = gr.Textbox(
                            label="Clinical Notes & Current Medications (Indexed for RAG)",
                            placeholder="e.g. Blurred vision, pioglitazone therapy, hypertension...",
                            lines=2,
                        )

                    # Panel 03: Model & RAG Execution
                    with gr.Group():
                        gr.HTML("""
                        <div class="panel-header" style="margin-top:0.8rem;">
                            <div>
                                <span class="step-badge">03</span>
                                <strong style="color:#f9fafb; margin-left:6px; font-size:0.92rem;">Model &amp; RAG Execution</strong>
                            </div>
                            <span class="xai-pill">256-D Vector DB</span>
                        </div>
                        """)
                        backbone_in = gr.Dropdown(
                            choices=list(BACKBONE_MAP.keys()),
                            value="EfficientNet-B0 (Hybrid CBAM)",
                            label="Hybrid CNN-CBAM Backbone",
                        )
                        top_k_in = gr.Slider(2, 5, value=3, step=1, label="Top-K Clinical Guideline Chunks")
                        gemini_key_in = gr.Textbox(label="Optional Gemini API Key (Blank = Local Grounded Synthesizer)", type="password")

                        with gr.Row():
                            run_btn = gr.Button("Run Screening & RAG", variant="primary", scale=2)
                            clear_btn = gr.ClearButton([fundus_in, p_name_in, p_id_in, p_symptoms_in], value="Reset", scale=1)

                # --- RIGHT COLUMN: CLINICAL WORKSTATION RESULTS ---
                with gr.Column(scale=2):
                    # Section Header
                    gr.HTML("""
                    <div class="panel-header">
                        <div>
                            <span class="step-badge">REPORT</span>
                            <strong style="color:#f9fafb; margin-left:6px; font-size:0.95rem;">Severity Assessment &amp; Vector DB Evidence</strong>
                        </div>
                        <span class="xai-pill">Grounded Clinical RAG Synthesizer</span>
                    </div>
                    """)

                    # Dual Image & Evidence Card Row
                    with gr.Row():
                        input_fundus_display = gr.Image(type="pil", label="Input Retinal Fundus Scan", height=280)
                        evidence_card_display = gr.HTML(EMPTY_STATE_HTML, label="Vector DB Similarity & RAG Evidence Profile")

                    # Severity & Grounded RAG Summary Card
                    report_card_out = gr.HTML()

                    # Top-3 Visual Cases & Guidelines
                    with gr.Accordion("Visual Vector DB — Top-3 Matched Verified Reference Cases (256-D Embedding)", open=True):
                        top3_cases_out = gr.HTML()

                    with gr.Accordion("Clinical Guideline Vector DB — Retrieved Evidence Chunks & Citations", open=False):
                        guidelines_out = gr.HTML()

                    # Safeguards Panel
                    safeguards_out = gr.HTML()

                    # PredictRx Retinal-Systemic Knowledge Graph
                    with gr.Accordion("PredictRx Retinal-Systemic Knowledge Graph", open=True):
                        kg_svg_out = gr.HTML()

                    # Download PDF Report Button
                    pdf_download_out = gr.File(label="Download Verified Zero-PHI Clinical DR RAG & Triage PDF Report")

            # Wire click handler
            run_btn.click(
                fn=run_screening_pipeline,
                inputs=[
                    fundus_in, use_bg_cb, backbone_in, top_k_in,
                    p_name_in, p_id_in, p_age_in, p_gender_in, p_eye_in,
                    p_hba1c_in, p_years_in, p_sbp_in, p_symptoms_in, gemini_key_in,
                ],
                outputs=[
                    report_card_out,
                    evidence_card_display,
                    top3_cases_out,
                    guidelines_out,
                    safeguards_out,
                    kg_svg_out,
                    pdf_download_out,
                    kpi_bar_html,
                ],
            )
            # Link input image to preview
            fundus_in.change(lambda img: img, inputs=[fundus_in], outputs=[input_fundus_display])

        # Tab 2: Vector DB Explorer & Follow-Up Q&A
        with gr.Tab("Vector DB Explorer & Knowledge Graph"):
            with gr.Row():
                qa_grade = gr.Dropdown(
                    choices=[f"Grade {g}: {DR_KNOWLEDGE_BASE[g]['name']}" for g in range(5)],
                    value="Grade 2: Moderate NPDR",
                    label="ICDR Severity Stage Context",
                )
                qa_hba1c = gr.Number(value=8.5, label="Simulated HbA1c (%)")
                qa_years = gr.Number(value=10, label="Diabetes Duration (yrs)")
                qa_sbp = gr.Number(value=145, label="Systolic BP (mmHg)")
            qa_question = gr.Textbox(
                value="What medication safeguards apply when lowering HbA1c rapidly or treating macular edema risk?",
                label="Ask the PredictRx Clinical Vector Database",
            )
            qa_gemini = gr.Textbox(label="Optional Gemini API Key", type="password")
            qa_btn = gr.Button("Query Clinical Vector DB", variant="primary")
            qa_answer_out = gr.HTML()
            qa_kg_out = gr.HTML()

            qa_btn.click(
                fn=run_standalone_qa,
                inputs=[qa_question, qa_grade, qa_hba1c, qa_years, qa_sbp, qa_gemini],
                outputs=[qa_answer_out, qa_kg_out],
            )

        # Tab 3: Architecture & Safeguards
        with gr.Tab("Architecture & Benchmarks"):
            gr.HTML(f"""
            <div class="panel-box">
                <h3 style="color:#f9fafb; margin-top:0;">Hybrid CNN-CBAM + Dual Vector DB RAG Architecture</h3>
                <p style="color:#9ca3af; font-size:0.86rem; line-height:1.6;">
                    Built for <b>AI Builders Congress — HealthSphere AI (PredictRx)</b>. Combines 5-stage ICDR Diabetic Retinopathy grading across <b>EfficientNet-B0, DenseNet121, and ResNet50</b> with sequential <b>CBAM Dual Attention</b>, L2-normalized <b>256-D retinal embeddings</b>, a <b>Visual Case Vector DB</b> ({len(REFERENCE_RETINAL_CASES)} prototypes), and a <b>Clinical Guideline &amp; Medication Safeguard Vector DB</b> ({len(CLINICAL_GUIDELINE_CORPUS)} authoritative documents).
                </p>
                <ul style="color:#d1d5db; font-size:0.84rem; line-height:1.65;">
                    <li><b>Pre-Inference OOD Safeguard:</b> Rejects monochrome, blurred, or non-retinal uploads automatically.</li>
                    <li><b>HITL Escalation Gate:</b> Flags low-confidence (&lt;70%) or Vision-Threatening DR (Grade ≥ 3) for urgent specialist verification.</li>
                    <li><b>Zero-PHI Data Sovereignty:</b> All image inference, vector retrieval, and PDF report compilation run locally in memory.</li>
                </ul>

                <h4 style="color:#f9fafb; margin-top:1.5rem; margin-bottom:0.5rem;">Empirical Hold-Out Validation Benchmarks (APTOS 2019)</h4>
                <table style="width:100%; border-collapse:collapse; color:#d1d5db; font-size:0.83rem; text-align:left;">
                    <thead>
                        <tr style="border-bottom:1px solid #2d3037; color:#9ca3af;">
                            <th style="padding:6px 8px;">Backbone</th>
                            <th style="padding:6px 8px;">Hold-Out Val QWK</th>
                            <th style="padding:6px 8px;">Val Accuracy</th>
                            <th style="padding:6px 8px;">Macro F1</th>
                            <th style="padding:6px 8px;">Best Epoch</th>
                            <th style="padding:6px 8px;">Rank</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #23262d;">
                            <td style="padding:6px 8px;"><b>Hybrid EfficientNet-B0 + CBAM</b></td>
                            <td style="padding:6px 8px; color:#34d399;"><b>0.8766</b></td>
                            <td style="padding:6px 8px;">80.87%</td>
                            <td style="padding:6px 8px;">0.6880</td>
                            <td style="padding:6px 8px;">Epoch 16/20</td>
                            <td style="padding:6px 8px; color:#34d399;">🏆 Top Performer</td>
                        </tr>
                        <tr style="border-bottom:1px solid #23262d;">
                            <td style="padding:6px 8px;"><b>Hybrid DenseNet121 + CBAM</b></td>
                            <td style="padding:6px 8px; color:#60a5fa;"><b>0.8727</b></td>
                            <td style="padding:6px 8px;">78.14%</td>
                            <td style="padding:6px 8px;">0.6281</td>
                            <td style="padding:6px 8px;">Early Stop 17</td>
                            <td style="padding:6px 8px; color:#60a5fa;">🥈 Runner-Up</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 8px;"><b>Hybrid ResNet50 + CBAM</b></td>
                            <td style="padding:6px 8px; color:#9ca3af;"><b>0.8588</b></td>
                            <td style="padding:6px 8px;">74.86%</td>
                            <td style="padding:6px 8px;">0.5814</td>
                            <td style="padding:6px 8px;">Early Stop 15</td>
                            <td style="padding:6px 8px; color:#9ca3af;">🥉 Verified Baseline</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
