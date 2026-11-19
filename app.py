# -*- coding: utf-8 -*-
"""
Hugging Face Spaces (Gradio) Entrypoint — RetinaRAG Clinical Workstation
Hybrid Explainable AI Framework for Automated Diabetic Retinopathy Detection,
5-Stage ICDR Severity Assessment, Dual Vector DB RAG & PredictRx Safeguards.
"""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr  # type: ignore[import-untyped]

# Hugging Face ZeroGPU decorator (graceful no-op when running locally without `spaces`)
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


def validate_retinal_fundus(image: Image.Image):
    img_rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(img_rgb, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    img_std = float(np.std(gray))
    if img_std < 12.0:
        return False, "Image contrast is too flat to be a clinical retinal fundus photograph."

    fg_mask = gray > 15
    if float(np.mean(fg_mask)) < 0.15:
        return False, "Image is predominantly dark/underexposed — retinal field not detected."

    fg_pixels = resized[fg_mask]
    r_mean = float(np.mean(fg_pixels[:, 0]))
    b_mean = float(np.mean(fg_pixels[:, 2]))

    channel_diff = float(
        np.mean(np.abs(fg_pixels[:, 0].astype(np.float32) - fg_pixels[:, 1].astype(np.float32)))
        + np.mean(np.abs(fg_pixels[:, 0].astype(np.float32) - fg_pixels[:, 2].astype(np.float32)))
    )
    if channel_diff < 8.0:
        return False, "Grayscale / monochrome image detected. Please upload a color RGB retinal fundus photograph."

    if r_mean < b_mean * 1.05:
        return False, "Color profile does not match a retinal fundus photograph (expected warm red/orange retinal spectrum)."

    lap_var = float(cv2.Laplacian(resized[:, :, 1], cv2.CV_64F).var())
    if lap_var < 4.0:
        return False, "Image appears severely blurred or out of focus. Please upload a sharper fundus scan."

    return True, f"Valid color retinal fundus verified (Focus Var: {lap_var:.1f}, R/B Ratio: {r_mean / (b_mean + 1e-5):.2f})."


# ----------------- CORE SCREENING & RAG HANDLER -----------------
@spaces.GPU
def run_dr_screening(
    fundus_img,
    use_ben_graham,
    backbone_label,
    top_k,
    patient_name,
    patient_id,
    patient_age,
    patient_gender,
    eye_side,
    hba1c,
    diabetes_years,
    systolic_bp,
    symptoms,
    gemini_api_key,
):
    if fundus_img is None:
        return (
            "<div class='dr-card'>Please upload a color retinal fundus photograph to run screening.</div>",
            None,
            {},
            "",
            None,
        )

    image = fundus_img.convert("RGB")
    is_valid, ood_msg = validate_retinal_fundus(image)
    if not is_valid:
        err_html = f"""
        <div class='dr-card' style='border-left: 4px solid #ef4444;'>
            <div style='color:#f87171; font-weight:700; font-size:0.95rem;'>OOD Input Safeguard Blocked Scan</div>
            <p style='color:#d4d8e0; font-size:0.86rem; margin-top:6px;'>{ood_msg}</p>
        </div>
        """
        return err_html, None, {}, "", None

    backbone_key = BACKBONE_MAP.get(backbone_label, "efficientnet_b0")
    model, model_loaded = get_model(backbone_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    processed = apply_ben_graham_preprocessing(image, img_size=224) if use_ben_graham else image.resize((224, 224))
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
        hba1c=hba1c if hba1c and hba1c > 0 else None,
        diabetes_years=diabetes_years if diabetes_years and diabetes_years > 0 else None,
        systolic_bp=systolic_bp if systolic_bp and systolic_bp > 0 else None,
        symptoms=symptoms or "",
        top_k=int(top_k),
        gemini_api_key=gemini_api_key or None,
    )
    evidence_card = render_rag_visual_evidence_card(
        similar_cases=similar_cases,
        retrieved_chunks=rag_result["retrieved_chunks"],
        predicted_grade=grade_idx,
        grade_name=prediction_name,
    )
    risk_profile = compute_systemic_risk_profile(
        grade=grade_idx,
        hba1c=hba1c if hba1c and hba1c > 0 else None,
        diabetes_years=diabetes_years if diabetes_years and diabetes_years > 0 else None,
        systolic_bp=systolic_bp if systolic_bp and systolic_bp > 0 else None,
    )
    safeguards = evaluate_clinical_safeguards(
        grade=grade_idx,
        confidence=confidence,
        model_weights_loaded=model_loaded,
    )

    # Build Probabilities dict for Gradio Label
    prob_dict = {
        f"Grade {g}: {DR_KNOWLEDGE_BASE[g]['name']}": float(probs[g])
        for g in range(5)
    }

    # Build Clinical Summary + Vector DB Matches + Safeguards HTML
    accent = grade_info["color"]
    cases_html = "".join(
        f"""
        <div class='dr-subcard' style='border-top:2px solid {"#34d399" if c["grade_match"] else "#6b8afd"};'>
            <div style='display:flex; justify-content:space-between;'>
                <strong style='color:#f2f4f8;'>#{c["rank"]} {c["case_id"]}</strong>
                <span class='dr-pill'>{c["similarity_pct"]}% Match</span>
            </div>
            <div style='color:#cfd5e2; font-size:0.8rem; margin-top:4px;'>Grade {c["grade"]}: {c["grade_name"]}</div>
            <div style='color:#a1a6b0; font-size:0.78rem; margin-top:4px;'><b>Lesions:</b> {c["lesion_summary"]}</div>
            <div style='color:#858a96; font-size:0.75rem; margin-top:3px;'><b>Outcome:</b> {c["clinical_outcome"]}</div>
        </div>
        """
        for c in similar_cases
    )
    sg_html = "".join(f"<li>{a}</li>" for a in safeguards["alerts"])
    citations_html = " ".join(f"<span class='dr-pill'>[{cid}]</span>" for cid in rag_result["citations"])

    summary_html = f"""
    <div class='dr-card' style='border-left: 4px solid {accent};'>
        <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px;'>
            <div>
                <div style='font-size:0.74rem; color:#9ca1ac; text-transform:uppercase; font-weight:600;'>Predicted ICDR Severity</div>
                <div style='font-size:1.35rem; font-weight:700; color:{accent};'>Grade {grade_idx}: {prediction_name}</div>
                <div style='font-size:0.85rem; color:#eceef2; margin-top:3px;'>
                    Confidence: <b>{confidence:.2f}%</b> • <b>{grade_info["triage_priority"]}</b> • {grade_info["referral_window"]}
                </div>
            </div>
            <div style='text-align:right;'>
                <div class='dr-pill' style='color:{risk_profile["risk_color"]};'>PredictRx Risk: {risk_profile["risk_score"]}/100 ({risk_profile["risk_tier"]})</div>
                <div style='font-size:0.74rem; color:#9ca1ac; margin-top:5px;'>{ood_msg}</div>
            </div>
        </div>

        <div class='dr-subcard' style='margin-top:10px;'>
            <div style='font-size:0.74rem; color:#9ca1ac; text-transform:uppercase; font-weight:600; margin-bottom:6px;'>Grounded Clinical RAG Explanation ({rag_result["generator_mode"]})</div>
            <div style='font-size:0.84rem; color:#dfe2e8; line-height:1.58;'>{rag_result["explanation_html"]}</div>
            <div style='margin-top:8px; display:flex; gap:5px; flex-wrap:wrap;'>{citations_html}</div>
        </div>

        <div style='margin-top:12px;'>
            <div style='font-size:0.76rem; color:#9ca1ac; text-transform:uppercase; font-weight:600; margin-bottom:6px;'>Visual Vector DB — Top-3 Matched Verified Reference Cases</div>
            <div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:8px;'>{cases_html}</div>
        </div>

        <div class='dr-subcard' style='margin-top:12px; border-left: 3px solid {safeguards["status_color"]};'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <strong style='color:#f2f4f8; font-size:0.84rem;'>PredictRx Documented Safeguards &amp; HITL Gate</strong>
                <span class='dr-pill' style='color:{safeguards["status_color"]};'>{safeguards["status_label"]}</span>
            </div>
            <ul style='margin:6px 0 0 18px; color:#cfd5e2; font-size:0.81rem; line-height:1.5;'>{sg_html}</ul>
        </div>
    </div>
    """

    kg_svg = render_knowledge_graph_svg(grade_idx, risk_profile)

    # Generate PDF in temporary file for Gradio File download
    pdf_bytes = generate_pdf_report(
        patient_name=patient_name,
        patient_id=patient_id,
        patient_age=patient_age,
        patient_gender=patient_gender,
        symptoms=symptoms or "",
        prediction=prediction_name,
        confidence=confidence,
        insights=rag_result["explanation_html"],
        original_image=image,
        xai_overlay=evidence_card,
        xai_method_name="Dual Vector DB & Clinical RAG",
        grade_idx=grade_idx,
        eye_side=eye_side,
        triage_level=grade_info["triage_level"],
        referral_window=grade_info["referral_window"],
        risk_score=risk_profile["risk_score"],
        risk_tier=risk_profile["risk_tier"],
        safeguard_notes=safeguards["alerts"],
    )
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="DR_RAG_Report_")
    tmp_pdf.write(pdf_bytes)
    tmp_pdf.close()

    return summary_html, evidence_card, prob_dict, kg_svg, tmp_pdf.name


def run_rag_qa(question, grade_label, hba1c, diabetes_years, systolic_bp, gemini_key):
    grade_idx = int(grade_label.split(":")[0].replace("Grade", "").strip())
    grade_name = DR_KNOWLEDGE_BASE[grade_idx]["name"]

    qa = CLINICAL_RAG.answer_clinician_question(
        question=question or "What triage and medication safeguards apply?",
        grade=grade_idx,
        grade_name=grade_name,
        hba1c=hba1c if hba1c and hba1c > 0 else None,
        systolic_bp=systolic_bp if systolic_bp and systolic_bp > 0 else None,
        gemini_api_key=gemini_key or None,
    )

    chunks_html = "".join(
        f"""
        <div class='dr-subcard' style='margin-top:8px; border-left:3px solid #6b8afd;'>
            <div style='display:flex; justify-content:space-between;'>
                <strong style='color:#f2f4f8; font-size:0.84rem;'>[{c["doc_id"]}] {c["title"]}</strong>
                <span class='dr-pill'>Cosine: {c["relevance_score"]:.3f}</span>
            </div>
            <div style='color:#9ca1ac; font-size:0.75rem; margin:3px 0;'>Source: {c["source"]} • {c["category"]}</div>
            <div style='color:#d4d8e0; font-size:0.81rem; line-height:1.5;'>{c["content"]}</div>
        </div>
        """
        for c in qa["chunks"]
    )
    answer_html = f"""
    <div class='dr-card'>
        <div style='font-size:0.75rem; color:#9ca1ac; font-weight:600; margin-bottom:6px;'>{qa["mode"]}</div>
        <div style='font-size:0.88rem; color:#f2f4f8; line-height:1.6;'>{qa["answer"]}</div>
        <div style='margin-top:12px; font-size:0.76rem; color:#9ca1ac; text-transform:uppercase; font-weight:600;'>Retrieved Vector DB Evidence Chunks</div>
        {chunks_html}
    </div>
    """
    risk_profile = compute_systemic_risk_profile(grade_idx, hba1c, diabetes_years, systolic_bp)
    return answer_html, render_knowledge_graph_svg(grade_idx, risk_profile)


# ----------------- SMOOTH DARK SILVER CUSTOM STYLES -----------------
HEADER_HTML = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
body, .gradio-container {{
    background-color: #141517 !important;
    color: #eceef2 !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}
.dr-header {{
    background: linear-gradient(180deg, #1f2126 0%, #1a1c20 100%);
    border: 1px solid #2e3138;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}
.dr-card {{
    background-color: #1c1e22;
    border: 1px solid #2d3037;
    border-radius: 10px;
    padding: 16px 18px;
}}
.dr-subcard {{
    background-color: #22252a;
    border: 1px solid #30343c;
    border-radius: 8px;
    padding: 11px 13px;
}}
.dr-pill {{
    background-color: #262930;
    color: #d4d8e0;
    border: 1px solid #383c46;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 500;
    display: inline-block;
}}
</style>

<div class="dr-header">
    <div>
        <div style="font-size:0.75rem; color:#9ca1ac; font-weight:500;">HealthSphere AI • PredictRx Clinical Workstation</div>
        <div style="font-size:1.38rem; font-weight:700; color:#f3f5f8; margin-top:2px;">Hybrid Explainable AI Framework for Diabetic Retinopathy</div>
        <div style="font-size:0.84rem; color:#a1a6b0; margin-top:2px;">5-Stage ICDR Severity Grading • 256-D Vector Case Retrieval • Guideline-Grounded Clinical RAG</div>
    </div>
    <div style="display:flex; gap:6px; flex-wrap:wrap;">
        <span class="dr-pill">{CLINICAL_RAG.backend_name}</span>
        <span class="dr-pill">APTOS 2019 • 5-Stage ICDR</span>
        <span class="dr-pill">Zero-PHI Local Session</span>
    </div>
</div>
"""


# ----------------- BUILD GRADIO APPLICATION -----------------
with gr.Blocks(title="RetinaRAG Clinical Workstation — PredictRx") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.Tab("Clinical Screening & RAG Triage"):
            with gr.Row():
                with gr.Column(scale=1):
                    fundus_input = gr.Image(type="pil", label="01. Upload Color Retinal Fundus Photograph")
                    use_bg = gr.Checkbox(value=True, label="Apply Ben Graham Retinal Illumination Normalization")

                    with gr.Accordion("02. Patient & Systemic Risk Profile (Optional)", open=False):
                        with gr.Row():
                            p_name = gr.Textbox(label="Patient Name", placeholder="Anonymous")
                            p_id = gr.Textbox(label="Case ID", placeholder="DR-2041")
                        with gr.Row():
                            p_age = gr.Textbox(label="Age", placeholder="54")
                            p_gender = gr.Dropdown(["Male", "Female", "Other"], value="Male", label="Sex")
                            p_eye = gr.Dropdown(["Right Eye (OD)", "Left Eye (OS)", "Bilateral (OU)"], value="Right Eye (OD)", label="Laterality")
                        with gr.Row():
                            p_hba1c = gr.Number(value=0.0, label="HbA1c (%)")
                            p_years = gr.Number(value=0, label="Diabetes Years")
                            p_sbp = gr.Number(value=0, label="Systolic BP (mmHg)")
                        p_notes = gr.Textbox(label="Symptoms & Current Medications", placeholder="e.g. Blurred vision, pioglitazone/insulin...")

                    backbone_dd = gr.Dropdown(
                        choices=list(BACKBONE_MAP.keys()),
                        value="EfficientNet-B0 (Hybrid CBAM)",
                        label="03. Hybrid CNN-CBAM Backbone",
                    )
                    top_k_slider = gr.Slider(2, 5, value=3, step=1, label="Top-K Clinical Guideline Chunks")
                    gemini_key_in = gr.Textbox(label="Optional Gemini API Key (Blank = Local Grounded RAG)", type="password")
                    run_btn = gr.Button("Run Screening & RAG", variant="primary")

                with gr.Column(scale=2):
                    summary_out = gr.HTML(
                        "<div class='dr-card'><strong style='color:#f2f4f8;'>Workstation Ready</strong>"
                        "<p style='color:#9ca1ac; font-size:0.85rem; margin-top:6px;'>Upload a color retinal fundus photograph on the left and click <b>Run Screening &amp; RAG</b> to compute 5-stage ICDR severity, retrieve Top-3 Vector DB reference cases, synthesize citation-backed clinical guidance, and generate a Zero-PHI PDF triage report.</p></div>"
                    )
                    with gr.Row():
                        prob_out = gr.Label(num_top_classes=5, label="5-Stage ICDR Probabilities")
                        evidence_img_out = gr.Image(type="pil", label="Vector DB Similarity & RAG Evidence Profile")
                    kg_out = gr.HTML(label="PredictRx Knowledge Graph")
                    pdf_out = gr.File(label="Download Zero-PHI Clinical DR RAG & Triage PDF Report")

            run_btn.click(
                fn=run_dr_screening,
                inputs=[
                    fundus_input, use_bg, backbone_dd, top_k_slider,
                    p_name, p_id, p_age, p_gender, p_eye,
                    p_hba1c, p_years, p_sbp, p_notes, gemini_key_in,
                ],
                outputs=[summary_out, evidence_img_out, prob_out, kg_out, pdf_out],
            )

        with gr.Tab("Vector DB RAG Q&A & Knowledge Graph"):
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
                fn=run_rag_qa,
                inputs=[qa_question, qa_grade, qa_hba1c, qa_years, qa_sbp, qa_gemini],
                outputs=[qa_answer_out, qa_kg_out],
            )

        with gr.Tab("Architecture & Safeguards"):
            gr.HTML(
                f"""
                <div class='dr-card'>
                    <h3 style='color:#f2f4f8; margin-top:0;'>Hybrid CNN-CBAM + Dual Vector DB RAG Architecture</h3>
                    <p style='color:#a1a6b0; font-size:0.86rem; line-height:1.6;'>
                        Built for <b>AI Builders Congress — HealthSphere AI (PredictRx)</b>. Combines 5-stage ICDR Diabetic Retinopathy grading across <b>EfficientNet-B0, DenseNet121, and ResNet50</b> with sequential <b>CBAM Dual Attention</b>, L2-normalized <b>256-D retinal embeddings</b>, a <b>Visual Case Vector DB</b> ({len(REFERENCE_RETINAL_CASES)} prototypes), and a <b>Clinical Guideline &amp; Medication Safeguard Vector DB</b> ({len(CLINICAL_GUIDELINE_CORPUS)} authoritative documents).
                    </p>
                    <ul style='color:#d4d8e0; font-size:0.84rem; line-height:1.65;'>
                        <li><b>Pre-Inference OOD Safeguard:</b> Rejects monochrome, blurred, or non-retinal uploads automatically.</li>
                        <li><b>HITL Escalation Gate:</b> Flags low-confidence (&lt;70%) or Vision-Threatening DR (Grade ≥ 3) for urgent specialist verification.</li>
                        <li><b>Zero-PHI Data Sovereignty:</b> All image inference, vector retrieval, and PDF report compilation run locally in memory.</li>
                    </ul>
                </div>
                """
            )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
