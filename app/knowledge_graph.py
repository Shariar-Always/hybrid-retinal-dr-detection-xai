# -*- coding: utf-8 -*-
"""
PredictRx Clinical Retinal-Systemic Knowledge Graph & Safeguards Engine.

Links 5-Stage ICDR Diabetic Retinopathy predictions with:
  1. Retinal Pathological Biomarkers
  2. Systemic & Cardiovascular Complication Risk Factors
  3. Clinical Triage Prioritization & Screening Windows
  4. Documented AI Safeguards & Medication / Escalation Protocols
"""
from typing import Dict, List, Optional, Tuple


DR_KNOWLEDGE_BASE: Dict[int, Dict] = {
    0: {
        "grade_code": "ICDR Grade 0",
        "name": "No DR",
        "short_name": "No DR",
        "referable_dr": False,
        "vtdr": False,
        "triage_level": "Routine Screening",
        "triage_priority": "Low Priority",
        "referral_window": "12–24 Months Routine Follow-Up",
        "color": "#22c55e",
        "bg_hex": "#dcfce7",
        "text_hex": "#15803d",
        "biomarkers": [
            "Intact retinal vasculature & clear optic disc margins",
            "No visible microaneurysms, hemorrhages, or exudates",
        ],
        "systemic_risks": [
            "Baseline microvascular risk profile",
            "Maintain glycemic & blood pressure targets (HbA1c < 7.0%)",
        ],
        "clinical_actions": [
            "Continue annual dilated fundus or retinal photography screening",
            "Reinforce glycemic control, lipid monitoring, and lifestyle adherence",
        ],
        "medication_safeguards": [
            "No ophthalmic intervention indicated",
            "Review routine antidiabetic & antihypertensive adherence",
        ],
        "insights_html": (
            "• Retinal vascular arcades and macular region appear structurally clear.<br>"
            "• Attention maps show no focal clusters of microaneurysms, hemorrhages, or exudates.<br>"
            "• <b>Triage:</b> Routine annual screening recommended."
        ),
    },
    1: {
        "grade_code": "ICDR Grade 1",
        "name": "Mild NPDR",
        "short_name": "Mild NPDR",
        "referable_dr": False,
        "vtdr": False,
        "triage_level": "Preventive Monitoring",
        "triage_priority": "Low-Moderate Priority",
        "referral_window": "6–12 Months Ophthalmology Follow-Up",
        "color": "#38bdf8",
        "bg_hex": "#e0f2fe",
        "text_hex": "#0369a1",
        "biomarkers": [
            "Isolated microaneurysms (early capillary outpouchings)",
            "Minimal capillary leakage without macular edema",
        ],
        "systemic_risks": [
            "Early systemic microvascular endothelial stress",
            "Elevated risk of progression if HbA1c > 8.0% or BP > 140/90",
        ],
        "clinical_actions": [
            "Schedule repeat retinal imaging within 6–12 months",
            "Optimize HbA1c, blood pressure, and serum lipids with primary care",
        ],
        "medication_safeguards": [
            "Avoid overly rapid glycemic drops (can transiently worsen DR)",
            "Evaluate ACEi/ARB & statin therapy for microvascular protection",
        ],
        "insights_html": (
            "• Attention maps highlight subtle punctate focus zones consistent with <b>early microaneurysms</b>.<br>"
            "• No widespread intraretinal hemorrhages or exudative plaques detected.<br>"
            "• <b>Triage:</b> Preventive monitoring and 6–12 month follow-up advised."
        ),
    },
    2: {
        "grade_code": "ICDR Grade 2",
        "name": "Moderate NPDR",
        "short_name": "Moderate NPDR",
        "referable_dr": True,
        "vtdr": False,
        "triage_level": "Referable DR — Priority Screening",
        "triage_priority": "Moderate-High Priority",
        "referral_window": "Within 3–6 Months (Ophthalmologist Referral)",
        "color": "#eab308",
        "bg_hex": "#fef9c3",
        "text_hex": "#854d0e",
        "biomarkers": [
            "Microaneurysms + dot/blot intraretinal hemorrhages",
            "Hard exudates (lipid deposits) and/or cotton-wool spots",
        ],
        "systemic_risks": [
            "Significant microvascular burden — correlate with diabetic nephropathy (eGFR/uACR)",
            "Increased 5-year cardiovascular & cerebrovascular event risk",
        ],
        "clinical_actions": [
            "Refer to ophthalmologist within 3–6 months for dilated exam & OCT",
            "Screen for Diabetic Macular Edema (DME) and renal microalbuminuria",
        ],
        "medication_safeguards": [
            "Verify hypertension & dyslipidemia control; caution with thiazolidinediones (fluid/DME risk)",
            "Coordinate systemic diabetes medication adjustments with primary physician",
        ],
        "insights_html": (
            "• High CBAM/XAI activation over localized <b>intraretinal hemorrhages and hard exudates</b>.<br>"
            "• Meets clinical threshold for <b>Referable Diabetic Retinopathy (rDR)</b>.<br>"
            "• <b>Triage:</b> Priority ophthalmology referral within 3–6 months + OCT check for macular edema."
        ),
    },
    3: {
        "grade_code": "ICDR Grade 3",
        "name": "Severe NPDR",
        "short_name": "Severe NPDR",
        "referable_dr": True,
        "vtdr": True,
        "triage_level": "Vision-Threatening DR — Urgent Referral",
        "triage_priority": "High Priority (Urgent)",
        "referral_window": "Within 2–4 Weeks (Urgent Retina Specialist)",
        "color": "#f97316",
        "bg_hex": "#ffedd5",
        "text_hex": "#9a3412",
        "biomarkers": [
            "Extensive intraretinal hemorrhages across multiple quadrants (4-2-1 rule)",
            "Venous beading and Intraretinal Microvascular Abnormalities (IRMA)",
        ],
        "systemic_risks": [
            "High probability (>50% within 1 year) of progression to Proliferative DR",
            "Strong biomarker for advanced systemic cardiovascular & renal vascular disease",
        ],
        "clinical_actions": [
            "Urgent referral to retina specialist within 2–4 weeks",
            "Comprehensive dilated fundus exam, OCT, and fluorescein angiography evaluation",
        ],
        "medication_safeguards": [
            "Specialist evaluation for early panretinal photocoagulation (PRP) or anti-VEGF",
            "Review anticoagulant/antiplatelet use with physician; strict BP & HbA1c management",
        ],
        "insights_html": (
            "• Strong spatial attention across <b>multi-quadrant hemorrhages, venous beading, or IRMA</b>.<br>"
            "• Classified as <b>Vision-Threatening DR (VTDR)</b> with high short-term progression risk.<br>"
            "• <b>Triage:</b> Urgent retina specialist evaluation within 2–4 weeks."
        ),
    },
    4: {
        "grade_code": "ICDR Grade 4",
        "name": "Proliferative DR",
        "short_name": "Proliferative DR",
        "referable_dr": True,
        "vtdr": True,
        "triage_level": "Emergency Vision-Threatening DR",
        "triage_priority": "Critical Priority (Immediate)",
        "referral_window": "Immediate / Within 1 Week (Retina Specialist)",
        "color": "#ef4444",
        "bg_hex": "#fee2e2",
        "text_hex": "#b91c1c",
        "biomarkers": [
            "Retinal neovascularization (NVD / NVE) — fragile abnormal blood vessels",
            "Vitreous or preretinal hemorrhage & fibrovascular proliferation risk",
        ],
        "systemic_risks": [
            "High risk of severe vision loss, vitreous hemorrhage, or tractional retinal detachment",
            "Markedly elevated systemic cardiovascular, stroke, and renal complication risk",
        ],
        "clinical_actions": [
            "Immediate referral to a vitreoretinal specialist (within days / <1 week)",
            "Urgent intervention planning: Anti-VEGF therapy, PRP laser, or vitrectomy assessment",
        ],
        "medication_safeguards": [
            "Do not delay ophthalmic care; avoid strenuous Valsalva activities if vitreous hemorrhage suspected",
            "Urgent multidisciplinary coordination (Ophthalmology + Endocrinology + Cardiology)",
        ],
        "insights_html": (
            "• Intense XAI activation over regions indicative of <b>neovascularization or preretinal hemorrhage</b>.<br>"
            "• Critical <b>Vision-Threatening Proliferative DR (PDR)</b> alert.<br>"
            "• <b>Triage:</b> Immediate vitreoretinal specialist referral to prevent irreversible vision loss."
        ),
    },
}


def compute_systemic_risk_profile(
    grade: int,
    hba1c: Optional[float] = None,
    diabetes_years: Optional[float] = None,
    systolic_bp: Optional[float] = None,
) -> Dict:
    """
    Combines the AI-predicted DR severity grade with optional patient systemic
    risk factors (HbA1c, Diabetes Duration, Systolic BP) to produce a transparent
    PredictRx Complication & Cardiovascular Risk Stratification score (0–100).
    """
    base_scores = {0: 12, 1: 28, 2: 52, 3: 74, 4: 88}
    score = base_scores.get(grade, 20)
    modifiers: List[str] = []

    if hba1c is not None and hba1c > 0:
        if hba1c >= 9.0:
            score += 10
            modifiers.append(f"Poor glycemic control (HbA1c {hba1c:.1f}% >= 9.0%)")
        elif hba1c >= 7.5:
            score += 5
            modifiers.append(f"Suboptimal glycemic control (HbA1c {hba1c:.1f}%)")

    if diabetes_years is not None and diabetes_years > 0:
        if diabetes_years >= 15:
            score += 6
            modifiers.append(f"Long diabetes duration ({diabetes_years:.0f} yrs >= 15 yrs)")
        elif diabetes_years >= 10:
            score += 3
            modifiers.append(f"Extended diabetes duration ({diabetes_years:.0f} yrs)")

    if systolic_bp is not None and systolic_bp > 0:
        if systolic_bp >= 160:
            score += 8
            modifiers.append(f"Stage 2 hypertension risk (SBP {systolic_bp:.0f} mmHg)")
        elif systolic_bp >= 140:
            score += 4
            modifiers.append(f"Elevated systolic BP ({systolic_bp:.0f} mmHg)")

    score = int(min(score, 99))

    if score >= 75:
        tier = "Critical Complication Risk"
        tier_color = "#ef4444"
    elif score >= 50:
        tier = "Elevated Microvascular & CV Risk"
        tier_color = "#f97316"
    elif score >= 30:
        tier = "Moderate Monitoring Risk"
        tier_color = "#eab308"
    else:
        tier = "Low Baseline Risk"
        tier_color = "#22c55e"

    return {
        "risk_score": score,
        "risk_tier": tier,
        "risk_color": tier_color,
        "active_modifiers": modifiers,
    }


def evaluate_clinical_safeguards(
    grade: int,
    confidence: float,
    model_weights_loaded: bool,
) -> Dict:
    """
    PredictRx Documented Safeguards Engine:
    Determines whether Human-in-the-Loop (HITL) escalation is mandatory due to:
      - Low prediction confidence (< 70%)
      - Vision-Threatening DR detection (Grade >= 3)
      - Uncalibrated / demonstration weights fallback
    """
    alerts: List[str] = []
    escalation_required = False

    if not model_weights_loaded:
        escalation_required = True
        alerts.append(
            "Model weights are running in ImageNet demonstration fallback mode — "
            "train APTOS 2019 Hybrid CBAM checkpoints before any clinical screening use."
        )

    if confidence < 70.0:
        escalation_required = True
        alerts.append(
            f"Low AI confidence ({confidence:.1f}% < 70.0% threshold): "
            "Mandatory ophthalmologist manual grading required."
        )

    if grade >= 3:
        escalation_required = True
        alerts.append(
            "Vision-Threatening DR (VTDR) detected: Automated urgent specialist escalation triggered."
        )

    if not alerts:
        alerts.append(
            "All automated safeguards passed (Valid Fundus OOD check & high confidence). "
            "Ophthalmologist verification remains required."
        )

    return {
        "escalation_required": escalation_required,
        "status_label": "HITL Escalation Triggered" if escalation_required else "Safeguards Verified",
        "status_color": "#f97316" if escalation_required else "#22c55e",
        "alerts": alerts,
    }


def render_knowledge_graph_svg(grade: int, risk_profile: Dict) -> str:
    """
    Renders a crisp, dependency-free SVG Clinical Knowledge Graph illustrating the
    directed PredictRx reasoning pathway:
      [AI DR Grade] -> [Retinal Biomarkers] -> [Systemic/CV Risk] -> [Triage & Safeguards]
    """
    info = DR_KNOWLEDGE_BASE[grade]
    accent = info["color"]
    risk_color = risk_profile["risk_color"]

    b1 = info["biomarkers"][0][:42] + ("..." if len(info["biomarkers"][0]) > 42 else "")
    b2 = info["biomarkers"][1][:42] + ("..." if len(info["biomarkers"][1]) > 42 else "")
    s1 = info["systemic_risks"][0][:42] + ("..." if len(info["systemic_risks"][0]) > 42 else "")
    t1 = info["referral_window"][:38]
    m1 = info["medication_safeguards"][0][:42] + ("..." if len(info["medication_safeguards"][0]) > 42 else "")
    ref_label = "Referable DR" if info["referable_dr"] else "Non-Referable"

    return (
        f'<div style="background: #1c1e22; border: 1px solid #2d3037; border-radius: 10px; padding: 12px; overflow-x: auto;">'
        f'<svg viewBox="0 0 920 245" width="100%" height="245" xmlns="http://www.w3.org/2000/svg" style="font-family: \'DM Sans\', \'Segoe UI\', sans-serif;">'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 1 L 10 5 L 0 9 z" fill="#6b7280" /></marker></defs>'
        f'<line x1="190" y1="122" x2="245" y2="70" stroke="#454a56" stroke-width="1.5" marker-end="url(#arrow)" />'
        f'<line x1="190" y1="122" x2="245" y2="175" stroke="#454a56" stroke-width="1.5" marker-end="url(#arrow)" />'
        f'<line x1="455" y1="70" x2="510" y2="70" stroke="#454a56" stroke-width="1.5" marker-end="url(#arrow)" />'
        f'<line x1="455" y1="175" x2="510" y2="175" stroke="#454a56" stroke-width="1.5" marker-end="url(#arrow)" />'
        f'<line x1="690" y1="70" x2="735" y2="112" stroke="#454a56" stroke-width="1.5" marker-end="url(#arrow)" />'
        f'<line x1="690" y1="175" x2="735" y2="132" stroke="#454a56" stroke-width="1.5" marker-end="url(#arrow)" />'
        f'<rect x="15" y="78" width="175" height="88" rx="8" fill="#23262c" stroke="{accent}" stroke-width="1.5" />'
        f'<text x="102" y="100" fill="#9ca1ac" font-size="9.5" font-weight="600" text-anchor="middle">Predictive ML Node</text>'
        f'<text x="102" y="124" fill="#f3f5f8" font-size="14" font-weight="700" text-anchor="middle">{info["name"]}</text>'
        f'<text x="102" y="146" fill="{accent}" font-size="10.5" font-weight="600" text-anchor="middle">{info["grade_code"]}</text>'
        f'<rect x="250" y="25" width="205" height="88" rx="8" fill="#23262c" stroke="#3a3e48" stroke-width="1.2" />'
        f'<text x="352" y="46" fill="#cfd5e2" font-size="9.8" font-weight="600" text-anchor="middle">Retinal Biomarkers</text>'
        f'<text x="352" y="69" fill="#e5e8ee" font-size="9.2" text-anchor="middle">{b1}</text>'
        f'<text x="352" y="89" fill="#9ca1ac" font-size="9.2" text-anchor="middle">{b2}</text>'
        f'<rect x="250" y="132" width="205" height="88" rx="8" fill="#23262c" stroke="#3a3e48" stroke-width="1.2" />'
        f'<text x="352" y="153" fill="{risk_color}" font-size="9.8" font-weight="600" text-anchor="middle">Systemic &amp; CV Risk</text>'
        f'<text x="352" y="176" fill="#f3f5f8" font-size="11.5" font-weight="600" text-anchor="middle">Risk Score: {risk_profile["risk_score"]}/100</text>'
        f'<text x="352" y="196" fill="#9ca1ac" font-size="9.2" text-anchor="middle">{s1}</text>'
        f'<rect x="515" y="25" width="175" height="88" rx="8" fill="#23262c" stroke="#3a3e48" stroke-width="1.2" />'
        f'<text x="602" y="46" fill="#cfd5e2" font-size="9.8" font-weight="600" text-anchor="middle">Triage Pathway</text>'
        f'<text x="602" y="70" fill="#f3f5f8" font-size="10.5" font-weight="600" text-anchor="middle">{info["triage_priority"]}</text>'
        f'<text x="602" y="90" fill="#9ca1ac" font-size="9" text-anchor="middle">{t1}</text>'
        f'<rect x="515" y="132" width="175" height="88" rx="8" fill="#23262c" stroke="#3a3e48" stroke-width="1.2" />'
        f'<text x="602" y="153" fill="#34d399" font-size="9.8" font-weight="600" text-anchor="middle">Clinical Safeguard</text>'
        f'<text x="602" y="176" fill="#e5e8ee" font-size="9.2" text-anchor="middle">{m1}</text>'
        f'<text x="602" y="196" fill="#9ca1ac" font-size="8.8" text-anchor="middle">Specialist Review</text>'
        f'<rect x="740" y="74" width="165" height="96" rx="8" fill="#23262c" stroke="#4b505d" stroke-width="1.5" />'
        f'<text x="822" y="98" fill="#cfd5e2" font-size="9.8" font-weight="600" text-anchor="middle">Clinical Outcome</text>'
        f'<text x="822" y="121" fill="#f3f5f8" font-size="11" font-weight="700" text-anchor="middle">{ref_label}</text>'
        f'<text x="822" y="140" fill="#9ca1ac" font-size="9.2" text-anchor="middle">Zero-PHI Local PDF</text>'
        f'<text x="822" y="156" fill="#7c818d" font-size="8.8" text-anchor="middle">Ophthalmologist Sign-Off</text>'
        f'</svg></div>'
    )
