import io
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage
import numpy as np


class InMemoryImage(RLImage):
    """
    Custom ReportLab Flowable Image class accepting a BytesIO buffer directly.
    Enables zero-persistence, in-memory PDF generation for data sovereignty
    and cloud compatibility.
    """
    def __init__(
        self,
        image_buffer,
        width=None,
        height=None,
        kind="direct",
        mask="auto",
        lazy=1,
        hAlign="CENTER",
        useDPI=False,
    ):
        self.hAlign = hAlign
        self._mask = mask
        self._drawing = None
        self._file = image_buffer
        self.filename = repr(image_buffer)
        self._dpi = useDPI
        self._img = ImageReader(image_buffer)
        self._setup(width, height, kind, 0)


def _grade_palette(grade_idx: int):
    palettes = {
        0: ("#dcfce7", "#22c55e", "#15803d"),
        1: ("#e0f2fe", "#0284c7", "#0369a1"),
        2: ("#fef9c3", "#eab308", "#854d0e"),
        3: ("#ffedd5", "#f97316", "#9a3412"),
        4: ("#fee2e2", "#ef4444", "#b91c1c"),
    }
    return palettes.get(grade_idx, ("#f1f5f9", "#64748b", "#334155"))


def generate_pdf_report(
    patient_name,
    patient_id,
    patient_age,
    patient_gender,
    symptoms,
    prediction,
    confidence,
    insights,
    original_image,
    xai_overlay,
    xai_method_name="RAG + Vector DB Evidence",
    grade_idx=0,
    eye_side="Unspecified",
    triage_level="Routine Screening",
    referral_window="12–24 Months Routine Follow-Up",
    risk_score=None,
    risk_tier=None,
    safeguard_notes=None,
):
    """
    Generates a print-ready, zero-persistence clinical PDF screening & triage report
    for Diabetic Retinopathy using ReportLab (entirely in memory).

    Returns:
        bytes: Binary PDF file payload.
    """
    orig_buffer = io.BytesIO()
    original_image.convert("RGB").save(orig_buffer, format="PNG")
    orig_buffer.seek(0)

    xai_buffer = io.BytesIO()
    if xai_overlay is None:
        original_image.convert("RGB").save(xai_buffer, format="PNG")
    elif isinstance(xai_overlay, np.ndarray):
        if xai_overlay.dtype in (np.float32, np.float64):
            xai_overlay = (xai_overlay * 255).astype(np.uint8)
        PILImage.fromarray(xai_overlay).save(xai_buffer, format="PNG")
    else:
        xai_overlay.save(xai_buffer, format="PNG")
    xai_buffer.seek(0)

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15.5,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=5,
    )
    normal_bold = ParagraphStyle(
        "NormalBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
    )
    normal_text = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
    )

    # --- HEADER BLOCK ---
    header_data = [[
        Paragraph(
            "DIABETIC RETINOPATHY RAG &amp; XAI CLINICAL SCREENING REPORT<br/>"
            "<font color='#0284c7' size='8'>HealthSphere AI — PredictRx | Hybrid CNN-CBAM + Dual Vector DB &amp; Clinical RAG + Knowledge Graph</font>",
            title_style,
        )
    ]]
    header_table = Table(header_data, colWidths=[540])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)

    divider = Table([[""]], colWidths=[540])
    divider.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 2, colors.HexColor("#0ea5e9")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 8))

    # --- PATIENT INFO TABLE ---
    story.append(Paragraph("Patient &amp; Screening Information", section_heading))
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    patient_data = [
        [
            Paragraph("Patient Name:", normal_bold),
            Paragraph(str(patient_name) if patient_name else "Anonymous (Local Session)", normal_text),
            Paragraph("Patient ID / Ref:", normal_bold),
            Paragraph(str(patient_id) if patient_id else "N/A", normal_text),
        ],
        [
            Paragraph("Age / Gender:", normal_bold),
            Paragraph(f"{patient_age or 'N/A'} / {patient_gender or 'N/A'}", normal_text),
            Paragraph("Eye Laterality:", normal_bold),
            Paragraph(str(eye_side), normal_text),
        ],
        [
            Paragraph("Screening Timestamp:", normal_bold),
            Paragraph(date_str, normal_text),
            Paragraph("Data Sovereignty:", normal_bold),
            Paragraph("Zero-PHI In-Memory Session", normal_text),
        ],
    ]
    if symptoms and symptoms.strip():
        patient_data.append([
            Paragraph("Clinical Notes / Risk Factors:", normal_bold),
            Paragraph(str(symptoms), normal_text),
            "",
            "",
        ])

    patient_table = Table(patient_data, colWidths=[115, 155, 115, 155])
    patient_styles = [
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]
    if len(patient_data) > 3:
        patient_styles.append(("SPAN", (1, 3), (3, 3)))
    patient_table.setStyle(TableStyle(patient_styles))
    story.append(patient_table)
    story.append(Spacer(1, 8))

    # --- DIAGNOSTIC & TRIAGE SUMMARY CARD ---
    story.append(Paragraph("Severity Assessment &amp; Triage Priority", section_heading))
    bg_color, border_color, text_color = _grade_palette(grade_idx)

    result_style_bold = ParagraphStyle(
        "ResultBold",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor(text_color),
        alignment=TA_CENTER,
    )
    result_style_normal = ParagraphStyle(
        "ResultNormal",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_CENTER,
    )

    risk_line = (
        f" | PredictRx Systemic Risk: {risk_score}/100 ({risk_tier})"
        if risk_score is not None and risk_tier
        else ""
    )
    result_data = [
        [Paragraph(f"ICDR Grade {grade_idx}: {prediction} — {triage_level}", result_style_bold)],
        [Paragraph(f"AI Confidence: {confidence:.2f}% | Recommended Window: {referral_window}{risk_line}", result_style_normal)],
    ]
    result_table = Table(result_data, colWidths=[540])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor(border_color)),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 8))

    # --- VISUAL INTERPRETABILITY (SIDE BY SIDE) ---
    story.append(Paragraph("Visual &amp; Case-Based Interpretability (Retinal Fundus &amp; Vector DB RAG Match)", section_heading))
    img_w, img_h = 215, 215
    img_orig = InMemoryImage(orig_buffer, width=img_w, height=img_h)
    img_xai = InMemoryImage(xai_buffer, width=img_w, height=img_h)

    images_table = Table([[img_orig, img_xai]], colWidths=[260, 260])
    images_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#cbd5e1")),
        ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    labels_table = Table(
        [[
            Paragraph(
                "Original Retinal Fundus Photograph",
                ParagraphStyle("Lbl1", fontName="Helvetica-Bold", fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor("#475569")),
            ),
            Paragraph(
                f"{xai_method_name} Similarity &amp; Evidence Profile",
                ParagraphStyle("Lbl2", fontName="Helvetica-Bold", fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor("#475569")),
            ),
        ]],
        colWidths=[260, 260],
    )
    story.append(images_table)
    story.append(labels_table)
    story.append(Spacer(1, 6))

    # --- RAG INSIGHTS & SAFEGUARDS ---
    story.append(Paragraph(f"{xai_method_name} Grounded Clinical Insights &amp; Documented Safeguards", section_heading))
    insights_clean = (
        insights.replace("•", "")
        .replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
    )
    insights_lines = [line.strip() for line in insights_clean.split("\n") if line.strip()]
    if safeguard_notes:
        for note in safeguard_notes:
            insights_lines.append(f"[Safeguard] {note}")

    bullet_style = ParagraphStyle(
        "InsightsBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=14,
        firstLineIndent=-9,
    )
    for line in insights_lines:
        story.append(Paragraph(f"• {line}", bullet_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))

    # --- SIGNATURE BLOCK & DISCLAIMER ---
    sig_data = [
        [
            Paragraph("Attending Ophthalmologist / Retina Specialist", normal_bold),
            Paragraph("Clinical Verification &amp; Audit", normal_bold),
        ],
        [
            Paragraph("<br/><br/>________________________________________<br/>Signature &amp; Registration No.", normal_text),
            Paragraph("<br/><br/>Date: ________________________<br/>Screening Status: Confirmed / Amended", normal_text),
        ],
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    footer_elements = [
        sig_table,
        Spacer(1, 8),
        Paragraph(
            "<b>Responsible AI &amp; Medical Disclaimer:</b> This report is generated by an Explainable AI screening and triage "
            "decision-support prototype. It does not constitute a definitive medical diagnosis. All severity grades, XAI focus maps, "
            "and referral priorities must be reviewed and confirmed by a qualified ophthalmologist prior to clinical management. "
            "No patient images or health identifiers are retained on external servers (local in-memory processing).",
            disclaimer_style,
        ),
    ]
    story.append(KeepTogether(footer_elements))

    doc.build(story)
    pdf_val = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_val
