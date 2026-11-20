# -*- coding: utf-8 -*-
"""
Automated Unit Test Suite for the Hybrid Explainable AI & Dual Vector DB RAG Framework
for Automated Diabetic Retinopathy Detection and Severity Assessment from Retinal Fundus Images.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_model import ChannelAttention, SpatialAttention, HybridDRPredictor
from src.preprocess import crop_fundus_borders, apply_ben_graham_preprocessing
from app.rag_engine import (
    RetinalVisualVectorDB,
    ClinicalGuidelineRAGEngine,
    render_rag_visual_evidence_card,
)
from app.report_generator import generate_pdf_report
from app.knowledge_graph import (
    DR_KNOWLEDGE_BASE,
    compute_systemic_risk_profile,
    evaluate_clinical_safeguards,
    render_knowledge_graph_svg,
)


class TestHybridDRArchitectureAndEmbeddings(unittest.TestCase):
    """Tests for CBAM attention blocks, 5-class logits, and 256-D Vector DB embeddings."""

    def test_cbam_attention_modules(self):
        dummy_features = torch.randn(2, 64, 14, 14)
        ca = ChannelAttention(in_planes=64, ratio=16)
        sa = SpatialAttention(kernel_size=7)

        ca_weights = ca(dummy_features)
        self.assertEqual(ca_weights.shape, (2, 64, 1, 1))

        refined = dummy_features * ca_weights
        sa_weights = sa(refined)
        self.assertEqual(sa_weights.shape, (2, 1, 14, 14))

    def test_all_hybrid_backbones_forward_and_embedding(self):
        x = torch.randn(2, 3, 224, 224)
        for backbone in ["densenet121", "resnet50", "efficientnet_b0"]:
            model = HybridDRPredictor(backbone=backbone, num_classes=5, pretrained=False)
            self.assertTrue(hasattr(model, "channel_attention"))
            self.assertTrue(hasattr(model, "spatial_attention"))
            model.eval()
            with torch.no_grad():
                logits, emb = model.forward_with_embedding(x)
            self.assertEqual(logits.shape, (2, 5))
            self.assertEqual(emb.shape, (2, 256))
            # Verify L2 normalization of extracted embeddings
            norms = torch.norm(emb, p=2, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))


class TestRetinalPreprocessing(unittest.TestCase):
    """Tests for retinal fundus border cropping and Ben Graham normalization."""

    def test_crop_and_ben_graham_preprocessing(self):
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[50:250, 50:250, 0] = 180
        arr[50:250, 50:250, 1] = 95
        arr[50:250, 50:250, 2] = 45

        cropped = crop_fundus_borders(arr, tol=7)
        self.assertEqual(cropped.shape[:2], (200, 200))

        pil_img = Image.fromarray(arr)
        processed = apply_ben_graham_preprocessing(pil_img, img_size=224)
        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.size, (224, 224))


class TestDualVectorDBAndClinicalRAG(unittest.TestCase):
    """Tests for Visual Case-Based Vector DB, Clinical Guideline RAG, and PDF Reporting."""

    @classmethod
    def setUpClass(cls):
        cls.visual_vdb = RetinalVisualVectorDB(embedding_dim=256)
        cls.clinical_rag = ClinicalGuidelineRAGEngine()
        cls.dummy_fundus = Image.new("RGB", (224, 224), color=(175, 90, 40))

    def test_visual_vector_db_case_retrieval(self):
        dummy_emb = np.random.randn(256).astype(np.float32)
        probs = [0.05, 0.10, 0.70, 0.10, 0.05]
        cases = self.visual_vdb.query_similar_cases(dummy_emb, predicted_grade=2, probs=probs, top_k=3)
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0]["grade"], 2)
        self.assertIn("similarity_pct", cases[0])

    def test_clinical_guideline_rag_retrieval_and_synthesis(self):
        dummy_emb = np.random.randn(256).astype(np.float32)
        probs = [0.02, 0.05, 0.13, 0.72, 0.08]
        cases = self.visual_vdb.query_similar_cases(dummy_emb, predicted_grade=3, probs=probs, top_k=3)
        rag_out = self.clinical_rag.generate_clinical_explanation(
            grade=3,
            grade_name="Severe NPDR",
            confidence=72.0,
            similar_cases=cases,
            hba1c=9.2,
            diabetes_years=14,
            systolic_bp=150,
            symptoms="Blurred vision",
            top_k=3,
        )
        self.assertEqual(len(rag_out["retrieved_chunks"]), 3)
        self.assertIn("ICDR-G3", rag_out["citations"])
        self.assertIn("Case-Based Vector Match", rag_out["explanation_html"])

    def test_interactive_clinician_rag_qa(self):
        qa = self.clinical_rag.answer_clinician_question(
            question="What medication safeguards apply to rapid HbA1c drop or pioglitazone?",
            grade=2,
            grade_name="Moderate NPDR",
            hba1c=9.5,
            systolic_bp=145,
        )
        self.assertIn("answer", qa)
        self.assertGreaterEqual(len(qa["chunks"]), 2)

    def test_rag_visual_evidence_card_and_pdf_report(self):
        dummy_emb = np.random.randn(256).astype(np.float32)
        probs = [0.05, 0.10, 0.70, 0.10, 0.05]
        cases = self.visual_vdb.query_similar_cases(dummy_emb, predicted_grade=2, probs=probs, top_k=3)
        chunks = self.clinical_rag.retrieve("Moderate NPDR referable DR medication safeguard", grade=2, top_k=3)

        card_img = render_rag_visual_evidence_card(cases, chunks, predicted_grade=2, grade_name="Moderate NPDR")
        self.assertIsInstance(card_img, Image.Image)
        self.assertGreaterEqual(card_img.size[0], 240)
        self.assertGreaterEqual(card_img.size[1], 240)

        pdf_bytes = generate_pdf_report(
            patient_name="Anonymous",
            patient_id="DR-1001",
            patient_age="58",
            patient_gender="Female",
            symptoms="Type 2 DM (12 yrs), mild visual blurring",
            prediction="Moderate NPDR",
            confidence=91.45,
            insights=DR_KNOWLEDGE_BASE[2]["insights_html"],
            original_image=self.dummy_fundus,
            xai_overlay=card_img,
            xai_method_name="Dual Vector DB & Clinical RAG",
            grade_idx=2,
            eye_side="Right Eye (OD)",
            triage_level=DR_KNOWLEDGE_BASE[2]["triage_level"],
            referral_window=DR_KNOWLEDGE_BASE[2]["referral_window"],
            risk_score=62,
            risk_tier="Elevated Microvascular & CV Risk",
            safeguard_notes=["Valid Fundus OOD check passed."],
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)


class TestPredictRxKnowledgeGraphAndSafeguards(unittest.TestCase):
    """Tests for PredictRx Knowledge Graph, Systemic Risk, and Safeguards."""

    def test_knowledge_base_has_all_five_icdr_grades(self):
        self.assertEqual(set(DR_KNOWLEDGE_BASE.keys()), {0, 1, 2, 3, 4})
        self.assertFalse(DR_KNOWLEDGE_BASE[0]["referable_dr"])
        self.assertTrue(DR_KNOWLEDGE_BASE[2]["referable_dr"])
        self.assertTrue(DR_KNOWLEDGE_BASE[4]["vtdr"])

    def test_systemic_risk_stratification(self):
        low = compute_systemic_risk_profile(grade=0, hba1c=6.5, diabetes_years=3, systolic_bp=120)
        high = compute_systemic_risk_profile(grade=4, hba1c=9.8, diabetes_years=18, systolic_bp=165)
        self.assertLess(low["risk_score"], high["risk_score"])
        self.assertGreaterEqual(high["risk_score"], 90)

    def test_clinical_safeguards_trigger_on_low_conf_and_vtdr(self):
        sg = evaluate_clinical_safeguards(grade=3, confidence=62.5, model_weights_loaded=True)
        self.assertTrue(sg["escalation_required"])
        self.assertGreaterEqual(len(sg["alerts"]), 2)

    def test_knowledge_graph_svg_renders(self):
        risk = compute_systemic_risk_profile(grade=3, hba1c=8.5, diabetes_years=12, systolic_bp=145)
        svg = render_knowledge_graph_svg(grade=3, risk_profile=risk)
        self.assertIn("<svg", svg)
        self.assertIn("Severe NPDR", svg)


if __name__ == "__main__":
    unittest.main()
