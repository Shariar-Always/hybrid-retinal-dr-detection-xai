# -*- coding: utf-8 -*-
"""
Hugging Face Spaces & Local Workstation Entrypoint — RetinaRAG Clinical Workstation
Launches the full interactive Streamlit Clinical Workstation UI from app/app.py.
"""
import sys
from pathlib import Path
import runpy

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

target_app = ROOT_DIR / "app" / "app.py"
runpy.run_path(str(target_app), run_name="__main__")
