from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT / ".env")

SOLODECK_AUTH_DB_PATH = Path(
    os.getenv("SOLODECK_AUTH_DB", str(DATA_DIR / "solodeck_auth.db"))
).expanduser()

UPLOAD_DIR = DATA_DIR / "uploads"
