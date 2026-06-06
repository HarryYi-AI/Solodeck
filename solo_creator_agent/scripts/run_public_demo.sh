#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOLODECK_PYTHON="${SOLODECK_PYTHON:-/workspace/ylj/miniconda3/envs/py310/bin/python}"

"${SOLODECK_PYTHON}" -m src.mock_data
"${SOLODECK_PYTHON}" -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
