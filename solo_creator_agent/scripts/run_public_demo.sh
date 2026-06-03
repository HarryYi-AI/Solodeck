#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m src.mock_data
python3 -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
