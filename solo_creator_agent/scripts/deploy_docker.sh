#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f ../.env ]; then
  echo "Missing ../.env. Copy .env.example to ../.env and fill API keys first."
  exit 1
fi

python3 -m src.mock_data
docker compose up -d --build
docker compose ps
