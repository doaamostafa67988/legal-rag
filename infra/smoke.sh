#!/usr/bin/env bash
# Is the environment Terraform built working? Health check, then a write through MLflow into Postgres.
#   infra/smoke.sh [http://localhost:5001]
set -euo pipefail
url="${1:-http://localhost:5001}"
echo "health: $(curl -fsS "$url/health")"
curl -fsS -X POST "$url/api/2.0/mlflow/experiments/create" \
  -H 'content-type: application/json' \
  -d "{\"name\": \"tf-smoke-$(date +%s)\"}"
echo
echo "ok: MLflow answers and can write to its database"
