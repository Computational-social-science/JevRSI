#!/usr/bin/env bash
# OASP nightly campaign — the 24/7 operational unit (implementation_plan.md §4).
#
# Design: one batch = one resumable pass. Safe to run unattended because every
# stage is idempotent: the runner and noise-floor scripts skip completed cells, and
# the analyser only reads. A crash, reboot, or full disk therefore costs at most the
# current cell, never the campaign.
#
# Install as a Hermes cron job, e.g.:
#   cronjob_manage(action='create', schedule='0 1 * * *',
#                  command='bash /d/2026-AI4S/autoresearch/scripts/nightly_campaign.sh')
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGDIR="runs/_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/nightly_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== OASP nightly campaign $(date -Iseconds) ==="
echo "root=$ROOT"

# --- preflight ------------------------------------------------------------
fail() { echo "[FAIL] $*"; exit 1; }

if ! curl -s --max-time 5 http://127.0.0.1:11434/api/tags >/dev/null; then
  fail "Ollama not reachable on 127.0.0.1:11434 — is the daemon running?"
fi
echo "[OK] Ollama reachable"

# Disk floor. D: sits at ~98% used; results are ~0.7 KB/cell, so this is a guard
# against a different process filling the volume mid-campaign, not against OASP.
FREE_GB=$(df -BG --output=avail /d 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 5 ]; then
  fail "only ${FREE_GB}GB free on D: — refusing to start (5GB floor)"
fi
echo "[OK] disk free: ${FREE_GB:-unknown}GB"

# Model presence: fail early rather than 2,000 cells into a run.
for m in gemma3:4b mistral:7b qwen3.8:27b; do
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m"; then
    fail "model '$m' not present in Ollama"
  fi
done
echo "[OK] required models present"

# --- campaign -------------------------------------------------------------
BENCH="${OASP_BENCH:-arc_challenge}"
MODELS="${OASP_MODELS:-mistral_7b gemma3_4b}"
RUN_ID="${OASP_RUN_ID:-campaign_${BENCH}}"

echo "--- matrix: run_id=$RUN_ID benchmark=$BENCH models=$MODELS (resumes) ---"
python -m oasp.runner --run-id "$RUN_ID" \
  --benchmark "$BENCH" --models $MODELS \
  --variants "${OASP_VARIANTS:-32}" \
  --items-dev "${OASP_ITEMS_DEV:-60}" --items-test "${OASP_ITEMS_TEST:-60}" \
  --splits dev --num-predict 192 --timeout 600 --concurrency 1
echo "matrix exit=$?"

echo "--- noise floor (M6) ---"
python scripts/run_noise_floor.py --run-id "noise_${BENCH}" \
  --benchmark "$BENCH" --models $MODELS --num-predict 192 --timeout 600
echo "noise exit=$?"

echo "--- analysis ---"
python -m oasp.analyze --run-id "$RUN_ID" --split dev --noise-run-id "noise_${BENCH}"
echo "analyze exit=$?"

echo "=== campaign done $(date -Iseconds) ==="
echo "report: analysis/$RUN_ID/report.md"
