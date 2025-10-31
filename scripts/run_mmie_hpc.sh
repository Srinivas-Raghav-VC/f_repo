#!/usr/bin/env bash
set -euo pipefail

# MMIE research-grade runner for HPC/long jobs with heartbeat emails and final report.
# Requirements:
# - Python env with repo deps installed
# - Optional mailer: either `mail`/`mailx` on PATH or SMTP env vars set
#
# Email configuration (set as env vars before running):
#   EMAIL_TO="you@example.com"            (required)
#   EMAIL_FROM="hpc-bot@example.com"      (optional; default = EMAIL_TO)
#   EMAIL_SUBJECT_PREFIX="[MMIE]"         (optional)
#
# SMTP (optional, if no `mail` command):
#   SMTP_SERVER=smtp.example.com
#   SMTP_PORT=587
#   SMTP_USER=...
#   SMTP_PASS=...
#   SMTP_STARTTLS=1                        (default 1)
#
# Gemini judge (optional): set GEMINI_API_KEY to enable judge-assisted selection automatically.
#
# Usage examples:
#   bash scripts/run_mmie_hpc.sh \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
#     --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
#     --auto --auto_plots --auto_bundle --device cuda

SEND_MAIL() {
  local subject="$1"; shift
  local body="$*"
  local to="${EMAIL_TO:-}"
  if [[ -z "${to}" ]]; then
    echo "[warn] EMAIL_TO unset; skipping email: $subject" >&2
    return 0
  fi
  local from="${EMAIL_FROM:-$to}"
  local prefix="${EMAIL_SUBJECT_PREFIX:-[MMIE]}"
  local full_subject="$prefix $subject"
  if command -v mail >/dev/null 2>&1; then
    echo -e "$body" | mail -s "$full_subject" -r "$from" "$to" || true
    return 0
  fi
  # Fallback: Python SMTP
  local py="import os, smtplib, ssl\nfrom email.mime.text import MIMEText\n\nserver=os.environ.get('SMTP_SERVER')\nport=int(os.environ.get('SMTP_PORT','587'))\nuser=os.environ.get('SMTP_USER')\npswd=os.environ.get('SMTP_PASS')\nstarttls=int(os.environ.get('SMTP_STARTTLS','1'))\nmsg=MIMEText(os.environ['MAIL_BODY'])\nmsg['Subject']=os.environ['MAIL_SUBJECT']\nmsg['From']=os.environ['MAIL_FROM']\nmsg['To']=os.environ['MAIL_TO']\nctx=ssl.create_default_context()\nwith smtplib.SMTP(server, port) as s:\n    if starttls: s.starttls(context=ctx)\n    if user and pswd: s.login(user, pswd)\n    s.send_message(msg)\n"
  if [[ -n "${SMTP_SERVER:-}" ]]; then
    MAIL_BODY="$body" MAIL_SUBJECT="$full_subject" MAIL_FROM="$from" MAIL_TO="$to" \
      python - <<PYEOF || true
$py
PYEOF
  else
    echo "[warn] no mail utility or SMTP configured; skipping email: $subject" >&2
  fi
}

now_utc() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/mmie_${STAMP}.log"

# Build the command line from arguments; add judge assist explicitly if key present
EXTRA_ARGS=("$@")
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  EXTRA_ARGS+=("--judge_assist_selection")
fi

# Ensure auto outputs go to a timestamped folder even if not provided in args
echo "[run] starting at $(now_utc)" | tee -a "$LOG"
echo "[run] python mmie.py ${EXTRA_ARGS[*]}" | tee -a "$LOG"

# Launch the job in background with full logging
(
  set -x
  python mmie.py "${EXTRA_ARGS[@]}"
) > >(tee -a "$LOG") 2> >(tee -a "$LOG" >&2) &
PID=$!

# Send start email
if [[ -n "${EMAIL_TO:-}" ]]; then
  GPU_INFO=$(nvidia-smi 2>/dev/null || true)
  SEND_MAIL "Run started ($STAMP)" "Started at: $(now_utc)\nPID: $PID\nCmd: python mmie.py ${EXTRA_ARGS[*]}\n\nGPU:\n$GPU_INFO\n\nLog: $(realpath "$LOG")"
fi

# Heartbeat: every 2h until process exits
INTERVAL=${HEARTBEAT_SECONDS:-7200}
while kill -0 "$PID" >/dev/null 2>&1; do
  sleep "$INTERVAL" || true
  if kill -0 "$PID" >/dev/null 2>&1; then
    if [[ -n "${EMAIL_TO:-}" ]]; then
      TAIL=$(tail -n 40 "$LOG" 2>/dev/null || true)
      SEND_MAIL "Heartbeat: job running (PID $PID)" "Time: $(now_utc)\nPID: $PID\nCmd: python mmie.py ${EXTRA_ARGS[*]}\n\nLast 40 log lines:\n$TAIL"
    fi
  fi
done

wait "$PID" || STATUS=$?
STATUS=${STATUS:-0}

# Locate latest auto run dir if present
LATEST_AUTO_DIR=$(ls -1dt auto_runs/* 2>/dev/null | head -n1 || true)
RESULT_PATH=""
if [[ -n "$LATEST_AUTO_DIR" ]]; then
  RESULT_PATH="$LATEST_AUTO_DIR/results.json"
fi

SUMMARY=""
if [[ -f "$RESULT_PATH" ]]; then
  SUMMARY=$(python - << 'PY'
import json,sys
p=sys.argv[1]
try:
  with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
  print(json.dumps({"decision": d.get("decision", {}), "gates": d.get("gates", {})}, indent=2))
except Exception as e:
  print(f"<no summary: {e}>")
PY
"$RESULT_PATH")
fi

SUBJ="Run completed (exit=$STATUS)"
BODY="Completed at: $(now_utc)\nExit status: $STATUS\nLog: $(realpath "$LOG")\nResults: ${RESULT_PATH:-<none>}\n\nSummary:\n${SUMMARY:-<none>}"
[[ -n "${EMAIL_TO:-}" ]] && SEND_MAIL "$SUBJ" "$BODY"

echo "[run] completed with status $STATUS at $(now_utc)" | tee -a "$LOG"
exit "$STATUS"

