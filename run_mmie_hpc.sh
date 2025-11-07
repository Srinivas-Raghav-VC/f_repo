#!/usr/bin/env bash
set -euo pipefail

# MMIE HPC Runner
# - Continues if you lose your network (runs in tmux if available, else nohup background)
# - Sends heartbeat email every 2 hours
# - Sends immediate email on crash and a final email on completion with a summary
# - Auto-enables judge-assisted selection when GEMINI_API_KEY is set

# Required env (export before run):
: "${EMAIL_TO:?export EMAIL_TO='srinivasraghav24@gmail.com'}"
: "${EMAIL_FROM:?export EMAIL_FROM='srinivasraghav24@gmail.com'}"

# Optional env:
EMAIL_SUBJECT_PREFIX="${EMAIL_SUBJECT_PREFIX:-[MMIE]}"
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-7200}
LOG_DIR=${LOG_DIR:-logs}
GMAIL_APP_PASSWORD="${GMAIL_APP_PASSWORD:-}"
# Or alternative SMTP if no Gmail app password / system mail:
SMTP_SERVER="${SMTP_SERVER:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASS="${SMTP_PASS:-}"
SMTP_STARTTLS="${SMTP_STARTTLS:-1}"

mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/mmie_${STAMP}.log"

# Email helper (prefers Gmail App Password, else /usr/bin/mail, else SMTP)
cat > notify_email.py << 'PY'
#!/usr/bin/env python3
import os, sys, smtplib, ssl, subprocess
from email.message import EmailMessage

def send(subject: str, body: str):
    to = os.environ['EMAIL_TO']
    sender = os.environ['EMAIL_FROM']
    prefix = os.environ.get('EMAIL_SUBJECT_PREFIX','[MMIE]')
    subj = f"{prefix} {subject}"

    # 1) Gmail App Password
    app = os.environ.get('GMAIL_APP_PASSWORD')
    if app:
        msg = EmailMessage(); msg['From']=sender; msg['To']=to; msg['Subject']=subj
        msg.set_content(body)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as s:
            s.login(sender, app); s.send_message(msg)
        return

    # 2) System mail
    if subprocess.call(['bash','-lc','command -v mail >/dev/null'])==0:
        p = subprocess.Popen(['mail','-s',subj,'-r',sender,to], stdin=subprocess.PIPE)
        p.communicate(body.encode()); return

    # 3) Generic SMTP
    server=os.environ.get('SMTP_SERVER'); port=int(os.environ.get('SMTP_PORT','587'))
    user=os.environ.get('SMTP_USER'); pw=os.environ.get('SMTP_PASS'); starttls=int(os.environ.get('SMTP_STARTTLS','1'))
    if server:
        msg = EmailMessage(); msg['From']=sender or user; msg['To']=to; msg['Subject']=subj
        msg.set_content(body)
        with smtplib.SMTP(server, port) as s:
            if starttls: s.starttls(context=ssl.create_default_context())
            if user and pw: s.login(user, pw)
            s.send_message(msg)
        return
    sys.stderr.write('[warn] no email method configured; skipping email\n')

if __name__=='__main__':
    subj=sys.argv[1] if len(sys.argv)>1 else 'MMIE notification'
    body=sys.stdin.read(); send(subj, body)
PY
chmod +x notify_email.py

# Resolve Python interpreter (prefer project venv, else python3)
if [[ -n "${VENV:-}" && -x "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"
elif [[ -x "./venv/bin/python" ]]; then
  PY="./venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[error] No Python interpreter found (need python3 or python)" >&2
  exit 1
fi
echo "[runner] using interpreter: $PY"

notify() {
  local subject="$1"; shift
  local body="$*"
  EMAIL_FROM="$EMAIL_FROM" EMAIL_TO="$EMAIL_TO" EMAIL_SUBJECT_PREFIX="$EMAIL_SUBJECT_PREFIX" \
  GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD" SMTP_SERVER="$SMTP_SERVER" SMTP_PORT="$SMTP_PORT" \
  SMTP_USER="$SMTP_USER" SMTP_PASS="$SMTP_PASS" SMTP_STARTTLS="$SMTP_STARTTLS" \
  "$PY" -u ./notify_email.py "$subject" <<< "$body" || true
}

now_utc() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

# Quick email-only test mode (no training). Usage:
#   EMAIL_TO=you@example.com EMAIL_FROM=you@gmail.com GMAIL_APP_PASSWORD=app_pwd \
#   ./run_mmie_hpc.sh --email-test
if [[ "${1:-}" == "--email-test" ]]; then
  BODY="This is a test message from run_mmie_hpc.sh
Time: $(now_utc)
Host: $(hostname)"
  if EMAIL_FROM="$EMAIL_FROM" EMAIL_TO="$EMAIL_TO" EMAIL_SUBJECT_PREFIX="$EMAIL_SUBJECT_PREFIX" \
     GMAIL_APP_PASSWORD="$GMAIL_APP_PASSWORD" SMTP_SERVER="$SMTP_SERVER" SMTP_PORT="$SMTP_PORT" \
     SMTP_USER="$SMTP_USER" SMTP_PASS="$SMTP_PASS" SMTP_STARTTLS="$SMTP_STARTTLS" \
     "$PY" -u ./notify_email.py "Email test" <<< "$BODY" ; then
    echo "[email-test] delivered to $EMAIL_TO from $EMAIL_FROM"
    exit 0
  else
    echo "[email-test] FAILED to send email (check GMAIL_APP_PASSWORD or SMTP_*)" >&2
    exit 1
  fi
fi

# Build mmie.py arguments; auto-enable judge assist when GEMINI_API_KEY is present
# You can disable judge via: export DISABLE_JUDGE=1  (or run with GEMINI_API_KEY unset)
EXTRA_ARGS=("$@")
if [[ -z "${DISABLE_JUDGE:-}" && -n "${GEMINI_API_KEY:-}" ]]; then EXTRA_ARGS+=("--judge_assist_selection"); fi

echo "[run] $(now_utc) $PY mmie.py ${EXTRA_ARGS[*]}" | tee -a "$LOG_FILE"

# Launch background process; prefer tmux if available so it survives disconnect
if command -v tmux >/dev/null 2>&1; then
  SESSION="mmie_${STAMP}"
  tmux new -s "$SESSION" -d "$PY -u mmie.py ${EXTRA_ARGS[*]} 2>&1 | tee -a '$LOG_FILE'" || true
  sleep 1
  PID=$(pgrep -f "python -u mmie.py" | head -n1 || true)
else
  ( nohup "$PY" -u mmie.py ${EXTRA_ARGS[*]} 2>&1 | tee -a "$LOG_FILE" & echo $! > "$LOG_FILE.pid" ) & disown || true
  sleep 1
  PID=$(cat "$LOG_FILE.pid" 2>/dev/null || true)
fi

notify "Run started" "Started: $(now_utc)
PID: ${PID:-unknown}
Cmd: python mmie.py ${EXTRA_ARGS[*]}
Log: $(realpath "$LOG_FILE")"

# Heartbeat
while [[ -n "${PID:-}" ]] && kill -0 "$PID" >/dev/null 2>&1; do
  sleep "$HEARTBEAT_SECONDS" || true
  if kill -0 "$PID" >/dev/null 2>&1; then
    TAIL=$(tail -n 60 "$LOG_FILE" 2>/dev/null || true)
    notify "Heartbeat (PID $PID)" "Time: $(now_utc)
Last 60 log lines:
$TAIL"
  fi
done

wait "$PID" 2>/dev/null || STATUS=$?
STATUS=${STATUS:-0}
if [[ "$STATUS" != 0 ]]; then
  notify "Run crashed (exit=$STATUS)" "Time: $(now_utc)
Log: $(realpath "$LOG_FILE")
Last 100 lines:
$(tail -n 100 "$LOG_FILE" 2>/dev/null || true)"
fi

# Final summary (best-effort)
LATEST_AUTO_DIR=$(ls -1dt auto_runs/* 2>/dev/null | head -n1 || true)
RESULT_PATH=""; SUMMARY="<none>"
if [[ -n "$LATEST_AUTO_DIR" && -f "$LATEST_AUTO_DIR/results.json" ]]; then
  RESULT_PATH="$LATEST_AUTO_DIR/results.json"
  SUMMARY=$($PY - << 'PY'
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

notify "Run completed (exit=$STATUS)" "Completed: $(now_utc)
Exit: $STATUS
Log: $(realpath "$LOG_FILE")
Results: ${RESULT_PATH:-<none>}

Summary:
${SUMMARY}"

echo "[run] done exit=$STATUS at $(now_utc)"
exit "$STATUS"
