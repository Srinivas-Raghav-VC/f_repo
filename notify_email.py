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
