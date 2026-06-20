#!/usr/bin/env python3
"""Send an experiment-watcher notification email (stdlib only).

Reads SMTP config from the environment (the caller `source`s .env and exports
SMTP_*). Body is read from stdin; subject from --subject. Exit 0 on success,
non-zero on any failure (so the watcher leaves the run un-notified and retries).

  # test the SMTP path end-to-end (no job needed):
  set -a; source .env; set +a
  echo "hello" | python3 scripts/notify_email.py --subject "[sim] watcher test"

  # preview without sending:
  echo "body" | python3 scripts/notify_email.py --subject "..." --dry-run
"""
import argparse
import os
import ssl
import sys
from email.message import EmailMessage


def main() -> int:
    ap = argparse.ArgumentParser(description="Send a watcher notification email.")
    ap.add_argument("--subject", help="email subject")
    ap.add_argument("--test", action="store_true", help="send a canned test email")
    ap.add_argument("--dry-run", action="store_true", help="print the message instead of sending")
    args = ap.parse_args()

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    to = os.environ.get("SMTP_TO") or user
    sender = os.environ.get("SMTP_FROM") or user

    if args.test:
        subject = "[sim] watcher test"
        body = "If you can read this, the watcher's SMTP path works."
    else:
        if not args.subject:
            print("notify_email: --subject required (or use --test)", file=sys.stderr)
            return 2
        subject = args.subject
        body = sys.stdin.read()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender or "watcher@localhost"
    msg["To"] = to or "unknown"
    msg.set_content(body or "(no body)")

    if args.dry_run:
        print(f"--- DRY RUN ---\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {subject}\n\n{body}")
        return 0

    if not (user and pw and to):
        print("notify_email: SMTP_USER/SMTP_PASS/SMTP_TO not set in environment (.env)", file=sys.stderr)
        return 3

    try:
        # Use certifi's CA bundle if present — some macOS Pythons ship without
        # root certificates, which otherwise fails TLS with CERTIFICATE_VERIFY_FAILED.
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
        if port == 465:
            with __import__("smtplib").SMTP_SSL(host, port, context=ctx, timeout=30) as s:
                s.login(user, pw)
                s.send_message(msg)
        else:
            with __import__("smtplib").SMTP(host, port, timeout=30) as s:
                s.starttls(context=ctx)
                s.login(user, pw)
                s.send_message(msg)
    except Exception as e:  # noqa: BLE001 — surface any SMTP/auth error as non-zero exit
        print(f"notify_email: send failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"notify_email: sent '{subject}' -> {to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
