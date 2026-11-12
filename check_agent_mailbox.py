#!/usr/bin/env python
"""Operator tool: read-only look at the agent mailbox for bounces and delivery errors.

Usage: python check_agent_mailbox.py [days]   (default 2)

Uses SMTP_HOST / SMTP_USER / SMTP_PASS from .env over IMAPS. Folders are opened read-only, so nothing
is marked as read, moved or deleted. Incoming mail is untrusted, so this is for humans: don't wire it
into an agent.
"""
import email
import email.policy
import imaplib
import os
import ssl
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

BOUNCE_HINTS = ("mailer-daemon", "postmaster", "undeliver", "delivery status", "failure notice", "returned mail")


def diagnostic_lines(msg) -> list[str]:
    """The lines of a bounce that say what went wrong."""
    wanted = ("final-recipient", "action:", "status:", "diagnostic-code", "remote-mta")
    lines: list[str] = []
    for part in msg.walk():
        if part.get_content_type() == "message/delivery-status":
            lines += [line.strip() for sub in part.get_payload()
                      for line in str(sub).splitlines() if line.lower().startswith(wanted)]
    return lines


def main() -> int:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    since = (date.today() - timedelta(days=days)).strftime("%d-%b-%Y")
    with imaplib.IMAP4_SSL(os.environ["SMTP_HOST"], 993, timeout=20, ssl_context=ssl.create_default_context()) as imap:
        imap.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        _, listing = imap.list()
        # Each row looks like: (\HasNoChildren) "." "INBOX.spam". The name is the last field.
        folders = [row.decode().rsplit(' "', 1)[-1].rsplit(" ", 1)[-1].strip('"') for row in listing]
        for folder in folders:
            status, _ = imap.select(f'"{folder}"', readonly=True)
            if status != "OK":
                print(f"== {folder}: could not open")
                continue
            _, data = imap.search(None, "SINCE", since)
            ids = data[0].split()
            print(f"== {folder}: {len(ids)} message(s) since {since}")
            for num in ids:
                # BODY.PEEK keeps the read-only promise even on servers that ignore the select mode.
                _, parts = imap.fetch(num, "(BODY.PEEK[])")
                msg = email.message_from_bytes(parts[0][1], policy=email.policy.default)
                sender, subject = str(msg["From"]), str(msg["Subject"])
                flag = "BOUNCE " if any(h in f"{sender} {subject}".lower() for h in BOUNCE_HINTS) else ""
                print(f"{flag}{msg['Date']} | {sender} | {subject}")
                for line in diagnostic_lines(msg):
                    print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
