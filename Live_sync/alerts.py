"""Email alerts for the live sync services (emanifest_sync.py and trip_sync.py).

What sends an email:
    - the service started (and whether the previous run stopped cleanly)
    - the service stopped (normal stop or crash, with the error)
    - a WARNING or ERROR in the log: company id, problem and the last log lines

No repeat emails for the same issue:
    An issue = company id + problem (numbers such as record Ids are ignored, so "Embedding failed
    for Id 123" and "... Id 456" are the same issue).
    - first time it appears                         -> email "NEW"
    - it appears again                              -> no email
    - it appeared in ALERT_REMIND_EVERY_CYCLES (60)
      cycles in a row (about 1 hour)                -> email "STILL HAPPENING" (again after 120, 180...)
    - it did not appear for ALERT_RESOLVE_AFTER_CYCLES
      (10) cycles in a row                          -> email "RESOLVED"; if it comes back later it is NEW again
    All alerts of one cycle go out in one email. The issue state is kept in
    logs/<service log name>_alerts.json, so a service restart does not re-send open issues.
    Start/stop emails are sent at most once per ALERT_LIFECYCLE_MIN_GAP_SECONDS (600) per kind,
    so a service that keeps crashing and restarting does not flood the mailbox.

Settings in .env:
    ALERT_SMTP_HOST, ALERT_SMTP_PORT (587 = STARTTLS, 465 = SSL), ALERT_SMTP_USER, ALERT_SMTP_PASSWORD
    ALERT_EMAIL_FROM (default: ALERT_SMTP_USER), ALERT_EMAIL_TO (comma-separated)
    optional: ALERT_REMIND_EVERY_CYCLES (60), ALERT_RESOLVE_AFTER_CYCLES (10), ALERT_LOG_LINES (40),
              ALERT_LIFECYCLE_MIN_GAP_SECONDS (600),
              ALERT_IGNORE (texts separated by |; log lines containing one are never emailed,
                            default: the Atlas "maximum number of FTS indexes" warning)
If SMTP is not configured, alerts are off and the sync runs as before.
"""

import json
import logging
import os
import re
import smtplib
import socket
import threading
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Records from this logger are never turned into alerts (no loop when an email fails).
log = logging.getLogger("Live_sync.alerts")


def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


SMTP_HOST = os.getenv("ALERT_SMTP_HOST", "").strip()
SMTP_PORT = _int("ALERT_SMTP_PORT", 587)
SMTP_USER = os.getenv("ALERT_SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("ALERT_SMTP_PASSWORD", "").strip()
EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "").strip() or SMTP_USER
EMAIL_TO = [x.strip() for x in os.getenv("ALERT_EMAIL_TO", "").split(",") if x.strip()]

REMIND_EVERY_CYCLES = max(1, _int("ALERT_REMIND_EVERY_CYCLES", 60))
RESOLVE_AFTER_CYCLES = max(1, _int("ALERT_RESOLVE_AFTER_CYCLES", 10))
LOG_LINES = _int("ALERT_LOG_LINES", 40)
LIFECYCLE_MIN_GAP_SECONDS = _int("ALERT_LIFECYCLE_MIN_GAP_SECONDS", 600)
IGNORE = [x.strip().lower() for x in
          os.getenv("ALERT_IGNORE", "maximum number of FTS indexes").split("|") if x.strip()]

# "company 2014" in sync messages, "on 5.emanifest" in the vector-index warning
_COMPANY = re.compile(r"company (\d+)|\bon (\d+)\.\w+")
_LONG_NUMBER = re.compile(r"\d{4,}")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Alerter(logging.Handler):
    """Logging handler that collects WARNING/ERROR lines; end_cycle() decides what to email."""

    def __init__(self, service: str, log_file: Path):
        super().__init__(logging.WARNING)
        self.service = service
        self.log_file = Path(log_file)
        self.state_file = self.log_file.with_name(f"{self.log_file.stem}_alerts.json")
        self.enabled = bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD and EMAIL_TO)
        self._seen: dict[str, dict] = {}
        self._seen_lock = threading.Lock()
        self.state = self._load_state()
        if not self.enabled:
            log.info("Email alerts are off (set ALERT_SMTP_* and ALERT_EMAIL_TO in .env to turn them on)")

    # ---------- collect ----------

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("Live_sync.alerts"):
            return
        try:
            message = " ".join(record.getMessage().split())
        except Exception:
            return
        if any(text in message.lower() for text in IGNORE):
            return
        match = _COMPANY.search(message)
        company = (match.group(1) or match.group(2)) if match else "-"
        problem = message[:500]
        key = f"{company}|{record.levelname}|{_LONG_NUMBER.sub('#', problem.split(', full error')[0])[:200]}"
        with self._seen_lock:
            entry = self._seen.setdefault(key, {"company": company, "level": record.levelname, "count": 0})
            entry["problem"] = problem
            entry["count"] += 1

    # ---------- per cycle ----------

    def end_cycle(self) -> None:
        """Call once after every sync cycle."""
        with self._seen_lock:
            seen, self._seen = self._seen, {}
        issues: dict = self.state["issues"]
        now = _now()
        new, still, resolved = [], [], []

        for key, info in seen.items():
            issue = issues.get(key)
            is_new = issue is None
            if is_new:
                issue = issues[key] = {"company": info["company"], "level": info["level"], "first_seen": now,
                                       "in_a_row": 0, "clean_cycles": 0, "occurrences": 0}
                new.append(issue)
            issue.update(problem=info["problem"], last_seen=now, clean_cycles=0)
            issue["in_a_row"] += 1
            issue["occurrences"] += info["count"]
            if not is_new and issue["in_a_row"] % REMIND_EVERY_CYCLES == 0:
                still.append(issue)

        for key in [k for k in issues if k not in seen]:
            issue = issues[key]
            issue["in_a_row"] = 0
            issue["clean_cycles"] += 1
            if issue["clean_cycles"] >= RESOLVE_AFTER_CYCLES:
                resolved.append(issues.pop(key))

        self._save_state()
        if not (new or still or resolved):
            return

        counts = [f"{len(items)} {label.lower()}" for label, items in
                  (("NEW", new), ("STILL HAPPENING", still), ("RESOLVED", resolved)) if items]
        companies = sorted({i["company"] for i in new + still + resolved})
        subject = f"[{self.service}] {', '.join(counts)} - company {', '.join(companies)}"
        body = [f"{self.service} on {socket.gethostname()} at {now}", ""]
        for label, items in (("NEW ISSUE", new), (f"STILL HAPPENING ({REMIND_EVERY_CYCLES}+ cycles in a row)", still),
                             (f"RESOLVED (not seen for {RESOLVE_AFTER_CYCLES} cycles)", resolved)):
            for issue in items:
                body += [f"== {label} ==",
                         f"Company id : {issue['company']}",
                         f"Level      : {issue['level']}",
                         f"Problem    : {issue['problem']}",
                         f"First seen : {issue['first_seen']}",
                         f"Last seen  : {issue['last_seen']}",
                         f"Cycles in a row: {issue['in_a_row']}   Total occurrences: {issue['occurrences']}", ""]
        if new or still:
            body += self._log_tail()
        self._send(subject, "\n".join(body))

    # ---------- start / stop ----------

    def service_started(self, company_ids: list[int]) -> None:
        unclean = self.state.get("running", False)
        self.state["running"] = True
        self._save_state()
        lines = [f"{self.service} started on {socket.gethostname()} at {_now()}",
                 f"Companies: {', '.join(map(str, company_ids))}"]
        if unclean:
            lines += ["", "NOTE: the previous run did not stop cleanly (crash, killed process or power loss)."]
        self._lifecycle("started", "restarted after an unclean stop" if unclean else "started", lines, unclean)

    def service_stopped(self, reason: str, crashed: bool = False) -> None:
        self.state["running"] = False
        self._save_state()
        lines = [f"{self.service} stopped on {socket.gethostname()} at {_now()}", f"Reason: {reason}"]
        self._lifecycle("stopped", "CRASHED" if crashed else "stopped", lines, crashed)

    def _lifecycle(self, kind: str, title: str, lines: list[str], with_logs: bool) -> None:
        last_key, skipped_key = f"last_{kind}_email", f"skipped_{kind}_emails"
        if time.time() - self.state.get(last_key, 0) < LIFECYCLE_MIN_GAP_SECONDS:
            self.state[skipped_key] = self.state.get(skipped_key, 0) + 1
            self._save_state()
            log.info("Service %s email skipped (one was sent less than %ds ago)", kind, LIFECYCLE_MIN_GAP_SECONDS)
            return
        if self.state.get(skipped_key):
            lines += ["", f"{self.state[skipped_key]} earlier '{kind}' email(s) were skipped (too frequent)."]
        if with_logs:
            lines += [""] + self._log_tail()
        if self._send(f"[{self.service}] Service {title}", "\n".join(lines)):
            self.state[last_key] = time.time()
            self.state[skipped_key] = 0
            self._save_state()

    # ---------- helpers ----------

    def _log_tail(self) -> list[str]:
        try:
            with open(self.log_file, "rb") as file:
                file.seek(max(0, file.seek(0, os.SEEK_END) - 64_000))
                lines = file.read().decode("utf-8", errors="replace").splitlines()[-LOG_LINES:]
        except OSError:
            return []
        return [f"---- last {len(lines)} log lines ({self.log_file.name}) ----", *lines]

    def _send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        message = EmailMessage()
        message["Subject"], message["From"], message["To"] = subject, EMAIL_FROM, ", ".join(EMAIL_TO)
        message.set_content(body)
        try:
            if SMTP_PORT == 465:
                server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
            else:
                server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
                server.starttls()
            with server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        except Exception as exc:
            log.warning("Alert email could not be sent (%s): %s", type(exc).__name__, exc)
            return False
        log.info("Alert email sent: %s", subject)
        return True

    def _load_state(self) -> dict:
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        state.setdefault("issues", {})
        return state

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not save alert state %s: %s", self.state_file, exc)
