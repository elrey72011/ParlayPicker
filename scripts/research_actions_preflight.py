"""Report Actions research configuration by name and presence only."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path


EVIDENCE_SECRETS = (
    "PARLAYPICKER_DRIVE_FOLDER_ID",
    "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
)
PROVIDER_SECRETS = ("ODDS_API_KEY", "CFBD_API_KEY")
OPTIONAL_VARIABLES = ("PARLAYPICKER_NETLIFY_SITE_ID",)
DEFAULT_SPORTS = "NFL,NCAAF,NBA,NCAAB,MLB,NHL"
PRESENCE_FLAGS = {
    "PARLAYPICKER_DRIVE_FOLDER_ID": "PREFLIGHT_DRIVE_FOLDER_PRESENT",
    "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT": "PREFLIGHT_SERVICE_ACCOUNT_PRESENT",
    "ODDS_API_KEY": "PREFLIGHT_ODDS_KEY_PRESENT",
    "CFBD_API_KEY": "PREFLIGHT_CFBD_KEY_PRESENT",
    "PARLAYPICKER_NETLIFY_SITE_ID": "PREFLIGHT_NETLIFY_SITE_PRESENT",
}


def configuration_status(environ: Mapping[str, str]) -> tuple[list[str], str]:
    """Return missing names and a value-free Markdown report."""
    enabled = environ.get("PREFLIGHT_RESEARCH_ENABLED", "").strip().lower() == "true"
    sports = {sport.strip().upper() for sport in
              environ.get("RESEARCH_SPORTS", DEFAULT_SPORTS).split(",") if sport.strip()}
    required_secrets = list(EVIDENCE_SECRETS)
    if any(sport != "MLB" for sport in sports):
        required_secrets.append("ODDS_API_KEY")
    if "NCAAF" in sports:
        required_secrets.append("CFBD_API_KEY")
    missing = [] if enabled else ["RESEARCH_SCHEDULER_ENABLED"]
    rows = ["| Setting | Scope | Status |", "| --- | --- | --- |",
            f"| RESEARCH_SCHEDULER_ENABLED | required repository variable | {'enabled' if enabled else 'missing or not true'} |"]
    for name in (*EVIDENCE_SECRETS, *PROVIDER_SECRETS):
        present = environ.get(PRESENCE_FLAGS[name], "").strip().lower() == "true"
        required = name in required_secrets
        if required and not present:
            missing.append(name)
        scope = "required Actions secret" if required else "not required for selected sports"
        rows.append(f"| {name} | {scope} | {'configured' if present else 'missing'} |")
    for name in OPTIONAL_VARIABLES:
        present = environ.get(PRESENCE_FLAGS[name], "").strip().lower() == "true"
        rows.append(f"| {name} | optional repository variable | {'configured' if present else 'not configured'} |")
    return missing, "\n".join(rows) + "\n"


def main() -> int:
    missing, report = configuration_status(os.environ)
    output = "# Research Actions configuration preflight\n\n" + report
    if missing:
        output += "\nOwner action required for: " + ", ".join(missing) + ".\n"
    else:
        output += "\nRequired settings are present; provider and Drive access still require live verification.\n"
    print(output, end="")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(output)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
