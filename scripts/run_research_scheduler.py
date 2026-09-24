"""Run one scheduled research cycle; nonzero exit means operator attention."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core.research_scheduler import run
from app_core.prospective_sport_adapters import DEFAULT_SPORTS, parse_sports
from app_core.research_schedule import is_open
from app_core.evidence_drive import DriveStore
from app_core.evidence_remote import settings


def main():
    configured_sports = os.getenv("RESEARCH_SPORTS", ",".join(DEFAULT_SPORTS))
    summary=Path(os.getenv("GITHUB_STEP_SUMMARY","research-scheduler-summary.md"))
    try:
        sports = parse_sports(configured_sports)
        if not is_open():
            result={"status":"outside_operating_window","errors":[]}
            summary.write_text("Research scheduler skipped: outside 11:45 a.m.-2:30 a.m. Eastern.\n",encoding="utf-8")
            print(json.dumps(result))
            return 0
        folder,_=settings()
        result=run(sports,Path(os.getenv("PARLAYPICKER_EVIDENCE_DIR","output/scheduled-research")),DriveStore(folder),folder,
                   os.getenv("CFBD_API_KEY"),os.getenv("ODDS_API_KEY"))
    except Exception as exc:
        result={"errors":["scheduler:"+type(exc).__name__]}
    site=os.getenv("PARLAYPICKER_NETLIFY_SITE_ID", "").strip()
    if site:
        try:
            from app_core.public_grading_scheduler import run as grade_public
            folder,_=settings()
            print("Public grading started", flush=True)
            # Public results cover every supported saved sport, independently of
            # the narrower research capture configuration (which excludes WNBA).
            from app_core.espn_results import ESPN_ENDPOINTS
            public=grade_public(site,folder,DriveStore(folder),set(ESPN_ENDPOINTS))
            result["public_grading"]=public
            result["errors"].extend("public_grading:"+e for e in public["errors"])
        except Exception as exc:
            result["public_grading"]={"status":"error", "error":type(exc).__name__}
            result["errors"].append("public_grading:"+type(exc).__name__)
    else:
        result["public_grading"]={"status":"not_configured", "action":"Set Actions variable PARLAYPICKER_NETLIFY_SITE_ID"}
    text=json.dumps(result,indent=2)
    summary.write_text("# Research scheduler\n\n```json\n"+text+"\n```\n",encoding="utf-8")
    print(text)
    return 1 if result["errors"] or result.get("requested_slate_success") is False else 0


if __name__=="__main__":raise SystemExit(main())
