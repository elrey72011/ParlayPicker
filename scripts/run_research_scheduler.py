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
from app_core.research_cycle_audit import sanitize_cycle


def write_audit(path, report):
    """Keep a complete sanitized checkpoint available if Actions cancels us."""
    audit = sanitize_cycle(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return audit


def missing_credentials(sports):
    needed = ["PARLAYPICKER_DRIVE_FOLDER_ID", "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT"]
    if any(sport != "MLB" for sport in sports):
        needed.append("ODDS_API_KEY")
    if "NCAAF" in sports:
        needed.append("CFBD_API_KEY")
    return [name for name in needed if not os.getenv(name, "").strip()]


def main():
    configured_sports = os.getenv("RESEARCH_SPORTS", ",".join(DEFAULT_SPORTS))
    summary=Path(os.getenv("GITHUB_STEP_SUMMARY","research-scheduler-summary.md"))
    audit_path=Path(os.getenv("RESEARCH_AUDIT_PATH",str(summary.with_name("research-cycle-audit.json"))))
    result={"requested_sports": [], "health": {}, "errors": [], "requested_slate_success": False}
    try:
        sports = parse_sports(configured_sports)
        result["requested_sports"] = sports
        if not is_open():
            result={"status":"outside_operating_window","requested_sports":sports,
                    "errors":[],"requested_slate_success":False,
                    "execution_state":"SKIPPED", "active_stage":"OPERATING_WINDOW"}
            write_audit(audit_path, result)
            summary.write_text("Research scheduler skipped: outside 11:45 a.m.-2:30 a.m. Eastern.\n",encoding="utf-8")
            print(json.dumps(result))
            return 0
        result["execution_state"] = "IN_PROGRESS"
        result["active_stage"] = "PREFLIGHT"
        write_audit(audit_path, result)
        missing = missing_credentials(sports)
        if missing:
            result={"status":"MISSING_CREDENTIALS","requested_sports":sports,
                    "missing_environment_variables":missing,"health":{},
                    "errors":["MISSING_CREDENTIALS"],"requested_slate_success":False}
            result["execution_state"] = "FAILED"
            audit=write_audit(audit_path, result)
            audit["missing_environment_variables"]=missing
            audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            summary.write_text("# Research scheduler\n\nAuthenticated cycle blocked: missing " +
                               ", ".join(missing) + ".\n",encoding="utf-8")
            print(json.dumps(audit,indent=2,allow_nan=False))
            return 1
        result["active_stage"] = "REMOTE_CONFIGURATION"
        write_audit(audit_path, result)
        folder,_=settings()
        result=run(sports,Path(os.getenv("PARLAYPICKER_EVIDENCE_DIR","output/scheduled-research")),DriveStore(folder),folder,
                   os.getenv("CFBD_API_KEY"),os.getenv("ODDS_API_KEY"),
                   lambda checkpoint: write_audit(audit_path, checkpoint))
        result.setdefault("requested_sports", sports)
    except Exception as exc:
        result={"requested_sports":result.get("requested_sports",[]),"health":{},
                "errors":["scheduler:"+type(exc).__name__],"requested_slate_success":False,
                "execution_state":"FAILED", "active_stage":result.get("active_stage")}
        write_audit(audit_path, result)
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
    result["execution_state"] = ("FAILED" if result["errors"] or
                                 result.get("requested_slate_success") is False else "COMPLETE")
    result["active_stage"] = "COMPLETE"
    audit=write_audit(audit_path, result)
    text=json.dumps(audit,indent=2,allow_nan=False)
    summary.write_text("# Research scheduler\n\n```json\n"+text+"\n```\n",encoding="utf-8")
    print(text)
    return 1 if result["errors"] or result.get("requested_slate_success") is False else 0


if __name__=="__main__":raise SystemExit(main())
