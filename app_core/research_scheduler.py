"""Bounded orchestration around frozen research modules; never fits models."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from app_core import mlb_prospective as mlb, mlb_prospective_store as ms
from app_core import ncaaf_prospective as ncaaf, ncaaf_prospective_store as ns
from app_core import nfl_market as nfl, nfl_market_store as fs
from app_core.mlb_history import timestamp
from app_core.research_api_budget import Budget, BudgetLimit
from app_core.research_schedule import is_open
from app_core.prospective_sport_adapters import get_adapter, parse_sports

PREFIX = "parlaypicker/research-scheduler-v1/"
_last_checkpoint_time = None


class ResearchStageError(Exception):
    """Sanitized stage information without provider URLs or credentials."""
    def __init__(self, stage, cause):
        self.stage = stage
        self.code = type(cause).__name__
        super().__init__(self.code)


def progress(sport, stage, **facts):
    print(json.dumps({"sport": sport, "stage": stage, **facts}), flush=True)


_CANONICAL_FAILURE_CODES = frozenset({
    "canonical_remote_schema_invalid", "canonical_remote_table_invalid",
    "canonical_remote_columns_invalid", "canonical_remote_blob_invalid",
    "canonical_remote_json_invalid", "canonical_remote_integrity_conflict",
    "canonical_remote_evidence_hash_conflict", "canonical_remote_identity_conflict",
    "canonical_remote_local_conflict", "canonical_remote_source_conflict",
    "canonical_remote_foreign_key_conflict", "canonical_remote_foreign_key_cycle",
    "canonical_remote_missing_source_dependencies",
    "canonical_remote_object_too_large", "canonical_remote_readback_conflict",
})


def canonical_failure_code(exc):
    """Expose only owned integrity codes, HTTP status, or exception class."""
    if isinstance(exc, ValueError) and str(exc) in _CANONICAL_FAILURE_CODES:
        return str(exc).upper()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if type(status) is int and 400 <= status <= 599:
        return f"DRIVE_STATUS_{status}"
    name = type(exc).__name__.upper()
    return name if re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", name) else "ERROR"


def utcnow():
    return datetime.now(timezone.utc)


def source_commit():
    """Bind frozen plans and run audits to the code actually being executed."""
    sha = os.getenv("GITHUB_SHA", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=5, check=True)
        sha = result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("source_commit_unavailable")
    return sha


class _ReceiptClient:
    """Supply bounded object listing for both DriveStore and test object stores."""
    def __init__(self, client, folder):
        self.client, self.folder = client, folder
        self._receipt_verified = set()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def read_objects(self, *, Prefix):
        if callable(getattr(self.client, "read_objects", None)):
            yield from self.client.read_objects(Prefix=Prefix)
        else:
            for page in self.client.get_paginator("list_objects_v2").paginate(
                    Bucket=self.folder, Prefix=Prefix):
                for item in page.get("Contents", []):
                    with self.client.get_object(Bucket=self.folder, Key=item["Key"])["Body"] as body:
                        yield item["Key"], body.read(40_000_001)


def due(start, now):
    return now < timestamp(start) <= now + timedelta(hours=2)


def latest_model(records):
    models=[r for r in records if r["kind"]=="model"]
    if not models:raise ValueError("missing_frozen_model")
    return models[-1]


def checkpoint(client, folder, state=None):
    global _last_checkpoint_time
    prefix=PREFIX+"state/"
    if state is None:
        keys=[o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=folder,Prefix=prefix) for o in page.get("Contents",[])]
        if not keys:return {}
        key=max(keys)
        with client.get_object(Bucket=folder,Key=key)["Body"] as b:raw=b.read(40_000_001)
        if len(raw)>40_000_000 or not key.endswith(hashlib.sha256(raw).hexdigest()+".json"):
            raise ValueError("scheduler_state_integrity")
        written=datetime.strptime(key[len(prefix):].split("-")[0],"%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc)
        _last_checkpoint_time=max(_last_checkpoint_time or written,written)
        return json.loads(raw)
    raw=json.dumps(state,sort_keys=True,allow_nan=False).encode()
    written=utcnow()
    if _last_checkpoint_time is not None:
        written=max(written,_last_checkpoint_time+timedelta(microseconds=1))
    _last_checkpoint_time=written
    key=prefix+written.strftime("%Y%m%dT%H%M%S%f")+"-"+hashlib.sha256(raw).hexdigest()+".json"
    client.put_object(Bucket=folder,Key=key,Body=raw,ContentType="application/json",IfNoneMatch="*")
    with client.get_object(Bucket=folder,Key=key)["Body"] as b:
        if b.read()!=raw:raise ValueError("scheduler_state_readback")


def upcoming_mlb():
    """Only request dates intersecting the two-hour capture window."""
    from zoneinfo import ZoneInfo
    at=utcnow()
    local=at.astimezone(ZoneInfo('America/New_York'))
    end=(at+timedelta(hours=2)).astimezone(ZoneInfo('America/New_York'))
    payload=mlb.fetch('api/v1/schedule',{'sportId':1,'gameType':'R','hydrate':'probablePitcher',
        'startDate':local.date().isoformat(),'endDate':end.date().isoformat()})
    return [g for g in mlb.games_from(payload) if g['status']['abstractGameState']=='Preview'
            and timestamp(g['gameDate'])>at]


def retry_schedule(fetch):
    """Retry only the read-only MLB schedule, without changing frozen model code."""
    import requests
    import time
    for attempt in range(3):
        try:
            return fetch()
        except (requests.Timeout, requests.ConnectionError):
            if attempt==2:
                raise
            time.sleep(attempt+1)


def run_mlb(path, state, backup):
    records=ms.records(path);model=latest_model(records)
    if model["data"]["runtime_hash"]!=mlb.runtime_hash():raise ValueError("stale_frozen_model")
    seen={e["game_id"] for r in records if r["kind"]=="capture" and r["data"]["model_id"]==model["id"] for e in r["data"]["events"]}
    counts={"captured":0,"graded":0,"blocked_captures":0,"errors":[]}
    try:
        games=sorted(retry_schedule(upcoming_mlb),key=lambda g:(g["gameDate"],g["gamePk"]))
    except Exception as exc:
        raise ResearchStageError("mlb_schedule", exc) from None
    candidates=[g for g in games if g["gamePk"] not in seen and due(g["gameDate"],utcnow())]
    capture_attempts=state.setdefault("mlb_capture_attempts",{})
    candidates.sort(key=lambda g:(capture_attempts.get(str(g["gamePk"]),""),g["gameDate"],g["gamePk"]))
    for g in candidates[:6]:
        if not is_open():break
        progress("MLB", "capture_started", game_id=g["gamePk"])
        capture_attempts[str(g["gamePk"])]=utcnow().isoformat()
        try:
            mlb.capture(g["gamePk"],path)
            # Storage may remove a game that starts during computation.
            saved=ms.records(path)[-1]["data"].get("events",[])
            counts["captured"]+=len(saved)
        except ValueError as exc:
            counts["blocked_captures"]+=1
            expected=("Both probable pitchers required","Insufficient team history","Insufficient pitcher history",
                      "Game is not pregame","Capture expired or game started")
            if str(exc) not in expected:counts["errors"].append("mlb_capture_validation")
        except Exception:
            counts["errors"].append("mlb_capture_provider")
        backup()
    records=ms.records(path)
    completed={r["data"]["game_id"] for r in records if r["kind"]=="scores"}
    events={e["game_id"]:e for r in records if r["kind"]=="capture" for e in r["data"]["events"] if timestamp(e["start"])<utcnow() and e["game_id"] not in completed}
    attempts=state.setdefault("mlb_grade_attempts",{})
    for gid,e in sorted(events.items(),key=lambda item:(attempts.get(str(item[0]),""),item[0]))[:6]:
        if not is_open():break
        attempts[str(gid)]=utcnow().isoformat()
        try:
            feed=mlb.fetch(f"api/v1.1/game/{gid}/feed/live")
            if feed["gameData"]["status"]["abstractGameState"]!="Final":continue
            game=mlb.normalize_game(feed)
            if game["game_id"]!=gid or any(game[s+"_id"]!=e[s+"_id"] for s in ("home","away")):
                raise ValueError("score_identity")
            starters=mlb.parse_boxscore(mlb.fetch(f"api/v1/game/{gid}/boxscore"),game)["starters"]
            ms.save("scores",{**game,"actual_starters":starters},path);counts["graded"]+=1
            backup()
        except Exception:
            counts["errors"].append("mlb_grading_failed")
    return counts


def run_ncaaf(path, state, cfbd_key, odds_key, backup, request_get=None):
    records=ns.records(path);model=latest_model(records)
    from app_core.research_model_recovery import recover_ncaaf
    recovered = recover_ncaaf(model, path)
    if recovered["id"] != model["id"]:
        backup()  # Verify the new immutable cohort remotely before capture.
    model = recovered
    if model["data"]["runtime_hash"]!=ncaaf.runtime_hash():raise ValueError("stale_frozen_model")
    if not cfbd_key or not odds_key:raise ValueError("missing_provider_keys")
    inputs=state.get("ncaaf_inputs")
    if inputs and (inputs["years"]!=[utcnow().year] or any(not 0 <= (utcnow()-timestamp(b["retrieved_at"])).total_seconds()<23*3600 for b in inputs["batches"])):
        inputs=None
    provider_args={"get":request_get} if request_get is not None else {}
    inputs,status=ncaaf.refresh(inputs,cfbd_key,**provider_args)
    state["ncaaf_inputs"]=inputs
    counts={"captured":0,"graded":0,"input_status":status,"errors":[]}
    if status not in ("ready","continue") and not status.startswith("api_budget:"):
        counts["errors"].append("ncaaf_refresh_failed")
    if status=="ready":
        seen={e["cfbd_id"] for r in records if r["kind"]=="capture" and r["data"]["model_id"]==model["id"] for e in r["data"]["events"]}
        targets=[g for g in inputs["batches"][0]["records"] if g.get("id") not in seen and g.get("startDate") and due(g["startDate"],utcnow())]
        eligible=[]
        for game in targets:
            _,features,_=ncaaf.build_dataset(inputs,feature_targets=[game])
            if features and ncaaf._eligible(features[0]):eligible.append(game)
        counts["insufficient_features"]=len(targets)-len(eligible)
        targets=eligible
        attempts=state.setdefault("ncaaf_capture_attempts",{})
        ordered=sorted(targets,key=lambda g:(attempts.get(str(g["id"]),""),g["startDate"],g["id"]))[:24]
        allowed={g["id"] for g in ordered}
        for gid in allowed:attempts[str(gid)]=utcnow().isoformat()
        if allowed:
            import requests
            def filtered_get(url,**kwargs):
                response=(request_get or requests.get)(url,**kwargs)
                if response.status_code!=200:return response
                values=response.json()
                if not isinstance(values,list):return response
                filtered=[]
                for event in values:
                    g=ncaaf._match(event,inputs["batches"][0]["records"])
                    if g and g["id"] in allowed:filtered.append(event)
                class Filtered:
                    status_code=200
                    def json(self):return filtered
                return Filtered()
            try:
                _,result=ncaaf.capture(inputs,model,odds_key,get=filtered_get,path=path)
                counts["captured"]=result["saved_games"]
                counts["excluded"]=result["skipped_games"]
            except BudgetLimit:
                counts["budget_paused"]=True
            except Exception:
                counts["errors"].append("ncaaf_capture_failed")
            backup()
    result=ncaaf.grade(cfbd_key,path=path,**provider_args)
    counts["graded"]=result["graded"]
    if result.get("error") and not result["error"].startswith("api_budget:"):counts["errors"].append("ncaaf_grading_failed")
    backup()
    return counts


def run(sports, root, client, folder, cfbd_key=None, odds_key=None, audit_callback=None):
    sports = parse_sports(sports)
    root.mkdir(parents=True,exist_ok=True)
    state=checkpoint(client,folder)
    last_run = state.get("last_run", {})
    previous_health = last_run.get("health", {}) if isinstance(last_run, dict) else {}
    if not isinstance(previous_health, dict):
        previous_health = {}
    budget=Budget(state,lambda:checkpoint(client,folder,state))
    report={"run_id":str(uuid.uuid4()),"source_commit":source_commit(),
            "started_at":utcnow().isoformat(),"requested_sports":sports,
            "sports":{},"health":{},"errors":[],"production_eligible":False,
            "execution_state":"IN_PROGRESS"}
    def audit_stage(sport, name):
        report["active_sport"] = sport
        report["active_stage"] = name
        progress(sport or "ALL", name)
        if audit_callback is not None:
            audit_callback(report)
    canonical_path = root / "prospective-evidence.sqlite3"
    canonical_session = {}
    receipt_client = _ReceiptClient(client, folder)
    try:
        from app_core.prospective_remote import (
            CanonicalMissingDependencies, sync as sync_canonical,
        )
        from app_core.prospective_validation_plans import freeze_current_validation_plans
        audit_stage(None, "CANONICAL_RESTORE")
        try:
            report["canonical_restore"] = sync_canonical(canonical_path, client, folder,
                                                          canonical_session)
        except CanonicalMissingDependencies as missing:
            # An interrupted older upload could have published facts before
            # their immutable source rows. Rebuild only those sources locally
            # from independent backups, then verify the entire remote again.
            from app_core.prospective_reconciliation import reconcile_sport
            from app_core.prospective_source_view import SOURCE_FILENAMES
            report["canonical_repair_sports"] = sorted(missing.sports)
            for repair_sport in sorted(missing.sports):
                audit_stage(repair_sport, "CANONICAL_SOURCE_REPAIR")
                adapter = get_adapter(repair_sport)
                adapter.restore(root / adapter.path_name, client, folder, {})
                if repair_sport == "MLB":
                    from app_core.mlb_receipt_remote import recover
                    recover(receipt_client, root / SOURCE_FILENAMES[repair_sport])
                source_path = root / (SOURCE_FILENAMES[repair_sport] if repair_sport == "MLB"
                                      else adapter.path_name)
                reconcile_sport(repair_sport, canonical_path, source_path)
            canonical_session.clear()
            audit_stage(None, "CANONICAL_RESTORE")
            report["canonical_restore"] = sync_canonical(canonical_path, client, folder,
                                                          canonical_session)
        audit_stage(None, "FREEZE_VALIDATION_PLANS")
        report["frozen_validation_plans"] = freeze_current_validation_plans(
            canonical_path, source_commit=report["source_commit"])
        audit_stage(None, "CANONICAL_BACKUP")
        report["canonical_backup"] = sync_canonical(canonical_path, client, folder,
                                                     canonical_session)
    except Exception as exc:
        code = canonical_failure_code(exc)
        failed_stage = report.get("active_stage") or "CANONICAL_RESTORE"
        progress("ALL", "failed", failed_stage=failed_stage, code=code)
        report["errors"].append("canonical:" + code)
        report["failure_stages"] = {"canonical": {"stage": failed_stage, "code": code}}
        report["health"] = {sport: {"restore": "not_attempted", "backup": "not_attempted",
                                    "verified_backup": False, "provider_blockers": [
                                        "CANONICAL_RESTORE_OR_PLAN_FREEZE_FAILED"],
                                    "production_eligible": False} for sport in sports}
        report["requested_slate_success"] = False
        report["finished_at"] = utcnow().isoformat()
        report["api_budget"] = budget.report()
        report["execution_state"] = "FAILED"
        state["last_run"] = report
        checkpoint(client, folder, state)
        audit_stage(None, "COMPLETE")
        return report
    for sport in sports:
        adapter=get_adapter(sport)
        path=root/adapter.path_name
        stage = "restore"
        sync_session = {}
        sync_status={"attempts":0}
        health={"last_attempt":utcnow().isoformat(),"restore":"not_attempted",
                "capture":"not_attempted","grade":"not_attempted",
                "close_capture":"not_attempted",
                "backup":"not_attempted","verified_backup":False,
                "discovered_events":0,"captured_events":0,"graded_events":0,
                "captured_quote_rows":0,"research_predictions":0,
                "research_close_candidates":0,"reconciliation_status":"not_attempted",
                "provider_blockers":[],"production_eligible":False}
        old = previous_health.get(sport, {})
        if isinstance(old, dict):
            for key in ("last_successful_restore", "last_successful_capture",
                        "last_successful_grade", "last_successful_backup", "last_verified_backup"):
                value = old.get(key)
                if isinstance(value, str) and timestamp(value) is not None and timestamp(value) <= utcnow():
                    health[key] = value
        report["health"][sport]=health
        def backup():
            operation = "restore" if sync_status["attempts"] == 0 else "backup"
            sync_status["attempts"] += 1
            audit_stage(sport, "NATIVE_RESTORE" if operation == "restore" else "NATIVE_BACKUP")
            progress(sport, "sync_started", operation=operation)
            result = (adapter.backup if operation == "backup" else adapter.restore)(path,client,folder,sync_session)
            progress(sport, "sync_completed", **(result or {}))
            health[operation]="success"
            health["last_successful_"+operation]=utcnow().isoformat()
            if operation == "backup":
                health["verified_backup"]=True
                health["last_verified_backup"]=utcnow().isoformat()
            health["backup_records_verified"]=(result or {}).get("records_verified",0)
            if audit_callback is not None:
                audit_callback(report)
            return result
        try:
            backup()  # Restore must succeed before any capture or grading.
            if sport == "MLB":
                from app_core.mlb_receipt_remote import recover
                from app_core.prospective_source_view import SOURCE_FILENAMES
                audit_stage(sport, "RECEIPT_RECOVERY")
                report.setdefault("receipt_restore", {})[sport] = recover(
                    receipt_client, root / SOURCE_FILENAMES[sport])
                if audit_callback is not None:
                    audit_callback(report)
            if sport in ("NFL", "NCAAF", "MLB"):
                from app_core.prospective_reconciliation import reconcile_sport
                from app_core.prospective_source_view import SOURCE_FILENAMES
                stage = "canonical_reconciliation"
                audit_stage(sport, "CANONICAL_RECONCILIATION")
                source_path = root / (SOURCE_FILENAMES[sport] if sport == "MLB" else adapter.path_name)
                report.setdefault("restore_reconciliation", {})[sport] = reconcile_sport(
                    sport, canonical_path, source_path)
                audit_stage(sport, "CANONICAL_BACKUP")
                report["canonical_backup"] = sync_canonical(canonical_path, client, folder,
                                                             canonical_session)
            try:
                stage = "capture_and_grade"
                audit_stage(sport, "CAPTURE_AND_GRADE")
                progress(sport, stage)
                if sport in ("NBA", "NCAAB", "NHL") and "after_capture" in inspect.signature(
                        adapter.run_cycle).parameters:
                    from app_core.prospective_research_models import run_sport_model_cycle
                    def model_after_capture():
                        model_report = run_sport_model_cycle(path, canonical_path, sport)
                        report["canonical_backup"] = sync_canonical(
                            canonical_path, client, folder, canonical_session)
                        return model_report
                    result=adapter.run_cycle(path,state,cfbd_key,odds_key,backup,budget,
                        after_capture=model_after_capture)
                else:
                    result=adapter.run_cycle(path,state,cfbd_key,odds_key,backup,budget)
                stage = "canonical_reconciliation"
                if sport in ("NFL", "NCAAF", "MLB"):
                    from app_core.prospective_reconciliation import reconcile_sport
                    from app_core.prospective_source_view import SOURCE_FILENAMES
                    source_path = root / (SOURCE_FILENAMES[sport] if sport == "MLB" else adapter.path_name)
                    audit_stage(sport, "CANONICAL_RECONCILIATION")
                    result["reconciliation"] = reconcile_sport(sport, canonical_path, source_path)
                    health["reconciliation_status"] = "success"
                else:
                    health["reconciliation_status"] = (
                        "research_model_cycle_completed" if "model_cycle" in result else "failed")
                report["sports"][sport]=result
                report["errors"].extend(sport+":"+reason for reason in result["errors"])
                health["capture"] = result.get("capture_status", "success" if not result["errors"] else "failed")
                health["grade"] = result.get("grade_status", "success" if not result["errors"] else "failed")
                health["close_capture"] = result.get("close_capture_status", "legacy_specialized")
                if health["capture"] == "success":
                    health["last_successful_capture"] = utcnow().isoformat()
                if health["grade"] == "success":
                    health["last_successful_grade"] = utcnow().isoformat()
                health["discovered_events"] = result.get("discovered", result.get("upcoming_games", 0))
                health["captured_events"] = result.get("captured",0)
                health["graded_events"] = result.get("graded",0)
                health["captured_quote_rows"] = result.get("captured_quote_rows",0)
                health["research_predictions"] = result.get("predictions",0)
                health["research_close_candidates"] = result.get("close_candidates",0)
                health["provider_blockers"] = sorted(set(result.get("blockers",[]) + result["errors"]))
            finally:
                # A backup error must be distinguishable from provider capture.
                previous_stage = stage
                stage = "backup"
                native_error = None
                try:
                    backup()
                    if sport == "MLB":
                        from app_core.mlb_receipt_remote import backup as backup_receipts, verify_backup
                        from app_core.prospective_source_view import SOURCE_FILENAMES
                        receipt_report = backup_receipts(receipt_client, folder,
                                                         root / SOURCE_FILENAMES[sport])
                        report.setdefault("receipt_backup", {})[sport] = verify_backup(
                            receipt_client, folder, receipt_report)
                except Exception as exc:
                    native_error = exc
                stage = "canonical_backup"
                audit_stage(sport, "CANONICAL_BACKUP")
                try:
                    report["canonical_backup"] = sync_canonical(canonical_path, client, folder,
                                                                 canonical_session)
                except Exception:
                    health["verified_backup"] = False
                    raise
                if native_error is not None:
                    stage = "backup"
                    raise native_error
                stage = previous_stage
        except Exception as exc:
            # Never include provider exception text: URLs may contain API keys.
            known={"missing_frozen_model","stale_frozen_model","missing_provider_keys"}
            code=str(exc) if str(exc) in known else type(exc).__name__
            if isinstance(exc, ResearchStageError):
                stage, code = exc.stage, exc.code
            progress(sport, "failed", failed_stage=stage, code=code)
            report["errors"].append(sport+":"+code)
            report.setdefault("failure_stages", {})[sport] = {"stage": stage, "code": code}
            health[stage if stage in ("restore","backup","canonical_backup") else "capture"]="failed"
            if stage in ("backup", "canonical_backup"):
                health["verified_backup"] = False
            health["provider_blockers"].append(code)
            if audit_callback is not None:
                audit_callback(report)
        health["api_budget"] = (budget.report()["by_sport"].get(sport)
                                if sport != "MLB" else
                                {"provider":"MLB_STATS_API", "status":"legacy_bounded_cycle_no_shared_ledger"})
        if audit_callback is not None:
            audit_callback(report)
    report["finished_at"]=utcnow().isoformat()
    report["api_budget"]=budget.report()
    report["requested_slate_success"] = not report["errors"] and all(
        report["health"][s]["restore"] == "success" and
        report["health"][s]["backup"] == "success" and
        report["health"][s].get("canonical_backup") != "failed" and
        report["health"][s]["capture"] == "success" and
        report["health"][s]["grade"] == "success" and
        report["health"][s]["close_capture"] in ("success", "legacy_specialized") and
        report["sports"].get(s,{}).get("budget_paused") is not True for s in sports)
    report["execution_state"] = "COMPLETE" if report["requested_slate_success"] else "FAILED"
    state["last_run"]=report
    audit_stage(None, "CHECKPOINT")
    checkpoint(client,folder,state)
    audit_stage(None, "COMPLETE")
    return report
