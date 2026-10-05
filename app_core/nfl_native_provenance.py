"""Native NFL aggregate dependencies in the existing private inference packet.

Returned-frame observations are not publisher first-publication clocks. Nothing
here acquires data, changes features, executes saved code or grants authority.
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import pandas as pd
from app_core.research_estimate_trace import encode, fact
from app_core.producer_provenance import clock, team, number
from app_core.research_replay import cell, value
from core.nfl_teams import nfl_stats_identity

VERSION = "nfl-native-aggregate-v1"
SCOPE = "nfl-score-aggregate-scope-v1"
COLUMN = "nfl_native_aggregate_receipt"
STATS = ("points_per_game", "points_allowed_per_game", "games_played", "win_pct", "last5_win_pct", "recent_point_margin")
CODE_PATHS = ("app_core/feature_processing.py", "app_core/nfl_native_provenance.py")
ROOT = Path(__file__).resolve().parents[1]
MAPPING = dict(ppg="points_per_game", oppg="points_allowed_per_game", games_played="games_played", win_pct="win_pct", recent_point_margin="recent_point_margin")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def code_identity():
    return {p: hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in CODE_PATHS}


def observe(frame, module):
    """Preserve original typed returned rows and the actual adapter observation."""
    observed = now()
    try:
        source = inspect.getsource(module.import_schedules)
    except (TypeError, OSError):
        source = None
    module_name=getattr(module, "__name__", None)
    module_version=getattr(module, "__version__", None)
    if not module_version and module_name:
        try:module_version=importlib.metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:pass
    consumed=("game_id","season","gameday","game_date","date","home_team","away_team","home_score","away_score","result","home_turnovers","away_turnovers","source_available_at")
    original=frame[[k for k in consumed if k in frame]].copy()
    return dict(version=VERSION, observed_at=observed, publisher_available_at=None,
        source=dict(module=module_name, version=module_version,
                    function="import_schedules", callable_source=source,
                    callable_source_sha256=hashlib.sha256(source.encode()).hexdigest() if source else None,
                    representation="actual_consumed_returned_frame_projection", wire_bytes_retained=False),
        columns=list(original.columns), rows=[{k:cell(v) for k,v in row.items()} for row in original.to_dict("records")])


def selected_rows(original, season, cutoff, team_name):
    """Reproduce the native date/completed-game selection on retained cells."""
    date_name = next((k for k in ("gameday", "game_date", "date") if k in original["columns"]), None)
    if date_name is None or cutoff is None:
        raise ValueError("aggregate cutoff/date evidence missing")
    stop = pd.Timestamp(cutoff)
    if stop.tzinfo is not None:
        stop = stop.tz_convert("UTC").tz_localize(None)
    candidates=[]
    for raw in original["rows"]:
        row={k:value(v) for k,v in raw.items()}
        dt=pd.to_datetime(row.get(date_name), errors="coerce", utc=True)
        if pd.isna(dt) or not dt.tz_localize(None)<stop.normalize():continue
        if "result" in row and pd.isna(row["result"]):continue
        h,a=pd.to_numeric(row.get("home_score"),errors="coerce"),pd.to_numeric(row.get("away_score"),errors="coerce")
        if pd.isna(h) or pd.isna(a):continue
        home=nfl_stats_identity(row.get("home_team"),schedule_code=True)
        away=nfl_stats_identity(row.get("away_team"),schedule_code=True)
        if team_name not in {home,away}:continue
        if home==away:raise ValueError("aggregate same-team event")
        if row.get("season") is not None and pd.notna(row["season"]) and int(row["season"])!=season:raise ValueError("aggregate season conflict")
        is_home=home==team_name
        candidates.append((dt,raw,float(h if is_home else a),float(a if is_home else h)))
    # The native adapter's stable date ordering is retained, including tied dates.
    return sorted(candidates,key=lambda x:x[0])


def aggregates(rows):
    if not rows:raise ValueError("aggregate has no completed members")
    count=len(rows); recent=rows[-5:]
    return dict(points_per_game=sum(r[2] for r in rows)/count,
        points_allowed_per_game=sum(r[3] for r in rows)/count, games_played=count,
        win_pct=sum(int(r[2]>r[3]) for r in rows)/count,
        last5_win_pct=sum(int(r[2]>r[3]) for r in recent)/len(recent),
        recent_point_margin=sum(r[2]-r[3] for r in recent)/len(recent))


def retain_stats(stats, observation, season, cutoff):
    """Add parallel provenance after the unchanged native aggregation."""
    for stat in stats:
        try:
            members=selected_rows(observation,season,cutoff,stat["team_norm"])
            fragment=dict(observation,rows=[r[1] for r in members])
            payload=dict(version=VERSION,kind="team_season_completed_before_utc_day",
                team=stat["team_norm"],season=season,cutoff=cutoff,original=fragment,
                values={k:cell(stat[k]) for k in STATS},adapter_code=code_identity(),
                transformation="native-stable-date-completed-score-aggregate-and-last5-v1")
            stat[COLUMN]=dict(payload=payload,sha256=digest(payload))
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            stat[COLUMN]=None  # Missing provenance never changes a numerical result.
    return stats


def _bind(frame, home_keys, away_keys, leagues, lookup):
    """Bind native aggregates to the exact target event and mapped feature slots."""
    out=frame.copy(); retained=frame["nfl_feature_dependencies"].copy() if "nfl_feature_dependencies" in frame else pd.Series(None,index=frame.index,dtype=object)
    from app_core.nfl_inference_evidence import FEATURES
    from app_core.producer_provenance import _offer
    for idx,row in frame.iterrows():
        if leagues.at[idx]!="NFL":continue
        line=number(row.get("spread_line"))
        if row.get("market_type") not in {"spread_home","spread_away"} or line is None or abs(line%1)!=.5:continue
        dependencies={}
        event=_offer(row,now())["event"]
        mapped=now()
        for name in FEATURES:
            if fact(row.get(name))["state"]!="VALUE":continue
            slots=("home","away") if name=="feature_diff_last5" else (name.split("_")[1],)
            stat_name="last5_win_pct" if name=="feature_diff_last5" else MAPPING[name.split("_",2)[2]]
            inputs=[]
            for slot in slots:
                key=(home_keys if slot=="home" else away_keys).at[idx]
                stat=lookup.get(("NFL",key),{})
                receipt=stat.get(COLUMN)
                if isinstance(receipt,dict):inputs.append(dict(slot=slot,stat=stat_name,receipt=receipt))
            if len(inputs)!=len(slots):continue
            available=[clock(x["receipt"]["payload"]["original"]["observed_at"]) for x in inputs]
            av=max(available) if all(available) else None
            scope=dict(contract=SCOPE,feature=name,event=event,available_at=av,observed_at=mapped,
                availability_basis="original_adapter_first_observation",publisher_available_at=None)
            original=dict(version=VERSION,event_id=event["provider_event_id"],value=fact(row[name])["value"],scope=scope,
                inputs=inputs,transformation="mapped-float-clamp-win-v1" if name!="feature_diff_last5" else "mapped-safe-home-minus-away-last5-v1",
                adapter_code=code_identity())
            raw=encode(original).encode()
            payload=dict(feature=name,value=fact(row[name])["value"],source_id="parlaypicker:nfl-native-derivation:v1:"+hashlib.sha256(raw).hexdigest(),
                provider_event_id=event["provider_event_id"],available_at=av,observed_at=mapped,scope=scope,scope_path=["scope"],
                source_artifact=dict(bytes_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()),
                value_path=["value"],event_path=["event_id"])
            dependencies[name]=dict(payload=payload,sha256=digest(payload))
        retained.at[idx]=encode(dependencies) if dependencies else None
    if leagues.eq("NFL").any():out["nfl_feature_dependencies"]=retained
    return out


def bind(frame, home_keys, away_keys, leagues, lookup):
    try:return _bind(frame,home_keys,away_keys,leagues,lookup)
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        out=frame.copy()
        if leagues.eq("NFL").any():
            if "nfl_feature_dependencies" not in out:out["nfl_feature_dependencies"]=None
            out.loc[leagues.eq("NFL"),"nfl_feature_dependencies"]=None
        return out  # Capture failure stays UNKNOWN; feature/probability values survive.


def validate(dependency, original, expected_event, name, consumed, errors, unknown):
    """Semantic applicability/derivation validation, never source authentication."""
    prefix="features.native:"+name+":"
    def reject(reason):errors.append(prefix+reason)
    def missing(reason):unknown.append(prefix+reason)
    try:
        scope=dependency["scope"]
        if encode(original.get("scope"))!=encode(scope) or dependency.get("scope_path")!=["scope"]:reject("original_scope")
        if scope.get("contract")!=SCOPE or scope.get("feature")!=name:reject("scope_contract")
        if scope.get("event")!=expected_event:reject("target_event")
        if original.get("event_id")!=(expected_event or {}).get("provider_event_id"):reject("target_id")
        if scope.get("availability_basis")!="original_adapter_first_observation" or scope.get("publisher_available_at") is not None:reject("availability_basis")
        if original.get("version")!=VERSION:reject("version")
        if original.get("adapter_code")!=code_identity():reject("adapter_identity")
        inputs=original.get("inputs")
        slots=["home","away"] if name=="feature_diff_last5" else [name.split("_")[1]]
        stat_name="last5_win_pct" if name=="feature_diff_last5" else MAPPING[name.split("_",2)[2]]
        if inputs is None or inputs==[]:missing("input_coverage");return
        if not isinstance(inputs,list):reject("input_slots");return
        found=[x.get("slot") for x in inputs]
        if len(found)<len(slots) and all(x in slots for x in found) and len(set(found))==len(found):missing("input_coverage");return
        if found!=slots:reject("input_slots");return
        computed=[];observations=[]
        target=clock((expected_event or {}).get("start"))
        if target is None:missing("target_clock");return
        for member,slot in zip(inputs,slots):
            if member.get("stat")!=stat_name:reject("stat_mapping")
            retained=member["receipt"];p=retained["payload"]
            if set(retained)!={"payload","sha256"} or digest(p)!=retained["sha256"]:reject("aggregate_integrity")
            if p.get("version")!=VERSION or p.get("kind")!="team_season_completed_before_utc_day" or p.get("transformation")!="native-stable-date-completed-score-aggregate-and-last5-v1":reject("aggregate_contract")
            if p.get("adapter_code")!=code_identity():reject("aggregate_adapter_identity")
            required=("team","season","cutoff","original","values")
            absent=[k for k in required if p.get(k) is None]
            if absent:
                for k in absent:missing("aggregate_"+k)
                continue
            if team(p.get("team"),"NFL")!=(expected_event or {}).get(slot):reject("aggregate_team")
            if not p.get("cutoff"):missing("aggregate_cutoff");continue
            cutoff=pd.Timestamp(p["cutoff"])
            if cutoff.tzinfo is not None:cutoff=cutoff.tz_convert("UTC").tz_localize(None)
            if cutoff.normalize()>pd.Timestamp(target).tz_localize(None).normalize():reject("cutoff_after_event")
            original_source=p["original"]
            source=original_source.get("source",{})
            for k in ("module","version","callable_source","callable_source_sha256"):
                if not source.get(k):missing("source_identity:"+k)
            if source.get("callable_source") and hashlib.sha256(source["callable_source"].encode()).hexdigest()!=source.get("callable_source_sha256"):
                reject("source_callable_integrity")
            if source.get("function")!="import_schedules" or source.get("representation")!="actual_consumed_returned_frame_projection" or source.get("wire_bytes_retained") is not False:reject("source_representation")
            observed=clock(original_source.get("observed_at"))
            if observed is None:missing("original_observation");continue
            observations.append(observed)
            publisher=original_source.get("publisher_available_at")
            if publisher is not None and (clock(publisher) is None or clock(publisher)>observed):reject("publisher_availability")
            if not original_source.get("rows") or not original_source.get("columns"):
                missing("member_coverage");continue
            date_name=next((k for k in ("gameday","game_date","date") if k in original_source["columns"]),None)
            if date_name is None:
                missing("member_date");continue
            incomplete=False
            for raw in original_source["rows"]:
                for key in ("home_team","away_team","home_score","away_score",date_name):
                    if key not in raw or fact(value(raw[key]))["state"]!="VALUE":
                        missing("member_"+key);incomplete=True
                if date_name in raw:
                    member_date=pd.to_datetime(value(raw[date_name]),errors="coerce",utc=True)
                    if pd.notna(member_date) and member_date>pd.Timestamp(observed):reject("member_after_observation")
                if "source_available_at" in raw and fact(value(raw["source_available_at"]))["state"]=="VALUE":
                    member_availability=clock(value(raw["source_available_at"]))
                    if member_availability is None or member_availability>observed:reject("member_availability")
            if incomplete:continue
            absent=[k for k in STATS if k not in p["values"] or fact(value(p["values"][k]))["state"]!="VALUE"]
            if absent:
                for k in absent:missing("aggregate_value:"+k)
                continue
            rows=selected_rows(original_source,p["season"],p["cutoff"],p["team"])
            if len(rows)!=len(original_source["rows"]):reject("unrelated_or_ineligible_member")
            identifiers=[]
            for _,raw,_,_ in rows:
                season=value(raw.get("season",{"type":"none"}))
                if fact(season)["state"]!="VALUE":missing("member_season")
                event_id=value(raw.get("game_id",{"type":"none"}))
                if fact(event_id)["state"]!="VALUE":missing("member_event_id")
                else:identifiers.append(event_id)
            if len(identifiers)!=len(set(identifiers)):reject("duplicate_members")
            actual=aggregates(rows)
            if any(cell(actual[k])!=p["values"][k] for k in STATS):reject("aggregate_values")
            n=float(value(p["values"][stat_name]))
            computed.append(min(1.,max(0.,n)) if stat_name=="win_pct" else n)
        if len(computed)!=len(slots):missing("derivation_inputs");return
        result=computed[0]-computed[1] if name=="feature_diff_last5" else computed[0]
        transform="mapped-safe-home-minus-away-last5-v1" if name=="feature_diff_last5" else "mapped-float-clamp-win-v1"
        if original.get("transformation")!=transform:reject("transformation")
        if fact(result)!=consumed or fact(original.get("value"))!=consumed:reject("derived_value")
        if not observations:missing("original_observation");return
        for key in ("available_at","observed_at"):
            if not scope.get(key) or not dependency.get(key):
                missing("scope_"+key);return
            if clock(scope[key]) is None or clock(dependency[key]) is None:
                reject("invalid_scope_clock:"+key);return
        av=clock(scope.get("available_at"));ob=clock(scope.get("observed_at"))
        if av!=max(observations) or av!=clock(dependency.get("available_at")):reject("availability_origin")
        if ob!=clock(dependency.get("observed_at")) or ob is None or av is None or not av<=ob<target:reject("availability_window")
    except (KeyError,ValueError,TypeError,AttributeError,IndexError,OverflowError,OSError):reject("schema")
