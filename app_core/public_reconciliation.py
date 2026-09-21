"""Read-only reconciliation of immutable public sources; no fetching or writes."""
from collections import defaultdict
from copy import deepcopy
import re
from app_core.public_history import digest, selections, grade_leg
from app_core.locked_picks import locked_selections
from app_core.result_reconciliation import latest_scores, match_result


def load_source(store):
    # UI restore_history can confirm deployments; this path never calls it.
    return dict(publications=store.publications(), revisions=store.all("scores"), locks=store.all("locks"))


def counts(rows, day):
    n = lambda outcome: sum(r["outcome"] == outcome for r in rows)
    w, l = n("WIN"), n("LOSS")
    return dict(date=day, records=len(rows), wins=w, losses=l, pushes=n("PUSH"),
                pending=n("PENDING"), needs_review=n("NEEDS_REVIEW"), win_rate=w/(w+l) if w+l else None)


def selection_facts(item):
    legs = []
    for leg in item.get("legs", []):
        market, pick = leg.get("market") or "", leg.get("pick") or ""
        match = re.fullmatch(r"(?:Over|Under)\s+(\d+(?:\.\d+)?)", pick, re.I) if market.startswith("total_") else re.fullmatch(r".+\s+([+-]\d+(?:\.\d+)?)", pick)
        legs.append(dict(league=leg.get("sport"), game=leg.get("game"), selection=pick,
                         market_type=market, line=float(match[1]) if match else None,
                         odds=leg.get("odds"), sportsbook=leg.get("quote_source"),
                         original_win_estimate=leg.get("win_estimate"),
                         start=leg.get("start")))
    return dict(id=item.get("id"), category=item.get("category"), group=item.get("group"),
                date=item.get("date"), published_at=item.get("published_at"), legs=legs)


def prepare_source(source):
    publications, errors = [], []
    for pub in source.get("publications", []):
        if digest(pub["package"]) != pub["package_hash"]:
            errors.append(dict(reason="identity_conflict", package_hash=pub["package_hash"]))
        else:
            publications.append(pub)
    entries = selections(publications)
    # The history selector uses its existing identity. Detect collisions in the
    # first confirmed package instead of accepting its first row arbitrarily.
    first = {r['id']:r for r in entries}
    for pub in publications:
        for family, legs in pub['package']['games'].items():
            for leg in legs:
                single = {**pub, 'package': dict(pub['package'])}
                single['package']['games'] = {'overall':[], 'sides':[], 'totals':[]}
                single['package']['games'][family] = [leg]
                single['package']['parlays'] = []
                single['package']['research_parlays'] = []
                single['package'].pop('top_ten_policy',None)
                for item in selections([single]):
                    original = first.get(item['id'])
                    if original and item['published_at'] == original['published_at'] and item != original:
                        entries.append(item)
    entries += deepcopy(source.get("locks", []))
    scores = latest_scores(source.get("revisions", []))
    return entries, scores, errors, digest(source)


def reconcile_many(source, days, progress=None):
    prepared = prepare_source(source)
    results = []
    days = sorted(set(days))
    for index, day in enumerate(days):
        results.append(_reconcile_day(prepared, day))
        if progress:
            progress(index + 1, len(days), day)
    return results


def reconcile(source, day):
    return _reconcile_day(prepare_source(source), day)


def _reconcile_day(prepared, day):
    entries, scores, errors, source_hash = prepared
    by_id = defaultdict(list)
    for index, item in enumerate(entries):
        if item.get("date") == day:
            key = item.get("id") or ("missing_identity",index)
            by_id[key].append(item)
    records = []
    for identity, variants in sorted(by_id.items(), key=lambda x: str(x[0])):
        item = variants[0]
        reasons = []
        conflict = not item.get("id") or len({digest(v) for v in variants}) != 1
        if len(variants) > 1:
            reasons.append("duplicate_record")
        if item.get("group") == "Locked":
            try:
                locked_selections([item])
            except (ValueError, KeyError, TypeError):
                conflict = True
        if conflict:
            reasons.append("identity_conflict")
        record = selection_facts(item)
        outcomes = []
        for leg, facts in zip(item.get("legs", []), record["legs"]):
            if facts["line"] is None and facts["market_type"].startswith(("spread_", "total_")):
                reasons.append("missing_original_line")
            if facts["odds"] is None:
                reasons.append("missing_original_price")
            try:
                _, failure = match_result(leg, scores)
                outcome, score = grade_leg(leg, scores)
            except (ValueError, KeyError, TypeError):
                failure, outcome, score = "INVALID_SAVED_SELECTION", "PENDING", None
            if failure and failure not in {"NO_FINAL_PROVIDER_RESULT", "NO_MATCH", "PROVIDER_ID_NOT_FOUND", "EVENT_ID_NOT_FOUND"}:
                reasons.append("identity_conflict")
            if outcome == "PENDING":
                reasons.append("pending" if failure else "missing_outcome")
            outcomes.append(outcome)
            facts.update(final_score=score, outcome=outcome, provider_reason=failure)
        if not outcomes or any(r in reasons for r in ("identity_conflict", "missing_original_line", "missing_original_price", "missing_outcome")):
            outcome = "NEEDS_REVIEW"
        else:
            outcome = "PENDING" if "PENDING" in outcomes else "LOSS" if "LOSS" in outcomes else "PUSH" if "PUSH" in outcomes else "WIN"
        record.update(outcome=outcome, reason_codes=sorted(set(reasons)) or ["reconciled"])
        if conflict:
            record["conflicting_selections"] = [selection_facts(v) for v in variants]
        records.append(record)
    published = lambda r: r["group"] in {"Approved", "Research"}
    cohorts = {"Locked Overall": [r for r in records if r["group"] == "Locked" and r["category"] == "overall"]}
    cohorts.update({"Published " + label: [r for r in records if published(r) and r["category"] == category]
                    for category, label in (("overall", "Overall"), ("sides", "Sides"), ("totals", "Totals"))})
    cohorts.update({group:[r for r in records if r["group"] == group] for group in ("Approved", "Research")})
    return dict(date=day, source_hash=source_hash, status="NEEDS_REVIEW" if errors or any(r["outcome"] == "NEEDS_REVIEW" for r in records) else "RECONCILED",
                source_errors=errors, cohorts={k:counts(v,day) for k,v in cohorts.items()}, records=records,
                overlap_notice="Categories and groups overlap; never sum these cohorts as independent wagers.")


def markdown(result):
    lines = ["# Public results reconciliation", "", result["date"], "", result["overlap_notice"], "",
             "| Cohort | Records | W | L | Push | Pending | Review | Win rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["cohorts"].items():
        rate = f"{row['win_rate']:.1%}" if row['win_rate'] is not None else "Unavailable"
        lines.append(f"| {name} | {row['records']} | {row['wins']} | {row['losses']} | {row['pushes']} | {row['pending']} | {row['needs_review']} | {rate} |")
    lines.extend(["", "Exact original selections and structured review reasons are in the accompanying JSON.", "Source SHA256: " + result["source_hash"], "Status: " + result["status"]])
    return "\n".join(lines) + "\n"
