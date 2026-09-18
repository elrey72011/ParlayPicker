"""Descriptive research results, isolated from model/wager validation and ranking."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from contextlib import closing
from core.wager_decisions import aware
from core.exposure_ledger import digest


def build(rows, now=None):
    clock = now or datetime.now(timezone.utc)
    rejected, groups = Counter(), defaultdict(list)
    for row in rows:
        if str(row.get('selected_as_best_pick')).lower() not in {'true', '1'}:
            rejected['not_explicit_saved_best_pick'] += 1
            continue
        pred, saved, start = (aware(row.get(k)) for k in
                              ('prediction_generated_at','capture_recorded_at','game_start_utc'))
        if not pred or not saved or not start or not pred <= saved < start <= clock:
            rejected['unverified_pregame_capture_or_future_game'] += 1
            continue
        sport, event = row.get('sport'), row.get('matchup_id')
        if sport not in {'MLB','NBA','WNBA','NFL','NCAAF','NCAAB','NHL'} or not event:
            rejected['missing_sport_event_identity'] += 1
            continue
        groups[(sport,event)].append(row)
    accepted = []
    for group in groups.values():
        # Freeze the latest selection BEFORE inspecting settlement or consensus.
        latest = max(aware(r['capture_recorded_at']) for r in group)
        finalists = {digest(r): r for r in group if aware(r['capture_recorded_at']) == latest}
        if len(finalists) != 1:
            rejected['ambiguous_latest_selection'] += 1
            continue
        row = next(iter(finalists.values()))
        outcome_at = aware(row.get('outcome_recorded_at'))
        if (row.get('result_source') not in {'ESPN','MLB'} or not row.get('result_provider_event_id')
                or not outcome_at or not aware(row['game_start_utc']) <= outcome_at <= clock):
            rejected['verified_provider_settlement_required'] += 1
            continue
        if str(row.get('quote_binding_verified')).lower() not in {'true','1'}:
            rejected['exact_quote_binding_required'] += 1
            continue
        quote_at = aware(row.get('odds_recorded_at'))
        from core.line_evidence import line_rejected
        if not quote_at or quote_at > aware(row['prediction_generated_at']) or line_rejected(row):
            rejected['invalid_original_quote_or_line'] += 1
            continue
        if row.get('market_type') not in {'spread_home','spread_away','total_over','total_under'}:
            rejected['unsupported_market'] += 1
            continue
        if row.get('candidate_outcome') not in {'WIN','LOSS','PUSH'}:
            rejected['unresolved_outcome'] += 1
            continue
        if row.get('consensus_agreement') not in {'Agrees','Disagrees','Neutral','No Kalshi'}:
            rejected['missing_original_consensus'] += 1
            continue
        accepted.append(row)
    from core.empirical_tiers import bucket_key
    from zoneinfo import ZoneInfo
    buckets = defaultdict(list)
    for row in accepted:
        buckets[bucket_key(row['sport'],row['market_type'],row['consensus_agreement'])].append(row)
    summaries = {}
    for key, cohort in sorted(buckets.items()):
        counts = Counter(r['candidate_outcome'] for r in cohort)
        decided = counts['WIN'] + counts['LOSS']
        anchor = max(aware(r['game_start_utc']).astimezone(ZoneInfo('America/New_York')).date() for r in cohort)
        summaries[key] = dict(wins=counts['WIN'],losses=counts['LOSS'],pushes=counts['PUSH'],games=len(cohort),
                              win_rate=counts['WIN']/decided if decided else None,
                              latest_settled_slate=anchor.isoformat(),
                              source_hash=digest(cohort))
    return dict(schema='research-performance-v1',status='DESCRIPTIVE_ONLY' if accepted else 'BLOCKED_NO_RESEARCH_RESULTS',
                eligible_games=len(accepted),exclusions=dict(rejected),buckets=summaries,
                ranking_authority=False,wager_authority=False,
                note='Latest saved pregame best pick per sport/event. No model calibration claim; no active bucket artifact replacement.')


def rebuild(database):
    from app_core.prediction_evidence import materialize, connect
    from app_core.candidate_evidence_schema import evidence_value
    frame, _ = materialize(database)  # verifies immutable snapshot and score payload hashes
    if frame.empty:
        return build([])
    with closing(connect(database)) as db:
        captured = dict(db.execute('SELECT snapshot_id, generated_at FROM snapshots'))
    rows = []
    for source in frame.to_dict('records'):
        row = evidence_value(source)
        row['capture_recorded_at'] = captured.get(row.get('snapshot_id'))
        rows.append(row)
    return build(rows)
