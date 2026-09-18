"""Rebuild auditable, sport-isolated ranking candidates; never activate authority."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
from core.wager_decisions import aware
from core.exposure_ledger import digest
from core.empirical_tiers import bucket_key


def producer_requirements(cohort):
    """Explain original producer facts without inferring validation or authority."""
    counts = Counter()
    statuses = Counter()
    for row in cohort:
        status = row.get('mlb_challenger_status')
        if isinstance(status, str) and status:
            statuses[status] += 1
        bad = set(row.get('exclusion_reasons', []))
        if 'missing_model_version' in bad:
            counts['model_artifact_not_configured' if status == 'NOT_CONFIGURED'
                   else 'production_model_provenance_missing'] += 1
        if bad & {'missing_calibration_version', 'calibration_not_available'}:
            counts['matching_calibration_provenance_required'] += 1
        if bad & {'missing_training_cutoff', 'unverified_training_cutoff', 'model_not_available'}:
            counts['verified_training_and_availability_times_required'] += 1
        if 'identity_unverified' in bad:
            counts['verified_event_identity_required'] += 1
        if bad & {'missing_evidence_version', 'evidence_not_available', 'missing_selection_policy_version', 'missing_sport_policy_version'}:
            counts['original_evidence_and_policy_provenance_required'] += 1
        if 'missing_conservative_probability' in bad:
            counts['prospective_uncertainty_evidence_required'] += 1
        if bad & {'missing_outcome', 'missing_or_future_outcome_timestamp'}:
            counts['verified_settlement_required'] += 1
    return {'counts': dict(counts), 'challenger_status_counts': dict(statuses),
            'note': 'Counts are candidate rows, not independent games. Challenger facts cannot authorize baseline probabilities. Missing provenance is not repaired retrospectively.'}


def build(rows, excluded=()):
    """Input rows must come from activation_validation.read_dataset.

    Select one latest saved best pick per sport/game across model versions.
    Never choose between repeated predictions using their eventual outcome.
    """
    rejected = Counter(reason for row in excluded for reason in row['exclusion_reasons'])
    games = defaultdict(list)
    for row in rows:
        if str(row.get('selected_as_best_pick')).lower() != 'true':
            rejected['not_saved_best_pick'] += 1
            continue
        key = (row.get('sport'), row.get('game_id'))
        if not all(key):
            rejected['missing_event_identity'] += 1
            continue
        games[key].append(row)
    selected = []
    for group in games.values():
        latest = max(aware(r['prediction_generated_at']) for r in group)
        finalists = [r for r in group if aware(r['prediction_generated_at']) == latest]
        unique = {digest(r): r for r in finalists}
        if len(unique) != 1:
            rejected['ambiguous_latest_selection'] += len(finalists)
            continue
        row = next(iter(unique.values()))
        if row.get('candidate_outcome') not in {'WIN', 'LOSS'}:
            rejected['not_decided_win_loss'] += 1
            continue
        if row.get('consensus_agreement') not in {'Agrees', 'Neutral', 'Disagrees', 'No Kalshi'}:
            rejected['missing_saved_consensus'] += 1
            continue
        selected.append(row)
    sports = {}
    for sport in sorted({r['sport'] for r in selected}):
        cohort = [r for r in selected if r['sport'] == sport]
        anchor = max(aware(r['game_start_utc']).date() for r in cohort)
        buckets = defaultdict(list)
        for row in cohort:
            buckets[bucket_key(sport, row['market_type'], row['consensus_agreement'])].append(row)
        def summarize(items):
            weights = [0.5 ** ((anchor - aware(r['game_start_utc']).date()).days / 21) for r in items]
            rate = sum(w * (r['candidate_outcome'] == 'WIN') for w, r in zip(weights, items)) / sum(weights)
            n = max(1, round(sum(weights)**2 / sum(w*w for w in weights)))
            return dict(n=n, wins=round(n*rate), win_rate=rate, raw_n=len(items))
        sports[sport] = dict(overall=summarize(cohort), buckets={k:summarize(v) for k,v in buckets.items()},
                            meta=dict(recency_anchor=anchor.isoformat(), source_hash=digest(cohort),
                                      status='CANDIDATE_NOT_ACTIVATED', unit='latest_saved_best_pick_per_game'))
    # Separate current producer health from thousands of immutable legacy rows.
    latest = {}
    for row in [*rows, *excluded]:
        sport = row.get('sport') or row.get('league') or 'UNKNOWN'
        at = aware(row.get('prediction_generated_at'))
        if at is None:
            continue
        if sport not in latest or at > latest[sport][0]:
            latest[sport] = (at, [row])
        elif at == latest[sport][0]:
            latest[sport][1].append(row)
    producer_health = {}
    for sport, (at, cohort) in latest.items():
        issues = Counter(reason for row in cohort for reason in row.get('exclusion_reasons', []))
        producer_health[sport] = dict(prediction_generated_at=at.isoformat(), rows=len(cohort),
                                     exclusions=dict(issues), producer_requirements=producer_requirements(cohort))
    return dict(schema=1, latest_producer_health=producer_health, status='CANDIDATE_NOT_ACTIVATED' if sports else 'BLOCKED_NO_ELIGIBLE_EVIDENCE',
                eligible_games=len(selected), exclusions=dict(rejected), sports=sports,
                activation_blocker='Requires sport-isolated runtime integration and prospective validation before replacing the legacy pooled overlay.')


def rebuild(database, output):
    from core.activation_validation import read_dataset
    output = Path(output)
    try:
        rows, excluded = read_dataset(database)
        report = build(rows, excluded)
    except Exception as exc:
        report = dict(schema=1, status='ERROR', error_type=type(exc).__name__, eligible_games=0,
                      activation_blocker='Evidence could not be verified; active ranking artifacts unchanged.')
    report['checked_at'] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    return report
