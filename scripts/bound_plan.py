"""Draft, review and freeze a plan version bound to a research model and calibration (roadmap C2, D11).

    python scripts/bound_plan.py draft  --directory DIR --output OUT [--sport NCAAB --market SPREAD]
                                        [--validation-start ISO] [--cache-dir DIR]
    python scripts/bound_plan.py freeze --directory DIR --output OUT --sport NCAAB --market SPREAD
                                        --inputs JSON --expected-hash SHA256 [--cache-dir DIR]

``draft`` restores the canonical store **read-only** (every Drive upload is refused),
reports each research market's C2 status, and drafts the proposed bound plan with
its hash and the exact freeze inputs. It changes nothing.

``freeze`` is the owner's step after reviewing a draft. It restores the canonical
store, redrafts the plan from the reviewed inputs (no dependence on observed rates),
freezes it only when the redrafted plan hashes to ``--expected-hash``, and then
backs the new row up to Drive with read-back verification. A plan never freezes
without that reviewed hash. Freezing grants no wager authority: the plan still has
to validate, be reviewed and be activated by the owner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import prospective_calibration as calibration  # noqa: E402
from app_core import prospective_research_models as research  # noqa: E402

CANONICAL = 'prospective-evidence.sqlite3'


def _drive(cache_dir):
    from app_core.evidence_drive import DRIVE_CACHE_ENV, DriveStore
    from app_core.evidence_remote import settings
    folder = settings()[0]
    if not folder or not os.getenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', '').strip():
        raise SystemExit('MISSING_CREDENTIALS: PARLAYPICKER_DRIVE_FOLDER_ID / PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT')
    if cache_dir:
        os.environ[DRIVE_CACHE_ENV] = str(cache_dir)
    return DriveStore(folder), folder


def _empty(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SystemExit('the restore directory must start empty, so no stray local row can be uploaded')
    return root / CANONICAL


def statuses(path, now):
    return [calibration.binding_status(path, sport, market, now=now)
            for sport, markets in research.SCOPES.items() for market in markets]


def draft(path, *, sport=None, market=None, validation_start=None, now=None, methodology='market_skill',
          model='research', min_edge=None):
    now = now or datetime.now(timezone.utc)
    result = {'mode': 'DRAFT_READ_ONLY', 'as_of': now.isoformat(), 'markets': statuses(path, now)}
    if sport and market:
        start = validation_start or (now + timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                                     microsecond=0).isoformat()
        try:
            model_id = None
            if model == 'pinnacle':
                from app_core import pinnacle_model
                found = pinnacle_model.registered(path, sport, market)
                if found is None:
                    raise ValueError('no Pinnacle model registered for this market yet')
                model_id = found['model_id']
            spec = calibration.draft_bound_plan(path, sport, market, validation_start=start,
                                                methodology=methodology, model_id=model_id,
                                                min_edge=min_edge if methodology == 'market_skill' else None)
        except ValueError as exc:
            result['draft'] = {'sport': sport, 'market_family': market, 'status': 'NOT_DRAFTABLE', 'reason': str(exc)}
        else:
            result['draft'] = {'sport': sport, 'market_family': market, 'status': 'DRAFTED_NOT_FROZEN',
                               'plan_hash': calibration.plan_hash(spec),
                               'freeze_inputs': calibration.freeze_inputs(spec), 'plan': spec}
        if model == 'pinnacle':
            # Read-only: how fast this plan would fill, before the bet threshold is frozen.
            from app_core import pinnacle_model
            try:
                result['replay_estimate'] = pinnacle_model.replay_estimate(path, sport, market, now=now)
            except Exception as exc:  # an estimate must never block the draft
                result['replay_estimate'] = {'status': 'ERROR', 'error': type(exc).__name__}
    return result


def markdown(result):
    cols = ('sport', 'market_family', 'models', 'most_training_events', 'candidate_decided', 'calibration_id',
            'games_per_day_14d', 'proposed_window_days', 'bound_plan_id', 'next_blocker')
    lines = [f"## Bound plan: {result['mode']}", '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for row in result.get('markets', []):
        lines.append('| ' + ' | '.join('' if row.get(c) is None else str(row.get(c)) for c in cols) + ' |')
    item = result.get('draft')
    if item:
        lines += ['', f"### Draft {item['sport']} {item['market_family']}: {item['status']}"]
        if item['status'] == 'DRAFTED_NOT_FROZEN':
            plan = item['plan']
            lines += ['', f"- plan: `{plan['validation_plan_id']}` (supersedes `{plan['supersedes_plan_id']}`)",
                      f"- model: `{plan['model_id']}`", f"- calibration: `{plan['calibration_id']}`",
                      f"- validation: {plan['validation_start']} to {plan['validation_end']}",
                      f"- holdout: {plan['holdout_start']} to {plan['holdout_end']}",
                      f"- minimum independent sample: {plan['minimum_independent_sample']}",
                      *([f"- bet rule: EV at least {plan['market_skill_methodology']['min_edge']:.0%} at the entry "
                         f"price, at least {plan['market_skill_methodology']['min_bets']} bets per window"]
                        if 'market_skill_methodology' in plan else []),
                      f"- **plan hash:** `{item['plan_hash']}`", '',
                      'To freeze after review, run the workflow with action `freeze`, this hash, and these inputs:',
                      '', '```json', json.dumps(item['freeze_inputs'], sort_keys=True), '```']
        else:
            lines.append(f"- reason: {item.get('reason')}")
    replay = result.get('replay_estimate')
    if replay:
        lines += ['', f"### Replay estimate {replay.get('sport', '')} {replay.get('market_family', '')}: "
                      f"{replay.get('status')}"]
        if replay.get('status') == 'ESTIMATED':
            lines += ['', f"Over {replay['days_observed']} days of captured games (results not read): "
                          f"{replay['plan_games']} of {replay['games_started']} games had a US book at Pinnacle's line "
                          f"({replay['plan_share']:.0%}), {replay['plan_games_per_day']} a day; "
                          f"{replay['min_sample']} games take about {replay['days_for_min_sample']} days.", '',
                      '| bet edge vs Pinnacle | bets | bets/day | days for '
                      f"{replay['min_bets']} bets | days per window |", '|---|---|---|---|---|']
            for edge, row in replay['by_edge'].items():
                lines.append(f"| {edge} | {row['bets']} | {row['bets_per_day']} | {row['days_for_min_bets']} | "
                             f"{row['days_per_window']} |")
            lines += ['', 'A plan has a validation and a holdout window. The frozen threshold is the template\'s '
                          '2% unless the plan is changed before freezing.']
    frozen = result.get('frozen')
    if frozen:
        lines += ['', f"### Frozen `{frozen['validation_plan_id']}` (hash `{frozen['plan_hash']}`)",
                  f"- Drive backup: {json.dumps(result.get('backup'))}"]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('action', choices=('draft', 'freeze'))
    parser.add_argument('--directory', type=Path, required=True, help='empty directory to restore into')
    parser.add_argument('--output', type=Path, required=True, help='JSON report path')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--sport')
    parser.add_argument('--market')
    parser.add_argument('--validation-start', help='ISO timestamp; default is the next UTC midnight')
    parser.add_argument('--methodology', choices=('market_skill', 'v1'), default='market_skill',
                        help='market_skill: scored against the de-vigged entry price (D19 option B); v1: absolute thresholds')
    parser.add_argument('--model', choices=('research', 'pinnacle'), default='research',
                        help="research: the pinned research score model; pinnacle: Pinnacle's no-vig price")
    parser.add_argument('--min-edge', type=float, choices=(0.01, 0.02), default=0.02,
                        help='market_skill only: the minimum EV for a bet (locks when the plan is frozen)')
    parser.add_argument('--inputs', help='freeze inputs JSON printed by the reviewed draft')
    parser.add_argument('--expected-hash', help='plan hash printed by the reviewed draft')
    args = parser.parse_args(argv)
    path = _empty(args.directory)
    client, folder = _drive(args.cache_dir)
    from app_core.prospective_remote import sync
    if args.action == 'draft':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from market_readiness_report import ReadOnlyDrive
        guarded = ReadOnlyDrive(client, args.cache_dir)
        restore = sync(path, guarded, folder, {})
        result = draft(path, sport=args.sport, market=args.market, validation_start=args.validation_start,
                       methodology=args.methodology, model=args.model, min_edge=args.min_edge)
        result['restore'] = dict(restore, writes_refused=guarded.writes_refused)
    else:
        if not (args.sport and args.market and args.inputs and args.expected_hash):
            raise SystemExit('freeze needs --sport, --market, --inputs and --expected-hash from a reviewed draft')
        inputs = json.loads(args.inputs)
        if (inputs.get('sport'), inputs.get('market')) != (args.sport, args.market):
            raise SystemExit('the inputs are for a different market')
        session = {}
        restore = sync(path, client, folder, session)
        frozen = calibration.freeze_reviewed_plan(path, expected_hash=args.expected_hash.strip(), **inputs)
        backup = sync(path, client, folder, session)  # uploads only the new rows, with read-back
        now = datetime.now(timezone.utc)
        result = {'mode': 'FROZEN_AFTER_REVIEW', 'as_of': now.isoformat(), 'frozen': frozen,
                  'restore': restore, 'backup': backup, 'markets': statuses(path, now)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    text = markdown(result)
    summary = os.getenv('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
