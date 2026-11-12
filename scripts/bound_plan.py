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
from app_core import market_skill_validation  # noqa: E402
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
          model='research', min_edge=None, entry_rule=None, min_bets=None, pool=False, pool_sports=None,
          entry_books=None, entry_within_minutes=None):
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
            skill = methodology in ('market_skill', 'market_skill_v2')
            v2 = methodology == 'market_skill_v2'
            if entry_books and not (skill and model == 'pinnacle'):
                raise ValueError('entry books apply only to a market-skill plan with --model pinnacle')
            if entry_within_minutes and not (v2 and model == 'pinnacle' and entry_rule):
                raise ValueError('a late entry window needs --methodology market_skill_v2, --model pinnacle '
                                 'and --entry-rule first_edge')
            if pool and not (v2 and model == 'pinnacle'):
                raise ValueError('pooling needs --methodology market_skill_v2 and --model pinnacle (owner D27)')
            spec = calibration.draft_bound_plan(path, sport, market, validation_start=start,
                                                methodology=methodology, model_id=model_id,
                                                min_edge=min_edge if skill else None,
                                                # A Pinnacle plan enters only where its picks can be staked.
                                                # Owner 2026-11-11: optionally only some of them (Novig, where
                                                # the stake is placed and nearly every replay bet was).
                                                entry_books=((entry_books or pinnacle_model.PLAN_BOOKS)
                                                             if model == 'pinnacle' and skill else None),
                                                # The staking terms a Pinnacle plan needs to ever stake.
                                                staking_terms=('pinnacle-provisional-v1' if model == 'pinnacle'
                                                               and skill else None),
                                                entry_rule=entry_rule if skill else None,
                                                # Owner 2026-11-09: new plans count a push as a zero-return bet
                                                # (market_skill_v2 always does).
                                                push_rule=(market_skill_validation.PUSH_ZERO_RETURN
                                                           if methodology == 'market_skill' else None),
                                                # Owner D26, D27, D29: bets per window, the NBA and NCAAB pool,
                                                # and no entry at a junk price.
                                                min_bets=min_bets if v2 else None,
                                                pool=(market_skill_validation.pool_scopes(sport, market, pool_sports)
                                                      if pool else None),
                                                max_entry_overround=(pinnacle_model.MAX_OVERROUND
                                                                     if v2 and model == 'pinnacle' else None),
                                                # Owner 2026-11-11: enter only this close to the start.
                                                entry_within_minutes=entry_within_minutes,
                                                # Size the windows by the rate seen so far, not up to a later start.
                                                as_of=now)
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
                artifact = None
                drafted = result['draft'].get('plan')
                if drafted:
                    # The plan bets with its calibrated probability: replay that too.
                    from app_core import prospective_evidence as evidence
                    found = evidence.load_record(path, 'prospective_calibration', drafted['calibration_id'])
                    artifact = json.loads(found['payload']).get('calibration_artifact') if found else None
                result['replay_estimate'] = pinnacle_model.replay_estimate(path, sport, market, now=now,
                                                                           books=(entry_books
                                                                                  or pinnacle_model.PLAN_BOOKS),
                                                                           calibration=artifact,
                                                                           entry_within_minutes=entry_within_minutes)
            except Exception as exc:  # an estimate must never block the draft
                result['replay_estimate'] = {'status': 'ERROR', 'error': type(exc).__name__}
    return result


def entry_books(text):
    """The stake books named in ``text`` (comma-separated labels), or ``None`` for all of them.
    A name that is not a stake book is refused, so a typo cannot freeze a plan that never bets."""
    from app_core import pinnacle_model
    wanted = {name.strip().lower() for name in (text or '').split(',') if name.strip()}
    if not wanted:
        return None
    labels = {label.lower(): label for label in pinnacle_model.PLAN_BOOKS}
    unknown = sorted(wanted - set(labels))
    if unknown:
        raise SystemExit(f"unknown stake book(s): {', '.join(unknown)}; choose from {', '.join(sorted(labels.values()))}")
    return frozenset(labels[name] for name in wanted)


ENTRY_RULE_TEXT = {
    'EARLIEST_SHARED_LINE': 'the earliest capture where one of its books posts Pinnacle\'s line',
    'FIRST_EDGE_CAPTURE': 'the first capture where one of its books posts Pinnacle\'s line and clears the bet '
                          'threshold (else the earliest such capture); bets on the first prediction that clears it',
}


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
                      *([f"- gates (market-skill-v2): mean bet CLV at least "
                         f"{plan['market_skill_methodology']['min_clv_t']:g} standard errors above zero, a close for "
                         f"{plan['market_skill_methodology']['min_bet_close_coverage']:.0%} of bets, bet ROI at least "
                         f"{plan['market_skill_methodology']['min_bet_roi']:.0%}; Brier and log loss against the "
                         "market are reported only"]
                        if plan.get('market_skill_methodology', {}).get('version') == 'market-skill-v2' else []),
                      *([f"- pool: bets counted across {', '.join(plan['market_skill_methodology']['pool']['scopes'])}; "
                         f"each needs {plan['market_skill_methodology']['pool']['min_scope_bets']} bets of its own. "
                         "Freeze the same draft for every scope, with the same validation start"]
                        if 'pool' in plan.get('market_skill_methodology', {}) else []),
                      *([f"- staking terms: {plan['deployment_criteria']['staking_terms']} (tenth Kelly, "
                         f"{plan['deployment_criteria']['validated_policy']['provisional_stake_cap']:.1%} of bankroll per pick, "
                         f"{plan['deployment_criteria']['validated_policy']['sport_exposure_cap']:.0%} open at once)"]
                        if 'staking_terms' in plan.get('deployment_criteria', {}) else []),
                      *([f"- entry rule: {ENTRY_RULE_TEXT[plan['market_skill_methodology'].get('entry_rule', 'EARLIEST_SHARED_LINE')]}"]
                        if 'entry_books' in plan.get('market_skill_methodology', {}) else []),
                      *([f"- entry books: {', '.join(plan['market_skill_methodology']['entry_books'])} "
                         "(the books its picks can be staked at)"]
                        if 'entry_books' in plan.get('market_skill_methodology', {}) else []),
                      *([f"- entry window: only captures taken within "
                         f"{plan['market_skill_methodology']['entry_within_minutes']} minutes of the start "
                         "(an earlier price is never predicted or bet)"]
                        if 'entry_within_minutes' in plan.get('market_skill_methodology', {}) else []),
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
            if replay.get('entry_within_minutes'):
                lines += ['', f"Only captures within {replay['entry_within_minutes']} minutes of each start are "
                              'replayed (the drafted entry window).']
            lines += ['', f"Over {replay['days_observed']} days of captured games (results not read): "
                          f"{replay['plan_games']} of {replay['games_started']} games had a US book at Pinnacle's line "
                          f"({replay['plan_share']:.0%}), {replay['plan_games_per_day']} a day; "
                          f"{replay['min_sample']} games take about {replay['days_for_min_sample']} days.", '',
                      '| bet edge vs Pinnacle | bets | bets/day | days for '
                      f"{replay['min_bets']} bets | days per window |", '|---|---|---|---|---|']
            for edge, row in replay['by_edge'].items():
                lines.append(f"| {edge} | {row['bets']} | {row['bets_per_day']} | {row['days_for_min_bets']} | "
                             f"{row['days_per_window']} |")
            if replay.get('variants'):
                cal = replay.get('calibration')
                lines += ['', '#### By entry rule and probability', '',
                          *([f"Calibration (Platt on {cal.get('decided')} games): a = {cal.get('a')}, "
                             f"b = {cal.get('b')}.", ''] if cal else []),
                          '| entry rule | probability | edge | bets | bets/day | days for '
                          f"{replay['min_bets']} bets | days per window |", '|---|---|---|---|---|---|---|']
                for variant in replay['variants']:
                    for edge, row in variant['by_edge'].items():
                        lines.append(f"| {variant['entry_rule']} | {variant['probability']} | {edge} | {row['bets']} | "
                                     f"{row['bets_per_day']} | {row['days_for_min_bets']} | {row['days_per_window']} |")
                lines += ['', 'CALIBRATED is the probability the plan bets with. FIRST_EDGE_CAPTURE enters at the '
                              'first capture that clears the edge (a draft with entry rule first_edge).']
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
    parser.add_argument('--methodology', choices=('market_skill', 'market_skill_v2', 'v1'), default='market_skill',
                        help='market_skill: scored against the de-vigged entry price (D19 option B); '
                             'market_skill_v2: CLV-led gates (D26); v1: absolute thresholds')
    parser.add_argument('--min-bets', type=int, choices=(50, 60, 75, 100), default=75,
                        help='draft, market_skill_v2: bets needed in each window (locks at the freeze)')
    parser.add_argument('--pool', action='store_true',
                        help='draft, market_skill_v2 with the Pinnacle model: count bets across sports (D27)')
    parser.add_argument('--pool-sports', default='NBA,NCAAB',
                        help='draft with --pool: the sports pooled, e.g. NBA,NCAAB,NHL (every one must freeze the same plan)')
    parser.add_argument('--model', choices=('research', 'pinnacle'), default='research',
                        help="research: the pinned research score model; pinnacle: Pinnacle's no-vig price")
    parser.add_argument('--min-edge', type=float, choices=(0.01, 0.02), default=0.02,
                        help='market_skill only: the minimum EV for a bet (locks when the plan is frozen)')
    parser.add_argument('--entry-rule', choices=('earliest', 'first_edge'), default='earliest',
                        help='draft, market_skill: enter at the earliest shared-line capture, or at the first '
                             'capture that clears the bet threshold (FIRST_EDGE_CAPTURE)')
    parser.add_argument('--entry-within-minutes', type=int, choices=(60,),
                        help='draft, market_skill_v2 with first_edge: enter only at captures taken this close '
                             'to the start (locks at the freeze); blank is any time before the start')
    parser.add_argument('--entry-books', default='',
                        help='draft, market_skill with the Pinnacle model: comma-separated stake books the plan '
                             'may enter at, e.g. Novig (blank: Novig, DraftKings, FanDuel and BetMGM)')
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
                       methodology=args.methodology, model=args.model, min_edge=args.min_edge,
                       entry_rule=('FIRST_EDGE_CAPTURE' if args.entry_rule == 'first_edge' else None),
                       min_bets=args.min_bets, pool=args.pool, entry_books=entry_books(args.entry_books),
                       entry_within_minutes=args.entry_within_minutes,
                       pool_sports=[s.strip().upper() for s in args.pool_sports.split(',') if s.strip()])
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
