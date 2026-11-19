"""Verified immutable closing observations in the existing evidence database."""
import json
from contextlib import closing
from core.exposure_ledger import digest
from core.wager_decisions import aware,finite,decimal_price
from core.clv import line_clv,price_clv
from app_core.public_quote_policy import canonical_book_label

def record_close(database,candidate,quote,*,captured_at):
    # Normalize the closing record, never the caller's raw quote or saved candidate.
    quote = dict(quote, sportsbook=canonical_book_label(quote.get('sportsbook')))
    start=aware(candidate.get('game_start_utc'));at=aware(quote.get('quote_recorded_at'));capture=aware(captured_at)
    if None in (start,at,capture) or not at<=capture<start or not 0<(start-at).total_seconds()<=1800 or (capture-at).total_seconds()>1800: raise ValueError('Not a verified pregame closing observation')
    if candidate.get('provider_namespace') not in {'odds_api','espn','mlb'} or candidate.get('provider_namespace')!=quote.get('provider_namespace'): raise ValueError('Closing provider namespace missing or different')
    for key in ('game_id','sport','market_type','sportsbook','provider_event_id'):
        source=canonical_book_label(candidate.get('quote_bookmaker')) if key=='sportsbook' else candidate.get(key)
        if not source or source!=quote.get(key):raise ValueError('Closing identity/book mismatch')
    # Exact direction is market_type, selected-team identity is immutable game ID + direction.
    if decimal_price(quote.get('price')) is None or finite(quote.get('line')) is None:raise ValueError('Invalid close')
    opening=finite(candidate.get('market_line_used'))
    if opening is None:  # the bound quote's own line: capture_run refuses a binding whose two lines differ
        opening=finite(candidate.get('total_line' if str(candidate.get('market_type')).startswith('total') else 'spread_line'))
    value={'snapshot_id':candidate['snapshot_id'],'candidate_id':candidate['candidate_id'],'quote':quote,'closing_capture_at':capture.isoformat(),
        'line_clv':line_clv(candidate['market_type'],opening,quote['line']),
        'price_clv':price_clv(candidate.get('odds_american'),quote['price']) if opening==quote['line'] else None,'quote_verified':True}
    value['beat_close']=value['line_clv']>0 if value['line_clv'] not in (None,0) else value['price_clv']>0 if value['price_clv'] is not None else None
    key=digest(value)
    from app_core.prediction_evidence import connect
    with closing(connect(database)) as db,db:db.execute('INSERT OR IGNORE INTO closing_observations VALUES (?,?,?,?)',(key,value['snapshot_id'],value['candidate_id'],json.dumps(value,sort_keys=True)))
    return key

def observations(database):
    from app_core.prediction_evidence import connect
    with closing(connect(database)) as db: rows=db.execute('SELECT observation_id,payload FROM closing_observations').fetchall()
    out=[]
    for key,raw in rows:
        value=json.loads(raw)
        if digest(value)!=key:raise ValueError('Closing payload changed')
        out.append(value)
    return out


def capture_live(database, *, fetch=None, now=None, unmatched=None, within_seconds=1800):
    """Explicit job; never called by rendering. Missing provider identity blocks.

    A close is recorded for each game's saved best pick, from the newest saved run that holds one:
    the pick the paper track record counts (``core.research_performance.accepted_rows``). TheOver's
    model pick is read as well where it feeds the blend, whether or not it is the game's best pick:
    it is the bet of the plan the owner approved on 2026-11-19
    (``docs/roadmap/THEOVER-COLLEGE-TOTALS-PLAN-2026-11-19.md``). Every candidate of every run would
    be a few hundred new evidence files a night for nothing that reads them. A saved run that does
    not mark its best picks is taken whole, as before.

    ``unmatched``: an optional dict filled with why each unmatched pick found no single live quote
    (``_why_unmatched``) and why a found quote was refused (``_why_refused``); diagnostic only, the
    result is unchanged. ``within_seconds``: how near its start a game must be; ``record_close``
    accepts 30 minutes at most.
    """
    from datetime import datetime, timezone
    from app_core import theover_model_totals as theover
    from app_core.prediction_evidence import load_snapshots
    current = now or datetime.now(timezone.utc)
    window = min(float(within_seconds), 1800.0)
    candidates, picks = [], {}
    for _, audit, _ in load_snapshots(database):  # oldest first
        marked = 'selected_as_best_pick' in audit
        model_pick = (theover.backed(audit) & theover.in_blend(audit)).to_numpy() if marked else ()
        chosen = {}
        for position, r in enumerate(audit.astype(object).where(audit.notna(), None).to_dict('records')):
            start = aware(r.get('game_start_utc'))
            if not start or not 0 < (start-current).total_seconds() <= window:
                continue
            if not marked:
                candidates.append(r)
            elif str(r.get('selected_as_best_pick')).lower() in {'true', '1'} or model_pick[position]:
                chosen.setdefault((r.get('sport', r.get('league')), r.get('matchup_id')), []).append(r)
        picks.update(chosen)  # a later run's picks for a game replace an earlier run's
    candidates += [r for rows in picks.values() for r in rows]
    if not candidates:
        return {'verified':0, 'unavailable':0, 'reason':'no_candidates_in_closing_window'}
    if fetch is None:
        from core.streamlit_pipeline import fetch_live_odds_dataframe
        fetch = fetch_live_odds_dataframe
    live = fetch(sorted({r.get('sport',r.get('league')) for r in candidates}))
    # The capture time is read after the answer. Loading the saved runs and the request take seconds, and
    # a quote the book updated meanwhile would otherwise read as recorded after its own capture.
    captured = now or datetime.now(timezone.utc)
    saved = 0; blocked = {}; detail = {} if unmatched is None else unmatched
    rows = live.to_dict('records')
    quotes = [json.loads(row['provider_quotes']) if isinstance(row.get('provider_quotes'), str) and row['provider_quotes'] else []
              for row in rows]
    for r in candidates:
        game = [i for i, row in enumerate(rows) if _same_game(row, r)]
        matching = [q for i in game for q in quotes[i]
                    if (q.get('provider_namespace') and q.get('provider_namespace') == r.get('provider_namespace')
                        and q.get('provider_event_id') == r.get('provider_event_id')
                        and canonical_book_label(q.get('book')) == canonical_book_label(r.get('quote_bookmaker')) and q.get('market_type') == r.get('market_type'))]
        if len(matching) != 1:
            blocked['missing_or_ambiguous_exact_quote'] = blocked.get('missing_or_ambiguous_exact_quote',0)+1
            why = _why_unmatched(r, [dict(rows[i], matchup_id=r.get('matchup_id')) for i in game], len(matching))
            detail[why] = detail.get(why, 0) + 1
            continue
        q = matching[0]
        quote = dict(game_id=r['game_id'], sport=r['sport'], market_type=r['market_type'],
                     sportsbook=canonical_book_label(q.get('book')), provider_event_id=q['provider_event_id'],
                     provider_namespace=q['provider_namespace'], quote_recorded_at=q.get('recorded_at'),
                     line=q.get('point'), price=q.get('price'))
        try:
            record_close(database,r,quote,captured_at=captured.isoformat()); saved += 1
        except ValueError as exc:
            blocked['invalid_or_stale_close'] = blocked.get('invalid_or_stale_close',0)+1
            why = _why_refused(r, quote, captured, exc)
            detail[why] = detail.get(why, 0) + 1
    return {'verified':saved, 'unavailable':sum(blocked.values()), 'reasons':blocked}


def _same_game(row, pick):
    """Whether a live row is the saved pick's game: the same game key, or the provider's own event.

    The live frame keys a game by its two provider team names; a saved pick is keyed by date and
    normalised names, or by ESPN's event. The two keys of one game never agree, so until 2026-11-19
    every pick read "game_not_in_live_odds" and no close was ever saved. A live row also carries the
    provider's event id as its ``game_id``, the id a saved pick's bound quote was read from.
    """
    if row.get('matchup_id') == pick.get('matchup_id'):
        return True
    feed = row.get('odds_feed_source') or 'the_odds_api'
    return (bool(pick.get('provider_event_id')) and row.get('game_id') == pick.get('provider_event_id')
            and ('odds_api' if feed == 'the_odds_api' else feed) == pick.get('provider_namespace'))


def _why_refused(pick, quote, captured, error):
    """Which rule of ``record_close`` a found quote failed (diagnostic only)."""
    start, at = aware(pick.get('game_start_utc')), aware(quote.get('quote_recorded_at'))
    if str(error) != 'Not a verified pregame closing observation':
        return 'refused_' + str(error).lower().replace(' ', '_').replace('/', '_')
    if at is None:
        return 'refused_quote_has_no_time'
    if at > captured:
        return 'refused_quote_recorded_after_capture'
    if start is None or captured >= start:
        return 'refused_captured_at_or_after_start'
    return 'refused_quote_older_than_30_minutes_before_start'


def _why_unmatched(pick, rows, matches):
    """The first check a saved pick fails against the live odds (diagnostic only)."""
    if matches > 1:
        return 'several_quotes_same_book_and_market'
    if not pick.get('provider_namespace') or not pick.get('provider_event_id'):
        return 'saved_pick_has_no_provider_identity'
    same_game = [row for row in rows if row.get('matchup_id') == pick.get('matchup_id')]
    if not same_game:
        return 'game_not_in_live_odds'
    quotes = [q for row in same_game for q in json.loads(row.get('provider_quotes') or '[]')]
    if not quotes:
        return 'live_game_has_no_provider_quotes'
    same_event = [q for q in quotes if q.get('provider_namespace') == pick.get('provider_namespace')
                  and q.get('provider_event_id') == pick.get('provider_event_id')]
    if not same_event:
        return 'live_quotes_have_another_provider_event'
    book = canonical_book_label(pick.get('quote_bookmaker'))
    same_book = [q for q in same_event if canonical_book_label(q.get('book')) == book]
    if not same_book:
        return 'book_not_quoting_this_game'
    return 'book_not_quoting_this_market'

