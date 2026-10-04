"""Pure replay of named home Run Line inputs; no fitting or runtime activation.

The legacy adapters and their feature-version/order remain untouched. This new
schema binds the names to the actual replayed indices, including the signed
selected-home line. A feature packet grants no model, storage or wager authority.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import re

from app_core import mlb_production_readiness as legacy
from app_core import mlb_spread_total_model as old_model
from app_core.mlb_history import timestamp
from app_core.mlb_event_matcher import scheduled_eastern_date
from app_core.mlb_team_aliases import MLB_TEAM_ALIASES

VERSION = "mlb-novig-home-half-runline-asof-v1"
NAMES = ("home_ppg", "away_ppg", "home_oppg", "away_oppg",
         "home_win_pct", "away_win_pct", "exact_line", "price_implied_probability")
LEGACY_NAMES = NAMES[:6] + ("reference_line",)


def _clock(value):
    if not isinstance(value, str) or datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("EXPLICIT_UTC_OFFSET_REQUIRED")
    return timestamp(value)


def schema():
    definitions = (
        "home-team mean runs scored in exactly ten retained prior observed finals",
        "away-team mean runs scored in exactly ten retained prior observed finals",
        "home-team mean runs allowed in those same prior finals",
        "away-team mean runs allowed in those same prior finals",
        "home-team wins divided by ten in those same prior finals",
        "away-team wins divided by ten in those same prior finals",
        "signed line applied to the selected home-team run margin; exactly +/-1.5",
        "one divided by original selected decimal price; market-derived, not no-vig",
    )
    return {"feature_version": VERSION,
            "fields": [{"index": i, "name": name, "definition": definition}
                       for i, (name, definition) in enumerate(zip(NAMES, definitions))],
            "scope": {"sport": "MLB", "sportsbook": "Novig", "selection": "spread_home",
                      "signed_lines": [-1.5, 1.5], "target_period": "FULL_GAME"},
            "availability": "observed by captured prediction cutoff, strictly before first pitch",
            "missing_policy": "BLOCK_INPUT; missing period/rules leaves target UNKNOWN",
            "authority": "RESEARCH_INPUT_ONLY_NO_INFERENCE_OR_QUALIFICATION_ASSERTION"}


def _matched_market(payload, raw):
    """Select exact original book/side/line/price before checking its clock."""
    q = payload['quote']
    exact = []
    for book in raw.get('bookmakers', []):
        if legacy.canonical_book_label(book.get('key')) != q['sportsbook']:
            continue
        for market in book.get('markets', []):
            if market.get('key') != 'spreads':
                continue
            for entry in market.get('outcomes', []):
                if (str(entry.get('name', '')).casefold() == str(raw.get('home_team', '')).casefold()
                        and entry.get('point') is not None and not isinstance(entry['point'], bool)
                        and float(entry['point']) == float(q['line'])
                        and legacy.decimal_price(entry.get('price')) == float(q['decimal_odds'])):
                    exact.append((book, market))
    if len(exact) != 1:
        raise ValueError('PRICE_CONFLICT')
    book, market = exact[0]
    updated = market.get('last_update') or book.get('last_update')
    if not updated:
        raise ValueError('QUOTE_UPDATE_CLOCK_MISSING')
    if (_clock(updated) != _clock(q['provider_updated_at'])
            or _clock(q['provider_updated_at']) > _clock(q['observed_at'])):
        raise ValueError('QUOTE_UPDATE_CLOCK_CONFLICT')
    return market


def _team_key(value):
    """Retain full MLB identity; generic result names collapse shared cities."""
    def key(name):
        return re.sub(r'[^a-z0-9]+', ' ', name.casefold()).strip()
    if not isinstance(value, str) or not value.strip():
        return None
    aliases = {key(name): key(full) for name, full in MLB_TEAM_ALIASES.items()}
    name = aliases.get(key(value), key(value))
    name = aliases.get(name, name)  # Existing Oakland -> Athletics aliases.
    return name if name in set(aliases.values()) else None


def _ordered_event_mapping(payload, raw, schedule):
    """Replay capture's ordered alias/date bridge without a live clock or heuristic."""
    pair = tuple(_team_key(raw.get(side + '_team')) for side in ('home', 'away'))
    if not all(pair) or pair[0] == pair[1]:
        raise ValueError('ODDS_SCHEDULE_MATCHUP_MISMATCH')
    day = scheduled_eastern_date({'start': raw['commence_time']})
    candidates = []
    for date in schedule['dates']:
        for game in date['games']:
            if game.get('gameType') != 'R':
                continue
            game_pair = tuple(_team_key(game['teams'][side]['team'].get('name'))
                              for side in ('home', 'away'))
            if not all(game_pair):
                raise ValueError('SCHEDULE_IDENTITY_INCOMPLETE')
            _clock(game['gameDate'])
            if game_pair == pair and scheduled_eastern_date({'start': game['gameDate']}) == day:
                candidates.append(game)
    if len(candidates) > 1:
        # Do not pick the nearest start, trust a claim, or ignore a completed game.
        raise ValueError('DOUBLEHEADER_IDENTITY_AMBIGUOUS')
    if len(candidates) != 1 or str(candidates[0]['gamePk']) != str(payload['provider_event_id']):
        raise ValueError('ODDS_SCHEDULE_MATCHUP_MISMATCH')
    claims = raw.get('provider_ids') or {}
    if not isinstance(claims, dict):
        raise ValueError('PROVIDER_ID_CONFLICT')
    if claims.get('mlb') is not None and str(claims['mlb']) != str(payload['provider_event_id']):
        raise ValueError('PROVIDER_ID_CONFLICT')


def _target_binding(payload, observations, market):
    """Bind only captured declarations; never infer full-game/book rules."""
    q = payload["quote"]
    periods = [market.get('period')]
    declared = q.get("market_period")
    if declared is not None and declared != "FULL_GAME":
        raise ValueError("UNSUPPORTED_MARKET_PERIOD")
    if any(x is not None and x != "FULL_GAME" for x in periods):
        raise ValueError("MARKET_PERIOD_CONFLICT")
    rule_id = q.get("settlement_rules_hash")
    ref = payload["source_observations"].get("settlement_rules")
    if rule_id is not None or ref is not None:
        if not rule_id or rule_id != ref or rule_id not in observations:
            raise ValueError("SETTLEMENT_RULE_LINEAGE_UNVERIFIED")
        rule = observations[rule_id]
        facts = rule.get("payload", {})
        if (old_model.digest(rule) != rule_id or rule.get("source") != "novig_rules_document"
                or facts.get("sportsbook") != "Novig" or facts.get("market_period") != "FULL_GAME"
                or not isinstance(facts.get("document_utf8"), str)
                or hashlib.sha256(facts["document_utf8"].encode()).hexdigest() != facts.get("document_sha256")
                or not facts.get("rule_version") or not facts.get("settlement_rules")
                or _clock(rule["observed_at"]) > _clock(payload["prediction_cutoff"])):
            raise ValueError("SETTLEMENT_RULE_LINEAGE_UNVERIFIED")
    complete = declared == "FULL_GAME" and periods == ["FULL_GAME"] and bool(rule_id)
    return {"status": "CAPTURED_TARGET_BOUND" if complete else "UNKNOWN",
            "market_period": declared, "source_market_period": periods[0] if len(periods) == 1 else None,
            "settlement_rules_observation_hash": rule_id,
            "source_rights_approved": False, "accepted_production_reader": False}


def replay_features(snapshot, observations):
    """Replay original source bytes without restoring a database or executing inference."""
    if tuple(old_model.FEATURE_COLUMNS) != LEGACY_NAMES:
        raise ValueError("LEGACY_FEATURE_ORDER_CHANGED")
    payload, values = legacy.exact_feature_values(snapshot)
    q = payload["quote"]
    for key in ("captured_at", "prediction_cutoff", "game_start_utc"):
        _clock(payload[key])
    for key in ("observed_at", "source_game_start_utc", "provider_updated_at"):
        _clock(q[key])
    if (q["market_type"] != "spread_home" or q.get("sportsbook") != "Novig"
            or isinstance(q["line"], bool) or float(q["line"]) not in (-1.5, 1.5)):
        raise ValueError("UNSUPPORTED_EXACT_HOME_SCOPE")
    if q.get("provider_namespace") != "odds_api" or not q.get("provider_event_id"):
        raise ValueError("QUOTE_PROVIDER_IDENTITY_MISSING")
    raw_quote = observations.get(payload["source_observations"]["quotes"], {}).get("payload", {})
    market = _matched_market(payload, raw_quote)
    for problem in (legacy._quote_source_check(payload, observations),
                    legacy._prior_check(payload, observations)):
        if problem:
            raise ValueError(problem)
    problem, number = legacy._schedule_check(payload, observations)
    if problem:
        raise ValueError(problem)
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("GAME_NUMBER_MISSING")
    _ordered_event_mapping(payload, raw_quote, observations[payload['source_observations']['schedule']]['payload'])
    for side in ("home", "away"):
        team = payload[side + "_team_id"]
        games = [g for g in payload["prior_games"] if team in (g["home_id"], g["away_id"])]
        if len(games) != 10:
            raise ValueError("EXACT_TEN_PRIOR_GAMES_REQUIRED")
    if timestamp(q["observed_at"]) >= min(timestamp(payload["game_start_utc"]),
                                          timestamp(q["source_game_start_utc"])):
        raise ValueError("QUOTE_NOT_PREGAME")
    if abs((timestamp(payload["game_start_utc"]) - timestamp(q["source_game_start_utc"])).total_seconds()) > 600:
        raise ValueError("EVENT_START_CONFLICT")
    dependencies = set(payload["source_observations"].values()) | {
        g["observation_hash"] for g in payload["prior_games"]}
    if any(k not in observations or old_model.digest(observations[k]) != k for k in dependencies):
        raise ValueError("SOURCE_LINEAGE_UNVERIFIED")
    if any(_clock(observations[k]["observed_at"]) > _clock(payload["prediction_cutoff"])
           for k in dependencies):
        raise ValueError("SOURCE_OBSERVED_AFTER_CUTOFF")
    identity = {k: payload[k] for k in ("provider_namespace", "provider_event_id", "home_team_id",
                "away_team_id", "season", "game_start_utc", "prediction_cutoff", "captured_at")}
    identity.update(game_number=number, quote=deepcopy(q))
    packet = {"feature_version": VERSION, "schema_hash": old_model.digest(schema()),
              "ordered_names": list(NAMES), "ordered_values": list(values),
              "named_values": dict(zip(NAMES, values)), "identity": identity,
              "receipt_sha256": snapshot["sha256"], "source_hashes": sorted(dependencies),
              "target_binding": _target_binding(payload, observations, market),
              "status": "FEATURE_INPUTS_VERIFIED", "inference_status": "NOT_EXECUTED",
              "production_eligible": False, "wager_approved": False, "recommended_stake": 0}
    return {"payload": packet, "sha256": old_model.digest(packet)}


def verify_packet(packet, snapshot, observations, *, require_target_binding=False):
    """Pure consumer replay. No legacy relabeling or unchecked named/vector inputs."""
    expected = replay_features(snapshot, observations)
    if packet != expected:
        raise ValueError("FEATURE_PACKET_CONTRACT_MISMATCH")
    if require_target_binding and expected["payload"]["target_binding"]["status"] != "CAPTURED_TARGET_BOUND":
        raise ValueError("TARGET_BINDING_UNKNOWN")
    return deepcopy(expected)


def verify_probability_vector(vector):
    """Validate a declared half-line target vector without asserting inference success."""
    legacy.validate_vector(vector, "MLB/RUN_LINE")
    if vector["PUSH"] != 0:
        raise ValueError("HALF_LINE_PUSH_CONFLICT")
    return deepcopy(vector)
