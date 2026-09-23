"""Pure wager-decision contract. No network, order placement, or implicit promotion.

A separately validated policy and game-specific conservative probability are
required. The existing app can inspect this contract before activating a policy.
"""
from datetime import datetime, timezone
import math
from collections import defaultdict
from core.market_policy import production_market, MARKET_POLICY_VERSION
from core.sport_policy import SportPolicy, DEPLOYMENT_STATES
from core.price_value import price_value


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def aware(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def decimal_price(value):
    odds = finite(value)
    return None if odds is None or abs(odds) < 100 else 1 + (odds / 100 if odds > 0 else 100 / -odds)


def moneyline_context(home_odds, away_odds):
    home, away = decimal_price(home_odds), decimal_price(away_odds)
    if home is None or away is None:
        return {"available": False, "home_probability": None, "away_probability": None, "wager_eligible": False}
    hp, ap = 1 / home, 1 / away
    return {"available": True, "home_probability": hp / (hp + ap), "away_probability": ap / (hp + ap), "wager_eligible": False}


def candidate_decision(row, policy: SportPolicy, now, *, outage_policy=None):
    """Only reduce eligibility. Trusted adapter must supply evidence identifiers.

    Probability and policy validation are independent. Bucket rate alone is not
    a candidate probability. All amounts are fractions of the supplied bankroll.
    """
    reasons = []
    market = str(row.get("market_type", "")).casefold()
    p = finite(row.get("conservative_probability"))
    mean = finite(row.get("mean_probability"))
    decimal = decimal_price(row.get("odds_american"))
    start, quote = aware(row.get("start")), aware(row.get("quote_time"))
    freeze = aware(row.get("evidence_frozen_at"))
    if now.tzinfo is None:
        raise ValueError("Decision time requires a timezone")
    if row.get("sport") != policy.sport:
        reasons.append("sport_policy_mismatch")
    if not production_market(market):
        reasons.append("unsupported_production_market")
    if not row.get("game_id") or row.get("identity_verified") is not True:
        reasons.append("unverified_mapping")
    from app_core.public_quote_policy import supported_quote
    if not supported_quote({"sport":row.get("sport", ""), "quote_source":row.get("book")}) or decimal is None or row.get("exact_quote_verified") is not True:
        reasons.append("invalid_exact_price")
    if finite(row.get("line")) is None:
        reasons.append("missing_exact_line")
    if row.get("alternate") and row.get("alternate_quote_verified") is not True:
        reasons.append("unverified_alternate_quote")
    if start is None or start <= now:
        reasons.append("started_or_missing_start")
    if quote is None or not 0 <= (now - quote).total_seconds() <= 1800:
        reasons.append("stale_or_missing_quote")
    if not policy.validation_id:
        reasons.append("unvalidated_sport_policy")
    if row.get("model_validated") is not True or not row.get("model_version"):
        reasons.append("unvalidated_model")
    if row.get("calibration_validated") is not True or not row.get("calibration_version"):
        reasons.append("unvalidated_calibration")
    if freeze is None or freeze > now or not row.get("evidence_snapshot_id"):
        reasons.append("missing_or_future_evidence")
    if row.get("critical_feature_error") is not False:
        reasons.append("critical_features_unverified")
    valid_p = p is not None and mean is not None and 0 < p <= mean < 1
    if not valid_p:
        reasons.append("missing_or_invalid_conservative_probability")
    line = finite(row.get("line"))
    push = finite(row.get("push_probability"))
    if push is None and line is not None and abs(line % 1) == 0.5:
        push = 0.0
    valid_push = push is not None and 0 <= push < 1 and (mean is None or mean + push <= 1)
    if not valid_push:
        reasons.append("missing_or_invalid_push_probability")
    pricing = price_value(p, push, decimal, minimum_edge=policy.min_conservative_edge) if valid_p and valid_push and decimal is not None else None
    ev = pricing['expected_value'] if pricing else None
    break_even = pricing['break_even'] if pricing else None
    edge = pricing['edge'] if pricing else None
    if ev is None or ev <= 0:
        reasons.append("nonpositive_conservative_ev")
    if edge is None or edge < policy.min_conservative_edge:
        reasons.append("insufficient_conservative_edge")
    tier = row.get("maturity", "RESEARCH")
    required_state = {"PROVISIONAL": 1, "STANDARD": 2, "PREMIUM": 3}.get(tier, 99)
    if DEPLOYMENT_STATES.get(policy.deployment_state, 0) < required_state:
        reasons.append("deployment_state_below_maturity")
    effective_n = finite(row.get("evidence_effective_sample_size"))
    if effective_n is None or effective_n <= 0:
        reasons.append("missing_effective_evidence")
    elif effective_n < (policy.provisional_minimum_evidence if tier == "PROVISIONAL" else policy.minimum_evidence):
        reasons.append("insufficient_evidence_for_maturity")
    caps = {"PROVISIONAL": policy.provisional_stake_cap if policy.provisional_allowed else 0,
            "STANDARD": policy.standard_stake_cap, "PREMIUM": policy.premium_stake_cap}
    if policy.kelly_fraction <= 0 or policy.sport_exposure_cap <= 0:
        reasons.append("zero_validated_allocation")
    if tier not in caps or caps[tier] <= 0:
        reasons.append("unvalidated_maturity_or_cap")
    review = row.get("gemini_status", "UNAVAILABLE")
    reduction = finite(row.get("gemini_stake_multiplier", 1))
    outage = review in {"UNAVAILABLE", "TIMEOUT", "SERVICE_ERROR"}
    outage_cap = None
    config = outage_policy or {}
    if outage and config.get("mode") == "capped":
        outage_cap = finite(config.get("cap"))
        reduction = finite(config.get("multiplier"))
        if outage_cap is None or not 0 < outage_cap <= .01 or reduction is None or not 0 < reduction < 1:
            reasons.append("gemini_hold")
    elif review not in {"CONFIRM", "APPROVE", "REDUCE"} or reduction is None or not 0 < reduction <= 1:
        reasons.append("gemini_hold")
    if row.get("unresolved_material_news"):
        reasons.append("wait_for_material_news")
    stake = 0.0
    kelly = 0.0
    if not reasons:
        kelly = pricing['full_kelly'] * policy.kelly_fraction
        stake = min(kelly, caps[tier], policy.sport_exposure_cap) * reduction
        if outage_cap is not None:
            stake = min(stake, outage_cap)
    action = "PASS" if reasons else "REDUCE" if review == "REDUCE" else "BET ALT LINE" if row.get("alternate") else "BET NOW"
    if reasons == ["wait_for_material_news"]:
        action = "WAIT"
    return dict(row, deployment_state=policy.deployment_state, wager_contract_version="live-v1", production_eligible=not reasons and stake > 0,
                production_gate_reason="; ".join(reasons), raw_kelly=kelly / policy.kelly_fraction if policy.kelly_fraction else 0,
                gemini_review_status=review, gemini_stake_multiplier=reduction,
                gemini_outage_capped=outage and not reasons, market_policy_version=MARKET_POLICY_VERSION, sport_policy_version=policy.version,
                conservative_ev=ev, conservative_edge=edge, break_even_probability=break_even,
                strategic_action=action, reason_for_pass=reasons, recommended_fraction=stake,
                minimum_decimal_price=(pricing['minimum_decimal_price'] if pricing else None), push_probability=push)


def select_matchups(rows, policies, now):
    """One row per (sport, game ID), including ML-only/no-edge matchups."""
    grouped = defaultdict(list)
    for row in rows:
        if not row.get("game_id") or not row.get("sport"):
            raise ValueError("Matchup selection requires a sport and stable game ID")
        grouped[(row.get("sport"), row.get("game_id"))].append(row)
    decisions = []
    for (sport, game), pool in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        policy = policies.get(sport)
        production = [row for row in pool if production_market(row.get("market_type"))]
        scored = [candidate_decision(row, policy, now) for row in production] if policy else []
        qualified = [row for row in scored if row["recommended_fraction"] > 0]
        def order(row):
            return (-(finite(row.get("conservative_ev")) or 0), -(finite(row.get("conservative_probability")) or 0), str(row.get("selection", "")), str(row.get("book", "")))
        lean = sorted(production, key=lambda row: (-(finite(row.get("mean_probability")) or 0), str(row.get("selection", ""))))
        if qualified:
            selected = sorted(qualified, key=order)[0]
        else:
            selected = {"sport": sport, "game_id": game, "selection": "PASS — NO QUALIFYING EDGE", "recommended_fraction": 0.0,
                        "strategic_action": "PASS", "reason_for_pass": sorted({reason for row in scored for reason in row["reason_for_pass"]}) or ["no_valid_production_market_or_policy"]}
        selected = dict(selected, best_research_lean=lean[0].get("selection") if lean else None,
                        markets_evaluated=sorted({row["market_type"] for row in production}), candidate_decisions=scored)
        decisions.append(selected)
    return decisions


def allocate_exposure(decisions, bankroll, *, total_cap, game_cap, sport_caps, committed=None, team_cap=None, daily_cap=None, weekly_cap=None):
    """Deterministic downward-only allocation. Include existing straight/parlay exposure.

    Committed keys: 'total', 'game:<sport>:<id>', 'sport:<sport>',
    'team:<sport>:<id>'. Fractions of current bankroll; a parlay's full
    risk must be counted against EACH underlying game and team in this input.
    """
    bankroll, total_cap, game_cap = finite(bankroll), finite(total_cap), finite(game_cap)
    if bankroll is None or bankroll <= 0 or total_cap is None or game_cap is None or not 0 <= total_cap <= 1 or not 0 <= game_cap <= 1:
        raise ValueError("Invalid bankroll or exposure caps")
    team_cap = game_cap if team_cap is None else finite(team_cap)
    if team_cap is None or not 0 <= team_cap <= 1:
        raise ValueError("Invalid team cap")
    period_caps = {}
    for key, value in (("daily", daily_cap), ("weekly", weekly_cap)):
        if value is not None:
            value = finite(value)
            if value is None or not 0 <= value <= 1:
                raise ValueError("Invalid period cap")
            period_caps[key] = value
    used = dict(committed or {})
    if any(finite(v) is None or finite(v) < 0 for v in used.values()):
        raise ValueError("Invalid committed exposure")
    used = {key: finite(value) for key, value in used.items()}
    result = []
    for row in sorted(decisions, key=lambda x: (-(finite(x.get("conservative_ev")) or 0), str(x.get("game_id")))):
        sport, game = row.get("sport"), row.get("game_id")
        cap = finite(sport_caps.get(sport))
        if cap is None or not 0 <= cap <= 1:
            cap = 0.0
        keys = {"total": total_cap, f"sport:{sport}": cap, f"game:{sport}:{game}": game_cap}
        keys.update(period_caps)
        teams = row.get("team_ids")
        valid_teams = isinstance(teams, (list, tuple)) and len(teams) == 2 and all(isinstance(t, str) and t.strip() for t in teams) and len(set(teams)) == 2
        if valid_teams:
            keys.update({f"team:{sport}:{team}": team_cap for team in teams})
        requested = max(0.0, finite(row.get("recommended_fraction")) or 0.0)
        if not valid_teams:
            requested = 0.0
            row = dict(row, production_gate_reason=(str(row.get('production_gate_reason') or '')
                       + '; missing_stable_team_ids').strip('; '))
        if not production_market(row.get("market_type")) or row.get("strategic_action") not in {"BET NOW", "BET ALT LINE", "REDUCE"} or row.get("reason_for_pass"):
            requested = 0.0
        stake = max(0.0, min([requested] + [limit - used.get(key, 0) for key, limit in keys.items()]))
        for key in keys:
            used[key] = used.get(key, 0) + stake
        result.append(dict(row, recommended_fraction=stake, recommended_stake=bankroll * stake))
    return result
