"""Private mapping validator with the same deterministic identity checks.

The frozen independent reader is untouched. Only the exact review trust boundary
uses the recorded owner policy; aliases and orientation checks are identical.
"""
from copy import deepcopy
from app_core.ncaaf_compatible_observation import _canonical, _name, mapping_binding
from app_core import ncaaf_model_compatibility as model, ncaaf_history as history


def check_mapping(event, schedule, crosswalk, review, *, as_of, review_policy):
    require = model.require
    require(isinstance(event, dict) and isinstance(schedule, list) and schedule,
            "NCAAF_COMPAT_EVENT_MISSING")
    require(isinstance(crosswalk, list) and bool(crosswalk), "NCAAF_COMPAT_CROSSWALK_MISSING")
    names, ids = {}, {}
    for entry in crosswalk:
        require(set(entry) == {"provider_name", "schedule_name", "team_id"}
                and type(entry["team_id"]) is int and entry["team_id"] > 0,
                "NCAAF_COMPAT_CROSSWALK_CONFLICT")
        key = _canonical(entry["provider_name"])
        require(key and key == _canonical(entry["schedule_name"]), "NCAAF_COMPAT_ALIAS_UNREVIEWED")
        require(key not in ids or ids[key] == entry["team_id"], "NCAAF_COMPAT_ALIAS_COLLISION")
        require(_name(entry["provider_name"]) not in names, "NCAAF_COMPAT_ALIAS_COLLISION")
        ids[key] = entry["team_id"]
        names[_name(entry["provider_name"])] = entry
    require(event.get("sport") == "NCAAF" and event.get("provider_namespace") == "odds_api"
            and bool(event.get("provider_event_id")) and bool(event.get("canonical_event_id")),
            "NCAAF_COMPAT_EVENT_IDENTITY_MISSING")
    home, away = [names.get(_name(event.get(k))) for k in ("home_team", "away_team")]
    require(home is not None and away is not None and home["team_id"] != away["team_id"],
            "NCAAF_COMPAT_TEAM_MAPPING_UNRESOLVED")
    kickoff = history.timestamp(event.get("start_utc"))
    require(kickoff is not None, "NCAAF_COMPAT_EVENT_CLOCK_MISSING")
    matches = [g for g in schedule if g.get("homeId") == home["team_id"]
        and g.get("awayId") == away["team_id"] and g.get("homeTeam") == home["schedule_name"]
        and g.get("awayTeam") == away["schedule_name"]
        and history.timestamp(g.get("startDate")) == kickoff]
    require(len(matches) == 1, "NCAAF_COMPAT_EVENT_AMBIGUOUS_OR_ORIENTATION_CONFLICT")
    game = matches[0]
    require(type(game.get("id")) is int and type(event.get("schedule_game_id")) is int
            and game.get("id") == event.get("schedule_game_id")
            and event["canonical_event_id"] == "cfbd:" + str(game["id"])
            and game.get("completed") is False and game.get("startTimeTBD") is False
            and type(game.get("neutralSite")) is bool and type(event.get("neutral_site")) is bool
            and game["neutralSite"] == event["neutral_site"], "NCAAF_COMPAT_EVENT_FACT_CONFLICT")
    binding = mapping_binding(event, schedule, crosswalk)
    require(isinstance(review, dict) and review.get("binding") == binding,
            "NCAAF_COMPAT_EVENT_REVIEW_MISSING")
    reviewed = history.timestamp(review.get("reviewed_at"))
    require(reviewed is not None and reviewed <= as_of and bool(review.get("review_id"))
            and review_policy.mapping(review),
            "NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED")
    return deepcopy(game)
