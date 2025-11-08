
import os, math, json, time, requests
from typing import Optional, Tuple, Dict, Any, List
import streamlit as st
import pandas as pd

# ============================
# Helpers: odds conversions
# ============================

def american_to_decimal(odds: float) -> float:
    if odds is None:
        return None
    if odds > 0:
        return 1.0 + odds/100.0
    else:
        return 1.0 + 100.0/(-odds)

def american_to_prob(odds: float) -> float:
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)

def fair_probs_from_two_prices(a_odds: Optional[float], b_odds: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    Returns de-juiced probabilities (p_a, p_b) that sum to 1.0 when both sides exist.
    If one side is missing, returns a single-sided approximation for that side and None for the other.
    \"\"\"
    pa = american_to_prob(a_odds) if a_odds is not None else None
    pb = american_to_prob(b_odds) if b_odds is not None else None
    if pa is not None and pb is not None and pa > 0 and pb > 0:
        s = pa + pb
        if s > 0:
            return pa / s, pb / s
    # single-sided fallback (conservative): cap at 0.55 for big plus money to avoid junk
    if pa is not None:
        return min(pa, 0.55), None
    if pb is not None:
        return None, min(pb, 0.55)
    return None, None

def market_prob_for_outcome(outcome_price: Optional[float], opp_price: Optional[float]) -> Optional[float]:
    \"\"\"Convenience wrapper to get the market (no-vig) probability for the chosen side.\"\"\"
    fa, fb = fair_probs_from_two_prices(outcome_price, opp_price)
    if outcome_price is None:
        return None
    if opp_price is not None:
        # assume outcome_price is the "a" side by call-site convention
        return fa
    # single-sided approx
    return american_to_prob(outcome_price)

def best_price(existing: Optional[float], candidate: Optional[float]) -> Optional[float]:
    \"\"\"Choose the better price for the bettor (higher decimal odds).\"\"\"
    if candidate is None:
        return existing
    if existing is None:
        return candidate
    # compare decimal returns
    dec_existing = american_to_decimal(existing)
    dec_candidate = american_to_decimal(candidate)
    return candidate if dec_candidate > dec_existing else existing

# ============================
# Fetch from The Odds API
# ============================

def fetch_oddsapi_snapshot(api_key: str, sport_key: str, regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict[str, Any]]:
    base = "https://api.the-odds-api.com/v4/sports"
    url = f"{base}/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "unix"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ============================
# Normalize into best prices
# ============================

def normalize_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    \"\"\"
    For each event, compute the best available price for ML home/away,
    best prices for spread (each side) at each point, and totals (O/U) at each total.
    Keep opponent price for fair-prob calculation when possible.
    \"\"\"
    norm = []
    for ev in raw_events:
        event = {
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "h2h": {"home": {"price": None, "book": None}, "away": {"price": None, "book": None}},
            "spreads": {},  # point -> {"home":{"price","book"}, "away":{"price","book"}}
            "totals": {},   # point -> {"over":{"price","book"}, "under":{"price","book"}}
        }
        bms = ev.get("bookmakers", [])
        for bm in bms:
            book = bm.get("title") or bm.get("key")
            for mk in bm.get("markets", []):
                key = mk.get("key")
                outcomes = mk.get("outcomes", [])
                if key == "h2h":
                    home_price = None
                    away_price = None
                    for o in outcomes:
                        name = o.get("name")
                        price = o.get("price")
                        if name == ev.get("home_team"):
                            home_price = price
                        elif name == ev.get("away_team"):
                            away_price = price
                    # pick best for bettors
                    if home_price is not None:
                        best = best_price(event["h2h"]["home"]["price"], home_price)
                        if best != event["h2h"]["home"]["price"]:
                            event["h2h"]["home"] = {"price": best, "book": book}
                    if away_price is not None:
                        best = best_price(event["h2h"]["away"]["price"], away_price)
                        if best != event["h2h"]["away"]["price"]:
                            event["h2h"]["away"] = {"price": best, "book": book}
                elif key == "spreads":
                    # outcomes have team name, price, point
                    for o in outcomes:
                        point = o.get("point")
                        price = o.get("price")
                        name = o.get("name")
                        side = "home" if name == ev.get("home_team") else "away"
                        d = event["spreads"].setdefault(point, {"home": {"price": None, "book": None},
                                                                "away": {"price": None, "book": None}})
                        best = best_price(d[side]["price"], price)
                        if best != d[side]["price"]:
                            d[side] = {"price": best, "book": book}
                elif key == "totals":
                    # outcomes have name "Over"/"Under", price, point
                    for o in outcomes:
                        point = o.get("point")
                        price = o.get("price")
                        name = (o.get("name") or "").lower()
                        side = "over" if name.startswith("over") else "under"
                        d = event["totals"].setdefault(point, {"over": {"price": None, "book": None},
                                                               "under": {"price": None, "book": None}})
                        best = best_price(d[side]["price"], price)
                        if best != d[side]["price"]:
                            d[side] = {"price": best, "book": book}
        norm.append(event)
    return norm

# ============================
# Build candidate legs with Market (No-Vig) %
# ============================

def build_candidates(event) -> List[Dict[str, Any]]:
    legs = []
    # Moneyline
    hp = event["h2h"]["home"]["price"]
    ap = event["h2h"]["away"]["price"]
    if hp is not None:
        legs.append({
            "type": "ML",
            "selection": f"{event['away_team']} @ {event['home_team']} — {event['home_team']} ML @{hp} ({event['h2h']['home']['book']})",
            "price": hp,
            "opp_price": ap,
            "market_prob": market_prob_for_outcome(hp, ap),
        })
    if ap is not None:
        legs.append({
            "type": "ML",
            "selection": f"{event['away_team']} @ {event['home_team']} — {event['away_team']} ML @{ap} ({event['h2h']['away']['book']})",
            "price": ap,
            "opp_price": hp,
            "market_prob": market_prob_for_outcome(ap, hp),
        })

    # Spreads (pair at each point)
    for point, sides in event["spreads"].items():
        hp = sides["home"]["price"]
        ap = sides["away"]["price"]
        if hp is not None:
            legs.append({
                "type": "Spread",
                "selection": f"{event['away_team']} @ {event['home_team']} — {event['home_team']} {point:+} @{hp} ({sides['home']['book']})",
                "price": hp,
                "opp_price": ap,
                "market_prob": market_prob_for_outcome(hp, ap),
            })
        if ap is not None:
            legs.append({
                "type": "Spread",
                "selection": f"{event['away_team']} @ {event['home_team']} — {event['away_team']} {(-point):+} @{ap} ({sides['away']['book']})",
                "price": ap,
                "opp_price": hp,
                "market_prob": market_prob_for_outcome(ap, hp),
            })

    # Totals (pair Over/Under for same number)
    for point, sides in event["totals"].items():
        op = sides["over"]["price"]
        up = sides["under"]["price"]
        if op is not None:
            legs.append({
                "type": "Total",
                "selection": f"{event['away_team']} @ {event['home_team']} — Over {point} @{op} ({sides['over']['book']})",
                "price": op,
                "opp_price": up,
                "market_prob": market_prob_for_outcome(op, up),
            })
        if up is not None:
            legs.append({
                "type": "Total",
                "selection": f"{event['away_team']} @ {event['home_team']} — Under {point} @{up} ({sides['under']['book']})",
                "price": up,
                "opp_price": op,
                "market_prob": market_prob_for_outcome(up, op),
            })

    return legs

# ============================
# UI and app logic
# ============================

st.set_page_config(page_title="ParlayDesk — AI-Enhanced Odds", layout="wide")

st.title("ParlayDesk — AI‑Enhanced Odds (No‑Vig Market% Filter)")

with st.sidebar:
    st.subheader("Data & Filters")
    api_key = st.text_input("Odds API Key", value=os.getenv("ODDS_API_KEY", ""), type="password")
    sport_key = st.selectbox(
        "Sport",
        options=[
            "americanfootball_nfl", "basketball_nba", "baseball_mlb",
            "icehockey_nhl", "soccer_usa_mls", "tennis_atp_french_open"
        ],
        index=0
    )
    min_mkt_prob = st.slider(
        "Minimum Market (No‑Vig) % for legs",
        min_value=0.50, max_value=0.90, value=0.60, step=0.01,
        help="Only include legs the market prices as at least this probability (de‑juiced)."
    )
    max_legs = st.number_input("Max legs to show", min_value=2, max_value=20, value=10, step=1)
    target_parlay_legs = st.number_input("Target legs to build", min_value=2, max_value=12, value=2, step=1)
    stake = st.number_input("Stake ($)", min_value=1.0, value=100.0, step=1.0)

if not api_key:
    st.info("Enter your Odds API key in the sidebar to fetch live lines.")
    st.stop()

try:
    raw_events = fetch_oddsapi_snapshot(api_key, sport_key)
except Exception as e:
    st.error(f"Failed to fetch odds: {e}")
    st.stop()

events = normalize_events(raw_events)

# Build and filter candidates
all_legs = []
for ev in events:
    all_legs.extend(build_candidates(ev))

# Filter to keep 'high' market% legs
filtered = [leg for leg in all_legs if leg.get("market_prob") is not None and leg["market_prob"] >= float(min_mkt_prob)]
# Sort by market_prob desc, then by decimal odds (safer legs first)
filtered.sort(key=lambda x: (x["market_prob"], american_to_decimal(x["price"])), reverse=True)
filtered = filtered[: int(max_legs)]

st.markdown(f"**Found {len(filtered)} high‑confidence legs (Market (No‑Vig) ≥ {int(min_mkt_prob*100)}%).**")

if not filtered:
    st.warning("No legs met the threshold. Try lowering the slider or switching sports.")
    st.stop()

# Build a simple parlay: take top N by Market% (safe legs)
chosen = filtered[: int(target_parlay_legs)]

# Table
rows = []
parlay_decimal = 1.0
for i, leg in enumerate(chosen, 1):
    dec = american_to_decimal(leg["price"])
    parlay_decimal *= dec
    odds_str = f"{int(leg['price']):+d}" if float(leg['price']).is_integer() else f"{leg['price']:+.0f}"
    rows.append({
        "Leg": i,
        "Type": leg["type"],
        "Selection": leg["selection"],
        "Odds": odds_str,
        "Market (No‑Vig) %": f"{leg['market_prob']*100:.1f}%"
    })

parlay_prob_naive = 1.0
for leg in chosen:
    # multiply fair (no‑vig) probs to get a conservative parlay probability
    p = max(min(leg["market_prob"], 0.99), 0.001)
    parlay_prob_naive *= p

profit_on_stake = stake * (parlay_decimal - 1.0)
expected_return = stake * parlay_prob_naive * (parlay_decimal - 1.0)

df = pd.DataFrame(rows)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Decimal Odds", f"{parlay_decimal:.3f}")
with col2:
    st.metric("Parlay Market (No‑Vig) Prob.", f"{parlay_prob_naive*100:.2f}%")
with col3:
    st.metric("Profit on $" + str(int(stake)), f"${profit_on_stake:,.2f}")
with col4:
    st.metric("Expected Value", f"${expected_return:,.2f}")

st.subheader("Parlay Legs")
st.dataframe(df, use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="parlay_legs.csv", mime="text/csv")

st.caption("Market (No‑Vig) % uses de‑juiced probabilities by pairing opposing sides when available; single‑sided lines use a conservative cap.")
