"""NBA official injury reports: archive, parse and status changes (owner 2026-11-08, option 3).

Why: the NCAAB Pinnacle plan found no price edge at the books a pick can be staked at
(2026-11-08: 0 bets at 1% or 2% over 41 games). The next edge source is news: a player's
status changes before every book has moved. Before any model, the archive answers whether
books lag (``docs/roadmap/NEWS-EDGE-PLAN-2026-11-08.md``).

Source: the league's official report, published every 15 minutes as a PDF at a fixed URL
per slot (``slot_url``; this season's slots stay downloadable, so missed runs backfill).
Research only: no Odds API credits, no picks, no stake.

* ``parse``: rows of (game date, game time, matchup, team, player, status, reason) from the
  PDF's word positions (columns found from each page's header).
* ``changes``: status changes between two consecutive reports.
* ``record_day``: one complete ET day, write-once to Drive: a manifest with every slot's URL,
  HTTP status, SHA-256, parsed rows and changes, plus the PDF of every report whose rows
  differ from the previous kept one (an unchanged report names the report it repeats), so
  every parsed row replays from retained bytes.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import io
import json
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{hour:02d}_{minute:02d}{half}.pdf"
PREFIX = "parlaypicker/nba-injury-reports-v1"
PARSER_VERSION = "nba-injury-pdf-v1"
STATUSES = ("Available", "Probable", "Questionable", "Doubtful", "Out")
COLUMNS = ("game_date", "game_time", "matchup", "team", "player", "status", "reason")
HEADER = {"Date": "game_date", "Time": "game_time", "Matchup": "matchup", "Team": "team", "Player": "player",
          "Current": "status", "Reason": "reason"}
# Unavailable for the game: a change into or out of this set can move a line.
UNAVAILABLE = {"Out", "Doubtful"}


def slots(day: date) -> list[datetime]:
    """Every 15-minute report slot of an ET day, as ET datetimes."""
    start = datetime.combine(day, time(0, 0), tzinfo=ET)
    return [start + timedelta(minutes=15 * i) for i in range(96)]


def slot_url(slot: datetime) -> str:
    local = slot.astimezone(ET)
    hour = local.hour % 12 or 12
    return URL.format(day=local.date().isoformat(), hour=hour, minute=local.minute,
                      half="AM" if local.hour < 12 else "PM")


def _words(pdf_bytes):
    from pypdf import PdfReader
    pages = []
    for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
        words = []

        def visit(text, cm, tm, font, size):
            if text.strip():
                words.append((float(tm[4]), float(tm[5]), text.strip()))
        page.extract_text(visitor_text=visit)
        pages.append(words)
    return pages


def _lines(words, tolerance=1.5):
    lines = []
    for x, y, text in sorted(words, key=lambda w: (w[1], w[0])):
        if lines and abs(lines[-1][0] - y) <= tolerance:
            lines[-1][1].append((x, text))
        else:
            lines.append([y, [(x, text)]])
    return [(y, sorted(parts)) for y, parts in lines]


def _columns(lines):
    """Column start x from the header line (``Game Date ... Reason``), or None."""
    for y, parts in lines:
        texts = [t for _, t in parts]
        if "Matchup" in texts and "Reason" in texts:
            starts = {}
            for i, (x, text) in enumerate(parts):
                if text in HEADER and HEADER[text] not in starts:
                    # "Game Date" and "Game Time": the column starts at "Game".
                    first = parts[i - 1][0] if text in ("Date", "Time") and i and parts[i - 1][1] == "Game" else x
                    starts[HEADER[text]] = first
            if len(starts) == len(COLUMNS):
                return y, sorted((x, name) for name, x in starts.items())
    return None


def rows_from_words(pages) -> list[dict]:
    """Report rows from per-page word positions [(x, y, text)] (y grows down the page)."""
    rows, context, columns = [], dict.fromkeys(("game_date", "game_time", "matchup", "team")), None
    for words in pages:
        lines = _lines(words)
        found = _columns(lines)
        header_y = None
        if found:
            header_y, columns = found
        if columns is None:
            continue
        page_rows, loose = [], []
        for y, parts in lines:
            texts = [t for _, t in parts]
            if (header_y is not None and y <= header_y + 1) or "Report:" in texts or texts[0] == "Page":
                continue  # title, page number and header (pages without a header still have a title)
            cells = {}
            for x, text in parts:
                name = next((n for start, n in reversed(columns) if x >= start - 3), None)
                if name:
                    cells.setdefault(name, []).append(text)
            cell = {k: " ".join(v) for k, v in cells.items()}
            if set(cell) <= {"reason"}:
                loose.append((y, cell))
                continue
            for key in context:
                if cell.get(key):
                    context[key] = cell[key]
            status = cell.get("status", "")
            if status in STATUSES and cell.get("player"):
                page_rows.append({"y": y, **context, "player": " ".join(cell["player"].split()),
                                  "status": status, "reason": [(y, cell.get("reason", ""))]})
            elif "NOT YET SUBMITTED" in cell.get("reason", "").upper():
                page_rows.append({"y": y, **context, "player": None, "status": "NOT_YET_SUBMITTED",
                                  "reason": [(y, "NOT YET SUBMITTED")]})
            else:
                loose.append((y, cell))
        for y, cell in loose:  # wrapped reasons sit above and below their row
            text = cell.get("reason")
            if text and page_rows:
                nearest = min(page_rows, key=lambda r: abs(r["y"] - y))
                if abs(nearest["y"] - y) <= 25:
                    nearest["reason"].append((y, text))
        for row in page_rows:
            row["reason"] = " ".join(" ".join(t for _, t in sorted(row["reason"]) if t).split())
            row.pop("y")
            rows.append(row)
    return rows


def parse(pdf_bytes: bytes) -> list[dict]:
    return rows_from_words(_words(pdf_bytes))


def _key(row):
    return (row["game_date"], row["matchup"], row["team"], row["player"])


def changes(previous: list[dict], current: list[dict]) -> list[dict]:
    """Status changes from one report to the next: new listings, changed statuses, removals.

    ``material`` marks a change into or out of Out/Doubtful (a player newly unavailable or
    newly back), the changes that can move a line.
    """
    before = {_key(r): r["status"] for r in previous if r["player"]}
    after = {_key(r): r["status"] for r in current if r["player"]}
    out = []
    for key in sorted(set(before) | set(after), key=lambda k: tuple("" if v is None else v for v in k)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        material = (old in UNAVAILABLE) != (new in UNAVAILABLE)
        out.append(dict(zip(("game_date", "matchup", "team", "player"), key), previous=old, status=new,
                        material=material))
    return out


def day_key(day: date) -> str:
    return f"{PREFIX}/{day.isoformat()}/manifest.json"


def pdf_key(day: date, sha256: str) -> str:
    return f"{PREFIX}/{day.isoformat()}/pdf/{sha256}.pdf"


def collect_day(day: date, fetch, *, now: datetime) -> dict:
    """Every slot of ``day``: ``fetch(url) -> (http_status, bytes)``. Only slots already past ``now``."""
    reports, previous = [], None
    for slot in slots(day):
        if slot.astimezone(timezone.utc) > now:
            break
        url = slot_url(slot)
        try:
            status, body = fetch(url)
        except Exception as exc:  # recorded, never guessed
            reports.append({"slot": slot.isoformat(), "url": url, "http_status": None,
                            "error": type(exc).__name__})
            continue
        entry = {"slot": slot.isoformat(), "url": url, "http_status": status}
        if status == 200 and body:
            entry["sha256"] = hashlib.sha256(body).hexdigest()
            entry["bytes"] = len(body)
            entry["_body"] = body
            try:
                entry["rows"] = parse(body)
            except Exception as exc:
                entry["parse_error"] = type(exc).__name__
            else:
                entry["changes"] = changes(previous, entry["rows"]) if previous is not None else []
                previous = entry["rows"]
        reports.append(entry)
    return {"day": day.isoformat(), "parser": PARSER_VERSION, "collected_at": now.isoformat(), "reports": reports}


def record_day(client, day: date, fetch, *, now: datetime) -> dict:
    """Write one complete ET day to Drive once (PDFs by hash, then the manifest). Never overwrites."""
    if datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=ET) > now:
        return {"day": day.isoformat(), "status": "DAY_NOT_OVER"}
    try:
        client.get_object(Key=day_key(day))
        return {"day": day.isoformat(), "status": "ALREADY_RECORDED"}
    except Exception:
        pass
    bundle = collect_day(day, fetch, now=now)
    stored, last_rows, last_slot = set(), None, None
    for entry in bundle["reports"]:
        body = entry.pop("_body", None)
        if body is None:
            continue
        # Keep the bytes of every report whose rows differ from the last kept one (and any that
        # failed to parse); an unchanged report points at the kept report it repeats.
        if "rows" in entry and entry["rows"] == last_rows:
            entry["same_rows_as"] = last_slot
            continue
        last_rows, last_slot = entry.get("rows"), entry["slot"]
        entry["stored"] = True
        if entry["sha256"] in stored:
            continue
        try:
            client.put_object(Key=pdf_key(day, entry["sha256"]), Body=body, IfNoneMatch="*")
        except Exception as exc:
            if type(exc).__name__ != "AlreadyExists":
                raise
        stored.add(entry["sha256"])
    found = [e for e in bundle["reports"] if e.get("http_status") == 200]
    if not found:
        return {"day": day.isoformat(), "status": "NO_REPORTS"}  # off day or archive gone: try again later
    body = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    client.put_object(Key=day_key(day), Body=body, IfNoneMatch="*")
    return {"day": day.isoformat(), "status": "RECORDED", "reports": len(found),
            "changes": sum(len(e.get("changes", [])) for e in found),
            "material_changes": sum(1 for e in found for c in e.get("changes", []) if c["material"])}
