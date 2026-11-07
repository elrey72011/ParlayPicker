"""TheOver page signals parsed from the raw page sheet of the owner's workbook (owner 2026-11-07).

The workbook's first sheet is the summary table the app has always read (league, teams,
pick, line, public %, model record). The raw page, pasted into ``Sheet1`` one cell per
line, also carries signals the table drops:

* "Sharp money agrees" (sometimes "Sharp money agrees · N plays") vs "No sharp signal yet";
* "PURPLE STAR";
* "Fade the public";
* on totals, the public split by bets *and* by money: "87% Over" / "% of bets" /
  "Under 13%", then "79% Over" / "% of money" / "Under 21%".

Each game block starts at a kickoff time cell ("20:00:00"); the first two "0" cells in a
block close the away and home team (full name, short name, nickname). Research only:
none of these is a probability and none is blended.
"""
from __future__ import annotations

import re

import pandas as pd

TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
PCT_OVER = re.compile(r"^(\d{1,3})%\s*Over$", re.IGNORECASE)
SHARP = re.compile(r"^Sharp money agrees(?:\s*·\s*(\d+)\s*plays?)?$", re.IGNORECASE)
BETTING = re.compile(r"^(\d{1,3})%\s*betting\s+(over|under)$", re.IGNORECASE)
COLUMNS = ["away_names", "home_names", "sharp_money_agrees", "sharp_plays", "purple_star",
           "fade_public", "bets_pct_over", "money_pct_over"]


def _cells(sheet: pd.DataFrame) -> list[str]:
    values = []
    for row in sheet.astype(str).values.tolist():
        for cell in row:
            cell = cell.strip()
            if cell and cell.lower() not in ("nan", "none"):
                values.append(cell)
    return values


def _teams(block):
    zeros = [i for i, cell in enumerate(block) if cell == "0"]
    if len(zeros) < 2 or zeros[0] < 3 or zeros[1] - zeros[0] < 4:
        return None
    away = tuple(block[zeros[0] - 3:zeros[0]])
    home = tuple(block[zeros[1] - 3:zeros[1]])
    return away, home, zeros[1]


def parse(sheet: pd.DataFrame) -> pd.DataFrame:
    """One row per game block found on the raw page; empty when the sheet is not a TheOver page."""
    cells = _cells(sheet)
    starts = [i for i, cell in enumerate(cells) if TIME.match(cell)]
    rows = []
    for n, start in enumerate(starts):
        block = cells[start:starts[n + 1] if n + 1 < len(starts) else len(cells)]
        teams = _teams(block)
        if teams is None:
            continue
        away, home, end = teams
        before, after = block[:end], block[end + 1:]
        # A block without "Sharp money agrees" has no sharp signal (sides say so explicitly).
        row = {"away_names": away, "home_names": home, "sharp_money_agrees": False, "sharp_plays": None,
               "purple_star": any(c.upper() == "PURPLE STAR" for c in before + after),
               "fade_public": any(c.lower() == "fade the public" for c in after),
               "bets_pct_over": None, "money_pct_over": None}
        for cell in after:
            sharp = SHARP.match(cell)
            if sharp:
                row["sharp_money_agrees"] = True
                row["sharp_plays"] = int(sharp.group(1)) if sharp.group(1) else None
            elif cell.lower() == "no sharp signal yet":
                row["sharp_money_agrees"] = False
        for i, cell in enumerate(after[:-1]):
            match = PCT_OVER.match(cell)
            if match and after[i + 1].lower() == "% of bets":
                row["bets_pct_over"] = int(match.group(1)) / 100.0
            elif match and after[i + 1].lower() == "% of money":
                row["money_pct_over"] = int(match.group(1)) / 100.0
        if row["bets_pct_over"] is None:
            # Some games show only "70% betting over" (no bets/money breakdown).
            for cell in after:
                betting = BETTING.match(cell)
                if betting:
                    share = int(betting.group(1)) / 100.0
                    row["bets_pct_over"] = share if betting.group(2).lower() == "over" else 1 - share
                    break
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def _token(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _names_match(table_name, page_names) -> bool:
    table = _token(table_name)
    if not table:
        return False
    tokens = [_token(n) for n in page_names if _token(n)]
    return any(table == t for t in tokens) or (len(table) >= 4 and _token(page_names[0]).startswith(table))


def attach(table: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Add the page signals to the summary table, by away and home team. Unmatched rows stay empty."""
    out = table.copy()
    for column in ("SharpMoneyAgrees", "SharpPlays", "PurpleStar", "FadePublic", "BetsPctOver", "MoneyPctOver"):
        out[column] = pd.NA
    out["PageMatched"] = False
    if signals is None or signals.empty or not {"HomeTeam", "AwayTeam"} <= set(out.columns):
        return out
    for index, row in out.iterrows():
        found = [s for s in signals.itertuples(index=False)
                 if _names_match(row["AwayTeam"], s.away_names) and _names_match(row["HomeTeam"], s.home_names)]
        if len(found) != 1:
            continue
        s = found[0]
        out.at[index, "PageMatched"] = True
        out.at[index, "SharpMoneyAgrees"] = s.sharp_money_agrees
        out.at[index, "SharpPlays"] = s.sharp_plays
        out.at[index, "PurpleStar"] = s.purple_star
        out.at[index, "FadePublic"] = s.fade_public
        out.at[index, "BetsPctOver"] = s.bets_pct_over
        out.at[index, "MoneyPctOver"] = s.money_pct_over
    return out


def read_workbook(uploaded_file) -> tuple[pd.DataFrame, int]:
    """(summary table with the page signals attached, games matched) from an .xlsx upload."""
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None, engine="openpyxl")
    names = list(sheets)
    table_name = next((n for n in names if n.lower() in ("totalsraw", "sidesraw", "table1")), names[0])
    raw = sheets[table_name]
    header = raw.iloc[0].astype(str).str.strip().tolist()
    table = raw.iloc[1:].reset_index(drop=True)
    table.columns = header
    page = next((sheets[n] for n in names if n != table_name and sheets[n].shape[1] == 1), None)
    signals = parse(page) if page is not None else pd.DataFrame(columns=COLUMNS)
    attached = attach(table, signals)
    matched = int(attached["PageMatched"].sum()) if len(attached) else 0
    return attached, matched
