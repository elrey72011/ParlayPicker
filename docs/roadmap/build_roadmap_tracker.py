"""Build docs/roadmap/ParlayPicker-roadmap-tracker.xlsx from the roadmap (2026-11-05 snapshot).

Run: python docs/roadmap/build_roadmap_tracker.py
The workbook is a tracking copy; docs/roadmap/ROADMAP.md stays the source of truth.
"""
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

OUT = Path(__file__).with_name("ParlayPicker-roadmap-tracker.xlsx")
FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3864")
INPUT_FILL = PatternFill("solid", start_color="FFF2CC")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BLUE = Font(name=FONT, color="0000FF", size=10)
BLACK = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)

STATUSES = ["Done", "In progress", "Waiting on data", "Owner action", "Not started", "Blocked"]

STAGES = [
    ("M0", "CI repair and protected-change guard", "—", "Claude", "Done", "#2455", None),
    ("M1", "Shared pipeline (shadow only)", "—", "Claude", "Done", "#2456; frozen (D10)", None),
    ("E1", "Legacy scoreboard and closing-capture workflow", "—", "Claude", "Done", "#2457; scoreboard reworked under C3", None),
    ("P1", "EV shadow selector on the authority chain", "—", "Claude", "Done", "#2458; off by default", None),
    ("C0", "Market readiness from Drive evidence", "—", "Claude", "Done", "#2460; market-readiness-2026-11-05.md", None),
    ("C1", "Certified closing prices into the prospective store", "C0", "Claude", "Done", "#2461; on via PARLAYPICKER_CERTIFIED_CLOSE_SPORTS", None),
    ("C2", "Bind models to new plan versions (settle, calibrate, bind, freeze)", "C0, D11, D12", "Claude",
     "Waiting on data", "#2462 merged; fills in as games are recorded", None),
    ("C3", "Plan-progress view; rework the scoreboard", "C0", "Claude", "Done",
     "#2463 progress panel; #2465 paper track record; #2466 research picks", None),
    ("C4", "Validation, review and activation", "C1, C2", "Owner", "Waiting on data", "First market: NCAAB spread", "2026-12-15"),
    ("C5", "Model improvements with fresh holdouts", "C2", "Claude", "In progress",
     "NCAAF backtest: no edge (D18); price screen (#2463); research picks (#2466); board Kalshi/de-vig fixes (G9); NCAAB line screen", None),
    ("C6", "Selector switch (app shows qualified picks)", "C4, D7", "Owner", "Not started", "After C4", None),
]

# sport, market, priority, games/day estimate, target sample, window days estimate, status, notes
MARKETS = [
    ("NCAAB", "SPREAD", 1, 30, 200, 11, "Waiting on data", "First market (D14). Capture starts with RESEARCH_SPORTS fix"),
    ("NCAAB", "TOTAL", 2, 30, 200, 11, "Waiting on data", "Runs in parallel with spread"),
    ("NBA", "SPREAD", 3, 7, 200, 45, "Waiting on data", "Certified closes allowed (D13)"),
    ("NBA", "TOTAL", 4, 7, 200, 45, "Waiting on data", ""),
    ("NHL", "PUCK_LINE", 5, 8, 200, 40, "Waiting on data", "Settlement on (D17); checked per game vs NHL official"),
    ("NHL", "TOTAL", 6, 8, 200, 40, "Waiting on data", "Settlement on (D17)"),
    ("NCAAF", "SPREAD", 7, None, 200, None, "Blocked", "Historical model shows no edge (D18); needs a better model (C5)"),
    ("NCAAF", "TOTAL", 8, None, 200, None, "Blocked", "Needs two seasons"),
    ("NFL", "SPREAD", 9, None, 200, None, "Blocked", "Needs two seasons"),
    ("NFL", "TOTAL", 10, None, 200, None, "Blocked", "Needs two seasons"),
    ("MLB", "RUN_LINE", 11, None, 200, None, "Not started", "Offseason; 2027"),
    ("MLB", "TOTAL", 12, None, 200, None, "Not started", "Offseason; 2027"),
]

ACTIONS = [
    ("Set RESEARCH_SPORTS=NFL,NCAAF,NBA,NCAAB,MLB,NHL", "Now", "Done", "2026-11-05"),
    ("Review D15 (NCAAF runtime recovery) and merge PR #2462", "Before merge", "Done", "2026-11-05"),
    ("Set PARLAYPICKER_CERTIFIED_CLOSE_SPORTS=NBA,NCAAB", "After merge", "Done", "2026-11-05"),
    ("Set PARLAYPICKER_RESEARCH_CAPTURE_HOURS=18 (D16)", "After merge", "Done", "2026-11-05"),
    ("Set PARLAYPICKER_NHL_SETTLEMENT=official_check (D17)", "After merge", "Done", "2026-11-05"),
    ("Add Streamlit secret PARLAYPICKER_GITHUB_READ_TOKEN", "For the progress panel", "Done", "2026-11-05"),
    ("Merge PRs #2463, #2464, #2465", "After CI", "Done", "2026-11-05"),
    ("Merge PR #2466 (research picks)", "After CI", "Open", None),
    ("Keep pressing Refresh daily (feeds the paper track record)", "Daily", "Open", None),
    ("Check the paper track record (Activation grading run summary)", "Weekly", "Open", None),
    ("Run Bound plan → draft for NCAAB SPREAD, review, then freeze with the hash", "When readiness shows READY_FOR_OWNER_REVIEW", "Open", None),
    ("Decide D11: who reviews plan bindings and deployment reviews", "Before first freeze", "Open", None),
    ("Validation review and market activation (C4)", "After validation and holdout", "Open", None),
]

DECISIONS = [
    ("D1", "Qualification thresholds", "Superseded", "Qualification follows the frozen validation plans"),
    ("D2", "NBA/NHL in parallel with NFL", "Moot", "Scheduler already covers six sports"),
    ("D3", "Moneyline in scope; WNBA in the policy", "Open", "Moneyline stays context-only"),
    ("D4", "Sharp-price and injury sources", "Decided", "Option A: Pinnacle fair-price reference for NBA/NCAAB spreads; 1,500 credits per 31 days; set PARLAYPICKER_PINNACLE_SPORTS"),
    ("D5", "Budgets for authentic NFL acceptance", "Open", ""),
    ("D6", "Hosted NCAAF custody", "Open", ""),
    ("D7", "Switch the selector to EV with No Bet", "Open", "Only after C4; shadow comparison built (Research picks report)"),
    ("D8", "Closing capture every 15 instead of 30 minutes", "Open", ""),
    ("D9", "NBA ESPN download", "Decided", "Offline benchmark; done"),
    ("D10", "Retire or keep the shadow shared pipeline", "Open", "Recommend: keep off; retire after C3"),
    ("D11", "Who reviews plan bindings and deployment reviews", "Decided", "Option B: owner plus automated audit (Deployment review workflow)"),
    ("D12", "Football plan family", "Decided", "Option C: design v3 (two seasons needed)"),
    ("D13", "Odds API credits for certified closes", "Decided", "Option B: NBA spread, 100/day, 2,500 per 31 days"),
    ("D14", "Plan windows and first market", "Decided", "Windows sized by game count; NCAAB spread first"),
    ("D15", "NCAAF runtime after alias-only edits", "Decided", "Merged #2462; NCAAF capture works again"),
    ("D16", "Research capture window", "Decided", "PARLAYPICKER_RESEARCH_CAPTURE_HOURS=18"),
    ("D17", "NHL settlement against the official final", "Decided", "PARLAYPICKER_NHL_SETTLEMENT=official_check"),
    ("D18", "NCAAF model trained on history", "Decided", "Backtest: no edge over 2025 lines; not bound to a plan"),
]


def header(ws, row, titles, widths):
    for col, (title, width) in enumerate(zip(titles, widths), 1):
        cell = ws.cell(row=row, column=col, value=title)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BOX
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def style_row(ws, row, ncols, inputs=()):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.border = BOX
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if col in inputs:
            cell.font = BLUE
            cell.fill = INPUT_FILL
        elif cell.font is None or not cell.font.bold:
            cell.font = BLACK


def status_colors(ws, rng):
    for text, color in (("Done", "C6EFCE"), ("Decided", "C6EFCE"), ("In progress", "DDEBF7"),
                        ("Owner action", "FCE4D6"), ("Blocked", "F8CBAD"), ("Waiting on data", "FFF2CC")):
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{text}"'],
                                                      fill=PatternFill("solid", start_color=color)))


def build():
    wb = Workbook()

    # Summary
    sm = wb.active
    sm.title = "Summary"
    sm["A1"] = "ParlayPicker roadmap tracker"
    sm["A1"].font = Font(name=FONT, size=14, bold=True)
    sm["A2"] = "Snapshot of docs/roadmap/ROADMAP.md, updated 2026-11-05 evening. Edit the yellow cells (blue text) on each tab; totals update automatically."
    sm["A2"].font = BLACK
    rows = [
        ("Roadmap stages", "=COUNTA(Stages!A2:A100)"),
        ("Stages done", '=COUNTIF(Stages!E2:E100,"Done")'),
        ("Stages complete (%)", "=IFERROR(B5/B4,0)"),
        ("Markets tracked", "=COUNT(Markets!C2:C100)"),
        ("Markets qualified", '=COUNTIF(Markets!N2:N100,"Qualified")'),
        ("Owner actions open", '=COUNTIF(\'Owner actions\'!C2:C100,"Open")'),
        ("Owner actions done", '=COUNTIF(\'Owner actions\'!C2:C100,"Done")'),
        ("Decisions still open or proposed", '=COUNTIF(Decisions!C2:C100,"Open")+COUNTIF(Decisions!C2:C100,"Proposed")'),
        ("Earliest estimated qualification", "=IFERROR(SMALL(Markets!M2:M100,1),\"—\")"),
    ]
    for i, (label, formula) in enumerate(rows, start=4):
        sm.cell(row=i, column=1, value=label).font = BOLD
        sm.cell(row=i, column=2, value=formula).font = BLACK
        for col in (1, 2):
            sm.cell(row=i, column=col).border = BOX
    sm["B6"].number_format = "0%"
    sm["B12"].number_format = "yyyy-mm-dd"
    sm.column_dimensions["A"].width = 36
    sm.column_dimensions["B"].width = 18
    sm["A14"] = "Legend"
    sm["A14"].font = BOLD
    sm["A15"] = "Yellow cell, blue text"
    sm["A15"].fill = INPUT_FILL
    sm["A15"].font = BLUE
    sm["B15"] = "Input you update"
    sm["A16"] = "Black text"
    sm["B16"] = "Formula; do not edit"
    sm["A17"] = "Status lists"
    sm["B17"] = "Pick from the drop-down in each Status column"
    for r in range(15, 18):
        sm.cell(row=r, column=2).font = BLACK
    sm["A16"].font = BLACK
    sm["A17"].font = BLACK

    # Stages
    st = wb.create_sheet("Stages")
    header(st, 1, ["ID", "Stage", "Depends on", "Owner", "Status", "Notes / PR", "Target date", "Done date"],
           [7, 48, 14, 10, 16, 44, 13, 13])
    for r, (sid, name, deps, owner, status, notes, target) in enumerate(STAGES, start=2):
        st.append([sid, name, deps, owner, status, notes,
                   date.fromisoformat(target) if target else None,
                   date(2026, 11, 5) if status == "Done" else None])
        style_row(st, r, 8, inputs=(5, 6, 7, 8))
        st.cell(row=r, column=7).number_format = "yyyy-mm-dd"
        st.cell(row=r, column=8).number_format = "yyyy-mm-dd"
    st["G10"].comment = Comment("Estimate from the 2026-11-05 plan: ~30 days for NCAAB spread "
                                "if runs complete and the model passes. Update as data arrives.", "Claude")
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=True)
    st.add_data_validation(dv)
    dv.add("E2:E100")
    status_colors(st, "E2:E100")

    # Markets
    mk = wb.create_sheet("Markets")
    titles = ["Sport", "Market", "Priority", "Est. games/day", "Training games (now)",
              "Training progress (of 120)", "Calibration decided (now)", "Calibration progress (of 40)",
              "Plan frozen date", "Validation days (est.)", "Holdout days (est.)", "Settled in validation (now)",
              "Est. qualification date", "Status", "Notes"]
    header(mk, 1, titles, [8, 11, 8, 10, 12, 13, 12, 13, 13, 11, 11, 13, 14, 16, 44])
    for r, (sport, market, prio, rate, target, days, status, notes) in enumerate(MARKETS, start=2):
        mk.cell(row=r, column=1, value=sport)
        mk.cell(row=r, column=2, value=market)
        mk.cell(row=r, column=3, value=prio)
        mk.cell(row=r, column=4, value=rate)
        mk.cell(row=r, column=5, value=0)
        mk.cell(row=r, column=6, value=f"=MIN(1,E{r}/120)")
        mk.cell(row=r, column=7, value=0)
        mk.cell(row=r, column=8, value=f"=MIN(1,G{r}/40)")
        mk.cell(row=r, column=10, value=days)
        mk.cell(row=r, column=11, value=days)
        mk.cell(row=r, column=12, value=0)
        mk.cell(row=r, column=13, value=f'=IF(OR(I{r}="",J{r}="",K{r}=""),"",I{r}+1+J{r}+K{r})')
        mk.cell(row=r, column=14, value=status)
        mk.cell(row=r, column=15, value=notes)
        style_row(mk, r, 15, inputs=(4, 5, 7, 9, 10, 11, 12, 14, 15))
        for col in (6, 8):
            mk.cell(row=r, column=col).number_format = "0%"
        for col in (9, 13):
            mk.cell(row=r, column=col).number_format = "yyyy-mm-dd"
    mk["D2"].comment = Comment("Estimate. Readiness report 'games_per_day_14d' gives the observed rate.", "Claude")
    mk["J2"].comment = Comment("D14: window_days = max(7, ceil(200 / (rate x 0.8) x 1.25)). "
                               "The frozen plan fixes the real value; copy it here after freezing.", "Claude")
    mk["M2"].comment = Comment("Plan frozen date + 1 day + validation days + holdout days. "
                               "Blank until the plan is frozen.", "Claude")
    mdv = DataValidation(type="list", formula1='"' + ",".join(STATUSES + ["Qualified", "Failed validation"]) + '"',
                         allow_blank=True)
    mk.add_data_validation(mdv)
    mdv.add("N2:N100")
    status_colors(mk, "N2:N100")
    mk.conditional_formatting.add("N2:N100", CellIsRule(operator="equal", formula=['"Qualified"'],
                                                        fill=PatternFill("solid", start_color="A9D08E")))
    for col in ("F", "H"):
        mk.conditional_formatting.add(f"{col}2:{col}100", FormulaRule(formula=[f"{col}2>=1"],
                                      fill=PatternFill("solid", start_color="C6EFCE")))
    mk.cell(row=len(MARKETS) + 3, column=1,
            value="Example: after freezing NCAAB SPREAD, type the frozen date in column I; "
                  "column M then shows the estimated qualification date.").font = Font(name=FONT, size=9, italic=True)

    # Owner actions
    oa = wb.create_sheet("Owner actions")
    header(oa, 1, ["Action", "When", "Status", "Done date", "Notes"], [62, 34, 10, 13, 30])
    for r, (action, when, status, done) in enumerate(ACTIONS, start=2):
        oa.append([action, when, status, date.fromisoformat(done) if done else None, ""])
        style_row(oa, r, 5, inputs=(3, 4, 5))
        oa.cell(row=r, column=4).number_format = "yyyy-mm-dd"
    adv = DataValidation(type="list", formula1='"Open,Done,Skipped"', allow_blank=True)
    oa.add_data_validation(adv)
    adv.add("C2:C100")
    status_colors(oa, "C2:C100")

    # Decisions
    de = wb.create_sheet("Decisions")
    header(de, 1, ["ID", "Decision", "Status", "Summary", "Decided date"], [7, 44, 12, 60, 13])
    for r, (did, name, status, summary) in enumerate(DECISIONS, start=2):
        de.append([did, name, status, summary, None])
        style_row(de, r, 5, inputs=(3, 4, 5))
        de.cell(row=r, column=5).number_format = "yyyy-mm-dd"
    ddv = DataValidation(type="list", formula1='"Open,Proposed,Decided,Superseded,Moot"', allow_blank=True)
    de.add_data_validation(ddv)
    ddv.add("C2:C100")
    status_colors(de, "C2:C100")

    wb.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
