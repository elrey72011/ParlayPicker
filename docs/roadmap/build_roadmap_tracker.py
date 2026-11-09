"""Build docs/roadmap/ParlayPicker-roadmap-tracker.xlsx from the roadmap (2026-11-05 snapshot; validation roadmap rows added 2026-11-09).

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
    ("C4", "Validation, review and activation", "C1, C2", "Owner", "Waiting on data", "First market: NCAAB spread. Go-live checklist in the app's readiness report", "2026-12-15"),
    ("C5", "Model improvements with fresh holdouts", "C2", "Claude", "In progress",
     "NCAAF backtest: no edge (D18); price screen (#2463); research picks (#2466); board Kalshi/de-vig fixes (G9); NCAAB line screen", None),
    ("C6", "Selector switch (app shows qualified picks)", "C4, D7", "Claude", "In progress",
     "Bridge + pipeline hookup (#2472); decision capture: fresh quotes and both-side predictions (step 1). "
     "App store refresh (read-only, every 10 min). Left: owner activation after C4", None),
    # Validation roadmap (docs/roadmap/VALIDATION-ROADMAP-2026-11-09.md). "V" keeps these apart from the old M0/M1.
    ("V-M1", "A validated, activated pick reaches a stake (bridge fixes, sport cap, checklist)", "D23", "Claude",
     "Done", "#2513; end-to-end test reaches a $5.00 stake on a $1,000 bankroll", None, "2026-11-09"),
    ("V-M2", "Price watch: does the Pinnacle/Novig lag have volume?", "D28, D29", "Claude", "Waiting on data",
     "#2513; watch runs through the night of 2026-11-16 (200 credits a day, 1,400 in all). Go: 2 bets a day "
     "across NBA and NCAAB with mean CLV above zero", "2026-11-16"),
    ("V-M3", "market-skill-v2: CLV-led plan design, NBA and NCAAB pooling", "D26, D27", "Claude", "Done",
     "#2513; 75 bets a window by default (100 if the per-bet CLV spread is near 3%)", None, "2026-11-09"),
    ("V-M4", "Freeze, validate, review, activate the first v2 plan", "V-M1, V-M2 go, V-M3", "Owner", "Blocked",
     "M4-RUNBOOK-2026-11-09.md. Not draftable on 2026-11-09: replay shows 0 bets; NBA has no calibration. "
     "Two windows of 15 to 59 days each, by bet rate", None),
    ("V-M5", "Second-line candidates and offline screens", "—", "Claude", "Waiting on data",
     "#2514; props, graded picks and the price screen by book show no edge. Research score models need 120 "
     "training games each (NHL settlement is on)", None),
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
    ("Merge PRs #2513 (roadmap, M1 to M3) and #2514 (M5 screens)", "After CI", "Done", "2026-11-09"),
    ("Read the M2 go/no-go report; decide whether capture continues past the price watch", "2026-11-16", "Open", None),
    ("Choose 75 or 100 bets a window for the first v2 plan (from the measured CLV spread)", "Before the freeze", "Open", None),
    ("Draft the pooled v2 plan for NBA and NCAAB with one validation start, review, freeze both",
     "If M2 says go", "Open", None),
]

DECISIONS = [
    ("D1", "Qualification thresholds", "Superseded", "Qualification follows the frozen validation plans"),
    ("D2", "NBA/NHL in parallel with NFL", "Moot", "Scheduler already covers six sports"),
    ("D3", "Moneyline in scope; WNBA in the policy", "Deferred",
     "Moneylines after the first Pinnacle spread results; WNBA in spring (owner 2026-11-06)"),
    ("D4", "Sharp-price and injury sources", "Decided", "Option A: Pinnacle fair-price reference for NBA/NCAAB spreads; 1,500 credits per 31 days; set PARLAYPICKER_PINNACLE_SPORTS"),
    ("D5", "Budgets for authentic NFL acceptance", "Closed", "Not needed (owner 2026-11-06)"),
    ("D6", "Hosted NCAAF custody", "Closed", "Not needed (owner 2026-11-06)"),
    ("D7", "Switch the selector to EV with No Bet", "Open", "Only after C4; shadow comparison built (Research picks report)"),
    ("D8", "Closing capture every 15 instead of 30 minutes", "Waiting", "Decide after the morning report's close-capture results"),
    ("D9", "NBA ESPN download", "Decided", "Offline benchmark; done"),
    ("D10", "Retire or keep the shadow shared pipeline", "Decided", "Retire the frozen shadow pipeline (C3 landed)"),
    ("D11", "Who reviews plan bindings and deployment reviews", "Decided", "Option B: owner plus automated audit (Deployment review workflow)"),
    ("D12", "Football plan family", "Decided", "This season: research-grade NCAAF picks (Pinnacle value board); plan family before 2027"),
    ("D13", "Odds API credits for certified closes", "Decided", "Option B: NBA spread, 100/day, 2,500 per 31 days"),
    ("D14", "Plan windows and first market", "Decided", "Windows sized by game count; NCAAB spread first"),
    ("D15", "NCAAF runtime after alias-only edits", "Decided", "Merged #2462; NCAAF capture works again"),
    ("D16", "Research capture window", "Decided", "PARLAYPICKER_RESEARCH_CAPTURE_HOURS=18"),
    ("D17", "NHL settlement against the official final", "Decided", "PARLAYPICKER_NHL_SETTLEMENT=official_check"),
    ("D18", "NCAAF model trained on history", "Decided", "Backtest: no edge over 2025 lines; not bound to a plan"),
    ("D19", "Plan probability thresholds for spreads and totals", "Decided",
     "Option B: scored against the market's no-vig price (market-skill plans)"),
    ("D20", "How a push counts in a plan's bets", "Decided", "A zero-return bet, for plans frozen from 2026-11-09"),
    ("D21", "Bind a Pinnacle plan to its prediction code", "Decided", "New plans record the code hash; predictions stop if it changes"),
    ("D22", "Recheck a frozen plan's hash; as-of filtering", "Decided", "Recheck yes; as-of filtering no"),
    ("D23", "What a stake needs besides the market activation", "Decided",
     "The verified market activation supplies the sport exposure cap; no second legacy policy (built, #2513)"),
    ("D24", "Two app sessions sizing against the same headroom", "Decided", "No change: one owner uses the app"),
    ("D25", "Football Stage 1 schedule", "Decided", "Paused; manual runs kept"),
    ("D26", "Gate design for plans frozen from now on", "Decided",
     "CLV-led market-skill-v2: mean bet CLV 2 standard errors above zero in each window (built, #2513)"),
    ("D27", "One plan counting bets across NBA and NCAAB", "Decided",
     "Yes, for the Pinnacle/Novig lag only; each sport's results reported (built, #2513)"),
    ("D28", "Credits for the fixed-interval price watch", "Decided",
     "Both sports, 7 days: 200 credits a day, 1,400 in all; review on 2026-11-16 (running)"),
    ("D29", "Novig junk-price filter", "Decided",
     "Approved: two-sided overround above 1.15 is dropped from the replay and from new plans' entries (built, #2513)"),
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
    for r, (sid, name, deps, owner, status, notes, target, *finished) in enumerate(STAGES, start=2):
        done = date.fromisoformat(finished[0]) if finished else date(2026, 11, 5)
        st.append([sid, name, deps, owner, status, notes,
                   date.fromisoformat(target) if target else None,
                   done if status == "Done" else None])
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
