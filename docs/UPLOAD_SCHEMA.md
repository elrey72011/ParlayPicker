# Upload CSV Schema and Header Aliases

ParlayPicker normalizes uploaded CSV headers to canonical analysis columns.

## Canonical identity columns
- `league`
- `home_team`
- `away_team`
- `market_type`

## Canonical market columns
- `spread_line`
- `total_line`
- `theover_probability`
- `odds_american`
- `ml_probability`

## Common aliases
- `Home`, `Home Team`, `HomeTeam` → `home_team`
- `Away`, `Away Team`, `AwayTeam`, `Visitor`, `Visitor Team` → `away_team`
- `Sport` → `league`
- `Market Type` → `market_type`
- `WinProbability`, `Win Probability` → `theover_probability`
- `Odds`, `American Odds`, `Odds American` → `odds_american`
- `Spread Line` → `spread_line`
- `Total Line` → `total_line`

## Notes
- Header normalization lowercases names and removes punctuation (for example, `Home-Team-Name` is normalized before alias mapping).
- If required canonical fields are still missing after normalization, the pipeline logs a warning with the source columns to simplify debugging.
