"""Descriptive prospective comparison; never promotes Gemini predictions."""
import pandas as pd


def review_comparison(frame):
    required = {'gemini_reviewed_at', 'game_start_utc', 'gemini_agreement', 'matchup_id', 'best_pick', 'gemini_review_model', 'gemini_review_input_hash'}
    if frame is None or not required.issubset(frame.columns):
        return pd.DataFrame()
    data = frame.copy()
    reviewed = pd.to_datetime(data.gemini_reviewed_at, utc=True, errors='coerce')
    starts = pd.to_datetime(data.game_start_utc, utc=True, errors='coerce')
    valid = reviewed.notna() & starts.notna() & reviewed.lt(starts)
    valid &= data.gemini_agreement.isin(['agree', 'disagree'])
    valid &= data.gemini_review_input_hash.fillna('').astype(str).str.len().eq(64)
    valid &= data.matchup_id.fillna('').astype(str).str.strip().ne('')
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame()
    data['_review_time'] = reviewed.loc[data.index]
    # First recorded pregame selection per game/model; reruns do not inflate samples.
    data = data.sort_values('_review_time').drop_duplicates(['matchup_id', 'game_start_utc', 'gemini_review_model'])
    data['_outcome'] = data.get('candidate_outcome', data.get('Outcome', pd.Series('', index=data.index))).astype(str).str.upper().replace({'W':'WIN', 'L':'LOSS'})
    rows = []
    for model, group in data.groupby('gemini_review_model'):
        for label, sample in [('All reviewed model selections', group), ('Gemini agrees (subset)', group[group.gemini_agreement.eq('agree')])]:
            wins = int(sample._outcome.eq('WIN').sum())
            losses = int(sample._outcome.eq('LOSS').sum())
            rows.append({'Model':model, 'Sample':label, 'Selections':len(sample), 'Wins':wins, 'Losses':losses,
                         'Win rate': wins/(wins+losses) if wins+losses else None,
                         'Other / unresolved':len(sample)-wins-losses})
    return pd.DataFrame(rows)
