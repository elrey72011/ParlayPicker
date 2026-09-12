"""Book labels allowed for exact quoted public game selections."""
COLLEGE_BOOKS = {'DraftKings', 'FanDuel', 'BetMGM'}


def supported_quote(row):
    source = row.get('quote_source')
    if 'quote_time_basis' in row:
        return row['quote_time_basis'] == 'espn_observed' and row.get('sport', '').upper() == 'NCAAF' and source == 'DraftKings'

    return source == 'Novig' or (row.get('sport', '').upper() == 'NCAAF' and source in COLLEGE_BOOKS)
