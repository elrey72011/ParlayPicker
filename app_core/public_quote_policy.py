"""Book labels allowed for exact quoted public game selections."""
COLLEGE_BOOKS = {'DraftKings', 'FanDuel', 'BetMGM'}


def supported_quote(row):
    source = row.get('quote_source')
    return source == 'Novig' or (row.get('sport', '').upper() == 'NCAAF' and source in COLLEGE_BOOKS)
