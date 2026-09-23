"""Book labels allowed for exact quoted public game selections."""
FALLBACK_BOOKS = {'DraftKings', 'FanDuel', 'BetMGM'}


CANONICAL_BOOKS = {
    'novig': 'Novig', 'novig_us': 'Novig',
    'draftkings': 'DraftKings', 'fanduel': 'FanDuel', 'betmgm': 'BetMGM',
}


def canonical_book_label(value):
    """Normalize known sportsbook identities, without authorizing unknown books."""
    label = value.strip() if isinstance(value, str) else ''
    return CANONICAL_BOOKS.get(label.casefold(), label)


def supported_quote(row):
    source = canonical_book_label(row.get('quote_source'))
    if 'quote_time_basis' in row:
        return row['quote_time_basis'] == 'espn_observed' and row.get('sport', '').upper() == 'NCAAF' and source == 'DraftKings'

    return source == 'Novig' or (row.get('sport', '').upper() in {'NCAAF', 'NFL', 'MLB', 'WNBA'} and source in FALLBACK_BOOKS)
