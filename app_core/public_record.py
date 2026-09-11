"""Owner-selected public record epoch; historical Drive records stay intact."""
START_DATE = '2026-09-11'


def current_records(rows):
    """Use the recorded Eastern game date, not publication or grading time."""
    return [row for row in rows if row['date'] >= START_DATE]
