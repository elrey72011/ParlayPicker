"""Freshness policy for current quotes and newly published boards."""
QUOTE_MAX_AGE_MINUTES = 30
QUOTE_MAX_AGE_SECONDS = QUOTE_MAX_AGE_MINUTES * 60


def package_age_minutes(package):
    """Honor archived publication policy; missing legacy values mean 15 minutes."""
    value = package.get("stale_after_minutes", 15)
    if isinstance(value, bool) or value not in (15, QUOTE_MAX_AGE_MINUTES):
        raise ValueError("Unsupported quote freshness policy")
    return value
