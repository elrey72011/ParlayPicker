"""Eastern operating window, independent of UTC daylight-saving offsets."""
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo


def is_open(at=None):
    at=at or datetime.now(timezone.utc)
    if at.tzinfo is None:raise ValueError("Timezone-aware clock required")
    local=at.astimezone(ZoneInfo("America/New_York"))
    clock=local.time().replace(tzinfo=None)
    return clock>=time(11,45) or clock<time(2,31)
