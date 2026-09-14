"""Owner validation/recommendation command; no sportsbook execution."""
from _activation_cli import main
if __name__ == "__main__":
    try:
        raise SystemExit(main("validate_sport_deployment"))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("BLOCKED:", str(exc))
        raise SystemExit(1)
