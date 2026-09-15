"""Train an opt-in research challenger from audited pregame receipt JSON."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.mlb_spread_total_model import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-through", required=True, help="Last Eastern development date")
    parser.add_argument("--validation-through", required=True, help="Last Eastern validation date")
    parser.add_argument("--output", type=Path, default=Path("models/mlb_spread_total"))
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.dataset.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("nonempty point-in-time receipt list required")
        path = train(payload, args.output, source=args.source,
                     train_through=args.train_through, validation_through=args.validation_through)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Training blocked: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
