"""Submit or inspect non-authoritative Gemini research Batch API jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.gemini_research_batch import get_research_batch, submit_research_batch


def _summary(job):
    state = getattr(getattr(job, "state", None), "name", None)
    return {
        "name": getattr(job, "name", None),
        "display_name": getattr(job, "display_name", None),
        "state": state or str(getattr(job, "state", "") or ""),
        "error": str(getattr(job, "error", "") or ""),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--input", type=Path, required=True)
    submit.add_argument("--display-name")
    submit.add_argument("--model")
    status = subparsers.add_parser("status")
    status.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    if args.command == "submit":
        rows = json.loads(args.input.read_text(encoding="utf-8"))
        job = submit_research_batch(
            rows,
            display_name=args.display_name,
            model=args.model,
        )
    else:
        job = get_research_batch(args.name)
    print(json.dumps(_summary(job), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
