#!/usr/bin/env python3
"""Emit a read-only source trace for probability math and release consumers."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKED_SYMBOLS = frozenset(
    {
        "apply_bucket_calibration",
        "apply_calibration",
        "calibration_provenance",
        "conditional_probabilities",
        "fit_isotonic_calibration",
        "load_calibration",
        "price_value",
        "verify_reviewed_submission",
    }
)


def build_trace(root: Path = ROOT) -> dict:
    calls: dict[str, list[dict[str, object]]] = {
        symbol: [] for symbol in sorted(TRACKED_SYMBOLS)
    }
    parse_errors: list[dict[str, str]] = []
    excluded_parts = {
        ".git", ".venv", "archive", "node_modules", "outputs", "site-packages",
        "test-results",
    }
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if excluded_parts.intersection(relative.parts) or any(
            part.startswith(".") for part in relative.parts
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append(
                {"path": relative.as_posix(), "error": type(exc).__name__}
            )
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in calls:
                calls[name].append(
                    {
                        "path": relative.as_posix(),
                        "line": int(node.lineno),
                    }
                )
    return {
        "schema_version": 1,
        "trace_kind": "static_python_call_sites",
        "read_only": True,
        "tracked_symbols": sorted(TRACKED_SYMBOLS),
        "calls": calls,
        "parse_errors": parse_errors,
        "qualification_authority": "integrations/subscriber_release/authority.py",
        "subscriber_contract": "services/subscriber/contracts.py",
        "calibration_runtime": "core/probability_calibration.py",
        "notes": [
            "This trace locates source-level consumers; it does not activate calibration or markets.",
            "Dynamic imports and non-Python consumers require separate runtime evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    payload = build_trace()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
