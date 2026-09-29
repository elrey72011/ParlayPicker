#!/usr/bin/env python3
"""Run one authenticated, bounded slice of the read-only evidence census."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from app_core.evidence_drive import DriveStore
from app_core.evidence_remote import settings
from app_core.read_only_census import SCHEMA, run_census


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    parser.add_argument("--max-objects", type=int, default=4000)
    parser.add_argument("--max-bytes", type=int, default=500_000_000)
    parser.add_argument("--deadline-seconds", type=int, default=2400)
    parser.add_argument("--full-verify", action="store_true")
    return parser.parse_args(argv)


def _failure_output(path, checkpoint_path, source_revision, exc):
    # Do not serialize secrets, request objects, environment values, or a full
    # traceback.  The exception class and a bounded message are sufficient for
    # an operational blocker without leaking credentials.
    value = {
        "schema": SCHEMA,
        "execution_kind": "AUTHENTICATED_READ_ONLY_REMOTE_CENSUS",
        "status": "BLOCKED",
        "source_revision": source_revision,
        "terminal_reason": type(exc).__name__,
        "sanitized_error": "EXTERNAL_CONFIGURATION_OR_RUNTIME_FAILURE",
        "checkpoint_retained": Path(checkpoint_path).is_file(),
        "hard_kill_artifact_retention": "NOT_GUARANTEED",
        "launch_authority": "NONE",
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    try:
        folder, _ = settings()
        report = run_census(
            DriveStore(folder), checkpoint_path=args.checkpoint,
            output_path=args.output, source_revision=args.source_revision,
            max_objects=args.max_objects, max_bytes=args.max_bytes,
            deadline_seconds=args.deadline_seconds, full_verify=args.full_verify,
            progress=lambda item: print(json.dumps(item, sort_keys=True), flush=True))
    except (Exception, KeyboardInterrupt) as exc:
        _failure_output(args.output, args.checkpoint, args.source_revision, exc)
        print(f"read-only census blocked: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"],
                      "terminal_reason": report["terminal_reason"],
                      "output": args.output, "checkpoint": args.checkpoint},
                     sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
