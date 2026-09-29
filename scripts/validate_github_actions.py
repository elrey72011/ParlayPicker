#!/usr/bin/env python3
"""Validate workflow structure and context availability used by this repo.

This is intentionally stricter than YAML parsing: it walks the workflow
structure and applies GitHub's context-availability boundary for the runner
context.  ``runner`` exists only once a job is running, so it cannot be used
to construct job-level environment values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml


EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
CONTEXT = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\.")


def _location(path: tuple[str, ...]) -> str:
    return ".".join(path) or "<workflow>"


def _runner_available(path: tuple[str, ...]) -> bool:
    return (len(path) >= 5 and path[0] == "jobs" and path[2] == "steps"
            and path[3].isdigit())


def _walk(value, path: tuple[str, ...], diagnostics: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _walk(item, (*path, str(key)), diagnostics)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk(item, (*path, str(index)), diagnostics)
        return
    if not isinstance(value, str):
        return
    for expression in EXPRESSION.findall(value):
        contexts = {match.group(1) for match in CONTEXT.finditer(expression)}
        if "runner" in contexts and not _runner_available(path):
            diagnostics.append(
                f"{_location(path)}: context 'runner' is unavailable outside "
                "jobs.<job_id>.steps[*]"
            )


def validate_workflow(path: str | Path) -> list[str]:
    """Return structural/context diagnostics for one GitHub workflow."""

    source = Path(path)
    try:
        workflow = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return [f"{source}: workflow YAML is invalid: {type(exc).__name__}"]
    if not isinstance(workflow, dict):
        return [f"{source}: workflow root must be a mapping"]
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        return [f"{source}: workflow must declare at least one job"]
    diagnostics: list[str] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            diagnostics.append(f"jobs.{job_id}: job must be a mapping")
            continue
        steps = job.get("steps")
        if steps is not None and not isinstance(steps, list):
            diagnostics.append(f"jobs.{job_id}.steps: steps must be a list")
    _walk(workflow, (), diagnostics)
    return diagnostics


def workflow_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values or [".github/workflows"]:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.yml")))
            paths.extend(sorted(candidate.glob("*.yaml")))
        else:
            paths.append(candidate)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    diagnostics: list[str] = []
    paths = workflow_paths(args.paths)
    if not paths:
        diagnostics.append("no workflow files found")
    for path in paths:
        diagnostics.extend(validate_workflow(path))
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    print(f"validated {len(paths)} GitHub workflow definition(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
