"""Unapproved, read-only policy proposal. Never replaces the paid-launch guard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import types

GUARD = "scripts/check_launch_change_scope.py"
MANIFEST = "docs/paid-launch/launch-baseline-manifest.json"
TEST = "tests/test_board_diagnostics.py"


@dataclass(frozen=True)
class Binding:
    base: str
    head: str
    before_blob: str
    after_blob: str
    manifest_sha256: str
    guard_sha256: str


# Exact correction requested by the owner; approval of this proposal is separate.
PROPOSED = Binding(
    base="d8f580734c28b712f93e0e4a647e9b21ab1f2928",
    head="dc211cc9438390c73848ce1a43d512ada00338d8",
    before_blob="cfa07b6b6c622f083cb7d2d7e3a0713780758013",
    after_blob="e610143aff5611235f9cfb44da13e2d54e1c6b48",
    manifest_sha256="2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343",
    guard_sha256="d9e4b9c3954d7b1803d23527a77b7e7c34ed5d73e5936d7fe07111da3c930ee8",
)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.PIPE)


def _text(repo: Path, *args: str) -> str:
    return _git(repo, *args).decode("utf-8").strip()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assess(repo: Path, binding: Binding) -> dict:
    """Internal fixture seam; the public entry point has no policy override."""
    report = {"proposal_status": "REJECTED", "approval": "REQUIRED",
              "gate_authority": False, "reasons": [], "original_guard": None}
    try:
        head = _text(repo, "rev-parse", "HEAD")
        base = _text(repo, "rev-parse", binding.base + "^{commit}")
        target = _text(repo, "rev-parse", binding.head + "^{commit}")
        if head != target:
            parents = _text(repo, "show", "-s", "--format=%P", "HEAD").split()
            if (parents != [base, target]
                    or _text(repo, "rev-parse", "HEAD^{tree}") !=
                    _text(repo, "rev-parse", target + "^{tree}")):
                raise ValueError("CANDIDATE_IDENTITY_NOT_EXACT")
        if (_text(repo, "diff", "--name-only") or
                _text(repo, "diff", "--cached", "--name-only")):
            raise ValueError("TRACKED_CHECKOUT_DIRTY")
        if (_text(repo, "rev-parse", base + ":" + TEST) != binding.before_blob
                or _text(repo, "rev-parse", target + ":" + TEST) != binding.after_blob):
            raise ValueError("TEST_BLOB_PAIR_NOT_EXACT")
        canonical = {}
        for path, expected in ((MANIFEST, binding.manifest_sha256),
                               (GUARD, binding.guard_sha256)):
            raw = _git(repo, "show", "HEAD:" + path)
            if (_sha(raw) != expected or
                    _git(repo, "show", base + ":" + path) != raw):
                raise ValueError("ORIGINAL_BASELINE_OR_GUARD_CHANGED")
            if (repo / path).read_bytes().replace(b"\r\n", b"\n") != raw:
                raise ValueError("BASELINE_OR_GUARD_CHECKOUT_CHANGED")
            canonical[path] = raw
        manifest = json.loads(canonical[MANIFEST])
        conversions = []
        for path, expected in manifest["tooling_sha256"].items():
            blob = _git(repo, "show", "HEAD:" + path)
            checkout = (repo / path).read_bytes()
            if _sha(blob) != expected or checkout.replace(b"\r\n", b"\n") != blob:
                raise ValueError("TOOLING_SUBSTANTIVE_MISMATCH")
            if checkout != blob:
                conversions.append(path)
        # Execute only the verified original guard, with its original manifest.
        # Its result stays intact, including raw checkout hash failures.
        legacy = types.ModuleType("verified_original_scope_guard")
        legacy.__file__ = str(repo / GUARD)
        exec(compile(canonical[GUARD], legacy.__file__, "exec"), legacy.__dict__)
        legacy.ROOT = repo
        _, original = legacy.run(repo / MANIFEST, base)
        report["original_guard"] = original
        report["checkout_line_ending_conversions"] = conversions
        allowed_reasons = {"EXISTING_TEST_EXPECTATION_CHANGED"}
        if conversions:
            allowed_reasons.add("SCOPE_GUARD_TOOLING_HASH_MISMATCH")
        if (set(original["reason_codes"]) != allowed_reasons or
                original["existing_test_changes"] != [TEST] or
                set(original["tooling_hash_mismatches"]) != set(conversions)):
            raise ValueError("OTHER_GUARD_REJECTION_OR_TEST_CHANGE")
        report["proposal_status"] = "MATCHED_FOR_REVIEW"
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        report["reasons"].append(str(exc))
    return report


def assess(repo: Path) -> dict:
    return _assess(repo.resolve(), PROPOSED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(args.repo), indent=2, sort_keys=True))
    # Even an exact match requires review; this executable cannot green a gate.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
