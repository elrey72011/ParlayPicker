from __future__ import annotations

import subprocess

from scripts import refresh_calibration


def test_bucket_refit_continues_when_global_calibration_promotion_is_blocked(
    monkeypatch, tmp_path
):
    calls: list[str] = []
    results = iter(
        [
            subprocess.CompletedProcess([], 2),
            subprocess.CompletedProcess([], 0),
        ]
    )

    def fake_run(command, *, check):
        calls.append(str(command[1]))
        result = next(results)
        if check:
            result.check_returncode()
        return result

    monkeypatch.setattr(refresh_calibration.subprocess, "run", fake_run)

    promoted = refresh_calibration._run_refits(tmp_path)

    assert promoted is False
    assert calls[0].endswith("fit_calibration.py")
    assert calls[1].endswith("fit_bucket_stats.py")
