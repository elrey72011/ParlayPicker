"""The second and third clean-up passes of 18 November 2026 (owner): library files nothing used are gone, and nothing
asks for them.

Second pass: the runner behind the removed Run Player Props button, the scoring behind the removed Strategy Lab tab, and
code from July that no entry point reached. Third pass (the owner's word, because a safety test ran through it): the
grading behind the removed Results tab, and the operations module of the shadow pipeline retired under D10. Files the
pipeline loads through a package import stay, and so does anything a binding names. An old feature builder and the
football evidence freeze stay too: tests/test_wager_integrity_audit.py runs through them, and that file must keep its
name because a frozen browser test imports a helper from it.
"""
import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED = ("app_core.prop_analysis", "app_core.strategy_lab_realized", "app_core.diagnostics", "app_core.new_summary_logic",
           "app_core.moneyline_parlay", "core.ev_engine", "core.models", "core.models.analysis_models",
           "core.models.game_models", "core.models.prediction_models",
           # third pass
           "app_core.performance_pipeline", "app_core.results_ingestion", "app_core.export_utils", "app_core.results_fetcher",
           "app_core.pipeline.operations")
# Every place tracked Python lives, except the archive of retired tests.
CODE = ("app", "app_core", "core", "scripts", "services", "integrations", "parlaypicker", "tools", "tests", "docs")


def _imported(path):
    """Dotted names a file imports, with `from package import name` read as package.name too."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="ignore"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.add(node.module or "")
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_the_removed_files_are_gone():
    for name in REMOVED:
        assert not (ROOT / (name.replace(".", "/") + ".py")).exists(), name
    assert not (ROOT / "core" / "models" / "__init__.py").exists()
    for name in REMOVED:
        if name != "core.models" and not name.startswith("core.models."):
            assert importlib.util.find_spec(name) is None, name


def test_nothing_outside_the_archive_imports_them():
    files = [*sorted(ROOT.glob("*.py")), *(path for top in CODE for path in sorted((ROOT / top).rglob("*.py")))]
    assert len(files) > 1000
    for path in files:
        asked = _imported(path) & set(REMOVED)
        assert not asked, (path.relative_to(ROOT).as_posix(), sorted(asked))


def test_the_schema_files_the_pipeline_loads_through_its_package_stay():
    # A first count listed these three as unused. Importing core.schema.base_schema, as the pipeline does, runs the
    # package's __init__, which imports all three: removing them would stop the pipeline from importing.
    from core.schema.base_schema import ensure_base_schema
    assert callable(ensure_base_schema)
    assert {"core.schema.schema_registry", "core.schema.schema_utils", "core.schema.schema_validator"} <= set(sys.modules)
    for name in ("schema_registry", "schema_utils", "schema_validator"):
        assert (ROOT / "core" / "schema" / f"{name}.py").exists(), name


def test_the_evidence_store_is_still_graded_by_the_daily_job():
    # The removed Results code also wrote final scores into the prediction evidence store. The route that remains is the
    # daily Activation grading job: bootstrap_evidence.py --grade calls the store's own refresh_outcomes.
    from app_core import prediction_evidence
    assert callable(prediction_evidence.refresh_outcomes) and callable(prediction_evidence.record_scores)
    assert "refresh_outcomes" in (ROOT / "scripts" / "bootstrap_evidence.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "activation-grading.yml").read_text(encoding="utf-8")
    assert "scripts/bootstrap_evidence.py --restore --grade --backup" in workflow and "schedule:" in workflow
