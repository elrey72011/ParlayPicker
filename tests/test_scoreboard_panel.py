"""SYNTHETIC AppTest of the Results-tab scoreboard (roadmap E1.3). Display only."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _script(tmp_path, baseline):
    script = tmp_path / 'SYNTHETIC-scoreboard.py'
    script.write_text('import sys\nsys.path.insert(0, %r)\nfrom pathlib import Path\n'
                      'from app.ui.scoreboard_panel import render_scoreboard\n'
                      'render_scoreboard(baseline=Path(%r))\n' % (str(ROOT), str(baseline)), encoding='utf-8')
    return AppTest.from_file(str(script))


def test_bundled_baseline_renders_a_verdict(tmp_path):
    app = _script(tmp_path, ROOT / 'data' / 'backtest_exports')
    app.run(timeout=60)
    assert not app.exception and not app.metric  # lazy: nothing computed on page load
    app.button(key='scoreboard_compute').click().run(timeout=120)
    assert not app.exception
    text = ' '.join(m.value for m in app.markdown)
    assert 'NO DEMONSTRATED EDGE' in text and 'graded picks' in text
    assert [m.label for m in app.metric] == ['Win rate', 'Model log loss', 'Market log loss', 'ROI / unit']
    assert len(app.dataframe) == 3


def test_empty_folder_reports_no_graded_picks(tmp_path):
    empty = tmp_path / 'SYNTHETIC-empty'
    empty.mkdir()
    app = _script(tmp_path, empty)
    app.run(timeout=60)
    app.button(key='scoreboard_compute').click().run(timeout=60)
    assert not app.exception and any('No graded picks' in i.value for i in app.info)


def test_closing_line_value_needs_an_uploaded_evidence_database(tmp_path):
    app = _script(tmp_path, ROOT / 'data' / 'backtest_exports')
    app.run(timeout=60)
    app.button(key='scoreboard_compute').click().run(timeout=120)
    assert not app.exception
    assert any('Upload the evidence database' in c.value for c in app.caption)
