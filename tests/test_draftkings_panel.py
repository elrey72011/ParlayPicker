import ast
from pathlib import Path
from streamlit.testing.v1 import AppTest


def app():
    from app.ui.draftkings import render_draftkings_builders
    render_draftkings_builders({})


def test_builders_visible_without_props_or_uploads():
    at = AppTest.from_function(app).run()
    assert not at.exception
    assert len(at.expander) == 2
    assert "NFL Classic" in at.expander[0].label
    assert "MLB Classic" in at.expander[1].label
    assert len(at.get("file_uploader")) == 4
    assert len(at.checkbox) == 2


def test_builder_call_is_outside_prop_condition():
    tree = ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8-sig"))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "render_draftkings_builders"]
    assert len(calls) == 1
    node = calls[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            assert "prop_card" not in ast.unparse(node.test)
