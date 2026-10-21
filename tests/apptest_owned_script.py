"""Public AppTest construction from a script owned by the calling test fixture."""
from pathlib import Path
from textwrap import dedent


def app_from_owned_script(directory, code, *, default_timeout=3):
    from streamlit.testing.v1 import AppTest

    # The fixture owns this directory through every run/rerun and its teardown.
    # A missing owner is an error: do not recreate it or use Streamlit's TMP_DIR.
    script = Path(directory) / "app.py"
    script.write_text(dedent(code), encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=default_timeout)
