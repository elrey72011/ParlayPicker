"""Synthetic test-process configuration; never inspect owner credentials."""
import os
from pathlib import Path
from streamlit import config
root=Path(os.environ['CAPACITY_LOCAL_SESSION_ROOT']).resolve()
assert root.is_dir()
empty=root/'empty-secrets.toml'
with empty.open('x',encoding='utf-8') as f:f.write('# Intentionally empty synthetic test-process secrets.\n')
config.set_option('secrets.files',[str(empty)])
assert config.get_option('secrets.files')==[str(empty)]
