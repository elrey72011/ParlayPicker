# fix_mojibake.py
from pathlib import Path

SRC = Path("streamlit_app.py")
BAK = Path("streamlit_app.py.bak")

text = SRC.read_text(encoding="utf-8", errors="strict")

# mojibake repair: interpret existing characters as latin-1 bytes, then decode as utf-8
fixed = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")

BAK.write_text(text, encoding="utf-8")
SRC.write_text(fixed, encoding="utf-8")

print("✅ Fixed mojibake. Backup written to streamlit_app.py.bak")
