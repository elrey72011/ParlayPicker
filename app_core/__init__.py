"""Core modules shared between the Streamlit app and auxiliary scripts.

Keep package init lightweight to avoid reload/import-order failures in Streamlit.
Do not add eager imports here.
"""

__all__: list[str] = []
