"""Isolated subscriber service.

This package deliberately does not import ``app``, ``app_core`` or the
Streamlit entry points.  The research system is an upstream, read-only
authority for this service.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
