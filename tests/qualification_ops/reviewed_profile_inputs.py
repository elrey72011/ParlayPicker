"""Authenticate the predecessor workflow for the frozen capacity assertion."""
from pathlib import Path
from scripts import ncaaf_filtered_history_scope as history


def reviewed_workflow_bytes(root, relative):
    root = Path(root)
    binding = history.authenticate(root)
    current = history.physical(root / relative)
    # This test checks an earlier capacity/workflow contract. Its expectation is
    # unchanged; the live pin/upload successor is separately checked and sealed.
    return history.capacity_predecessor(binding, relative, current)
