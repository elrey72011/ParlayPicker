"""Shared, sport-neutral pipeline contract (roadmap M1).

Modules here never perform provider requests, inference or schema setup on
import. Sport-specific behavior lives in adapters; durable state lives in
``store``.
"""
