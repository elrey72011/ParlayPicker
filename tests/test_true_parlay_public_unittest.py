"""Synthetic publication controls; no live price or validation artifact is used."""

import copy
from datetime import datetime, timezone
import json
import sys
import types
import unittest
from unittest.mock import patch

from app_core.true_parlay_public import build_product_board


NOW = datetime(2026, 9, 23, 18, tzinfo=timezone.utc)


def public_row(i, market='spread_away'):
    return dict(sport='MLB', game=f'A{i} at B{i}', pick=f'A{i} +1.5' if market == 'spread_away' else 'Over 8.5',
                market=market, odds=-110, status='PASS', start='2026-09-24T19:00:00+00:00',
                as_of='2026-09-23T17:58:00+00:00', quote_source='Novig',
                quote_time='2026-09-23T17:59:00+00:00')


class TrueParlayPublicTests(unittest.TestCase):
    def test_three_products_and_funnel_reconcile_without_activation(self):
        games = {'overall': [public_row(1), public_row(2)],
                 'sides': [public_row(1), public_row(2)],
                 'totals': [public_row(1, 'total_over'), public_row(2, 'total_over')]}
        records, funnel = build_product_board(games, [], NOW)
        self.assertEqual(funnel['evaluated_by_product'],
                         {'CROSS_GAME_PARLAY': 3, 'SAME_GAME_PARLAY': 2, 'STANDARD_PARLAY': 1})
        self.assertEqual(funnel['counts']['candidate_combinations'], len(records))
        self.assertEqual(funnel['counts']['actionable_now'], 0)
        self.assertTrue(all(r['recommended_stake'] == 0 and not r['production_eligible'] for r in records))
        self.assertTrue(all('PRICE_UNAVAILABLE' in r['blockers'] and 'PRODUCT_UNVALIDATED' in r['blockers']
                            for r in records))
        self.assertEqual((records, funnel), build_product_board(games, [], NOW))

    def test_bounded_generation_reports_truncation(self):
        rows = [public_row(i) for i in range(10)]
        records, funnel = build_product_board({'overall': rows, 'sides': rows, 'totals': rows}, [], NOW, limit=2)
        self.assertEqual(funnel['evaluated_by_product'],
                         {'STANDARD_PARLAY': 2, 'CROSS_GAME_PARLAY': 2})
        self.assertGreater(funnel['truncated']['STANDARD_PARLAY'], 0)
        self.assertEqual(funnel['counts']['candidate_combinations'], len(records))

    def test_package_manifest_parity_and_tamper_rejection(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest('Pandas runtime unavailable')
        from app_core.public_board import build_package, validate_package
        from scripts.publish_board import render, assets_from_html
        base = {'export_run_id': '20260923T180000Z', 'league': 'MLB', 'odds': -110,
                'Bettable': False, 'Play_Stake': 0, 'status': 'PASS',
                'win_probability': .6, 'ev': .14, 'start': '2026-09-24T19:00:00Z',
                'quote_source': 'Novig', 'quote_time': '2026-09-23T17:59:00Z'}
        rows = [dict(base, matchup_id=f'game-{i}', matchup=f'A{i} at B{i}',
                     pick=f'A{i} +1.5', market_type='spread_away') for i in (1, 2)]
        overall = pd.DataFrame(rows)
        totals = pd.DataFrame([dict(row, pick='Over 8.5', market_type='total_over') for row in rows])
        # Timing normalization imports the prediction stack even for zero props.
        # Isolate that unrelated dependency while exercising the real serializer.
        timing = types.ModuleType('app_core.public_prop_timing')
        timing.with_game_starts = lambda props, games: props
        with patch.dict(sys.modules, {'app_core.public_prop_timing': timing}):
            package = build_package(overall, overall, totals)
            validate_package(package)
            assets = assets_from_html(render(package))
        manifest = json.loads(assets['version.json'])
        self.assertEqual(json.loads(assets['board-data.json']), package)
        self.assertEqual(manifest['board_hash'], manifest['build_id'])
        self.assertIn('source_git_sha', manifest)
        self.assertIn('source_fingerprint', manifest)
        self.assertEqual(package['parlay_product_funnel']['counts']['actionable_now'], 0)
        tampered = copy.deepcopy(package)
        tampered['parlay_products'][0]['status'] = 'ACTIONABLE'
        with self.assertRaises(ValueError):
            validate_package(tampered)


if __name__ == '__main__':
    unittest.main()
