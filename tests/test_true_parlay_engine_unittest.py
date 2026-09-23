"""Synthetic mechanical controls. These fixtures are never validation evidence."""

import copy
from datetime import datetime, timedelta, timezone
import unittest

from core.exposure_ledger import digest
from core.true_parlay_engine import (
    american_price, component_hashes, evaluate_ticket, exposure_leg_key,
    leg_blockers, leg_hash, minimum_acceptable_price, optimize_portfolio,
    settle_ticket, ticket_hash,
)


NOW = datetime(2026, 9, 23, 18, tzinfo=timezone.utc)


def at(minutes):
    return (NOW + timedelta(minutes=minutes)).isoformat()


def leg(i, *, game=None, market='spread_away', selection=None):
    line = 8.5 if market.startswith('total_') else 1.5
    return dict(candidate_id=f'synthetic-c{i}-{market}', game_id=game or f'g{i}',
                sport='MLB', team_ids=[f'A{game or i}', f'B{game or i}'],
                market_type=market, selection=selection or f'A{i} +{line:.1f}', line=line,
                sportsbook='Novig', american_odds=-110, quote_timestamp=at(-1),
                analysis_timestamp=at(-2), start=at(120), identity_verified=True,
                quote_verified=True, production_eligible=True, model_version='m1',
                model_trained_through=at(-10000), model_available_at=at(-100),
                calibration_version='c1', calibration_available_at=at(-100),
                policy_version='s1', evidence_snapshot_id='e1', evidence_frozen_at=at(-3),
                critical_input_state='CLEAR', material_news_status='CLEAR',
                probability_semantics='UNCONDITIONAL', probability_mean=.60,
                probability_conservative=.55, probability_push=0., probability_loss=.40)


def policy(product='STANDARD_PARLAY', state='STANDARD_VALIDATED'):
    return dict(product_type=product, validation_state=state, validation_id=f'{product}-v1',
                policy_version='policy-v1', expires_at=at(100), frozen_at=at(-100),
                minimum_ev=.01, minimum_edge=.02, stake_cap=.01, provisional_cap=.0025,
                kelly_fraction=.5, exposure_caps=dict(sport=.05, leg=.05, parlay=.05,
                                                      sgp=.05, overlap=.05))


def joint(product, components, method='INDEPENDENT_VERIFIED'):
    return dict(product_type=product, component_hashes=component_hashes(components),
                model_id='joint-m1', model_version='j1', model_trained_through=at(-1000),
                model_available_at=at(-100), calibration_version='jc1',
                calibration_available_at=at(-100), validation_id=f'{product}-v1',
                evidence_snapshot_id='je1', evidence_frozen_at=at(-5),
                method=method, generated_at=at(-1), probability_semantics='UNCONDITIONAL',
                probability_mean=.36, probability_conservative=.35,
                probability_push=0., probability_loss=.64)


def quote(product, components, book='Novig', price=3.5):
    legs = [r for c in components for r in (c['legs'] if c.get('product_type') == 'SAME_GAME_PARLAY' else [c])]
    return dict(quote_id='synthetic-q1', provider_ticket_id='synthetic-provider-ticket',
                source='SPORTSBOOK', verification_state='VERIFIED', sportsbook=book,
                ticket_hash=ticket_hash(product, components, book),
                leg_hashes=sorted(leg_hash(r) for r in legs), quoted_at=at(-1),
                expires_at=at(10), decimal_odds=price,
                american_odds=round((price - 1) * 100) if price >= 2 else -round(100 / (price - 1)))


def exposure(used=None):
    result = dict(as_of=at(0), bankroll=1000., unit_value=10., currency='USD',
                  committed=used or {}, ledger_hash='synthetic-ledger', total_cap=.05,
                  daily_cap=.05, weekly_cap=.05, game_cap=.02, team_cap=.02)
    result['snapshot_hash'] = digest(result)
    return result


def case(product='STANDARD_PARLAY', components=None, method='INDEPENDENT_VERIFIED'):
    components = components or [leg(1), leg(2)]
    return dict(product_type=product, components=components, sportsbook='Novig',
                quote=quote(product, components), joint=joint(product, components, method),
                dependence=dict(classification='INDEPENDENT_VERIFIED', assessed_at=at(-2),
                                expires_at=at(30), evidence_id='dep1'),
                policy=policy(product), authorization=dict(status='ACTIVE',
                    product_types=[product], authorization_id='synthetic-owner-consent',
                    granted_at=at(-100), expires_at=at(100)),
                exposure=exposure(), review=dict(status='CONFIRM'), now=NOW)


class TrueParlayEngineTests(unittest.TestCase):
    def evaluate(self, value, **updates):
        value = copy.deepcopy(value)
        value.update(updates)
        return evaluate_ticket(**value)

    def test_standard_positive_control_requires_every_gate(self):
        result = self.evaluate(case())
        self.assertEqual(result['status'], 'ACTIONABLE')
        self.assertGreater(result['recommended_stake'], 0)
        self.assertEqual(result['blockers'], [])
        self.assertFalse(result['automated_bet_placement'])

    def test_any_stale_or_unvalidated_leg_fails(self):
        for change, reason in (({'quote_timestamp': at(-31)}, 'LEG_QUOTE_STALE'),
                               ({'analysis_timestamp': at(-31)}, 'LEG_ANALYSIS_STALE'),
                               ({'production_eligible': False}, 'LEG_NOT_PRODUCTION_ELIGIBLE'),
                               ({'start': at(-1)}, 'LEG_STARTED_OR_TIME_UNKNOWN')):
            with self.subTest(change=change):
                value = case()
                value['components'][0].update(change)
                result = self.evaluate(value)
                self.assertIn(reason, result['blockers'])
                self.assertEqual(result['recommended_stake'], 0)

    def test_standard_duplicate_game_and_leg(self):
        value = case(components=[leg(1), leg(2, game='g1')])
        self.assertIn('DUPLICATE_GAME', self.evaluate(value)['blockers'])
        value = case(components=[leg(1), leg(1)])
        self.assertIn('DUPLICATE_LEG', self.evaluate(value)['blockers'])

    def test_product_leg_count_bounds(self):
        self.assertNotIn('LEG_COUNT_INVALID', self.evaluate(case(components=[leg(i) for i in range(1, 5)]))['blockers'])
        self.assertIn('LEG_COUNT_INVALID', self.evaluate(case(components=[leg(i) for i in range(1, 6)]))['blockers'])
        self.assertNotIn('LEG_COUNT_INVALID', self.evaluate(case('CROSS_GAME_PARLAY', [leg(i) for i in range(1, 7)],
                                                                 'JOINT_COMPONENT_MODEL'))['blockers'])
        self.assertIn('LEG_COUNT_INVALID', self.evaluate(case('CROSS_GAME_PARLAY', [leg(i) for i in range(1, 8)],
                                                              'JOINT_COMPONENT_MODEL'))['blockers'])

    def test_sgp_requires_joint_model_and_actual_ticket_quote(self):
        components = [leg(1, game='g1'), leg(2, game='g1', market='total_over', selection='Over 8.5')]
        value = case('SAME_GAME_PARLAY', components, 'COPULA')
        self.assertEqual(self.evaluate(value)['status'], 'ACTIONABLE')
        value['joint'] = None
        missing = self.evaluate(value)
        self.assertIn('JOINT_MODEL_UNAVAILABLE', missing['blockers'])
        self.assertIsNone(missing['probability_partial'])
        value = case('SAME_GAME_PARLAY', components, 'COPULA')
        value['quote'] = None
        self.assertIn('PRICE_UNAVAILABLE', self.evaluate(value)['blockers'])
        value = case('SAME_GAME_PARLAY', components, 'INDEPENDENCE_PRODUCT')
        self.assertIn('SGP_INDEPENDENCE_FORBIDDEN', self.evaluate(value)['blockers'])

    def test_cross_game_sgp_is_one_component_block(self):
        sgp_legs = [leg(1, game='g1'), leg(2, game='g1', market='total_over', selection='Over 8.5')]
        sgp_case = case('SAME_GAME_PARLAY', sgp_legs, 'COPULA')
        sgp = self.evaluate(sgp_case)
        component = dict(product_type='SAME_GAME_PARLAY', parlay_id=sgp['parlay_id'],
                         ticket_hash=sgp['ticket_hash'], sportsbook='Novig', legs=sgp_legs,
                         production_eligible=True, status='ACTIONABLE',
                         validation_id=sgp['validation_id'], validation_state=sgp['validation_state'],
                         quote_verification_state='VERIFIED', quoted_at=at(-1), expires_at=at(10), blockers=[])
        value = case('CROSS_GAME_PARLAY', [component, leg(3)], 'JOINT_COMPONENT_MODEL')
        self.assertEqual(self.evaluate(value)['status'], 'ACTIONABLE')
        value['components'][0]['legs'][0]['line'] = 2.5
        self.assertIn('SGP_COMPONENT_HASH_MISMATCH', self.evaluate(value)['blockers'])
        value = case('CROSS_GAME_PARLAY', [component, leg(3)], 'JOINT_COMPONENT_MODEL')
        value['components'][0]['production_eligible'] = False
        self.assertIn('SGP_COMPONENT_UNVALIDATED', self.evaluate(value)['blockers'])

    def test_quote_identity_price_and_time_fail_closed(self):
        for mutation in ({'ticket_hash': 'wrong'}, {'leg_hashes': []},
                         {'sportsbook': 'FanDuel'}, {'quoted_at': at(1)},
                         {'quoted_at': at(-31)}, {'expires_at': at(-1)},
                         {'decimal_odds': float('nan')}, {'verification_state': 'PENDING'},
                         {'provider_ticket_id': None}):
            with self.subTest(mutation=mutation):
                value = case()
                value['quote'].update(mutation)
                self.assertIn('PRICE_UNAVAILABLE', self.evaluate(value)['blockers'])

    def test_line_selection_market_and_book_change_hash(self):
        value = case()
        original = value['quote']['ticket_hash']
        for change in ({'line': 2.5}, {'selection': 'B1 -1.5'},
                       {'market_type': 'total_over'}, {'sportsbook': 'FanDuel'}):
            changed = copy.deepcopy(value)
            changed['components'][0].update(change)
            self.assertNotEqual(ticket_hash('STANDARD_PARLAY', changed['components'], 'Novig'), original)
            self.assertIn('PRICE_UNAVAILABLE', self.evaluate(changed)['blockers'])

    def test_joint_input_binding_blocks_changed_line_even_at_better_price(self):
        value = case()
        value['components'][0]['line'] = 2.5
        value['quote'] = quote('STANDARD_PARLAY', value['components'], price=4.0)
        self.assertIn('JOINT_INPUT_MISMATCH', self.evaluate(value)['blockers'])

    def test_positive_legs_do_not_imply_positive_ticket(self):
        value = case()
        value['joint'].update(probability_mean=.22, probability_conservative=.20,
                              probability_push=0., probability_loss=.78)
        self.assertIn('NONPOSITIVE_CONSERVATIVE_EV', self.evaluate(value)['blockers'])
        self.assertEqual(self.evaluate(value)['recommended_stake'], 0)

    def test_dependence_unknown_and_material_need_validated_joint(self):
        for classification in ('UNKNOWN', 'MATERIAL_DEPENDENCE'):
            value = case()
            value['dependence']['classification'] = classification
            value['joint'] = None
            self.assertIn('DEPENDENCE_JOINT_MODEL_REQUIRED', self.evaluate(value)['blockers'])

    def test_review_only_reduces_and_cannot_override_failures(self):
        value = case()
        base = self.evaluate(value)['recommended_stake']
        value['review'] = dict(status='REDUCE', stake_multiplier=.25,
                               probability_conservative=1.0, stake_cap=1.0)
        self.assertLessEqual(self.evaluate(value)['recommended_stake'], base)
        value['policy']['validation_state'] = 'UNVALIDATED'
        self.assertEqual(self.evaluate(value)['recommended_stake'], 0)
        value['review'] = dict(status='HOLD')
        self.assertIn('REVIEW_HOLD', self.evaluate(value)['blockers'])

    def test_straight_exposure_and_overlap_reduce_capacity(self):
        value = case()
        used = {f'game:MLB:g1': .019, f'overlap:{exposure_leg_key(value["components"][0])}': .019}
        value['exposure'] = exposure(used)
        value['policy']['exposure_caps']['overlap'] = .02
        self.assertLessEqual(self.evaluate(value)['recommended_stake'], 1.01)
        self.assertGreater(self.evaluate(value)['recommended_stake'], 0)

    def test_full_ticket_counts_under_every_game_team_bucket(self):
        value = case()
        value['exposure'] = exposure({'team:MLB:A1': .02})
        result = self.evaluate(value)
        self.assertEqual(result['recommended_stake'], 0)
        self.assertIn('EXPOSURE_OR_KELLY_EXHAUSTED', result['blockers'])

    def test_missing_nan_future_artifacts_fail(self):
        for field, bad in (('model_version', None), ('calibration_version', None),
                           ('evidence_snapshot_id', None), ('probability_mean', float('nan')),
                           ('model_available_at', at(1)), ('calibration_available_at', at(1)),
                           ('evidence_frozen_at', at(1))):
            with self.subTest(field=field):
                value = case()
                value['components'][0][field] = bad
                self.assertEqual(self.evaluate(value)['recommended_stake'], 0)

    def test_price_floor_and_move_consistency(self):
        value = case()
        floor = minimum_acceptable_price(.45, .05, .01, .02)
        self.assertAlmostEqual(floor, .95 / .43)
        self.assertIsNotNone(american_price(floor))
        value['quote'] = quote('STANDARD_PARLAY', value['components'], price=2.0)
        result = self.evaluate(value)
        self.assertIn('PRICE_MOVED', result['blockers'])
        self.assertEqual(result['recommended_stake'], 0)
        value['policy']['validation_state'] = 'UNVALIDATED'
        value['quote'] = quote('STANDARD_PARLAY', value['components'], price=4.0)
        self.assertIn('PRODUCT_UNVALIDATED', self.evaluate(value)['blockers'])

    def test_zero_bankroll_missing_consent_and_product_state(self):
        value = case()
        value['authorization'] = None
        self.assertIn('OWNER_AUTHORIZATION_MISSING', self.evaluate(value)['blockers'])
        value = case()
        value['exposure'] = None
        self.assertIn('EXPOSURE_UNAVAILABLE', self.evaluate(value)['blockers'])
        value = case()
        value['policy']['validation_state'] = 'PROVISIONAL_VALIDATED'
        self.assertLessEqual(self.evaluate(value)['recommended_fraction'], .0025)
        value['policy']['validation_state'] = 'UNVALIDATED'
        self.assertEqual(self.evaluate(value)['recommended_stake'], 0)

    def test_other_product_validation_cannot_activate(self):
        value = case()
        value['policy']['product_type'] = 'SAME_GAME_PARLAY'
        self.assertIn('PRODUCT_UNVALIDATED', self.evaluate(value)['blockers'])
        value = case()
        value['policy']['validation_id'] = 'straight-v1'
        self.assertIn('JOINT_VALIDATION_MISMATCH', self.evaluate(value)['blockers'])

    def test_portfolio_order_and_pruning_diagnostic(self):
        first = case()
        second = case(components=[leg(1), leg(3)])
        second['joint']['probability_conservative'] = .34
        original = copy.deepcopy([first, second])
        output = optimize_portfolio([second, first], exposure(), now=NOW, max_tickets=1)
        self.assertEqual(output['records'][0]['parlay_id'], ticket_hash('STANDARD_PARLAY', first['components'], 'Novig'))
        self.assertEqual(output['diagnostics']['truncated'], 1)
        self.assertEqual([first, second], original)
        self.assertEqual(optimize_portfolio([], exposure(), now=NOW)['records'], [])

    def test_push_void_settlement_requires_captured_reduced_price(self):
        self.assertEqual(settle_ticket(['WIN', 'LOSS'], 3, 10)['net_return'], -10)
        self.assertEqual(settle_ticket(['PUSH', 'VOID'], 3, 10)['net_return'], 0)
        self.assertEqual(settle_ticket(['WIN', 'PUSH'], 3, 10)['status'], 'NEEDS_REVIEW')
        self.assertEqual(settle_ticket(['WIN', 'PUSH'], 3, 10, reduced_decimal_odds=1.9)['net_return'], 9)
        self.assertEqual(settle_ticket(['WIN', 'VOID'], 3, 10, void_rule='UNKNOWN')['status'], 'NEEDS_REVIEW')

    def test_integer_push_needs_joint_outcome_and_actual_reduced_payout(self):
        value = case()
        first = value['components'][0]
        first.update(line=1.0, selection='A1 +1.0', probability_push=.05,
                     probability_loss=.35)
        value['joint'] = joint('STANDARD_PARLAY', value['components'], 'JOINT_MODEL')
        value['joint'].update(probability_loss=.59, partial_outcomes=[
            dict(state_id='first_push_second_win', probability_mean=.05,
                 probability_conservative=.04)])
        value['quote'] = quote('STANDARD_PARLAY', value['components'])
        self.assertIn('SETTLEMENT_MODEL_UNAVAILABLE', self.evaluate(value)['blockers'])
        value['joint']['settlement_rules_id'] = 'synthetic-rule-v1'
        value['quote']['settlement_rules_id'] = 'synthetic-rule-v1'
        value['quote']['partial_payouts'] = [dict(state_id='first_push_second_win', decimal_return=1.9)]
        result = self.evaluate(value)
        self.assertEqual(result['status'], 'ACTIONABLE')
        self.assertAlmostEqual(result['conservative_ev'], .35 * 3.5 + .04 * 1.9 - 1)
        self.assertGreater(result['minimum_acceptable_decimal'], 1)

    def test_missing_identity_and_probability_mass(self):
        sample = leg(1)
        sample['candidate_id'] = None
        self.assertIn('LEG_IDENTITY_MISSING', leg_blockers(sample, NOW))
        sample = leg(1)
        sample['team_ids'] = ['A1', 'A1']
        self.assertIn('LEG_TEAM_IDENTITY_MISSING', leg_blockers(sample, NOW))
        sample = leg(1)
        sample['probability_push'] = .2
        self.assertIn('LEG_PROBABILITY_INVALID', leg_blockers(sample, NOW))
        sample = leg(2)
        sample.update(probability_push=.05, probability_loss=.35)
        self.assertIn('HALF_POINT_PUSH_IMPOSSIBLE', leg_blockers(sample, NOW))

    def test_malformed_external_inputs_fail_closed(self):
        value = case()
        value.update(components='not-a-list', policy='invalid', authorization='invalid',
                     review='invalid', dependence='invalid', joint='invalid')
        result = self.evaluate(value)
        self.assertIn('COMPONENTS_INVALID', result['blockers'])
        self.assertEqual(result['recommended_stake'], 0)
        value = case()
        value['components'] = [{'product_type': 'SAME_GAME_PARLAY', 'legs': 'invalid'}]
        self.assertIn('COMPONENTS_INVALID', self.evaluate(value)['blockers'])
        value = case()
        value['authorization']['product_types'] = 'STANDARD_PARLAY'
        self.assertIn('OWNER_AUTHORIZATION_MISSING', self.evaluate(value)['blockers'])

    def test_mutually_exclusive_opposites_rejected(self):
        home = leg(1, game='g1', market='spread_home', selection='A1 -1.5')
        away = leg(2, game='g1', market='spread_away', selection='B2 +1.5')
        home['line'] = -1.5
        away['line'] = 1.5
        value = case('SAME_GAME_PARLAY', [home, away], 'COPULA')
        self.assertIn('MUTUALLY_EXCLUSIVE_LEGS', self.evaluate(value)['blockers'])
        over = leg(1, game='g1', market='total_over', selection='Over 9.5')
        under = leg(2, game='g1', market='total_under', selection='Under 8.5')
        over['line'], under['line'] = 9.5, 8.5
        value = case('SAME_GAME_PARLAY', [over, under], 'COPULA')
        self.assertIn('MUTUALLY_EXCLUSIVE_LEGS', self.evaluate(value)['blockers'])


if __name__ == '__main__':
    unittest.main()
