const assert = require('node:assert/strict');
const fs = require('node:fs');

const html = fs.readFileSync('publishing/board.html', 'utf8');
const start = html.indexOf('const PARLAY_PRODUCT_NAMES=');
const end = html.indexOf("document.getElementById('parlayProductFilter').addEventListener", start);
assert.ok(start > 0 && end > start, 'product helpers remain in the public template');
for (const value of ['', 'STANDARD_PARLAY', 'SAME_GAME_PARLAY', 'CROSS_GAME_PARLAY']) {
  assert.ok(html.includes(`<option value="${value}">`), `${value || 'All'} filter is present`);
}

class Element {
  constructor(tag, label = '') { this.tag = tag; this.textContent = String(label); this.children = []; this.value = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
}
const elements = Object.fromEntries(['parlayProductFilter', 'parlayProductFunnel', 'parlayProductRows', 'parlayProductCurrent'].map(id => [id, new Element('div')]));
const document = { getElementById: id => elements[id] };
const el = (tag, label) => new Element(tag, label);
const data = { parlay_products: [], parlay_product_funnel: { counts: { candidate_combinations: 3, actionable_now: 1 }, blockers: { PRICE_UNAVAILABLE: 2 }, potential_combinations: { STANDARD_PARLAY: 4 }, evaluated_by_product: { STANDARD_PARLAY: 3 }, truncated: { STANDARD_PARLAY: 1 } } };
const code = html.slice(start, end);
const api = new Function('data', 'document', 'el', 'setTimeout', 'clearTimeout', `${code};return {parlayProductView,parlayProductRows,renderParlayProducts};`)(data, document, el, () => 1, () => {});
const text = node => [node.textContent, ...node.children.map(text)].join(' ');

const now = Date.parse('2030-01-01T12:00:00Z');
const at = delta => new Date(now + delta).toISOString();
const base = () => ({
  parlay_id: 'synthetic-standard-ticket', product_type: 'STANDARD_PARLAY', status: 'ACTIONABLE',
  sportsbook: 'DraftKings', quoted_american_odds: 120, quoted_decimal_odds: 2.2,
  quoted_at: at(-60_000), expires_at: at(60_000),
  quote_id: 'synthetic-quote', provider_ticket_id: 'synthetic-provider-ticket',
  quote_source: 'SPORTSBOOK', quote_verification_state: 'VERIFIED',
  legs: [
    { sport: 'NFL', game_id: 'synthetic-game-1', market_type: 'spread_home', selection: 'B -2.5', line: -2.5, start: at(3_600_000), quote_timestamp: at(-60_000), analysis_timestamp: at(-60_000), sportsbook: 'DraftKings' },
    { sport: 'NCAAF', game_id: 'synthetic-game-2', market_type: 'total_over', selection: 'Over 42.5', line: 42.5, start: at(7_200_000), quote_timestamp: at(-60_000), analysis_timestamp: at(-60_000), sportsbook: 'DraftKings' },
  ],
  probability_mean: 0.52, probability_conservative: 0.5, probability_push: 0,
  probability_method: 'validated_independent', validation_id: 'synthetic-validation',
  joint_model_id: null, break_even_probability: 1 / 2.2,
  conservative_edge: 0.05, conservative_ev: 0.1,
  minimum_acceptable_decimal: 2.1, minimum_acceptable_american: 110,
  dependence_status: 'INDEPENDENT_VERIFIED', recommended_stake: 5,
  recommended_fraction: 0.005, production_eligible: true,
  blockers: [], ticket_hash: 'synthetic-ticket-hash',
});
const view = (change = {}, when = now) => api.parlayProductView({ ...base(), ...change }, when);

assert.equal(view().status, 'ACTIONABLE');
assert.equal(view().currentStake, 5);
assert.deepEqual([...view({ blockers: ['PRODUCER_BLOCKER'] }).blockers], ['PRODUCER_BLOCKER']);
assert.equal(view({ blockers: ['PRODUCER_BLOCKER'] }).currentStake, 0);
assert.equal(view({}, now + 60_000).status, 'STALE');
assert.equal(view({}, now + 60_000).currentStake, 0);
assert.ok(view({}, now + 60_000).blockers.includes('EXPIRED_TICKET_QUOTE'));
assert.equal(view({ quoted_at: at(1_000) }).status, 'STALE');
assert.equal(view({ legs: [{ ...base().legs[0], start: at(-1) }, base().legs[1]] }).status, 'STALE');
assert.equal(view({ quoted_decimal_odds: null }).status, 'PRICE_UNAVAILABLE');
assert.equal(view({ quoted_decimal_odds: 2.3 }).status, 'PRICE_UNAVAILABLE');
assert.ok(view({ quoted_decimal_odds: 2.3 }).blockers.includes('TICKET_ODDS_CONFLICT'));
assert.equal(view({ quoted_decimal_odds: 2.0, quoted_american_odds: 100 }).status, 'PRICE_MOVED');
assert.equal(view({ status: 'PRICE MOVED' }).status, 'PRICE_MOVED');
assert.equal(view({ status: 'JOINT MODEL UNAVAILABLE' }).status, 'JOINT_MODEL_UNAVAILABLE');
assert.equal(view({ status: 'PRICE UNAVAILABLE' }).status, 'PRICE_UNAVAILABLE');
assert.equal(view({ legs: [{ ...base().legs[0], start: null }, base().legs[1]] }).status, 'STALE');
assert.ok(view({ legs: [{ ...base().legs[0], start: null }, base().legs[1]] }).blockers.includes('LEG_START_UNAVAILABLE'));
assert.equal(view({ legs: [{ ...base().legs[0], analysis_timestamp: null }, base().legs[1]] }).status, 'STALE');
assert.equal(view({ validation_id: null }).status, 'PASS');
assert.equal(view({ production_eligible: false }).currentStake, 0);
assert.equal(view({ recommended_stake: 0 }).currentStake, 0);
assert.equal(view({ conservative_ev: 0 }).currentStake, 0);
assert.equal(view({ status: 'RESEARCH' }).status, 'RESEARCH');
assert.equal(view({ status: 'RESEARCH' }).currentStake, 0);
assert.equal(view({ status: 'UNVALIDATED', product_type: 'SAME_GAME_PARLAY' }).status, 'UNVALIDATED');
assert.equal(view({ status: 'JOINT_MODEL_UNAVAILABLE', product_type: 'CROSS_GAME_PARLAY' }).status, 'JOINT_MODEL_UNAVAILABLE');

const rows = [base(), { ...base(), product_type: 'SAME_GAME_PARLAY', parlay_id: 'synthetic-sgp', status: 'UNVALIDATED' }, { ...base(), product_type: 'CROSS_GAME_PARLAY', parlay_id: 'synthetic-cross', status: 'RESEARCH' }];
assert.equal(api.parlayProductRows(rows, '').length, 3);
assert.equal(api.parlayProductRows(rows, 'SAME_GAME_PARLAY').length, 1);
assert.equal(api.parlayProductRows(rows, 'CROSS_GAME_PARLAY').length, 1);
assert.equal(api.parlayProductRows(rows, 'BOGUS').length, 0);

// Rendering does not alter saved eligibility when the user changes products.
const actualNow = Date.now();
const future = new Date(actualNow + 3_600_000).toISOString();
const displayed = { ...base(), quoted_at: new Date(actualNow - 60_000).toISOString(), expires_at: future,
  legs: base().legs.map(leg => ({ ...leg, start: future, quote_timestamp: new Date(actualNow - 60_000).toISOString(), analysis_timestamp: new Date(actualNow - 60_000).toISOString() })) };
data.parlay_products = [displayed, { ...displayed, product_type: 'SAME_GAME_PARLAY', status: 'UNVALIDATED', validation_id: null, parlay_id: 'synthetic-sgp' }];
api.renderParlayProducts();
assert.equal(elements.parlayProductRows.children.length, 2);
assert.match(text(elements.parlayProductFunnel), /candidate combinations: 3/);
assert.match(text(elements.parlayProductFunnel), /Pruned by limit: STANDARD PARLAY: 1/);
assert.match(text(elements.parlayProductFunnel), /Overlapping blockers \(non-additive\)/);
assert.match(text(elements.parlayProductRows), /Ticket hash: synthetic-ticket-hash/);
assert.match(text(elements.parlayProductCurrent), /1 currently actionable of 2/);
elements.parlayProductFilter.value = 'SAME_GAME_PARLAY';
api.renderParlayProducts();
assert.equal(elements.parlayProductRows.children.length, 1);
assert.match(text(elements.parlayProductRows), /UNVALIDATED/);
assert.match(text(elements.parlayProductCurrent), /1 currently actionable of 2/);
assert.equal(data.parlay_products[0].status, 'ACTIONABLE');

data.parlay_products = [];
data.parlay_product_funnel = null;
api.renderParlayProducts();
assert.match(text(elements.parlayProductRows), /No parlay product tickets/);
assert.match(text(elements.parlayProductFunnel), /No product funnel/);
console.log('PASS: product filters, ticket facts, funnel, blockers, expiry and fail-closed actionability');
