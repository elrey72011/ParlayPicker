// Run with Playwright installed; set PLAYWRIGHT_MODULE and BROWSER_CHANNEL if needed.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');

// Public results begin on September 11, 2026. Keep both test dates inside that epoch.
const row = (date, group, outcome) => ({
  date, group, outcome, category: 'overall', picks: 'Example pick', odds: '-110', final_score: '1-0',
});
const data = {
  built_at: '2026-09-14T14:00:00Z', stale_after_minutes: 15,
  games: {overall: [], sides: [], totals: []}, props: [], dfs: [], parlays: [],
  results: [
    row('2026-09-13', 'Research', 'WIN'),
    row('2026-09-12', 'Imported research', 'LOSS'),
    row('2026-09-13', 'Imported research', 'WIN'),
  ],
};
data.results.push({...row('2026-09-13', 'Research', 'WIN'), category: 'props', sport: 'MLB', market: 'batter_hits'});
data.results.push({...row('2026-09-13', 'Research', 'PENDING'), category: 'props', sport: 'MLB', market: 'pitcher_strikeouts'});
data.results.push({...row('2026-09-13', 'Research', 'NEEDS_REVIEW'), category: 'props', sport: 'MLB', market: 'pitcher_walks', final_score: 'Player missing or ambiguous in final box score'});
const html = fs.readFileSync('publishing/board.html', 'utf8').replace('__PUBLIC_DATA__', JSON.stringify(data));

(async () => {
  const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_CHANNEL ? {channel: process.env.BROWSER_CHANNEL} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.clock.install({time: new Date('2026-09-14T02:00:00Z')}); // September 13 Eastern.
    await page.route('https://board.test/**', route => route.fulfill({contentType: 'text/html', body: html}));
    await page.goto('https://board.test/');
    await page.locator('.site-nav [data-site-tab=results]').first().click();
    await page.selectOption('#resultPeriod', '1');
    assert.match(await page.locator('#resultRange').innerText(), /2026-09-12 through 2026-09-12/);
    await page.selectOption('#resultGroup', 'Imported research');
    assert.equal(await page.locator('#resultDetails tr').count(), 2);

    await page.clock.setFixedTime(new Date('2026-09-14T14:00:00Z')); // September 14 Eastern.
    await page.selectOption('#resultPeriod', '2');
    assert.match(await page.locator('#resultDetails').innerText(), /2026-09-12/);
    assert.doesNotMatch(await page.locator('#resultDetails').innerText(), /2026-09-13/);
    await page.reload();
    await page.locator('.site-nav [data-site-tab=results]').first().click();
    assert.equal(await page.locator('#resultPeriod').inputValue(), '2');
    assert.equal(await page.locator('#resultGroup').inputValue(), 'Imported research');
    await page.evaluate(() => render());
    assert.equal(await page.locator('#resultPeriod').inputValue(), '2');

    await page.selectOption('#resultPeriod', 'last2');
    assert.match(await page.locator('#resultDetails').innerText(), /2026-09-12/);
    assert.match(await page.locator('#resultDetails').innerText(), /2026-09-13/);
    assert.equal(await page.locator('#resultDetails tr').count(), 3);
    await page.selectOption('#resultPeriod', '1');
    await page.selectOption('#resultGroup', 'Research');
    assert.equal(await page.locator('#resultDetails tr').count(), 2);
    await page.selectOption('#resultPeriod', '2');
    await page.selectOption('#resultGroup', 'Imported research');
    assert.equal(await page.locator('#resultGroup').inputValue(), 'Imported research');

    await page.selectOption('#resultPeriod', '1');
    await page.selectOption('#resultKind', 'props');
    await page.selectOption('#resultGroup', 'Research');
    assert.equal(await page.locator('#resultDetails tr').count(), 4);
    assert.match(await page.locator('#resultSummary').innerText(), /Player Props/);
    assert.match(await page.locator('#resultSummary').innerText(), /needing review/);
    assert.match(await page.locator('#resultSummary').innerText(), /pending/);
    assert.match(await page.locator('#resultSummary').innerText(), /100.0%/);
    const summary = await page.locator('#resultSummary tr').nth(1).locator('td').allTextContents();
    assert.deepEqual(summary.slice(0, 8), ['Player Props', '1', '0', '0', '1', '1', '1', '100.0%']);
    await page.selectOption('#resultMarket', 'batter_hits');
    assert.equal(await page.locator('#resultDetails tr').count(), 2);
    assert.match(await page.locator('#resultDetails').innerText(), /Actual statistic/);
    await page.reload();
    assert.equal(await page.locator('#resultKind').inputValue(), 'props');
    assert.equal(await page.locator('#resultMarket').inputValue(), 'batter_hits');
    assert.deepEqual(errors, []);

    const blocked = await browser.newPage();
    await blocked.addInitScript(() => {
      Storage.prototype.getItem = () => { throw Error('blocked'); };
      Storage.prototype.setItem = () => { throw Error('blocked'); };
    });
    await blocked.route('https://board.test/**', route => route.fulfill({contentType: 'text/html', body: html}));
    await blocked.goto('https://board.test/');
    assert.equal(await blocked.locator('#resultSummary table').count(), 1);
    console.log('PASS: daily periods, Eastern dates, separated groups, reload/render persistence, mobile layout and blocked storage.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1);});
