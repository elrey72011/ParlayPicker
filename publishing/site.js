/* Public publication transport. Only same-origin saved JSON is fetched. */
(() => {
  'use strict';
  const api = window.parlayPicker;
  if (!api) return;
  const VERSION_CHECK_INTERVAL_MS = 45000;
  let currentVersion = JSON.parse(document.getElementById('board-version')?.textContent || 'null');
  let busy = false, checkFailed = false, toastTimer;
  const live = !!document.querySelector('meta[name="pp-live-publication"]') &&
    window.self === window.top && ['https:', 'http:'].includes(location.protocol);
  function renderFreshness() {
    const freshness = api.freshness(live ? currentVersion : null);
    const target = document.getElementById('siteFreshnessText');
    if (target) target.textContent = freshness.text + (live ? '' : ' · Preview') +
      (checkFailed ? ' · update check unavailable' : '');
    const container = document.getElementById('siteFreshness');
    if (container) container.dataset.state = freshness.state;
    const icon = document.getElementById('siteFreshnessIcon');
    if (icon) icon.textContent = freshness.state === 'fresh' ? '●' : '◷';
    const summary = document.getElementById('boardAnalysisAge');
    if (summary) summary.textContent = freshness.primary;
  }
  async function fetchText(name) {
    const url = new URL(name, location.href);
    if (url.origin !== location.origin) throw new Error('Invalid publication origin');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(url, {cache:'no-store', redirect:'error', signal:controller.signal,
        headers:{Accept:'application/json'}});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.text();
    } finally { clearTimeout(timeout); }
  }
  function validateVersion(version) {
    if (!version || !/^[a-f0-9]{64}$/.test(version.board_hash) ||
        version.build_id !== version.board_hash || !Number.isFinite(Date.parse(version.published_at)))
      throw new Error('Invalid publication version');
    if ('source_git_sha' in version && version.source_git_sha !== null &&
        !/^[a-f0-9]{40}$/.test(version.source_git_sha)) throw new Error('Invalid source revision');
    if ('source_git_dirty' in version && version.source_git_dirty !== null &&
        typeof version.source_git_dirty !== 'boolean') throw new Error('Invalid source state');
    if ('source_fingerprint' in version && !/^[a-f0-9]{64}$/.test(version.source_fingerprint))
      throw new Error('Invalid source fingerprint');
  }
  function validateData(next) {
    if (!next || ![1,2,3,4,5].includes(next.schema_version) ||
        !Number.isFinite(Date.parse(next.built_at)) || ![15,30].includes(next.stale_after_minutes) ||
        !next.games || !['overall','sides','totals'].every(key => Array.isArray(next.games[key])) ||
        !Array.isArray(next.props) || !Array.isArray(next.dfs) ||
        (next.results !== undefined && !Array.isArray(next.results)) ||
        (next.parlay_products !== undefined && !Array.isArray(next.parlay_products)) ||
        (next.parlay_product_funnel !== undefined &&
          (!next.parlay_product_funnel || typeof next.parlay_product_funnel !== 'object')))
      throw new Error('Invalid board data');
  }
  function showUpdateToast() {
    const toast = document.getElementById('updateToast');
    toast.textContent = 'New ParlayPicker data loaded'; toast.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
  }
  async function checkForPublicUpdate() {
    if (busy || !live) return;
    busy = true;
    try {
      const version = JSON.parse(await fetchText(`version.json?t=${Date.now()}`));
      validateVersion(version);
      if (currentVersion && Date.parse(version.published_at) < Date.parse(currentVersion.published_at))
        throw new Error('Publication is still updating');
      if (version.build_id !== currentVersion?.build_id) {
        const raw = await fetchText(`board-data.json?v=${encodeURIComponent(version.build_id)}&t=${Date.now()}`);
        const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw))),
          byte => byte.toString(16).padStart(2,'0')).join('');
        if (hash !== version.board_hash) throw new Error('Publication is still updating');
        const next = JSON.parse(raw); validateData(next);
        api.apply(next);
        currentVersion = version;
        showUpdateToast();
      } else {
        currentVersion = version;
      }
      checkFailed = false;
    } catch (error) {
      checkFailed = true;
    } finally { busy = false; renderFreshness(); }
  }
  renderFreshness();
  setInterval(renderFreshness, 60000);
  if (live) { checkForPublicUpdate(); setInterval(checkForPublicUpdate, VERSION_CHECK_INTERVAL_MS); }
})();
