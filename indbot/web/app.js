/* indbot dashboard — logic layer.
 *
 * Binds the engine's snapshot() to the markup in index.html. No framework, no build.
 *
 * Two rules run through everything here:
 *
 *   1. Money direction is NEVER carried by colour alone. Profit-green and loss-red
 *      measure an OKLab Delta E of 4.0 under deuteranopia — for roughly one man in
 *      twelve they are the same colour, while normal vision reads them at 33.9.
 *      So every signed figure gets an explicit +/- and every move gets a glyph with
 *      a word behind it. Colour is the third channel, never the first.
 *   2. A stale number must never look live. On websocket loss the body gets
 *      data-conn="down", the CSS greys every figure, and a banner says so. A frozen
 *      P&L that still reads as current is the most dangerous thing this page could do.
 */
'use strict';

/* ======================================================================= *
 * 1. DOM helpers
 * ======================================================================= */

const $$ = (sel, root) => Array.prototype.slice.call((root || document).querySelectorAll(sel));

/** getElementById that complains instead of returning null silently.
 *  A renamed id should fail loudly in the console, not blank a panel. */
const missingIds = new Set();
function el(id) {
  const node = document.getElementById(id);
  if (!node && !missingIds.has(id)) {
    missingIds.add(id);
    console.error('[indbot] missing element id: ' + id);
  }
  return node;
}

/** Build an element. h('p', {class, text, html, onclick, ...attrs}, children) */
function h(tag, props, children) {
  const node = document.createElement(tag);
  const p = props || {};
  for (const key of Object.keys(p)) {
    const v = p[key];
    if (v === null || v === undefined || v === false) continue;
    if (key === 'text') node.textContent = String(v);
    else if (key === 'html') node.innerHTML = v;
    else if (key === 'class') node.className = v;
    else if (key.startsWith('on') && typeof v === 'function') node.addEventListener(key.slice(2), v);
    else if (key === 'dataset') for (const d of Object.keys(v)) node.dataset[d] = v[d];
    else node.setAttribute(key, v === true ? '' : String(v));
  }
  const kids = Array.isArray(children) ? children : (children ? [children] : []);
  for (const kid of kids) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return node;
}

/** Replace a container's children in one shot. */
function fill(node) {
  if (!node) return;
  const kids = Array.prototype.slice.call(arguments, 1).flat().filter(Boolean);
  node.replaceChildren.apply(node, kids);
}

const emptyNote = (message) => h('p', { class: 'empty', text: message });

function svgEl(tag, props) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const key of Object.keys(props || {})) {
    const v = props[key];
    if (v !== null && v !== undefined) node.setAttribute(key, String(v));
  }
  return node;
}

/* ======================================================================= *
 * 2. Formatters
 * ======================================================================= */

const isNum = (v) => typeof v === 'number' && isFinite(v);

/** Rupees, Indian digit grouping (1,23,456 not 123,456).
 *  `sign: true` forces an explicit + or − — mandatory on anything directional. */
function inr(value, opts) {
  const o = opts || {};
  if (!isNum(value)) return '₹ —';
  const decimals = o.decimals === undefined ? 2 : o.decimals;
  const body = Math.abs(value).toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  let prefix = '';
  if (o.sign) prefix = value > 0 ? '+' : (value < 0 ? '−' : '±');
  else if (value < 0) prefix = '−';
  return prefix + '₹' + body;
}

function num(value, decimals) {
  if (!isNum(value)) return '—';
  const d = decimals === undefined ? 2 : decimals;
  return value.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
}

function pct(value, opts) {
  const o = opts || {};
  if (!isNum(value)) return '—';
  const d = o.decimals === undefined ? 2 : o.decimals;
  const prefix = o.sign ? (value > 0 ? '+' : (value < 0 ? '−' : '±')) : (value < 0 ? '−' : '');
  return prefix + Math.abs(value).toFixed(d) + '%';
}

/** The ▲/▼ glyph plus a screen-reader word — the non-colour channel for direction. */
function dirMark(value) {
  if (!isNum(value) || value === 0) {
    return h('span', { class: 'dir dir--flat' }, [
      h('span', { 'aria-hidden': 'true', text: '■' }),
      h('span', { class: 'sr-only', text: ' unchanged ' }),
    ]);
  }
  const up = value > 0;
  return h('span', { class: 'dir ' + (up ? 'dir--up' : 'dir--down') }, [
    h('span', { 'aria-hidden': 'true', text: up ? '▲' : '▼' }),
    h('span', { class: 'sr-only', text: up ? ' up ' : ' down ' }),
  ]);
}

const signClass = (v) => (!isNum(v) || v === 0 ? 'val--flat' : (v > 0 ? 'val--up' : 'val--down'));

/* English plurals the naive +"s" gets wrong. "3 entrys left" shipped once. */
const IRREGULAR_PLURALS = { entry: 'entries', loss: 'losses' };
const plural = (n, word) =>
  (n === 1 ? word : (IRREGULAR_PLURALS[word] || word + 's'));

function istTime(iso) {
  if (!iso) return '--:--:--';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '--:--:--';
  return d.toLocaleTimeString('en-IN', {
    hour12: false, timeZone: 'Asia/Kolkata',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function istClock(iso) {
  if (!iso) return '--:--';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '--:--';
  return d.toLocaleTimeString('en-IN', {
    hour12: false, timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit',
  });
}

const tsTime = (seconds) =>
  (isNum(seconds) ? istClock(new Date(seconds * 1000).toISOString()) : '--:--');

function labelise(key) {
  const s = String(key).replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** "5minute" is the API's spelling, not a thing to show a person. */
function prettyInterval(interval) {
  const m = /^(\d+)minute$/.exec(String(interval || ''));
  if (m) return m[1] + '-minute';
  if (interval === '1day') return 'daily';
  return String(interval || '?');
}

/* ======================================================================= *
 * 3. API
 * ======================================================================= */

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

const api = {
  async request(method, path, body) {
    let res;
    try {
      res = await fetch(path, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (err) {
      throw new ApiError('Cannot reach the local engine. Is run.py still running?', 0, null);
    }
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = { raw: text }; }
    if (!res.ok) {
      const raw = (data && (data.detail || data.error || data.message)) || res.statusText ||
        ('HTTP ' + res.status);
      throw new ApiError(typeof raw === 'string' ? raw : JSON.stringify(raw), res.status, data);
    }
    return data;
  },
  get(path) { return this.request('GET', path); },
  post(path, body) { return this.request('POST', path, body || {}); },
};

/** The manual CLOSE route. Unlike PANIC it squares off one position and leaves
 *  the day running. */
const closeEndpoint = (id) => '/api/positions/' + encodeURIComponent(id) + '/close';

/* ======================================================================= *
 * 4. State
 * ======================================================================= */

const state = {
  snap: null,
  connected: false,
  lastSnapAt: 0,
  tab: 'live',
  whyFilter: 'rejected',
  btJob: null,
  btTimer: null,
  btResult: null,
  prevSpots: {},          // underlying -> last spot, for the tick flash
  settingsDirty: false,
};

const STORE = {
  // Browser storage throws in a private window and can come back empty. Never let
  // a remembered tab break the page.
  get(key, fallback) {
    try {
      const v = localStorage.getItem('indbot.' + key);
      return v === null ? fallback : v;
    } catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('indbot.' + key, value); } catch (e) { /* ignore */ }
  },
};

function reduceMotion() {
  try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; }
  catch (e) { return false; }
}

/* ======================================================================= *
 * 5. Tabs
 * ======================================================================= */

const TABS = ['live', 'learning', 'backtest', 'settings'];

function setTab(name, opts) {
  if (TABS.indexOf(name) < 0) name = 'live';
  state.tab = name;
  document.body.dataset.tab = name;
  STORE.set('tab', name);
  for (const t of TABS) {
    const btn = el('tabbtn-' + t);
    const panel = el('tab-' + t);
    const on = t === name;
    if (btn) {
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
      btn.tabIndex = on ? 0 : -1;
    }
    if (panel) panel.hidden = !on;
  }
  if (opts && opts.focus) {
    const b = el('tabbtn-' + name);
    if (b) b.focus();
  }
  // Charts size themselves off their container, which is zero-width while hidden,
  // so anything with an SVG has to be redrawn when its tab becomes visible.
  if (state.snap) { renderLearning(state.snap); renderBacktestResult(); }
}

function initTabs() {
  const list = el('tablist');
  if (!list) return;
  list.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-tab]');
    if (btn) setTab(btn.dataset.tab);
  });
  list.addEventListener('keydown', (e) => {
    const i = TABS.indexOf(state.tab);
    let next = null;
    if (e.key === 'ArrowRight') next = TABS[(i + 1) % TABS.length];
    else if (e.key === 'ArrowLeft') next = TABS[(i - 1 + TABS.length) % TABS.length];
    else if (e.key === 'Home') next = TABS[0];
    else if (e.key === 'End') next = TABS[TABS.length - 1];
    if (next) { e.preventDefault(); setTab(next, { focus: true }); }
  });
  setTab(STORE.get('tab', 'live'));
}

/* ======================================================================= *
 * 6. Theme
 * ======================================================================= */

/* Dark is the default and the OS cannot override it — a desk that flips to a
   white screen because the OS decided it is daytime is the wrong behaviour with
   a position open. Light is a deliberate choice, so the toggle is a two-state. */
function applyTheme(mode) {
  const theme = mode === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  STORE.set('theme', theme);
  const btn = el('btn-theme');
  if (btn) {
    btn.textContent = theme.toUpperCase();
    btn.setAttribute('aria-label',
      'Colour theme: ' + theme + '. Activate to switch to '
      + (theme === 'dark' ? 'light' : 'dark') + '.');
  }
}

function initTheme() {
  applyTheme(STORE.get('theme', 'dark'));
  const btn = el('btn-theme');
  if (btn) {
    btn.addEventListener('click', () => {
      applyTheme(STORE.get('theme', 'dark') === 'dark' ? 'light' : 'dark');
    });
  }
}

/* ======================================================================= *
 * 7. Announcements & banners
 * ======================================================================= */

function announce(msg) { const n = el('aria-mode'); if (n) n.textContent = msg; }
function alertNow(msg) { const n = el('aria-alert'); if (n) n.textContent = msg; }

function banner(id, message) {
  const node = el(id);
  if (!node) return;
  if (message) { node.textContent = message; node.hidden = false; }
  else { node.textContent = ''; node.hidden = true; }
}

/* ======================================================================= *
 * 8. Connection
 * ======================================================================= */

function setConnection(up, note) {
  state.connected = up;
  document.body.dataset.conn = up ? 'up' : 'down';
  const text = el('conn-text');
  if (text) text.textContent = up ? 'live' : (note || 'disconnected');
  const wrap = el('conn');
  if (wrap) {
    wrap.title = up
      ? 'Receiving snapshots from the local engine.'
      : 'Not receiving snapshots. The figures below are the last ones received and are out of date.';
  }
  banner('stale-banner', up ? '' :
    'Disconnected from the engine. Every figure on this page is the last one received and is no '
    + 'longer updating — do not act on it. Reconnecting…');
}

/* ======================================================================= *
 * 9. Render — header, ticker, hero, day
 * ======================================================================= */

function renderHeader(s) {
  const mode = String(s.mode || 'paper');
  const armed = !!s.armed;
  document.body.dataset.mode = mode;
  document.body.dataset.armed = armed ? 'true' : 'false';

  const badgeMode = el('mode-badge-mode');
  if (badgeMode) badgeMode.textContent = mode.toUpperCase();
  const badgeArm = el('mode-badge-arm');
  if (badgeArm) {
    badgeArm.textContent = mode === 'live'
      ? (armed ? 'ARMED — real money' : 'live, disarmed')
      : (mode === 'backtest' ? 'replay only' : 'simulated');
  }

  // The rail is the one element meant to be readable from across a room.
  const railMode = el('mode-rail-mode');
  if (railMode) railMode.textContent = mode.toUpperCase();
  const railText = el('mode-rail-text');
  if (railText) {
    railText.textContent = mode === 'live'
      ? (armed ? 'Real orders will be placed on INDmoney.'
               : 'Live mode, not armed. No orders will be placed.')
      : (mode === 'backtest' ? 'Replaying history. Nothing is sent to the broker.'
                             : 'Simulated fills against the live book. Nothing is sent to the broker.');
  }
  const railArmed = el('mode-rail-armed');
  if (railArmed) railArmed.hidden = !(mode === 'live' && armed);

  for (const btn of $$('[data-mode-btn]')) {
    btn.setAttribute('aria-pressed', btn.dataset.modeBtn === mode ? 'true' : 'false');
  }
  const armBtn = el('btn-arm');
  if (armBtn) armBtn.hidden = (mode === 'live' && armed);
  const disarmBtn = el('btn-disarm');
  if (disarmBtn) disarmBtn.hidden = !(mode === 'live' && armed);

  const runBtn = el('btn-run');
  if (runBtn) {
    runBtn.textContent = s.running ? 'STOP SCAN' : 'START SCAN';
    runBtn.dataset.running = s.running ? 'true' : 'false';
  }

  const t = el('clock-time');
  if (t) t.textContent = istTime(s.now);
  const sess = el('clock-session');
  if (sess) {
    const session = s.session || {};
    sess.textContent = session.state
      ? String(session.state).replace(/_/g, ' ')
      : (s.market_open ? 'open' : 'closed');
  }

  announce('Mode ' + mode + (mode === 'live' ? (armed ? ', armed, real money' : ', disarmed') : ''));

  banner('halt-banner', s.halted
    ? 'Trading is halted for today. ' + (s.halt_reason || '')
      + ' The bot will still manage and exit any open position.'
    : '');

  const errs = Array.isArray(s.errors) ? s.errors : [];
  banner('error-banner', errs.length
    ? errs.slice(0, 3).map((e) => (e && (e.message || e.type)) || String(e)).join(' · ')
    : '');
}

function renderTicker(s) {
  const track = el('ticker-track');
  if (!track) return;
  const rows = Array.isArray(s.underlyings) ? s.underlyings : [];
  if (!rows.length) {
    fill(track, h('li', { class: 'tick tick--placeholder',
      text: 'Waiting for the first candle of each index…' }));
    return;
  }
  const flash = !reduceMotion();
  fill(track, rows.map((u) => {
    const name = String(u.name || '');
    const spot = isNum(u.spot) ? u.spot : null;
    const chg = isNum(u.change_pct) ? u.change_pct : null;
    const prev = state.prevSpots[name];
    let cls = 'tick';
    if (u.data_error) cls += ' tick--error';
    if (flash && isNum(prev) && isNum(spot) && prev !== spot) {
      cls += spot > prev ? ' tick--flashup' : ' tick--flashdown';
    }
    if (isNum(spot)) state.prevSpots[name] = spot;
    return h('li', { class: cls, title: u.data_error ? String(u.data_error) : (u.label || name) }, [
      h('span', { class: 'tick-name', text: name }),
      h('span', { class: 'tick-spot', text: isNum(spot) ? num(spot, 2) : '—' }),
      h('span', { class: 'tick-chg ' + signClass(chg) }, [
        dirMark(chg), h('span', { text: ' ' + pct(chg, { sign: true }) }),
      ]),
      u.regime ? h('span', { class: 'tick-regime', text: String(u.regime).replace(/_/g, ' ') }) : null,
    ]);
  }));
}

function renderHero(s) {
  const day = s.day || {};
  const pnl = isNum(day.realized_pnl) ? day.realized_pnl : null;

  const node = el('hero-pnl');
  if (node) {
    node.textContent = pnl === null ? '₹ —' : inr(pnl, { sign: true, decimals: 0 });
    node.className = 'hero-value ' + signClass(pnl);
  }

  const delta = el('hero-delta');
  if (delta) {
    const closed = day.closed_count || 0;
    const wins = day.wins || 0;
    const losses = day.losses || 0;
    fill(delta, closed
      ? [dirMark(pnl), h('span', {
          text: ' from ' + closed + ' closed ' + plural(closed, 'trade') + ' — '
            + wins + ' ' + plural(wins, 'win') + ', ' + losses + ' ' + plural(losses, 'loss'),
        })]
      : [h('span', { text: 'No trade has closed today.' })]);
  }

  const sub = el('hero-sub');
  if (sub) {
    const open = (s.positions || []).length;
    sub.textContent = open
      ? 'Closed trades only. ' + open + ' open ' + plural(open, 'position')
        + ' is marked separately below.'
      : 'Closed trades only. No position is open.';
  }
  renderHeroSpark(s);
}

function renderHeroSpark(s) {
  const box = el('hero-spark');
  const label = el('hero-sparklabel');
  if (!box) return;
  const trades = (s.recent_trades || []).filter((t) => isNum(t.pnl) && t.exit_ts);
  if (trades.length < 2) {
    fill(box);
    if (label) label.textContent = trades.length ? 'One closed trade so far.' : '';
    return;
  }
  const ordered = trades.slice().sort((a, b) => (a.exit_ts || 0) - (b.exit_ts || 0)).slice(-12);
  let run = 0;
  const pts = ordered.map((t) => (run += t.pnl));
  fill(box, sparkline(pts, 140, 44));
  if (label) {
    label.textContent = 'Cumulative over the last ' + ordered.length + ' closed trades: '
      + inr(run, { sign: true, decimals: 0 }) + '.';
  }
}

/** A bare sparkline: one 2px line, a zero rule, the last point marked. */
function sparkline(values, w, hgt) {
  const svg = svgEl('svg', {
    class: 'spark', viewBox: '0 0 ' + w + ' ' + hgt, width: w, height: hgt,
    'aria-hidden': 'true', focusable: 'false',
  });
  const lo = Math.min(0, Math.min.apply(null, values));
  const hi = Math.max(0, Math.max.apply(null, values));
  const span = (hi - lo) || 1;
  const x = (i) => (values.length === 1 ? w / 2 : (i / (values.length - 1)) * (w - 4) + 2);
  const y = (v) => hgt - 3 - ((v - lo) / span) * (hgt - 6);
  const zeroY = y(0);
  svg.appendChild(svgEl('line', { x1: 0, x2: w, y1: zeroY, y2: zeroY, class: 'spark-zero' }));
  svg.appendChild(svgEl('polyline', {
    class: 'spark-line', fill: 'none',
    points: values.map((v, i) => x(i) + ',' + y(v)).join(' '),
  }));
  const last = values[values.length - 1];
  svg.appendChild(svgEl('circle', {
    cx: x(values.length - 1), cy: y(last), r: 3,
    class: 'spark-dot ' + (last >= 0 ? 'is-up' : 'is-down'),
  }));
  return svg;
}

function tile(label, value, opts) {
  const o = opts || {};
  return h('div', { class: 'tile' + (o.tone ? ' tile--' + o.tone : '') }, [
    h('p', { class: 'tile-label', text: label }),
    h('p', { class: 'tile-value ' + (o.valueClass || ''), text: value }),
    o.note ? h('p', { class: 'tile-note', text: o.note }) : null,
  ]);
}

function renderDay(s) {
  const box = el('day-tiles');
  if (!box) return;
  const day = s.day || {};
  const cfg = s.config || {};
  const trades = day.trade_count || 0;
  const cap = cfg.max_trades_per_day || 0;
  const pnl = isNum(day.realized_pnl) ? day.realized_pnl : 0;
  const limit = cfg.max_daily_loss || 0;
  const used = limit > 0 ? Math.min(100, Math.max(0, (-pnl / limit) * 100)) : 0;
  const open = (s.positions || []).length;

  fill(box, [
    tile('Realised P&L', inr(pnl, { sign: true, decimals: 0 }), {
      valueClass: signClass(pnl),
      note: day.closed_count ? day.closed_count + ' closed' : 'nothing closed yet',
    }),
    tile('Loss limit used', num(used, 0) + '%', {
      tone: used >= 100 ? 'critical' : (used >= 66 ? 'warning' : null),
      note: 'of ' + inr(limit, { decimals: 0 }) + ' a day',
    }),
    tile('Trades used', trades + ' / ' + cap, {
      tone: cap && trades >= cap ? 'warning' : null,
      note: cap && trades >= cap ? 'cap reached'
        : Math.max(0, cap - trades) + ' ' + plural(Math.max(0, cap - trades), 'entry') + ' left',
    }),
    tile('Open now', String(open), { note: 'cap ' + (cfg.max_open_positions || 1) }),
    tile('Best / worst', isNum(day.best)
      ? inr(day.best, { sign: true, decimals: 0 }) + ' / ' + inr(day.worst, { sign: true, decimals: 0 })
      : '—', { note: 'closed trades today' }),
  ]);

  const note = el('day-note');
  if (note) {
    const tick = s.tick || {};
    const sess = s.session || {};
    const bits = [];
    if (!s.running) bits.push('The scan loop is stopped — press START SCAN.');
    if (tick.count) {
      bits.push('Tick ' + tick.count + ' at ' + istTime(tick.last_at)
        + (isNum(tick.last_ms) ? ' (' + num(tick.last_ms, 0) + ' ms)' : '')
        + ', every ' + (tick.poll_seconds || '?') + 's on ' + prettyInterval(tick.interval)
        + ' candles.');
    }
    if (sess.no_entry_before) {
      bits.push('Entries ' + sess.no_entry_before + '–' + sess.no_entry_after
        + ', everything flat by ' + sess.flat_by + '.');
    }
    note.textContent = bits.join(' ');
  }
}

/* ======================================================================= *
 * 10. Render — positions
 * ======================================================================= */

function renderPositions(s) {
  const box = el('position-list');
  if (!box) return;
  const rows = Array.isArray(s.positions) ? s.positions : [];
  const sub = el('positions-sub');
  if (sub) {
    sub.textContent = rows.length
      ? rows.length + ' open of ' + ((s.config || {}).max_open_positions || 1) + ' allowed'
      : '';
  }
  if (!rows.length) {
    fill(box, emptyNote('No position is open. The bot is waiting for a signal that clears the '
      + 'risk gate; when one does, the contract, the stop and the live P&L appear here.'));
    return;
  }
  fill(box, rows.map(positionCard));
}

function positionCard(p) {
  const pnl = isNum(p.pnl) ? p.pnl : null;
  const stop = isNum(p.risk_stop) ? p.risk_stop : p.stop_price;
  // Reward:risk, stated outright. It is the term of the high-conviction bargain,
  // so it should not have to be inferred from two other rows.
  const risk = isNum(p.entry_price) && isNum(stop) ? p.entry_price - stop : null;
  const reward = isNum(p.target_price) && isNum(p.entry_price)
    ? p.target_price - p.entry_price : null;
  const rr = (isNum(risk) && risk > 0 && isNum(reward)) ? reward / risk : null;
  const rows = [
    ['Entry', inr(p.entry_price) + ' × ' + (p.qty || 0) + ' units'],
    ['Mark', p.mark_ok ? inr(p.ltp) : 'NO QUOTE'],
    ['Stop', inr(stop) + (p.trail_armed ? ' (trailing)' : '')],
    ['Target', inr(p.target_price) + (isNum(rr) ? '  (' + rr.toFixed(1) + ':1)' : '')],
    ['At risk', inr(p.risk_at_stop, { decimals: 0 })],
    ['Opened', tsTime(p.entry_ts)],
  ];
  return h('article', { class: 'poscard ' + signClass(pnl) }, [
    h('header', { class: 'poscard-head' }, [
      h('h3', { class: 'poscard-title', text: String(p.label || '') }),
      h('p', { class: 'poscard-pnl ' + signClass(pnl) }, [
        dirMark(pnl),
        h('span', { text: ' ' + inr(pnl, { sign: true, decimals: 0 }) }),
        h('span', { class: 'poscard-pct',
          text: ' (' + pct(p.pnl_pct, { sign: true, decimals: 1 }) + ')' }),
      ]),
    ]),
    !p.mark_ok ? h('p', { class: 'warnline', text:
      'No live quote for this contract, so the P&L above is not current. The stop is still in '
      + 'force on the engine side.' }) : null,
    p.exit_fault ? h('p', { class: 'badline', text:
      'An exit for this position FAILED: '
      + (p.exit_fault.message || JSON.stringify(p.exit_fault))
      + ' — the position is still open. Check INDmoney.' }) : null,
    h('dl', { class: 'kv' }, rows.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })])),
    h('div', { class: 'poscard-actions' }, [
      h('button', {
        type: 'button', class: 'btn btn--danger btn--sm',
        disabled: p.exiting || null,
        onclick: () => confirmClose(p),
      }, p.exiting ? 'Closing…' : 'CLOSE THIS LOT'),
      h('span', { class: 'poscard-setup', text: 'from ' + String(p.signal_setup || '?') }),
    ]),
  ]);
}

/* ======================================================================= *
 * 11. Render — why no trade
 * ======================================================================= */

function initWhyFilter() {
  for (const chip of $$('[data-why-filter]')) {
    chip.addEventListener('click', () => {
      state.whyFilter = chip.dataset.whyFilter;
      for (const c of $$('[data-why-filter]')) {
        const on = c.dataset.whyFilter === state.whyFilter;
        c.setAttribute('aria-pressed', on ? 'true' : 'false');
        c.classList.toggle('chip--on', on);
      }
      if (state.snap) renderWhy(state.snap);
    });
  }
}

function parseReasons(raw) {
  if (Array.isArray(raw)) return raw.map(String);
  if (typeof raw === 'string' && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch (e) { /* not JSON — treat as one reason */ }
    return [raw];
  }
  return [];
}

/* What actually stopped a signal, in the order the tests are applied.
 * Ordered most-specific first: "could not read the chain" also mentions the
 * underlying, so the chain test has to run before anything looser. */
const BLOCK_KINDS = [
  { key: 'superseded', label: 'another signal scored higher that candle',
    test: /higher-scoring signal/i, real: false },
  { key: 'chain', label: 'a chain read failure',
    test: /could not read|option chain for|no tradable legs|security_id/i, real: true },
  { key: 'budget', label: 'the per-trade risk budget',
    test: /allowed to cost|a 30% stop risks|costs ₹|premium ₹|tighter than the/i, real: true },
  { key: 'strike', label: 'the delta, spread or liquidity filters',
    test: /delta|spread|open interest|\boi\b|volume/i, real: true },
  { key: 'session', label: 'the entry window',
    test: /market is closed|entries only|flat by|no_entry/i, real: true },
  { key: 'limits', label: "today's trade or loss limits",
    test: /halted|trades today|per-trade limit reached|daily loss/i, real: true },
  { key: 'score', label: 'the minimum confluence score',
    // "scored 0.10, under the 0.55 minimum" — the earlier /score .*(below|under)/
    // missed it, because "scored" has no space after "score".
    test: /scored? [\d.]+.*(below|under)|under the [\d.]+ minimum|min_score/i, real: true },
  { key: 'learner', label: "the learner's gate",
    test: /blocked:|expectancy/i, real: true },
];

function classifyBlock(reason) {
  const text = String(reason || '');
  for (const kind of BLOCK_KINDS) if (kind.test.test(text)) return kind;
  return { key: 'other', label: 'another reason', test: null, real: true };
}

function renderWhy(s) {
  const box = el('why-list');
  if (!box) return;
  const all = Array.isArray(s.recent_signals) ? s.recent_signals : [];
  const rows = all.filter((r) => {
    if (state.whyFilter === 'taken') return !!r.taken;
    if (state.whyFilter === 'rejected') return !r.taken;
    return true;
  });

  // Report the actual mix rather than asserting a cause. An earlier version
  // hard-coded "the per-trade budget is why", which read as authoritative while
  // being wrong: the budget was fourth, and the largest group was not a refusal
  // at all but the bot picking the best signal in a candle.
  //
  // The tally covers THIS session only. The journal outlives the process, so a
  // bug fixed an hour ago keeps appearing in it — four SENSEX chain failures went
  // on being reported as a live problem after the cause was fixed. Rows from an
  // earlier run stay visible in the feed, marked, but they do not feed a headline
  // that is supposed to describe what is happening now.
  const sessionStart = s.started_at ? Date.parse(s.started_at) / 1000 : 0;
  const inSession = (r) => !sessionStart
    || Number(r.recorded_ts || r.ts || 0) >= sessionStart - 1;

  const head = el('why-headline');
  if (head) {
    const blocked = all.filter((r) => !r.taken && inSession(r));
    const staleBlocked = all.filter((r) => !r.taken && !inSession(r));
    const tally = new Map();
    for (const r of blocked) {
      const kind = classifyBlock(r.block_reason);
      const cur = tally.get(kind.key) || { kind, n: 0 };
      cur.n += 1;
      tally.set(kind.key, cur);
    }
    const ranked = Array.from(tally.values()).sort((a, b) => b.n - a.n);
    const refusals = ranked.filter((x) => x.kind.real);
    const superseded = tally.get('superseded');

    if (!blocked.length) {
      // Nothing blocked this run. Say so, and account for any older rows below
      // rather than leaving the reader to wonder why the feed is not empty.
      head.textContent = staleBlocked.length
        ? 'Nothing has been blocked since the bot started. The '
          + staleBlocked.length + ' entries below are from an earlier run — some may '
          + 'name problems that have since been fixed.'
        : 'Nothing has been blocked since the bot started.';
      head.hidden = false;
    } else {
      const parts = [];
      if (refusals.length) {
        parts.push('Of ' + blocked.length + ' blocked since the bot started: '
          + refusals.map((x) => x.n + ' blocked by ' + x.kind.label).join(', ') + '.');
      }
      if (superseded) {
        parts.push(superseded.n + ' were not refused at all — a higher-scoring signal '
          + 'was taken for that candle instead, which is the bot choosing, not blocking.');
      }
      // Only make the budget argument when the data actually says so.
      const top = refusals[0];
      if (top && top.kind.key === 'budget' && top.n >= 2) {
        parts.push('One lot of an at-the-money option costs more than your '
          + inr((s.config || {}).max_trade_loss, { decimals: 0 })
          + ' per-trade limit allows — Settings shows the three ways to change that.');
      }
      if (staleBlocked.length) {
        parts.push('A further ' + staleBlocked.length + ' below are from an earlier run '
          + 'and are not counted here.');
      }
      head.textContent = parts.join(' ');
      head.hidden = false;
    }
  }

  if (!rows.length) {
    fill(box, emptyNote(state.whyFilter === 'taken'
      ? 'No signal has been taken yet.'
      : 'No signal has been evaluated yet. Once a candle closes and a setup fires, the reason it '
        + 'was taken or refused is written here as a sentence.'));
    return;
  }
  fill(box, rows.slice(0, 40).map((r) => signalRow(r, inSession(r))));
}

function signalRow(r, current) {
  const taken = !!r.taken;
  const reasons = parseReasons(r.reasons);
  const blocked = String(r.block_reason || '').trim();
  // A signal passed over because a better one fired in the same candle was not
  // refused, so it does not get the refusal styling or the "NOT TAKEN" label —
  // calling it a rejection overstates how often the bot says no.
  const kind = taken ? null : classifyBlock(blocked);
  const superseded = !!kind && kind.key === 'superseded';
  const cls = taken ? 'sigrow--taken'
    : (superseded ? 'sigrow--superseded' : 'sigrow--rejected');
  return h('article', { class: 'sigrow ' + cls }, [
    h('header', { class: 'sigrow-head' }, [
      h('span', { class: 'sigrow-side side--' + String(r.side || '').toLowerCase(),
        text: String(r.side || '') }),
      h('span', { class: 'sigrow-name',
        text: String(r.underlying || '') + ' · ' + String(r.setup || '') }),
      h('span', { class: 'sigrow-score', text: 'score ' + num(r.score, 2) }),
      h('span', { class: 'sigrow-time', text: tsTime(r.ts) }),
      h('span', {
        class: 'sigrow-verdict ' + (taken ? 'is-taken' : (superseded ? 'is-superseded' : 'is-blocked')),
        text: taken ? 'TAKEN' : (superseded ? 'SUPERSEDED' : 'NOT TAKEN'),
      }),
      // A row from before this process started may describe a problem already
      // fixed. Mark it, so old history cannot be read as a live fault.
      current === false
        ? h('span', { class: 'sigrow-stale', title: 'Recorded before the bot was last started',
            text: 'earlier run' })
        : null,
    ]),
    blocked ? h('p', { class: 'sigrow-block', text: blocked }) : null,
    reasons.length
      ? h('ul', { class: 'sigrow-reasons' }, reasons.map((x) => h('li', { text: x })))
      : null,
  ]);
}

/* ======================================================================= *
 * 12. Render — underlyings
 * ======================================================================= */

function renderUnderlyings(s) {
  const box = el('underlying-cards');
  if (!box) return;
  const rows = Array.isArray(s.underlyings) ? s.underlyings : [];
  const sub = el('underlyings-sub');
  if (sub) {
    const live = rows.filter((u) => u.enabled !== false).length;
    sub.textContent = rows.length
      ? live + ' of ' + rows.length + ' scanning on ' + prettyInterval((s.tick || {}).interval) + ' candles'
      : '';
  }
  if (!rows.length) {
    fill(box, emptyNote('No index has been scanned yet. Start the scan, or wait for the next poll.'));
    return;
  }
  fill(box, rows.map(underlyingCard));
}

function underlyingCard(u) {
  const chg = isNum(u.change_pct) ? u.change_pct : null;
  const sigs = Array.isArray(u.signals) ? u.signals : [];
  return h('article', { class: 'ucard' + (u.enabled === false ? ' ucard--off' : '') }, [
    h('header', { class: 'ucard-head' }, [
      h('h3', { class: 'ucard-name', text: String(u.name || '') }),
      h('p', { class: 'ucard-spot' }, [
        h('span', { class: 'ucard-price', text: isNum(u.spot) ? num(u.spot, 2) : '—' }),
        h('span', { class: 'ucard-chg ' + signClass(chg) }, [
          dirMark(chg), h('span', { text: ' ' + pct(chg, { sign: true }) }),
        ]),
      ]),
    ]),
    h('p', { class: 'ucard-meta' }, [
      u.regime ? h('span', { class: 'badge badge--' + String(u.regime),
        text: String(u.regime).replace(/_/g, ' ') }) : null,
      h('span', { text: 'lot ' + (u.lot_size || '?') }),
      isNum(u.or_high) ? h('span', { text: 'OR ' + num(u.or_low, 0) + '–' + num(u.or_high, 0) }) : null,
      u.last_candle_ts ? h('span', { text: 'bar ' + tsTime(u.last_candle_ts) }) : null,
      isNum(u.bars) ? h('span', { text: u.bars + ' bars' }) : null,
    ]),
    u.data_error ? h('p', { class: 'warnline', text: String(u.data_error) }) : null,
    sigs.length
      ? h('ul', { class: 'ucard-sigs' }, sigs.map((g) => h('li', {}, [
          h('span', { class: 'sigrow-side side--' + String(g.side || '').toLowerCase(),
            text: String(g.side || '') }),
          h('span', { text: ' ' + String(g.setup || '') + ' · ' + num(g.score, 2) }),
        ])))
      : h('p', { class: 'ucard-quiet', text: 'No setup firing.' }),
  ]);
}

/* ======================================================================= *
 * 13. Render — learning
 * ======================================================================= */

function renderLearning(s) {
  const head = el('learning-headline');
  if (head) {
    const learning = s.learning || {};
    const total = learning.total_trades || learning.n || 0;
    head.textContent = total
      ? total + ' closed ' + plural(total, 'trade') + ' across '
        + (learning.buckets || (s.leaderboard || []).length) + ' buckets. A bucket is judged only '
        + 'after ' + ((s.config || {}).min_samples_to_gate || 10) + ' trades.'
      : 'Nothing learned yet. Every closed trade is filed by setup, side, time of day and market '
        + 'regime; once a bucket has enough trades the bot starts gating on it.';
  }

  const rows = Array.isArray(s.leaderboard) ? s.leaderboard : [];
  const table = el('leaderboard-table');
  const empty = el('leaderboard-empty');
  if (empty) empty.hidden = rows.length > 0;
  if (table) {
    fill(table, rows.length ? [
      h('thead', {}, h('tr', {},
        ['Bucket', 'n', 'Wins', 'Win rate', 'Lower bound', 'Break-even', 'Expectancy', 'State']
          .map((t) => h('th', { scope: 'col', text: t })))),
      h('tbody', {}, rows.map((v) => h('tr', { class: 'lb lb--' + String(v.state || '') }, [
        h('th', { scope: 'row', class: 'lb-bucket', text: String(v.bucket || '') }),
        h('td', { class: 'numcol', text: String(v.n || 0) }),
        h('td', { class: 'numcol', text: String(v.wins || 0) }),
        h('td', { class: 'numcol', text: pct((v.win_rate || 0) * 100, { decimals: 1 }) }),
        h('td', { class: 'numcol', text: pct((v.lower_bound || 0) * 100, { decimals: 1 }) }),
        h('td', { class: 'numcol', text: pct((v.break_even || 0) * 100, { decimals: 1 }) }),
        h('td', { class: 'numcol ' + signClass(v.expectancy),
          text: inr(v.expectancy, { sign: true, decimals: 0 }) }),
        h('td', {}, h('span', { class: 'state state--' + String(v.state || ''),
          text: String(v.state || '') })),
      ]))),
    ] : []);
  }

  // The note is the whole point of the gate: it explains itself in one sentence.
  const noteBox = document.getElementById('lb-notes') || (() => {
    if (!table || !table.parentNode) return null;
    const ul = h('ul', { class: 'notelist notelist--lb', id: 'lb-notes' });
    table.parentNode.insertBefore(ul, table.nextSibling);
    return ul;
  })();
  if (noteBox) {
    const withNotes = rows.filter((v) => v.note).slice(0, 8);
    fill(noteBox, withNotes.map((v) => h('li', {}, [
      h('strong', { text: String(v.bucket) + ': ' }),
      h('span', { text: String(v.note) }),
    ])));
  }

  renderTrades(s);
  renderEquity(s);
}

function renderTrades(s) {
  const rows = Array.isArray(s.recent_trades) ? s.recent_trades : [];
  const table = el('trades-table');
  const empty = el('trades-empty');
  const sub = el('trades-sub');
  if (sub) sub.textContent = rows.length ? 'Last ' + rows.length + ' in ' + s.mode + ' mode' : '';
  if (empty) empty.hidden = rows.length > 0;
  if (!table) return;
  fill(table, rows.length ? [
    h('thead', {}, h('tr', {},
      ['Contract', 'Setup', 'In', 'Out', 'Entry', 'Exit', 'P&L', 'Why']
        .map((t) => h('th', { scope: 'col', text: t })))),
    h('tbody', {}, rows.map((t) => h('tr', {}, [
      h('th', { scope: 'row', text: String(t.trading_symbol || t.label || '') }),
      h('td', { text: String(t.setup || '') }),
      h('td', { class: 'numcol', text: tsTime(t.entry_ts) }),
      h('td', { class: 'numcol', text: t.exit_ts ? tsTime(t.exit_ts) : 'open' }),
      h('td', { class: 'numcol', text: inr(t.entry_price) }),
      h('td', { class: 'numcol',
        text: (t.exit_price === null || t.exit_price === undefined) ? '—' : inr(t.exit_price) }),
      h('td', { class: 'numcol ' + signClass(t.pnl) }, isNum(t.pnl)
        ? [dirMark(t.pnl), h('span', { text: ' ' + inr(t.pnl, { sign: true, decimals: 0 }) })]
        : [h('span', { text: '—' })]),
      h('td', { text: String(t.exit_reason || '') }),
    ]))),
  ] : []);
}

/* ======================================================================= *
 * 14. Charts
 * ======================================================================= */

/** A line chart: one series, solid hairline grid, neutral zero rule, selective
 *  endpoint labels, crosshair + tooltip. Returns an <svg>.
 *
 *  Deliberately NOT a dual-axis chart. Where two measures matter (equity and
 *  drawdown) the caller stacks two of these on a shared x-axis instead: two
 *  y-scales on one plot invents a correlation that is not in the data. */
function lineChart(points, opts) {
  const o = opts || {};
  const w = o.width || 720;
  const hgt = o.height || 220;
  const pad = { t: 14, r: 60, b: 30, l: 62 };
  const svg = svgEl('svg', {
    class: 'chart', viewBox: '0 0 ' + w + ' ' + hgt, width: '100%', height: hgt,
    role: 'img', 'aria-label': o.label || 'chart',
  });
  if (!points.length) return svg;

  const ys = points.map((p) => p.y);
  let lo = Math.min.apply(null, ys);
  let hi = Math.max.apply(null, ys);
  if (o.zeroAnchored) { lo = Math.min(0, lo); hi = Math.max(0, hi); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;

  const px = (i) => pad.l + (points.length === 1
    ? (w - pad.l - pad.r) / 2
    : (i / (points.length - 1)) * (w - pad.l - pad.r));
  const py = (v) => hgt - pad.b - ((v - lo) / (hi - lo)) * (hgt - pad.t - pad.b);

  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    const y = py(v);
    svg.appendChild(svgEl('line', { class: 'chart-grid', x1: pad.l, x2: w - pad.r, y1: y, y2: y }));
    const t = svgEl('text', { class: 'chart-tick', x: pad.l - 8, y: y + 4, 'text-anchor': 'end' });
    t.textContent = o.money ? inr(v, { decimals: 0 }) : num(v, 0);
    svg.appendChild(t);
  }
  svg.appendChild(svgEl('line', { class: 'chart-axis',
    x1: pad.l, x2: w - pad.r, y1: hgt - pad.b, y2: hgt - pad.b }));

  // Zero rule — position against this is the non-colour channel for sign.
  if (lo < 0 && hi > 0) {
    const zy = py(0);
    svg.appendChild(svgEl('line', { class: 'chart-zero', x1: pad.l, x2: w - pad.r, y1: zy, y2: zy }));
  }

  if (o.area !== false) {
    const zeroY = Math.min(hgt - pad.b, Math.max(pad.t, py(Math.max(lo, Math.min(hi, 0)))));
    const d = points.map((p, i) => (i ? 'L' : 'M') + px(i) + ',' + py(p.y)).join(' ');
    svg.appendChild(svgEl('path', {
      class: 'chart-area',
      d: d + ' L' + px(points.length - 1) + ',' + zeroY + ' L' + px(0) + ',' + zeroY + ' Z',
    }));
  }

  svg.appendChild(svgEl('polyline', {
    class: 'chart-line', fill: 'none',
    points: points.map((p, i) => px(i) + ',' + py(p.y)).join(' '),
  }));

  // Direct-label the endpoint only — a number on every point goes unread.
  const lastPt = points[points.length - 1];
  svg.appendChild(svgEl('circle', {
    class: 'chart-dot', cx: px(points.length - 1), cy: py(lastPt.y), r: 4,
  }));
  const lab = svgEl('text', {
    class: 'chart-endlabel', x: Math.min(w - 4, px(points.length - 1) + 8),
    y: py(lastPt.y) + 4, 'text-anchor': 'start',
  });
  lab.textContent = o.money ? inr(lastPt.y, { sign: o.sign !== false, decimals: 0 }) : num(lastPt.y, 1);
  svg.appendChild(lab);

  const xf = svgEl('text', { class: 'chart-tick', x: pad.l, y: hgt - 9, 'text-anchor': 'start' });
  xf.textContent = String(points[0].label || '');
  svg.appendChild(xf);
  if (points.length > 1) {
    const xl = svgEl('text', { class: 'chart-tick', x: w - pad.r, y: hgt - 9, 'text-anchor': 'end' });
    xl.textContent = String(lastPt.label || '');
    svg.appendChild(xl);
  }

  attachCrosshair(svg, points, { px, py, pad, w, hgt, money: o.money });
  return svg;
}

/** Crosshair + tooltip. An SVG chart in a browser is interactive; shipping it
 *  inert is a defect, not a simplification. */
function attachCrosshair(svg, points, geo) {
  const line = svgEl('line', { class: 'chart-cross', x1: 0, x2: 0,
    y1: geo.pad.t, y2: geo.hgt - geo.pad.b, visibility: 'hidden' });
  const dot = svgEl('circle', { class: 'chart-crossdot', r: 4, visibility: 'hidden' });
  const box = svgEl('g', { class: 'chart-tip', visibility: 'hidden' });
  box.appendChild(svgEl('rect', { rx: 4, x: 0, y: 0, width: 152, height: 36 }));
  const l1 = svgEl('text', { x: 9, y: 15 });
  const l2 = svgEl('text', { x: 9, y: 28, class: 'chart-tip-sub' });
  box.appendChild(l1);
  box.appendChild(l2);
  svg.appendChild(line);
  svg.appendChild(dot);
  svg.appendChild(box);

  const hit = svgEl('rect', {
    x: geo.pad.l, y: geo.pad.t,
    width: Math.max(0, geo.w - geo.pad.l - geo.pad.r),
    height: Math.max(0, geo.hgt - geo.pad.t - geo.pad.b),
    fill: 'transparent',
  });
  svg.appendChild(hit);

  const show = (clientX) => {
    const r = svg.getBoundingClientRect();
    if (!r.width) return;
    const vx = ((clientX - r.left) / r.width) * geo.w;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < points.length; i++) {
      const d = Math.abs(geo.px(i) - vx);
      if (d < bestD) { bestD = d; best = i; }
    }
    const p = points[best];
    const x = geo.px(best);
    const y = geo.py(p.y);
    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
    dot.setAttribute('cx', x);
    dot.setAttribute('cy', y);
    l1.textContent = geo.money ? inr(p.y, { sign: true, decimals: 0 }) : num(p.y, 2);
    l2.textContent = String(p.label || '');
    const bx = Math.min(geo.w - 158, Math.max(2, x + 10));
    box.setAttribute('transform', 'translate(' + bx + ',' + Math.max(2, y - 44) + ')');
    for (const n of [line, dot, box]) n.setAttribute('visibility', 'visible');
  };
  const hide = () => { for (const n of [line, dot, box]) n.setAttribute('visibility', 'hidden'); };

  hit.addEventListener('mousemove', (e) => show(e.clientX));
  hit.addEventListener('mouseleave', hide);
  hit.addEventListener('touchstart', (e) => { if (e.touches[0]) show(e.touches[0].clientX); },
    { passive: true });
  hit.addEventListener('touchmove', (e) => { if (e.touches[0]) show(e.touches[0].clientX); },
    { passive: true });
  hit.addEventListener('touchend', hide);
}

/** The table view every chart needs — how a CVD or screen-reader user reads it. */
function chartTable(points, headers, money) {
  return h('table', { class: 'datatable' }, [
    h('thead', {}, h('tr', {}, headers.map((t) => h('th', { scope: 'col', text: t })))),
    h('tbody', {}, points.map((p) => h('tr', {}, [
      h('th', { scope: 'row', text: String(p.label || '') }),
      h('td', { class: 'numcol ' + signClass(p.y),
        text: money ? inr(p.y, { sign: true, decimals: 0 }) : num(p.y, 2) }),
    ]))),
  ]);
}

function wireTableToggle(btnId, tableId) {
  const btn = el(btnId);
  const table = el(tableId);
  if (!btn || !table || btn.dataset.wired) return;
  btn.dataset.wired = '1';
  btn.addEventListener('click', () => {
    const show = table.hidden;
    table.hidden = !show;
    btn.setAttribute('aria-expanded', show ? 'true' : 'false');
    btn.textContent = show ? 'Hide table' : 'Show table';
  });
}

function renderEquity(s) {
  const box = el('equity-chart');
  if (!box) return;
  const empty = el('equity-empty');
  const tableBox = el('equity-table');
  const trades = (s.recent_trades || [])
    .filter((t) => isNum(t.pnl) && t.exit_ts)
    .sort((a, b) => (a.exit_ts || 0) - (b.exit_ts || 0));
  if (trades.length < 2) {
    fill(box);
    if (empty) empty.hidden = false;
    if (tableBox) fill(tableBox);
    return;
  }
  if (empty) empty.hidden = true;
  let run = 0;
  const points = trades.map((t) => {
    run += t.pnl;
    return { y: run, label: tsTime(t.exit_ts) };
  });
  fill(box, lineChart(points, {
    money: true, zeroAnchored: true, height: 220,
    label: 'Cumulative realised P&L over the last ' + points.length + ' closed trades',
  }));
  if (tableBox) fill(tableBox, chartTable(points, ['Closed at', 'Cumulative P&L'], true));
  wireTableToggle('equity-table-toggle', 'equity-table');
}

/* ======================================================================= *
 * 15. Backtest
 * ======================================================================= */

function initBacktest(s) {
  const form = el('backtest-form');
  if (form && !form.dataset.wired) {
    form.dataset.wired = '1';
    form.addEventListener('submit', (e) => { e.preventDefault(); runBacktest(); });
  }
  const cancel = el('bt-cancel');
  if (cancel && !cancel.dataset.wired) {
    cancel.dataset.wired = '1';
    cancel.addEventListener('click', cancelBacktest);
  }
  const sel = el('bt-underlying');
  if (sel && !sel.options.length) {
    const names = (s && s.config && s.config.underlyings) || ['NIFTY'];
    fill(sel, names.map((n) => h('option', { value: n, text: n })));
  }
  const start = el('bt-start');
  const end = el('bt-end');
  if (start && end && !start.value) {
    const now = new Date();
    end.value = now.toISOString().slice(0, 10);
    start.value = new Date(now.getTime() - 30 * 864e5).toISOString().slice(0, 10);
  }
}

async function runBacktest() {
  const msg = el('bt-msg');
  const btn = el('bt-run');
  const body = {
    underlying: (el('bt-underlying') || {}).value,
    start: (el('bt-start') || {}).value,
    end: (el('bt-end') || {}).value,
  };
  if (!body.underlying || !body.start || !body.end) {
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = 'Pick an index and both dates.'; }
    return;
  }
  if (btn) btn.disabled = true;
  if (msg) { msg.className = 'formmsg'; msg.textContent = 'Starting…'; }
  try {
    const out = await api.post('/api/backtest', body);
    state.btJob = out && (out.job_id || out.id);
    if (!state.btJob) throw new ApiError('The server did not return a job id.', 0, out);
    if (msg) {
      msg.textContent = 'Running. Historical option data is rate-limited to 5 requests a second, '
        + 'so a long window takes a while; progress is below.';
    }
    pollBacktest();
  } catch (err) {
    if (btn) btn.disabled = false;
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = 'Could not start: ' + err.message; }
  }
}

function cancelBacktest() {
  if (!state.btJob) return;
  api.post('/api/backtest/' + encodeURIComponent(state.btJob) + '/cancel').catch(() => {});
  if (state.btTimer) clearTimeout(state.btTimer);
  state.btJob = null;
  const btn = el('bt-run');
  if (btn) btn.disabled = false;
  const msg = el('bt-msg');
  if (msg) msg.textContent = 'Cancelled.';
}

async function pollBacktest() {
  if (!state.btJob) return;
  try {
    const out = await api.get('/api/backtest/' + encodeURIComponent(state.btJob));
    const status = String((out && out.status) || '');
    const prog = el('bt-progress');
    if (prog && out && out.progress) {
      const p = out.progress;
      const donePct = p.total ? Math.round((p.done / p.total) * 100) : 0;
      prog.hidden = false;
      prog.textContent = 'Day ' + (p.done || 0) + ' of ' + (p.total || '?') + ' (' + donePct + '%)'
        + (p.day ? ' — ' + p.day : '')
        + (isNum(p.trades) ? ' · ' + p.trades + ' trades' : '')
        + (isNum(p.pnl) ? ' · ' + inr(p.pnl, { sign: true, decimals: 0 }) : '');
    }
    if (status === 'done' || status === 'completed' || (out && out.result)) {
      state.btResult = (out && out.result) || out;
      state.btJob = null;
      const btn = el('bt-run');
      if (btn) btn.disabled = false;
      const msg = el('bt-msg');
      if (msg) { msg.className = 'formmsg good'; msg.textContent = 'Done.'; }
      renderBacktestResult();
      return;
    }
    if (status === 'error' || status === 'failed') {
      throw new ApiError((out && (out.error || out.message)) || 'The backtest failed.', 0, out);
    }
    state.btTimer = setTimeout(pollBacktest, 1200);
  } catch (err) {
    state.btJob = null;
    const btn = el('bt-run');
    if (btn) btn.disabled = false;
    const msg = el('bt-msg');
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = err.message; }
  }
}

function renderBacktestResult() {
  const r = state.btResult;
  const wrap = el('bt-result');
  const empty = el('bt-empty');
  if (!wrap) return;
  if (!r) {
    wrap.hidden = true;
    if (empty) empty.hidden = false;
    return;
  }
  wrap.hidden = false;
  if (empty) empty.hidden = true;

  const sum = r.summary || r;
  const pnl = isNum(sum.net_pnl) ? sum.net_pnl : (isNum(sum.pnl) ? sum.pnl : null);

  const head = el('bt-headline');
  if (head) {
    head.textContent = String(r.underlying || '') + ' ' + String(r.start || '') + ' to '
      + String(r.end || '') + ': ' + inr(pnl, { sign: true, decimals: 0 }) + ' over '
      + (sum.trades || sum.trade_count || 0) + ' trades.';
    head.className = 'headline ' + signClass(pnl);
  }

  const stats = el('bt-stats');
  if (stats) {
    const wr = sum.win_rate;
    fill(stats, [
      tile('Net P&L', inr(pnl, { sign: true, decimals: 0 }), { valueClass: signClass(pnl) }),
      tile('Trades', String(sum.trades || sum.trade_count || 0)),
      tile('Win rate', pct(isNum(wr) ? (wr <= 1 ? wr * 100 : wr) : null, { decimals: 1 })),
      tile('Max drawdown', inr(sum.max_drawdown, { decimals: 0 })),
      tile('Expectancy', inr(sum.expectancy, { sign: true, decimals: 0 }),
        { valueClass: signClass(sum.expectancy) }),
    ]);
  }

  // Equity and drawdown as TWO STACKED SMALL MULTIPLES sharing an x-axis.
  const curve = Array.isArray(r.equity_curve) ? r.equity_curve : [];
  const points = curve.map((p, i) => ({
    y: isNum(p.equity) ? p.equity
      : (isNum(p.cum_pnl) ? p.cum_pnl : (isNum(p.pnl) ? p.pnl : 0)),
    label: String(p.day || p.date || (i + 1)),
  }));

  const eq = el('bt-equity-chart');
  if (eq) {
    fill(eq, points.length > 1
      ? lineChart(points, { money: true, zeroAnchored: true, height: 200,
          label: 'Backtest equity curve' })
      : emptyNote('Not enough closed trades to draw an equity curve.'));
  }
  const dd = el('bt-dd-chart');
  if (dd) {
    if (points.length > 1) {
      let peak = -Infinity;
      const ddPts = points.map((p) => {
        peak = Math.max(peak, p.y);
        return { y: p.y - peak, label: p.label };
      });
      fill(dd, lineChart(ddPts, { money: true, zeroAnchored: true, height: 140,
        label: 'Drawdown from the running peak' }));
    } else {
      fill(dd);
    }
  }

  const tbl = el('bt-curve-table');
  if (tbl && points.length) {
    fill(tbl, chartTable(points, ['Day', 'Equity'], true));
    wireTableToggle('bt-curve-table-toggle', 'bt-curve-table');
  }

  // Caveats are not fine print here — they are what makes the number honest.
  const cav = el('bt-caveats');
  if (cav) {
    const caveats = Array.isArray(r.caveats) ? r.caveats : [];
    fill(cav, caveats.length
      ? caveats.map((c) => h('li', { text: String(c) }))
      : [h('li', { text: 'This result carries no caveat list, which is itself worth distrusting.' })]);
  }

  const skipped = el('bt-skipped');
  if (skipped) {
    const n = r.days_skipped || (r.stats && r.stats.days_skipped) || 0;
    skipped.textContent = n
      ? n + ' ' + plural(n, 'day') + ' were skipped for missing data. A silent gap reads as a '
        + 'clean backtest when it is not, so they are counted here.'
      : '';
  }

  const blocks = el('bt-blocks');
  if (blocks) {
    const rows = Array.isArray(r.blocked) ? r.blocked : [];
    fill(blocks, rows.length
      ? rows.slice(0, 20).map((b) => h('li', { text: String((b && b.reason) || b) }))
      : [h('li', { text: 'No signals were blocked by the risk gate in this window.' })]);
  }

  const trades = el('bt-trades');
  if (trades) {
    const rows = Array.isArray(r.trades) ? r.trades : [];
    fill(trades, rows.length ? [
      h('thead', {}, h('tr', {},
        ['Day', 'Contract', 'Setup', 'Entry', 'Exit', 'P&L', 'Why']
          .map((t) => h('th', { scope: 'col', text: t })))),
      h('tbody', {}, rows.slice(0, 200).map((t) => h('tr', {}, [
        h('th', { scope: 'row', text: String(t.day || t.date || '') }),
        h('td', { text: String(t.trading_symbol || '') }),
        h('td', { text: String(t.setup || '') }),
        h('td', { class: 'numcol', text: inr(t.entry_price) }),
        h('td', { class: 'numcol', text: inr(t.exit_price) }),
        h('td', { class: 'numcol ' + signClass(t.pnl),
          text: inr(t.pnl, { sign: true, decimals: 0 }) }),
        h('td', { text: String(t.exit_reason || '') }),
      ]))),
    ] : []);
  }
}

/* ======================================================================= *
 * 16. Settings
 * ======================================================================= */

/* One sentence per setting: what it does, and what changing it means. */
const HELP = {
  max_daily_loss: 'Realised loss that halts trading for the day. The halt is written to the '
    + 'database, so restarting the bot cannot resume trading on a day that already hit it.',
  max_trade_loss: 'The most one trade may lose at its stop. This is the setting that blocks '
    + 'at-the-money strikes — see the budget table beside this form.',
  max_trades_per_day: 'Entries per day. Exits are never counted and never blocked.',
  max_open_positions: 'How many positions may be open at once.',
  max_lots: 'Lots per entry. One lot is the minimum the exchange allows.',
  high_conviction_enabled: 'Let a high-scoring signal risk more than the ordinary '
    + 'per-trade limit, in exchange for a required reward-to-risk. Off means every trade '
    + 'uses max_trade_loss.',
  high_conviction_score: 'The score at or above which the raised cap applies. Note that '
    + 'score is the strategy’s own confluence number, not a measured probability — until '
    + 'the learner has closed trades in a bucket, a high score means "this setup ticked more '
    + 'boxes", not "this is more likely to win".',
  high_conviction_max_trade_loss: 'The most a high-conviction trade may lose. Capped at '
    + 'max_daily_loss, and never less than max_trade_loss.',
  high_conviction_rr: 'Reward-to-risk required of a high-conviction trade. At 2, a 30% stop '
    + 'gets a 60% target — the bigger stake has to buy a bigger payoff, not the same one.',
  stop_pct: 'Stop distance as a fraction of the premium paid. 0.30 means a 30% fall exits.',
  target_pct: 'Target as a fraction of the premium. With a 30% stop, 0.50 needs a 37.5% win rate '
    + 'just to break even.',
  trail_trigger_pct: 'Profit at which the stop is lifted to break-even, as a fraction of premium.',
  min_stop_pct: 'The tightest stop allowed. Below this the bid-ask spread alone would trigger the '
    + 'stop and you would pay the loss for nothing.',
  min_score: 'Confluence score a signal needs before the chain picker is even asked.',
  or_minutes: 'Length of the opening range, in minutes from 09:15.',
  interval: 'Candle size the strategy runs on.',
  min_delta: 'Lowest absolute delta to buy. Lower means cheaper and further out of the money.',
  max_delta: 'Highest absolute delta to buy.',
  max_spread_pct: 'Widest bid-ask spread, as a percentage of mid, that is still tradable.',
  min_oi: 'Least open interest a strike must have.',
  min_volume: "Least of today's volume a strike must have.",
  max_premium_per_lot: 'Hard premium cap per unit. Leave at 0 to derive it from the per-trade '
    + 'loss limit and the stop.',
  no_entry_before: 'No entries before this time — the first minutes after the open are noise.',
  no_entry_after: 'No entries after this time, so a trade has room to work before the flat-by.',
  flat_by: 'Everything is closed by this time, win or lose.',
  time_stop_minutes: 'Exit a position that has gone nowhere for this many minutes.',
  poll_seconds: 'Seconds between ticks. The data rate limit is 5 requests a second across all five '
    + 'indices, so do not set this too low.',
  learning_enabled: 'Whether the outcome gate may change a signal’s score at all.',
  min_samples_to_gate: 'Trades a bucket needs before the bot will judge it.',
  gate_confidence: 'Confidence level for the Wilson lower bound on a bucket’s win rate.',
  learn_across_modes: 'Off by default, so a paper losing streak cannot block a live setup.',
  cost_per_order: 'Modelled cost per order in paper mode and backtests.',
  slippage_ticks: 'Ticks of slippage assumed on a simulated fill.',
  backtest_spread_pct: 'Spread assumed in backtests, since expired-contract candles carry no book.',
  underlyings: 'Indices to scan. Drop FINNIFTY and MIDCPNIFTY if the log fills with liquidity '
    + 'rejections away from their expiry day.',
  enabled_setups: 'Which of the six setups may fire.',
};

const EDITABLE = Object.keys(HELP);

function renderSettings(s) {
  const box = el('settings-fields');
  if (!box) return;
  renderRiskTable(s);
  if (state.settingsDirty) return;   // never clobber a half-typed form on a tick
  const cfg = s.config || {};
  const fields = EDITABLE.filter((k) => Object.prototype.hasOwnProperty.call(cfg, k));
  fill(box, fields.map((key) => {
    const v = cfg[key];
    const id = 'set-' + key;
    let input;
    if (typeof v === 'boolean') {
      input = h('input', { type: 'checkbox', id, name: key, checked: v || null,
        'aria-describedby': id + '-help' });
    } else if (Array.isArray(v)) {
      input = h('input', { type: 'text', id, name: key, value: v.join(','),
        'aria-describedby': id + '-help', spellcheck: 'false' });
    } else if (typeof v === 'number') {
      input = h('input', { type: 'number', id, name: key, value: String(v),
        step: Number.isInteger(v) ? '1' : 'any', 'aria-describedby': id + '-help' });
    } else {
      input = h('input', { type: 'text', id, name: key,
        value: String(v === null || v === undefined ? '' : v),
        'aria-describedby': id + '-help', spellcheck: 'false' });
    }
    input.addEventListener('input', () => { state.settingsDirty = true; });
    return h('div', { class: 'field' + (typeof v === 'boolean' ? ' field--bool' : '') }, [
      h('label', { for: id, text: labelise(key) }),
      input,
      h('p', { class: 'field-hint', id: id + '-help', text: HELP[key] || '' }),
    ]);
  }));
}

function renderRiskTable(s) {
  const rows = Array.isArray(s.limits) ? s.limits : [];
  const table = el('risk-table');
  if (table) {
    fill(table, rows.length ? [
      h('thead', {}, h('tr', {},
        ['Index', 'Lot', 'Max premium / unit', 'Max cost / lot', 'Typical ATM', 'Fits?']
          .map((t) => h('th', { scope: 'col', text: t })))),
      h('tbody', {}, rows.map((r) => h('tr', { class: r.fits_typical_atm ? '' : 'row--blocked' }, [
        h('th', { scope: 'row', text: String(r.underlying || '') }),
        h('td', { class: 'numcol', text: String(r.lot_size || '?') }),
        h('td', { class: 'numcol', text: inr(r.max_premium_per_unit) }),
        h('td', { class: 'numcol', text: inr(r.max_lot_cost, { decimals: 0 }) }),
        h('td', { class: 'numcol', text: isNum(r.typical_atm_low)
          ? inr(r.typical_atm_low, { decimals: 0 }) + '–' + inr(r.typical_atm_high, { decimals: 0 })
          : '—' }),
        h('td', {}, h('span', {
          class: 'state state--' + (r.fits_typical_atm ? 'promoted' : 'blocked'),
          text: r.fits_typical_atm ? 'yes' : 'no',
        })),
      ]))),
    ] : []);
  }
  const notes = el('risk-notes');
  if (notes) fill(notes, rows.filter((r) => r.note).map((r) => h('li', { text: String(r.note) })));
}

function initSettings() {
  const form = el('settings-form');
  if (form && !form.dataset.wired) {
    form.dataset.wired = '1';
    form.addEventListener('submit', (e) => { e.preventDefault(); saveSettings(); });
  }
  const reset = el('btn-reset-settings');
  if (reset && !reset.dataset.wired) {
    reset.dataset.wired = '1';
    reset.addEventListener('click', () => {
      state.settingsDirty = false;
      if (state.snap) renderSettings(state.snap);
      const msg = el('settings-msg');
      if (msg) { msg.className = 'formmsg'; msg.textContent = 'Reverted to the running config.'; }
    });
  }
}

async function saveSettings() {
  const msg = el('settings-msg');
  const cfg = (state.snap && state.snap.config) || {};
  const patch = {};
  for (const key of EDITABLE) {
    const input = document.getElementById('set-' + key);
    if (!input) continue;
    const cur = cfg[key];
    if (typeof cur === 'boolean') {
      patch[key] = input.checked;
    } else if (Array.isArray(cur)) {
      patch[key] = input.value.split(',').map((x) => x.trim()).filter(Boolean);
    } else if (typeof cur === 'number') {
      const n = parseFloat(input.value);
      if (!isFinite(n)) {
        if (msg) {
          msg.className = 'formmsg bad';
          msg.textContent = labelise(key) + ' is not a number.';
        }
        return;
      }
      patch[key] = n;
    } else {
      patch[key] = input.value;
    }
  }
  delete patch.armed;   // arming is never a settings change
  if (msg) { msg.className = 'formmsg'; msg.textContent = 'Saving…'; }
  try {
    const out = await api.post('/api/config', patch);
    state.settingsDirty = false;
    const changed = (out && out.changed) || [];
    if (msg) {
      msg.className = 'formmsg good';
      msg.textContent = changed.length ? 'Saved: ' + changed.join(', ') + '.' : 'Nothing changed.';
    }
    await refreshNow();
  } catch (err) {
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = 'Not saved: ' + err.message; }
  }
}

/* ======================================================================= *
 * 17. Diagnostics & log
 * ======================================================================= */

/* ---------------------------------------------------------- access token */

/** Token status, plus a banner once it is within an hour of lapsing.
 *  Tokens die at 07:00 IST daily, so this is a standing chore and the UI should
 *  warn before it becomes a 403 in the middle of a session. */
function renderToken(s) {
  const t = s.token || {};
  const line = el('token-status-text');
  if (line) {
    line.textContent = t.present
      ? String(t.status_text || 'Present.')
      : 'No token. The bot cannot fetch prices or place orders until one is set.';
    line.className = t.expired ? 'bad' : (t.expiring_soon ? 'warn' : '');
  }

  const box = el('token-status');
  if (box) {
    const rows = [['Status', line ? line.textContent : '']];
    if (t.present) {
      rows.push(['Token', String(t.masked || '') + '  (' + (t.length || 0) + ' chars)']);
      if (t.client_id) rows.push(['Client id', String(t.client_id)]);
      if (t.issued_at_text) rows.push(['Issued', String(t.issued_at_text)]);
      if (t.expires_at_text) rows.push(['Expires', String(t.expires_at_text)]);
      if (t.source) rows.push(['Source', String(t.source)]);
      rows.push(['Broker link', t.client_attached ? 'connected' : 'not connected']);
    }
    fill(box, rows.flatMap(([k, v], i) => [
      h('dt', { text: k }),
      h('dd', { class: i === 0 && (t.expired || t.expiring_soon) ? 'bad' : '', text: v }),
    ]));
  }

  // An env-var token cannot be replaced from here; say so rather than letting
  // the form look like it will work.
  const save = el('token-save');
  const input = el('token-input');
  if (t.env_override) {
    if (save) save.disabled = true;
    if (input) {
      input.disabled = true;
      input.placeholder = 'Set by $INDSTOCKS_ACCESS_TOKEN — unset it to paste here';
    }
  } else {
    if (save) save.disabled = false;
    if (input) input.disabled = false;
  }

  banner('token-banner', t.expired
    ? 'The access token has expired, so no prices are being fetched and no order could be '
      + 'placed. Paste a fresh one in Settings → Access token.'
    : (t.expiring_soon
      ? 'The access token ' + String(t.status_text || 'expires shortly')
        + ' Refresh it in Settings → Access token before the next session.'
      : ''));
}

function initToken() {
  const form = el('token-form');
  if (form && !form.dataset.wired) {
    form.dataset.wired = '1';
    form.addEventListener('submit', (e) => { e.preventDefault(); saveToken(); });
  }
  const reveal = el('token-reveal');
  if (reveal && !reveal.dataset.wired) {
    reveal.dataset.wired = '1';
    reveal.addEventListener('click', () => {
      const input = el('token-input');
      if (!input) return;
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      reveal.textContent = show ? 'Hide' : 'Show';
      reveal.setAttribute('aria-pressed', show ? 'true' : 'false');
    });
  }
  const clear = el('token-clear');
  if (clear && !clear.dataset.wired) {
    clear.dataset.wired = '1';
    clear.addEventListener('click', async () => {
      const msg = el('token-msg');
      if (!window.confirm('Forget the stored token? The bot will stop fetching prices '
        + 'until you paste a new one.')) return;
      try {
        await api.request('DELETE', '/api/token');
        if (msg) { msg.className = 'formmsg'; msg.textContent = 'Stored token removed.'; }
        await refreshNow();
      } catch (err) {
        if (msg) { msg.className = 'formmsg bad'; msg.textContent = err.message; }
      }
    });
  }
}

async function saveToken() {
  const input = el('token-input');
  const msg = el('token-msg');
  const btn = el('token-save');
  const raw = input ? input.value.trim() : '';
  if (!raw) {
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = 'Paste a token first.'; }
    return;
  }
  if (btn) btn.disabled = true;
  if (msg) { msg.className = 'formmsg'; msg.textContent = 'Checking it against your profile…'; }
  try {
    const out = await api.post('/api/token', { token: raw });
    const info = (out && out.token) || {};
    if (input) { input.value = ''; input.type = 'password'; }
    const reveal = el('token-reveal');
    if (reveal) { reveal.textContent = 'Show'; reveal.setAttribute('aria-pressed', 'false'); }
    if (msg) {
      msg.className = 'formmsg good';
      msg.textContent = 'Token accepted' + (info.client_id ? ' for client ' + info.client_id : '')
        + '. ' + String(info.status_text || '');
    }
    alertNow('Access token updated.');
    await refreshNow();
  } catch (err) {
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = err.message; }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderDiag(s) {
  const box = el('diag-list');
  if (!box) return;
  const tick = s.tick || {};
  const inst = s.instruments || {};
  const funds = s.funds || null;
  const rows = [
    ['Mode', String(s.mode) + (s.armed ? ' (ARMED)' : '')],
    ['Engine', s.running ? 'scanning' : 'stopped'],
    ['Broker link', s.connected ? 'client attached' : 'no client — backtest, or no token'],
    ['Ticks', String(tick.count || 0) + (tick.last_at ? ' · last ' + istTime(tick.last_at) : '')],
    ['Candle interval', prettyInterval(tick.interval)],
    ['Session', String((s.session || {}).state || '?')],
    ['Instrument master', inst.sources ? Object.keys(inst.sources).join(', ') : 'not loaded'],
    ['Account', s.account ? String(s.account.user_id || s.account.ucc || 'known') : 'unknown'],
  ];
  if (funds && funds.detailed_avl_balance) {
    rows.push(['Option-buy balance',
      inr(funds.detailed_avl_balance.option_buy, { decimals: 0 })]);
  }
  fill(box, rows.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })]));
}

function renderLog(s) {
  const box = el('log-list');
  if (!box) return;
  const rows = Array.isArray(s.log) ? s.log.slice().reverse() : [];
  const empty = el('log-empty');
  if (empty) empty.hidden = rows.length > 0;
  fill(box, rows.slice(0, 200).map((r) =>
    h('li', { class: 'logline log--' + String(r.level || '').toLowerCase() }, [
      h('span', { class: 'log-at', text: String(r.at || tsTime(r.ts)) }),
      h('span', { class: 'log-level', text: String(r.level || '') }),
      h('span', { class: 'log-msg', text: String(r.msg || '') }),
    ])));
}

/* ======================================================================= *
 * 18. First-run card
 * ======================================================================= */

function renderSetupCard(s) {
  const card = el('setup-card');
  if (!card) return;
  // Backtest mode is NOT exempt: the backtester reads historical option premiums
  // over the same API, so without a token it cannot run either. Hiding this card
  // in backtest mode left a dashboard of zeroes and no explanation.
  const needsToken = !s.connected;
  const neverTicked = !(s.tick && s.tick.count);
  const show = needsToken || (neverTicked && !s.running);
  card.hidden = !show;
  document.body.dataset.boot = show ? 'true' : 'false';
  if (!show) return;
  const st = el('setup-state');
  if (st) {
    if (needsToken) {
      st.textContent = 'No broker connection. INDSTOCKS_ACCESS_TOKEN is missing or the token has '
        + 'expired, so no market data can be fetched — live, paper and backtest all read prices '
        + 'over the same API, so none of the three can run until this is fixed.';
    } else {
      st.textContent = 'Connected to the broker, but the scan loop has not run yet. '
        + 'Press START SCAN above.';
    }
  }
}

/* ======================================================================= *
 * 19. Modals
 * ======================================================================= */

let lastFocus = null;

function openModal(dlg) {
  if (!dlg) return;
  lastFocus = document.activeElement;
  if (!dlg.open) {
    try { dlg.showModal(); } catch (e) { dlg.setAttribute('open', ''); }
  }
}

function closeModal(dlg) {
  if (!dlg) return;
  try { dlg.close(); } catch (e) { dlg.removeAttribute('open'); }
  if (lastFocus && lastFocus.focus) lastFocus.focus();
}

function sheet(title, nodes) {
  const t = el('sheet-title');
  if (t) t.textContent = title;
  const body = el('sheet-body');
  if (body) fill(body, nodes);
  openModal(el('sheet'));
  const close = el('sheet-close');
  if (close) close.focus();
}

/* --- live arming gate ------------------------------------------------- */

function openLiveGate() {
  const s = state.snap || {};
  const cfg = s.config || {};
  const day = s.day || {};
  const phrase = s.arm_phrase || 'ARM LIVE TRADING';

  const ptext = el('arm-phrase-text');
  if (ptext) ptext.textContent = phrase;

  const body = el('live-gate-body');
  if (body) {
    const openCount = (s.positions || []).length;
    fill(body, [
      h('dl', { class: 'kv' }, [
        ['Account', s.account ? String(s.account.user_id || s.account.ucc || 'unknown') : 'unknown'],
        ['Max loss per trade', inr(cfg.max_trade_loss, { decimals: 0 })],
        ['Max loss per day', inr(cfg.max_daily_loss, { decimals: 0 })],
        ['Trades allowed today', String(cfg.max_trades_per_day) + ' (used ' + (day.trade_count || 0) + ')'],
        ['Open positions allowed', String(cfg.max_open_positions)],
        ['Lots per entry', String(cfg.max_lots)],
        ['Stop / target', pct((cfg.stop_pct || 0) * 100, { decimals: 0 }) + ' / '
          + pct((cfg.target_pct || 0) * 100, { decimals: 0 })],
        ['Everything flat by', String(cfg.flat_by)],
        ["Today's realised P&L", inr(day.realized_pnl, { sign: true, decimals: 0 })],
        ['Open right now', openCount + ' ' + plural(openCount, 'position')],
      ].flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })])),
      h('p', { class: 'modal-warn', text:
        'Once armed, the bot places real BUY and SELL orders on INDmoney without asking again. It '
        + 'stays armed until you disarm it, switch mode, or restart the process.' }),
    ]);
  }

  const input = el('arm-input');
  const armBtn = el('live-gate-arm');
  const hint = el('arm-hint');
  if (input) {
    input.value = '';
    input.oninput = () => {
      const ok = input.value.trim().toUpperCase() === phrase.toUpperCase();
      if (armBtn) armBtn.disabled = !ok;
      if (hint) {
        hint.textContent = ok
          ? 'Phrase matches. The ARM button is now enabled.'
          : 'Not armed. Type the phrase exactly.';
      }
    };
  }
  if (armBtn) armBtn.disabled = true;
  const msg = el('live-gate-msg');
  if (msg) msg.textContent = '';
  openModal(el('live-gate'));
  if (input) input.focus();
}

async function doArm() {
  const msg = el('live-gate-msg');
  const input = el('arm-input');
  const phrase = (state.snap && state.snap.arm_phrase) || 'ARM LIVE TRADING';
  if (msg) { msg.className = 'formmsg'; msg.textContent = 'Arming…'; }
  try {
    if (String((state.snap || {}).mode) !== 'live') await api.post('/api/mode', { mode: 'live' });
    // The server passes the phrase straight to engine.arm(), which is the one place
    // in the codebase that decides what arms this bot.
    await api.post('/api/arm', { confirm: input ? input.value.trim() : phrase });
    closeModal(el('live-gate'));
    await refreshNow();
    alertNow('Live trading is ARMED. Real orders will be placed.');
  } catch (err) {
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = 'Not armed: ' + err.message; }
  }
}

/* --- panic ------------------------------------------------------------ */

function openPanic() {
  const s = state.snap || {};
  const open = (s.positions || []).length;
  resetPanicDialog();
  const what = el('panic-gate-what');
  if (what) {
    what.textContent = open
      ? 'This will immediately SELL ' + open + ' open ' + plural(open, 'position')
        + ' at the bid and halt trading for the rest of today.'
      : 'No position is open. This will halt trading for the rest of today and disarm live mode.';
  }
  fill(el('panic-gate-detail'),
    (s.positions || []).map((p) => h('li', {
      text: p.label + ' — ' + p.qty + ' units, marked '
        + (p.mark_ok ? inr(p.ltp) : 'NO QUOTE') + ', ' + inr(p.pnl, { sign: true, decimals: 0 }),
    })).concat([
      h('li', { text: 'The day halt is written to the database, so a restart will not resume '
        + 'trading today.' }),
    ]));
  const msg = el('panic-gate-msg');
  if (msg) msg.textContent = '';
  openModal(el('panic-gate'));
  const cancel = el('panic-cancel');
  if (cancel) cancel.focus();
}

function resetPanicDialog() {
  const t = el('panic-gate-title');
  if (t) t.textContent = 'Panic';
  const btn = el('panic-confirm');
  if (btn) {
    btn.textContent = 'Yes — flatten and halt';
    btn.onclick = doPanic;
    btn.disabled = false;
  }
}

async function doPanic() {
  const msg = el('panic-gate-msg');
  const btn = el('panic-confirm');
  if (btn) btn.disabled = true;
  if (msg) { msg.className = 'formmsg'; msg.textContent = 'Flattening…'; }
  try {
    const out = await api.post('/api/panic');
    if (btn) btn.disabled = false;
    closeModal(el('panic-gate'));
    resetPanicDialog();
    await refreshNow();
    alertNow('Panic complete. Trading is halted for today.');
    const failed = (out && out.failed) || [];
    sheet('Panic complete', [
      h('p', { text: 'Flattened ' + (((out && out.flattened) || []).length)
        + ' position(s). Trading is halted for the rest of today.' }),
      failed.length ? h('p', { class: 'badline', text: failed.length
        + ' exit(s) FAILED and may still be open. Check INDmoney now.' }) : null,
    ].filter(Boolean));
  } catch (err) {
    if (btn) btn.disabled = false;
    if (msg) {
      msg.className = 'formmsg bad';
      msg.textContent = 'PANIC FAILED: ' + err.message + ' — square off on INDmoney by hand.';
    }
    alertNow('Panic failed. Square off on INDmoney by hand.');
  }
}

/* --- manual close ----------------------------------------------------- */

function confirmClose(p) {
  const t = el('panic-gate-title');
  if (t) t.textContent = 'Close ' + String(p.label);
  const what = el('panic-gate-what');
  if (what) {
    what.textContent = 'This will SELL ' + p.qty + ' units (' + p.lots + ' ' + plural(p.lots, 'lot')
      + ') of ' + p.label + ' at the bid, right now. Trading is NOT halted and the bot may take '
      + 'another trade today if the limits still allow it.';
  }
  fill(el('panic-gate-detail'), [
    h('li', { text: 'Marked at '
      + (p.mark_ok ? inr(p.ltp) + ' a unit' : 'NO QUOTE — the fill price is unknown') }),
    h('li', { text: 'Realises about ' + inr(p.pnl, { sign: true, decimals: 0 }) + ' at that mark.' }),
    h('li', { text: 'To stop for the day as well, cancel this and press PANIC instead.' }),
  ]);
  const msg = el('panic-gate-msg');
  if (msg) msg.textContent = '';
  const btn = el('panic-confirm');
  if (btn) {
    btn.textContent = 'Yes — sell this lot';
    btn.onclick = () => doClose(p);
    btn.disabled = false;
  }
  openModal(el('panic-gate'));
  const cancel = el('panic-cancel');
  if (cancel) cancel.focus();
}

async function doClose(p) {
  const msg = el('panic-gate-msg');
  const btn = el('panic-confirm');
  if (btn) btn.disabled = true;
  if (msg) { msg.className = 'formmsg'; msg.textContent = 'Selling…'; }
  try {
    const out = await api.post(closeEndpoint(p.id), {});
    const done = (out && out.closed) || {};
    if (btn) btn.disabled = false;
    closeModal(el('panic-gate'));
    resetPanicDialog();
    await refreshNow();
    alertNow('Closed ' + String(p.label) + '.');
    sheet('Closed ' + String(p.label), [
      h('p', { text: 'Exited at ' + inr(done.exit_price) + ' a unit for '
        + inr(done.pnl, { sign: true, decimals: 0 }) + '.' }),
      h('p', { class: 'field-hint',
        text: 'Reason recorded: ' + String(done.exit_reason || 'manual') + '.' }),
    ]);
  } catch (err) {
    // A failed exit leaves the position open. Never let that read as success.
    if (btn) btn.disabled = false;
    if (msg) { msg.className = 'formmsg bad'; msg.textContent = 'NOT closed: ' + err.message; }
    alertNow('The close order failed. The position is still open — check INDmoney.');
  }
}

/* ======================================================================= *
 * 20. Controls
 * ======================================================================= */

async function setMode(mode) {
  if (mode === 'live') { openLiveGate(); return; }
  try {
    await api.post('/api/mode', { mode });
    await refreshNow();
    announce('Mode ' + mode);
  } catch (err) {
    sheet('Could not switch mode', [h('p', { text: err.message })]);
  }
}

function initControls() {
  for (const btn of $$('[data-mode-btn]')) {
    btn.addEventListener('click', () => setMode(btn.dataset.modeBtn));
  }
  const arm = el('btn-arm');
  if (arm) arm.addEventListener('click', openLiveGate);

  const disarm = el('btn-disarm');
  if (disarm) {
    disarm.addEventListener('click', async () => {
      try { await api.post('/api/disarm'); await refreshNow(); alertNow('Disarmed.'); }
      catch (err) { sheet('Could not disarm', [h('p', { text: err.message })]); }
    });
  }

  const panic = el('btn-panic');
  if (panic) panic.addEventListener('click', openPanic);

  const run = el('btn-run');
  if (run) {
    run.addEventListener('click', async () => {
      const running = run.dataset.running === 'true';
      try { await api.post(running ? '/api/stop' : '/api/start'); await refreshNow(); }
      catch (err) {
        sheet('Could not ' + (running ? 'stop' : 'start'), [h('p', { text: err.message })]);
      }
    });
  }

  const gateArm = el('live-gate-arm');
  if (gateArm) gateArm.addEventListener('click', doArm);
  const gateCancel = el('live-gate-cancel');
  if (gateCancel) gateCancel.addEventListener('click', () => closeModal(el('live-gate')));
  const gateDry = el('live-gate-dry');
  if (gateDry) {
    gateDry.addEventListener('click', async () => {
      try {
        await api.post('/api/mode', { mode: 'live' });
        closeModal(el('live-gate'));
        await refreshNow();
        announce('Live mode, disarmed. No orders will be placed.');
      } catch (err) {
        const msg = el('live-gate-msg');
        if (msg) { msg.className = 'formmsg bad'; msg.textContent = err.message; }
      }
    });
  }

  const panicCancel = el('panic-cancel');
  if (panicCancel) {
    panicCancel.addEventListener('click', () => {
      closeModal(el('panic-gate'));
      resetPanicDialog();
    });
  }
  const sheetClose = el('sheet-close');
  if (sheetClose) sheetClose.addEventListener('click', () => closeModal(el('sheet')));

  // <dialog> closes on Escape natively; reset the shared panic dialog either way.
  for (const dlg of $$('dialog')) {
    dlg.addEventListener('cancel', () => { if (dlg.id === 'panic-gate') resetPanicDialog(); });
    dlg.addEventListener('close', () => { if (lastFocus && lastFocus.focus) lastFocus.focus(); });
  }
}

/* ======================================================================= *
 * 21. Render pipeline
 * ======================================================================= */

function render(s) {
  state.snap = s;
  state.lastSnapAt = Date.now();
  try {
    renderHeader(s);
    renderTicker(s);
    renderSetupCard(s);
    renderHero(s);
    renderDay(s);
    renderPositions(s);
    renderWhy(s);
    renderUnderlyings(s);
    renderLearning(s);
    renderSettings(s);
    renderToken(s);
    renderDiag(s);
    renderLog(s);
    initBacktest(s);
  } catch (err) {
    console.error('[indbot] render failed', err);
    banner('error-banner', 'The dashboard hit a rendering error: ' + err.message
      + '. The engine is unaffected; reload the page.');
  }
}

async function refreshNow() {
  try {
    render(await api.get('/api/state'));
    setConnection(true);
  } catch (err) {
    setConnection(false, 'no answer');
  }
}

/* ======================================================================= *
 * 22. Websocket
 * ======================================================================= */

let ws = null;
let wsRetry = 0;
let wsTimer = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try { ws = new WebSocket(proto + '//' + location.host + '/ws'); }
  catch (e) { scheduleReconnect(); return; }

  ws.onopen = () => { wsRetry = 0; setConnection(true); };
  ws.onmessage = (ev) => {
    let data = null;
    try { data = JSON.parse(ev.data); } catch (e) { return; }
    if (data && typeof data === 'object') { render(data); setConnection(true); }
  };
  ws.onclose = () => { setConnection(false); scheduleReconnect(); };
  ws.onerror = () => { try { ws.close(); } catch (e) { /* onclose handles it */ } };
}

function scheduleReconnect() {
  if (wsTimer) clearTimeout(wsTimer);
  wsRetry = Math.min(wsRetry + 1, 6);
  const delay = Math.min(15000, 500 * Math.pow(2, wsRetry));
  const text = el('conn-text');
  if (text) text.textContent = 'reconnecting in ' + Math.round(delay / 1000) + 's';
  wsTimer = setTimeout(() => { refreshNow(); connect(); }, delay);
}

/* Keep the clock moving between ticks, and treat a very old snapshot as a dead
 * connection even if the socket still claims to be open. */
setInterval(() => {
  if (state.connected) {
    const t = el('clock-time');
    if (t && state.snap) t.textContent = istTime(new Date().toISOString());
    if (state.lastSnapAt && Date.now() - state.lastSnapAt > 90000) {
      setConnection(false, 'no snapshot for 90s');
    }
  }
}, 1000);

/* ======================================================================= *
 * 23. Boot
 * ======================================================================= */

function boot() {
  initTheme();
  initTabs();
  initControls();
  initWhyFilter();
  initSettings();
  initToken();
  initBacktest(null);
  refreshNow().finally(connect);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
