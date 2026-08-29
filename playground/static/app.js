'use strict';

// The page is a renderer and nothing else: every number it draws arrives from
// the server already computed by the agent's own modules. Nothing is derived
// here, so the screen cannot disagree with the run.

const S = {
  mode: 'replay',
  turns: [],        // [{turn, user_message, message, explain, ...}]
  cards: {},        // asin -> display fields
  target: null,
  selected: 0,
  live: false,
  goal: null,
};

const $ = (id) => document.getElementById(id);
const el = (html) => { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstElementChild; };
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const n2 = (x) => (x == null ? '-' : Number(x).toFixed(2));
const n3 = (x) => (x == null ? '-' : Number(x).toFixed(3));
const pct = (x) => `${(Number(x) * 100).toFixed(1)}%`;
const comma = (x) => Number(x).toLocaleString('en-US');

async function api(route, body) {
  const res = await fetch(route, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  } : undefined);
  const data = await res.json();
  if (data && data.error) throw new Error(data.error);
  return data;
}

/* ------------------------------------------------------------------ modes */

document.querySelectorAll('.mode').forEach((b) => b.onclick = () => {
  document.querySelectorAll('.mode').forEach((x) => x.classList.remove('on'));
  b.classList.add('on');
  S.mode = b.dataset.mode;
  S.turns = []; S.selected = 0; S.live = false; S.target = null;
  $('replay-controls').classList.toggle('hidden', S.mode !== 'replay');
  $('live-controls').classList.toggle('hidden', S.mode !== 'live');
  $('composer').classList.add('hidden');
  $('verdict').innerHTML = '';
  $('chat').innerHTML = '<div class="empty">' + (S.mode === 'replay'
    ? 'Pick a session and press run.'
    : 'Start a session, then type as the customer.') + '</div>';
  $('pipe').innerHTML = '<div class="empty">The pipeline for the selected turn appears here.</div>';
});

/* ----------------------------------------------------------------- replay */

api('/api/samples').then((rows) => {
  $('samples').innerHTML = rows.map((r) =>
    `<option value="${r.sample_id}">${r.sample_id} · ${r.scenario_type} · ${esc(r.title.slice(0, 60))}</option>`
  ).join('');
});

$('run').onclick = async () => {
  $('run').disabled = true;
  $('chat').innerHTML = '<div class="empty">running through the evaluator…</div>';
  try {
    const out = await api('/api/replay', { sample_id: $('samples').value });
    S.turns = out.session.turns;
    S.cards = out.cards;
    S.target = out.target;
    S.live = false;
    S.selected = S.turns.length - 1;
    renderVerdict(out);
    renderChat();
    renderPipe();
  } catch (e) {
    $('chat').innerHTML = `<div class="banner err">${esc(e.message)}</div>`;
  }
  $('run').disabled = false;
};

function renderVerdict(out) {
  const s = out.session;
  const stat = (label, value, cls) =>
    `<div class="stat"><b class="${cls || ''}">${value}</b><span>${label}</span></div>`;
  $('verdict').innerHTML = [
    stat('outcome', s.hit ? 'HIT' : 'MISS', s.hit ? '' : 'mono'),
    stat('rank', s.best_rank == null ? '-' : s.best_rank),
    stat('turn', s.first_hit_turn == null ? '-' : s.first_hit_turn),
    stat('recip. rank', n3(s.reciprocal_rank)),
    stat('session score', n3(out.metrics.technical_score)),
  ].join('');
}

/* ------------------------------------------------------------------- live */

let searchTimer = null;
$('goal-search').oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = $('goal-search').value.trim();
    if (q.length < 3) return $('goal-results').classList.add('hidden');
    const rows = await api('/api/search?q=' + encodeURIComponent(q));
    $('goal-results').innerHTML = rows.map((r) =>
      `<div data-asin="${r.asin}">${esc(r.title)} <span class="muted">· ${esc(r.store)} · ${comma(r.reviews)} reviews</span></div>`
    ).join('') || '<div class="muted">nothing matched</div>';
    $('goal-results').classList.remove('hidden');
    $('goal-results').querySelectorAll('[data-asin]').forEach((d) => d.onclick = () => {
      S.goal = d.dataset.asin;
      $('goal-search').value = d.textContent.trim();
      $('goal-results').classList.add('hidden');
    });
  }, 220);
};

$('open').onclick = async () => {
  const out = await api('/api/chat/open', { goal: S.goal });
  S.turns = []; S.cards = {}; S.live = true; S.selected = 0;
  S.target = out.goal_card ? out.goal_card.asin : null;
  $('composer').classList.remove('hidden');
  $('verdict').innerHTML = '<div class="stat"><b class="muted">no score</b><span>no ground truth</span></div>';
  $('chat').innerHTML = '<div class="empty">Session open. Type as the customer.</div>';
  $('pipe').innerHTML = '<div class="empty">The pipeline appears after the first turn.</div>';
  $('turns').textContent = `turn 0 / 10`;
  $('say').focus();
};

$('send').onclick = say;
$('say').onkeydown = (e) => { if (e.key === 'Enter') say(); };

async function say() {
  const text = $('say').value.trim();
  if (!text) return;
  $('say').value = '';
  const entry = await api('/api/chat/send', { text });
  if (entry.exhausted) {
    $('turns').textContent = 'protocol limit reached: a scored session stops at turn 10';
    return;
  }
  Object.assign(S.cards, entry.cards || {});
  entry.slate = (entry.recommendations || []).map((r) => r.parent_asin);
  entry.target_rank = S.target ? (entry.slate.indexOf(S.target) + 1 || null) : null;
  S.turns.push(entry);
  S.selected = S.turns.length - 1;
  $('turns').textContent = `turn ${entry.turn} / 10`;
  renderChat();
  renderPipe();
}

/* ------------------------------------------------------------------- chat */

function renderChat() {
  const box = $('chat');
  box.innerHTML = '';
  S.turns.forEach((t, i) => {
    const e = t.explain || {};
    const r = e.ranking || {};
    const marks = [];
    if (t.target_rank) marks.push(`<span class="pill hit">target @ ${t.target_rank}</span>`);
    if (t.scorable === false) marks.push('<span class="pill warn">pre-pivot, unscorable</span>');
    if (r.goal && r.goal.rank && !t.target_rank) marks.push(`<span class="pill">goal ranked ${r.goal.rank} / ${r.goal.of}</span>`);
    if (e.degraded) marks.push('<span class="pill bad">degraded</span>');
    if (r.verified === false) marks.push('<span class="pill bad">unverified</span>');

    const block = el(`
      <div class="turnblock ${i === S.selected ? 'sel' : ''}">
        <div class="head">
          <b>Turn ${t.turn}</b>
          ${e.policy ? `<span class="pill">${e.policy.chosen}</span>` : ''}
          ${t.ask_attribute ? `<span class="pill">asks: ${esc(t.ask_attribute)}</span>` : '<span class="pill">asks nothing</span>'}
          <span style="margin-left:auto">${marks.join(' ')}</span>
        </div>
        <div class="said user"><span class="who">customer</span>${esc(t.user_message)}</div>
        <div class="said bot"><span class="who">agent</span>${esc(t.message)}</div>
      </div>`);
    block.onclick = () => { S.selected = i; renderChat(); renderPipe(); };
    box.appendChild(block);
  });
}

/* --------------------------------------------------------------- pipeline */

function panel(num, name, lede, body, open) {
  return `<details class="panel" ${open ? 'open' : ''}>
    <summary><span class="num">${num}</span><span class="name">${name}</span>
      <span class="lede">${lede}</span></summary>
    <div class="body">${body}</div></details>`;
}

function card(asin) {
  return S.cards[asin] || { asin, title: asin, store: '', price: null, rating: null, reviews: 0 };
}

function renderPipe() {
  const t = S.turns[S.selected];
  const box = $('pipe');
  if (!t) return;
  const e = t.explain || {};

  if (e.degraded) {
    box.innerHTML = `<div class="banner err"><b>This turn degraded.</b>
      <code>_serve</code> raised and the agent fell back rather than
      returning nothing, which is why the evaluator still scored a slate.
      Fallback used: <code>${esc((e.debug || {}).degraded)}</code>,
      from <code>${esc((e.debug || {}).error)}</code>.</div>`;
    return;
  }

  const r = e.ranking || {};
  const banner = r.verified
    ? `<div class="banner ok"><b>Verified.</b> This account was produced by
       replaying the agent's own stages, and the slate it reconstructs is
       identical to the one served. Every score below adds up to the score
       the evaluator saw.</div>`
    : `<div class="banner err"><b>Unverified.</b> The replayed stages did not
       reproduce the served slate, so treat the breakdown below as
       untrustworthy rather than as evidence.</div>`;

  box.innerHTML = banner
    + understandPanel(e)
    + statePanel(e)
    + policyPanel(e)
    + routePanel(e)
    + orchestrationPanel(e)
    + retrievalPanel(r)
    + rankingPanel(r, t)
    + shapePanel(r, t)
    + rerankPanel(r)
    + probePanel(e)
    + messagePanel(e);
}

/* 1 ------------------------------------------------------------ understand */

function understandPanel(e) {
  const u = e.understand || {};
  const body = `
    <dl class="kv">
      <dt>dialogue act</dt><dd><span class="pill">${esc(u.act)}</span></dd>
      <dt>confidence</dt><dd>${n2(u.confidence)}
        <span class="muted">${u.exact ? 'exact template match' : 'read from cues and catalog vocabulary'}</span></dd>
      <dt>category read</dt><dd>${u.category ? esc(u.category) : '<span class="muted">none stated this turn</span>'}</dd>
      <dt>buckets offered</dt><dd class="chips">${(u.buckets || []).map((b) =>
        `<span class="chip">${esc(b)}</span>`).join('') || '<span class="muted">-</span>'}</dd>
      <dt>constraints heard</dt><dd class="chips">${(u.constraints || []).map((c) =>
        `<span class="chip">${esc(c)}</span>`).join('') || '<span class="muted">nothing new</span>'}</dd>
      <dt>flags</dt><dd class="chips">
        ${u.pivot ? '<span class="chip neg"><b>pivot</b></span>' : ''}
        ${u.exhausted ? `<span class="chip">exhausted${u.exhausted_arm ? ': ' + esc(u.exhausted_arm) : ''}</span>` : ''}
        ${u.boundary_refusal ? '<span class="chip">boundary refusal</span>' : ''}
        ${!u.pivot && !u.exhausted && !u.boundary_refusal ? '<span class="muted">none</span>' : ''}
      </dd>
    </dl>`;
  return panel(1, 'Understand', `${esc(u.act)} · ${n2(u.confidence)}`, body, true);
}

/* 2 ----------------------------------------------------------------- state */

function statePanel(e) {
  const s = e.state || {};
  const slots = (s.slots || []).map((x) =>
    `<span class="chip ${x.negated ? 'neg' : ''}">
      <b>${esc(x.attribute)}</b> ${esc(x.value)} <span class="t">t${x.turn}</span>
    </span>`).join('') || '<span class="muted">nothing disclosed yet</span>';
  const superseded = (s.superseded || []).map((x) =>
    `<span class="chip dim">${esc(x)}</span>`).join('') || '<span class="muted">none</span>';
  const declined = (s.declined || []).map((x) =>
    `<span class="chip">${esc(x)}</span>`).join('') || '<span class="muted">none</span>';
  const spent = (s.spent_arms || []).map((x) =>
    `<span class="chip ${(s.declined || []).includes(x) ? '' : 'dim'}">${esc(x)}</span>`
  ).join('') || '<span class="muted">none</span>';

  const body = `
    <dl class="kv">
      <dt>typed slots</dt><dd class="chips">${slots}</dd>
      <dt>refused values</dt><dd>${s.refused_text
        ? `<span class="chip neg">${esc(s.refused_text)}</span>`
        : '<span class="muted">none</span>'}
        <div class="muted" style="font-size:11px;margin-top:4px">a value the customer rejected; subtracted from the blend</div></dd>
      <dt>declined arms</dt><dd class="chips">${declined}
        <div class="muted" style="font-size:11px;margin-top:4px">a dimension they would not discuss; never asked again, never subtracted</div></dd>
      <dt>spent arms</dt><dd class="chips">${spent}
        <div class="muted" style="font-size:11px;margin-top:4px">everything not worth asking again, declined or merely exhausted. <code>probe</code> reads this wider set; <code>policy</code> reads only the declined subset above, and reading one as the other sent 74% of hard turns down the wrong branch</div></dd>
      <dt>superseded</dt><dd class="chips">${superseded}</dd>
      <dt>counters</dt><dd class="muted">
        turn ${s.turn} · idle ${s.idle} · already shown ${s.shown}
        ${s.pivoted ? ` · pivoted at turn ${s.pivot_turn}` : ''}
        ${s.carried && s.carried.length ? ` · carried ${s.carried.length} from earlier visits` : ''}</dd>
      <dt>query text</dt><dd class="mono" style="font-size:12px">${esc(s.query_text) || '<span class="muted">empty</span>'}</dd>
    </dl>`;
  return panel(2, 'Session state', `${(s.slots || []).length} slots · ${(s.constraints || []).length} constraints`, body);
}

/* 3 ---------------------------------------------------------------- policy */

function policyPanel(e) {
  const p = e.policy || {};
  const rungs = (p.ladder || []).map((r) => `
    <div class="rung ${r.state}">
      <span class="nm">${esc(r.policy)}</span>
      <span class="why">${esc(r.test)}</span>
      <span class="tag ${r.state === 'fired' ? '' : 'muted'}">${r.state === 'fired' ? 'FIRED' : r.state === 'passed' ? 'tested, false' : 'not reached'}</span>
    </div>`).join('');
  const body = `<div class="muted" style="font-size:12px;margin-bottom:8px">
      Strict priority. The first rung that holds decides the turn, so the order
      is the design: a redirect outranks narrowing, and running out of turns
      outranks gathering information there is no turn left to spend.
    </div><div class="ladder">${rungs}</div>`;
  return panel(3, 'Dialogue policy', esc(p.chosen), body, true);
}

/* 4 ----------------------------------------------------------------- route */

function routePanel(e) {
  const r = e.route || {};
  const mark = (field, value) => {
    const shared = !(r.specialised || []).includes(field);
    return `<dd>${value == null ? '<span class="muted">module default</span>' : value}
      ${shared ? '<span class="muted" style="font-size:11px">· shared constant</span>' : '<span class="pill">route-specific</span>'}</dd>`;
  };
  const body = `<div class="muted" style="font-size:12px;margin-bottom:8px">
      Four routes, all shipping at the same constants. Every attempt to give
      them different ones measured negative, so they are named here rather
      than differentiated.
    </div>
    <dl class="kv">
      <dt>route</dt><dd><span class="pill">${esc(r.name)}</span></dd>
      <dt>alpha</dt>${mark('alpha', r.alpha)}
      <dt>defer turns</dt>${mark('defer_turns', r.defer_turns)}
      <dt>dense weight</dt>${mark('dense_weight', r.dense_weight)}
      <dt>reach</dt>${mark('reach', r.reach)}
      <dt>diversity</dt>${mark('diversity', r.diversity)}
    </dl>`;
  return panel(4, 'Retrieval route', `alpha ${r.alpha}`, body);
}

/* 5 --------------------------------------------------------- orchestration */

function orchestrationPanel(e) {
  const o = e.orchestration || {};
  if (!o.candidates) return '';
  const rows = o.candidates.map(c => {
    const why = !c.eligible
      ? '<span class="muted">no evidence of its own</span>'
      : (c.refuted ? '<span class="pill">head disproven</span>'
                   : '<span class="muted">head still unserved</span>');
    const mark = c.chosen ? ' class="best"'
      : (c.eligible ? '' : ' class="dead"');
    return `<tr${mark}>
      <td>${c.chosen ? '<b>' + esc(c.ordering) + '</b>' : esc(c.ordering)}</td>
      <td class="num">${pct(c.spent)}</td>
      <td>${why}</td></tr>`;
  }).join('');
  const headline = o.switched
    ? `<div class="banner ok"><b>Re-orchestrated.</b> The blend's own head has
       been served and did not end the session, so this turn ranks the same
       pool by <code>${esc(o.ordering)}</code> instead.</div>`
    : `<div class="muted" style="font-size:12px;margin-bottom:8px">
       A slate that was served and did not end the session is provably wrong,
       which is the only signal here that is about correctness rather than
       about confidence. While the blend's head still holds products no turn
       has served, there is nothing to switch away from.</div>`;
  const body = headline + `<dl class="kv">
      <dt>slates disproven</dt><dd>${o.refuted_slates}</dd>
      <dt>switch at</dt><dd>${pct(o.threshold)} of the top ${o.horizon}</dd>
      <dt>reason</dt><dd>${esc(o.reason)}</dd>
    </dl>
    <table><thead><tr><th>ordering</th><th class="num">head disproven</th>
      <th>status</th></tr></thead><tbody>${rows}</tbody></table>`;
  return panel(5, 'Orchestration',
               o.switched ? `switched to ${esc(o.ordering)}` : 'held', body,
               true);
}

/* 5 ------------------------------------------------------------- retrieval */

function retrievalPanel(r) {
  const z = r.sizes || {};
  const share = z.catalog ? z.pool / z.catalog : 0;
  const body = `
    <div class="funnel">
      <div class="step"><b>${comma(z.catalog || 0)}</b><span>catalog</span></div>
      <div class="arrow">→</div>
      <div class="bar"><i style="width:${Math.max(share * 100, 0.4)}%"></i></div>
      <div class="arrow">→</div>
      <div class="step"><b style="color:var(--blue)">${comma(z.pool || 0)}</b><span>candidates</span></div>
    </div>
    <div class="muted" style="font-size:12px;margin-top:10px">
      The coarse category is the strongest signal in the problem, and it is a
      hard filter rather than extra query terms. ${z.buckets} bucket${z.buckets === 1 ? '' : 's'}
      resolved, holding ${pct(share)} of the catalog.
      ${z.reached ? `Dense reach added ${z.reached} from outside the bucket.` : ''}
    </div>
    <div style="margin-top:10px"><span class="muted" style="font-size:11px">QUERY TOKENS</span>
      <div class="chips" style="margin-top:5px">${(r.query_tokens || []).map((t) =>
        `<span class="chip ${t.known ? '' : 'dim'}">${esc(t.token)}</span>`).join('')
        || '<span class="muted">no query text yet</span>'}</div>
      <div class="muted" style="font-size:11px;margin-top:4px">struck through = the index has never seen this word, so it scores nothing</div>
    </div>`;
  return panel(6, 'Retrieval', `${comma(z.catalog || 0)} → ${comma(z.pool || 0)}`, body, true);
}

/* 6 --------------------------------------------------------------- ranking */

const SEGS = [
  ['bm25', 'BM25 over title + features', 'var(--blue)'],
  ['prior', 'popularity prior × alpha', 'var(--purple)'],
  ['profile', 'profile affinity', 'var(--amber)'],
  ['dense', 'dense similarity', 'var(--teal)'],
  ['negation', 'refused penalty', 'var(--red)'],
  ['dense_negation', 'dense refused penalty', 'var(--red)'],
];

function rankingPanel(r, t) {
  const slots = r.slots || [];
  const top = Math.max(...slots.map((s) => Math.abs(s.score) || 0), 0.001);
  const legend = SEGS.filter(([k]) => slots.some((s) => s.breakdown && k in s.breakdown))
    .map(([k, label, col]) => `<span><b style="background:${col}"></b>${label}</span>`).join('');

  const rows = slots.map((s) => {
    const c = card(s.asin);
    const bits = SEGS.filter(([k]) => s.breakdown && s.breakdown[k])
      .map(([k]) => `<i class="seg-${k}" style="width:${Math.abs(s.breakdown[k]) / top * 100}%"></i>`).join('');
    const isTarget = S.target && s.asin === S.target;
    return `<div class="prod ${isTarget ? 'target' : ''}">
      <div class="slot">${s.position + 1}</div>
      <div class="main">
        <div class="ttl">${esc(c.title)}</div>
        <div class="stack">${bits}</div>
        <div class="meta">
          <span class="src ${s.source}">${s.source}</span>
          <span>${s.pooled ? 'score ' + n3(s.score) : 'no score: from outside the ranked pool'}</span>
          ${s.rank != null ? `<span>ranked ${s.rank + 1}</span>` : ''}
          <span>${comma(c.reviews)} reviews</span>
          ${c.rating ? `<span>${c.rating}★</span>` : ''}
          ${c.store ? `<span>${esc(c.store)}</span>` : ''}
          ${s.agrees ? '' : '<span class="pill bad">score does not reconcile</span>'}
        </div>
      </div></div>`;
  }).join('');

  const who = S.live ? 'goal product' : 'session target';
  let goal = '';
  if (r.goal && r.goal.in_pool) {
    goal = `<div class="banner ok" style="margin-top:10px">
      The ${who} sits at <b>rank ${r.goal.rank} of ${r.goal.of}</b> in the full
      ranking ${t.target_rank ? `and was served at slot ${t.target_rank}.`
        : 'and was not served this turn: the slots below the top pick are being held back.'}
    </div>`;
  } else if (r.goal) {
    goal = `<div class="banner err" style="margin-top:10px">
      The ${who} is <b>not in this turn's candidate pool</b>. The category read
      resolved a bucket that does not hold it, and the filter is hard, so no
      amount of ranking can reach it until the conversation names a different
      category.
    </div>`;
  }

  const body = `<div class="legend">${legend}</div>${rows}${goal}
    <div class="muted" style="font-size:12px;margin-top:10px">
      Bars are the blend's own terms, to scale. Both halves are load-bearing:
      the prior is immune to rewording and BM25 is immune to a change in which
      products get bought, so neither alone is a safety net.
    </div>`;
  return panel(7, 'Ranking', `alpha ${r.alpha} · ${slots.length} slots`, body, true);
}

/* 7 ----------------------------------------------------------- slate shape */

function shapePanel(r, t) {
  const head = r.head || 0;
  const band = r.band || [];
  const dropped = r.dropped_shown || [];
  const withheld = band.slice(head).map((b) =>
    `<div class="prod"><div class="slot muted">${b.rank + 1}</div>
      <div class="main"><div class="ttl muted">${esc(card(b.asin).title)}</div>
      <div class="meta"><span>score ${n3(b.score)}</span></div></div></div>`).join('');

  const body = `
    <dl class="kv">
      <dt>committed</dt><dd>${head} slot${head === 1 ? '' : 's'} to the top of the ranking</dd>
      <dt>explored</dt><dd>${Math.max((r.slots || []).length - head, 0)} slots drawn from past the slate</dd>
      <dt>contenders</dt><dd>${r.contenders} still scoring near the leader</dd>
    </dl>
    <div class="muted" style="font-size:12px;margin:10px 0">
      Showing a product is irreversible: the session ends the moment the
      customer sees the one they wanted, at whatever position it happened to
      occupy. So while the customer still has things to tell us, the slots
      below the top pick are worth more held back than spent, and they reach
      past the slate instead.
    </div>
    ${head < band.length ? `<div><span class="muted" style="font-size:11px">WITHHELD THIS TURN</span>${withheld}</div>` : ''}
    ${dropped.length ? `<div style="margin-top:10px"><span class="muted" style="font-size:11px">DROPPED, ALREADY SHOWN</span>
      ${dropped.slice(0, 6).map((d) => `<div class="prod"><div class="slot muted">${d.rank + 1}</div>
        <div class="main"><div class="ttl muted">${esc(card(d.asin).title)}</div></div></div>`).join('')}
      <div class="muted" style="font-size:11px;margin-top:4px">a slot spent on one of these is spent on a product the session already declined by not ending</div></div>` : ''}`;
  return panel(8, 'Slate shape', `head ${head} · ${Math.max((r.slots || []).length - head, 0)} exploring`, body);
}

/* 8 ---------------------------------------------------------------- rerank */

function rerankPanel(r) {
  const k = r.rerank || {};
  if (!k.phrases) {
    return panel(9, 'Phrase rerank', 'inactive', `<div class="muted">
      No constraint the customer has stated matches a whole phrase the catalog
      knows, so the blend's order stands untouched.</div>`);
  }
  const moves = (k.moves || []).filter((m) => m.evidence > 0 || m.from !== m.to).map((m) => {
    const d = m.from - m.to;
    return `<tr><td>${esc(card(m.asin).title.slice(0, 64))}</td>
      <td class="mono">${m.from + 1} → ${m.to + 1}</td>
      <td>${d > 0 ? `<span style="color:var(--green)">+${d}</span>` : d < 0 ? `<span style="color:var(--red)">${d}</span>` : '<span class="muted">-</span>'}</td>
      <td class="mono">${n3(m.evidence)}</td></tr>`;
  }).join('');
  const body = `<div class="muted" style="font-size:12px;margin-bottom:8px">
      A phrase held by one product is worth 1.0; one held by a thousand is worth
      0.001, which is why a bare material word cannot move a ranking. This stage
      only permutes the ten already chosen, so it can move precision and never
      coverage.</div>
    <table><thead><tr><th>product</th><th>slot</th><th>moved</th><th>evidence</th></tr></thead>
    <tbody>${moves || '<tr><td class="muted" colspan="4">evidence found, but it changed no order</td></tr>'}</tbody></table>`;
  return panel(9, 'Phrase rerank', k.active ? `${k.phrases} phrase(s), order changed` : `${k.phrases} phrase(s), order held`, body);
}

/* 9 ----------------------------------------------------------------- probe */

function probePanel(e) {
  const p = e.probe || {};
  const table = p.table || [];
  const top = Math.max(...table.map((x) => x.score), 0.0001);
  const rows = table.map((x) => `
    <tr class="${x.blocked || x.dead ? 'blocked' : ''} ${x.arm === p.asked ? 'best' : ''}">
      <td>${esc(x.arm)}${x.dead ? ' <span class="muted">· dead arm</span>' : ''}</td>
      <td>${pct(x.coverage)}</td>
      <td>${n2(x.spread)}</td>
      <td>${x.heard ? `${x.heard}× (${n2(x.decay)})` : '<span class="muted">-</span>'}</td>
      <td><div class="meter"><i style="width:${Math.max(x.score / top * 100, 0)}%"></i></div></td>
      <td class="mono">${n3(x.score)}</td>
      <td class="muted">${esc(x.blocked || '')}</td>
    </tr>`).join('');

  const body = `<dl class="kv">
      <dt>asked</dt><dd>${p.asked ? `<span class="pill hit">${esc(p.asked)}</span>` : '<span class="pill">nothing</span>'}</dd>
      <dt>because</dt><dd>${esc(p.reason)}</dd>
      <dt>offered</dt><dd>${(p.options || []).length
        ? (p.options || []).map((o) => `<span class="chip">${esc(o)}</span>`).join(' ')
        : '<span class="muted">no alternatives drawn from the pool</span>'}</dd>
    </dl>
    <div class="muted" style="font-size:12px;margin:10px 0">
      Each arm scores coverage × spread × decay: how much of the live pool still
      leads with something of that kind, how varied those values are, and how
      much the customer has already said about it. Two arms are dead by
      construction: the simulator's classifier can never return them, so asking
      always yields nothing.
    </div>
    <table><thead><tr><th>arm</th><th>coverage</th><th>spread</th><th>heard</th><th></th><th>score</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  return panel(10, 'Clarifying question', p.asked ? esc(p.asked) : 'silent', body, true);
}

/* 10 ------------------------------------------------------------- response */

function messagePanel(e) {
  const m = e.message || {};
  const seg = (cls, label, value) => value
    ? `<div class="seg ${cls}"><span>${label}</span>${esc(value)}</div>` : '';
  const body = `${seg('a', 'acknowledge', m.acknowledge)}
    ${seg('b', 'slate framing', m.slate)}
    ${seg('c', 'question', m.question)}
    <div class="muted" style="font-size:12px;margin-top:8px">
      Composed from state, deterministically, at zero tokens. The simulator
      never reads this prose, only the attribute enum beside it. It is built
      anyway because the transcript is what a human judge reads.
    </div>`;
  return panel(11, 'Response', 'composed from state', body);
}
