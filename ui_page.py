# ui_page.py

"""The /proxy/ui/ dashboard page: a single self-contained HTML document
(vanilla JS, no external assets) served by the proxy on the same port.

The page polls /proxy/ui/state every 3 s (active requests, history, slots)
and subscribes to /proxy/ui/events (SSE) for the live token stream of the
selected request.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama-kv-proxy</title>
<style>
:root {
  --bg: #0d1117; --panel: #161b22; --border: #30363d; --text: #e6edf3;
  --dim: #8b949e; --green: #3fb950; --yellow: #d29922; --red: #f85149;
  --orange: #db6d28; --purple: #bc8cff; --blue: #58a6ff;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
header { display: flex; align-items: center; gap: 10px; padding: 8px 14px;
  border-bottom: 1px solid var(--border); background: var(--panel); }
header h1 { font-size: 14px; margin: 0; font-weight: 600; }
#conn { width: 9px; height: 9px; border-radius: 50%; background: var(--red); }
#conn.ok { background: var(--green); }
#clock { margin-left: auto; color: var(--dim); }
main { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  gap: 10px; padding: 10px 14px; }
section { background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; min-width: 0; }
section h2 { font-size: 12px; margin: 0 0 6px; color: var(--dim);
  text-transform: uppercase; letter-spacing: .05em; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--dim); font-weight: 400; font-size: 11px;
  padding: 2px 6px 2px 0; white-space: nowrap; }
td { padding: 2px 6px 2px 0; vertical-align: top; white-space: nowrap; }
tr.req { cursor: pointer; }
tr.req:hover td { background: #1c2129; }
tr.req.sel td { background: #1f2a3a; }
.preview { max-width: 340px; overflow: hidden; text-overflow: ellipsis;
  color: var(--dim); }
.slotgrid { display: flex; flex-wrap: wrap; gap: 6px; }
.slotchip { border: 1px solid var(--border); border-radius: 4px; padding: 3px 8px;
  font-size: 12px; cursor: default; }
.slotchip.busy { border-color: var(--green); cursor: pointer; }
.slotchip.busy:hover { background: #1c2129; }
.slotchip .st { color: var(--dim); }
.st-free { color: var(--dim); } .st-busy { color: var(--green); }
.st-proc { color: var(--yellow); }
#detail { display: flex; flex-direction: column; gap: 8px; }
#detail .meta { color: var(--dim); }
#detail .meta b { color: var(--text); font-weight: 400; }
#prompt { max-height: 180px; overflow: auto; white-space: pre-wrap;
  word-break: break-word; background: var(--bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 8px; font-size: 12px; }
#tokens { flex: 1; min-height: 220px; max-height: 420px; overflow: auto;
  white-space: pre-wrap; word-break: break-word; background: var(--bg);
  border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px; }
#tokens .reason { color: var(--purple); font-style: italic; }
#tokens .content { color: var(--text); }
button { background: var(--panel); color: var(--text); border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 10px; cursor: pointer; font: inherit; font-size: 12px; }
button:hover { border-color: var(--blue); }
.status-queued { color: var(--yellow); } .status-generating { color: var(--green); }
.status-done { color: var(--dim); } .status-error { color: var(--red); }
.status-cancelled { color: var(--orange); }
#slots { grid-column: 1 / -1; }
.empty { color: var(--dim); padding: 4px 0; }
</style>
</head>
<body>
<header>
  <div id="conn"></div>
  <h1>llama-kv-proxy</h1>
  <span id="clock"></span>
</header>
<main>
  <section>
    <h2>Active requests</h2>
    <table id="active"><thead><tr>
      <th>rid</th><th>model</th><th>slot</th><th>status</th>
      <th>chars</th><th>tok</th><th>tps</th><th>age</th><th>prompt</th>
    </tr></thead><tbody></tbody></table>
    <div id="active-empty" class="empty">no active requests</div>
    <h2 style="margin-top:12px">History</h2>
    <table id="history"><thead><tr>
      <th>rid</th><th>model</th><th>slot</th><th>status</th>
      <th>dur</th><th>chars</th><th>tok</th><th>prompt</th>
    </tr></thead><tbody></tbody></table>
    <div id="history-empty" class="empty">no finished requests yet</div>
  </section>
  <section id="detail">
    <h2>Request detail</h2>
    <div id="detail-body">
      <div class="meta" id="d-meta"></div>
      <div style="display:flex;gap:6px;align-items:center">
        <button id="d-full" style="display:none">full prompt</button>
        <span class="meta" id="d-full-note"></span>
      </div>
      <div id="prompt"></div>
      <h2>Token stream</h2>
      <div id="tokens"></div>
    </div>
  </section>
  <section id="slots">
    <h2>Slots</h2>
    <div class="slotgrid" id="slotgrid"></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let selected = null;
let tails = new Map();      // rid -> {c: content, r: reasoning}
let fullLoaded = new Set();

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g,
    (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function slotStr(s) { return s ? s.backend + "/" + s.model + "/" + s.id : "—"; }
function tokOf(u) { return u && u.completion_tokens != null ? u.completion_tokens : null; }
function age(t0) { return (Date.now() / 1000 - t0).toFixed(0) + "s"; }
function dur(a, b) { return b ? (b - a).toFixed(1) + "s" : "—"; }
function tpsOf(r) {
  const t = tokOf(r.usage);
  if (t == null || !r.started_at) return "—";
  const el = (r.ended_at || Date.now() / 1000) - r.started_at;
  return el > 0 ? (t / el).toFixed(1) : "—";
}

function renderActive() {
  const tb = $("active").tBodies[0]; tb.innerHTML = "";
  const act = STATE.active || [];
  $("active-empty").style.display = act.length ? "none" : "";
  for (const r of act) {
    const tr = document.createElement("tr");
    tr.className = "req" + (r.rid === selected ? " sel" : "");
    tr.innerHTML =
      "<td>" + esc(r.rid.slice(0, 8)) + "</td><td>" + esc(r.model) + "</td>" +
      "<td>" + esc(slotStr(r.slot)) + "</td>" +
      '<td class="status-' + esc(r.status) + '">' + esc(r.status) + "</td>" +
      "<td>" + r.n_chars + "</td><td>" + (tokOf(r.usage) ?? "—") + "</td>" +
      "<td>" + tpsOf(r) + "</td><td>" + age(r.started_at) + "</td>" +
      '<td class="preview">' + esc(r.prompt_preview) + "</td>";
    tr.onclick = () => select(r.rid);
    tb.appendChild(tr);
  }
}

function renderHistory() {
  const tb = $("history").tBodies[0]; tb.innerHTML = "";
  const hist = STATE.history || [];
  $("history-empty").style.display = hist.length ? "none" : "";
  for (const r of hist.slice(0, 100)) {
    const tr = document.createElement("tr");
    tr.className = "req" + (r.rid === selected ? " sel" : "");
    tr.innerHTML =
      "<td>" + esc(r.rid.slice(0, 8)) + "</td><td>" + esc(r.model) + "</td>" +
      "<td>" + esc(slotStr(r.slot)) + "</td>" +
      '<td class="status-' + esc(r.status) + '">' + esc(r.status) + "</td>" +
      "<td>" + dur(r.started_at, r.ended_at) + "</td>" +
      "<td>" + r.n_chars + "</td><td>" + (tokOf(r.usage) ?? "—") + "</td>" +
      '<td class="preview">' + esc(r.prompt_preview) + "</td>";
    tr.onclick = () => select(r.rid);
    tb.appendChild(tr);
  }
}

function renderSlots() {
  const g = $("slotgrid"); g.innerHTML = "";
  for (const s of STATE.slots || []) {
    const busy = s.state && s.state !== "available";
    const chip = document.createElement("span");
    chip.className = "slotchip" + (busy && s.busy_rid ? " busy" : "");
    chip.innerHTML =
      "<b>" + s.backend + "/" + esc(s.model) + "/" + s.slot + "</b> " +
      '<span class="st st-' + (busy ? (s.state === "processing" ? "proc" : "busy") : "free") + '">' +
      esc(s.state || "?") + "</span>" +
      (s.busy_rid ? ' <b style="color:var(--green)">' + esc(s.busy_rid.slice(0, 8)) + "</b>" : "");
    if (busy && s.busy_rid) chip.onclick = () => select(s.busy_rid);
    g.appendChild(chip);
  }
}

function findReq(rid) {
  return (STATE.active || []).find((r) => r.rid === rid) ||
         (STATE.history || []).find((r) => r.rid === rid);
}

function select(rid) {
  selected = rid;
  const r = findReq(rid);
  if (!r) return;
  const t = tails.get(rid) || { c: r.tail || "", r: r.tail_reason || "" };
  tails.set(rid, t);
  renderDetailMeta(r);
  $("prompt").textContent = r.prompt_preview || "(no prompt)";
  fullLoaded.delete(rid);
  $("d-full-note").textContent = "";
  $("d-full").style.display = "";
  renderTokens(true);
  // History rows carry no token tail (the state poll stays lean): fetch the
  // stored response tail on demand and show it in the stream pane.
  if (r.tail === undefined) loadResponse(rid);
  renderActive(); renderHistory();
}

async function loadResponse(rid) {
  try {
    const resp = await fetch("/proxy/ui/request/" + encodeURIComponent(rid));
    if (!resp.ok || rid !== selected) return;
    const j = await resp.json();
    tails.set(rid, { c: j.response || "", r: j.response_reasoning || "" });
    if (rid === selected) renderTokens(true);
  } catch (e) { /* ignore */ }
}

function renderDetailMeta(r) {
  $("d-meta").innerHTML =
    "rid <b>" + esc(r.rid) + "</b> · model <b>" + esc(r.model) + "</b> · " +
    "slot <b>" + esc(slotStr(r.slot)) + "</b> · " +
    '<span class="status-' + esc(r.status) + '">' + esc(r.status) + "</span> · " +
    "stream <b>" + (r.stream ? "yes" : "no") + "</b> · words <b>" + r.n_words + "</b> · " +
    "ttft <b>" + (r.ttft != null ? r.ttft.toFixed(2) + "s" : "—") + "</b> · " +
    "chars <b>" + r.n_chars + "</b> · tok <b>" + (tokOf(r.usage) ?? "—") + "</b> · " +
    "tps <b>" + tpsOf(r) + "</b>" +
    (r.error ? ' · <span style="color:var(--red)">' + esc(r.error) + "</span>" : "");
}

function renderTokens(forceBottom) {
  const el = $("tokens");
  const t = selected ? tails.get(selected) : null;
  if (!t) {
    el.innerHTML = '<span class="empty">select a request</span>';
    return;
  }
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
  el.innerHTML =
    '<span class="reason">' + esc(t.r) + "</span>" +
    '<span class="content">' + esc(t.c) + "</span>";
  // One frame later, so the scroll sees the layout of the fresh content.
  if (forceBottom || atBottom) {
    requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  }
}

$("d-full").onclick = async () => {
  if (!selected) return;
  try {
    const resp = await fetch("/proxy/ui/request/" + encodeURIComponent(selected));
    if (!resp.ok) return;
    const j = await resp.json();
    $("prompt").textContent = j.prompt;
    fullLoaded.add(selected);
    $("d-full-note").textContent = "full prompt loaded";
  } catch (e) { /* ignore */ }
};

let STATE = { active: [], history: [], slots: [] };
async function poll() {
  try {
    const resp = await fetch("/proxy/ui/state");
    if (!resp.ok) throw new Error(resp.status);
    STATE = await resp.json();
    $("conn").classList.add("ok");
    renderActive(); renderHistory(); renderSlots();
    if (selected) {
      const r = findReq(selected);
      if (r) renderDetailMeta(r);
    }
  } catch (e) {
    $("conn").classList.remove("ok");
  }
}

const es = new EventSource("/proxy/ui/events");
es.onmessage = (m) => {
  let ev;
  try { ev = JSON.parse(m.data); } catch (e) { return; }
  if (ev.type === "tokens") {
    const t = tails.get(ev.rid) || { c: "", r: "" };
    if (ev.content) t.c += ev.content;
    if (ev.reasoning) t.r += ev.reasoning;
    tails.set(ev.rid, t);
    if (ev.rid === selected) {
      renderTokens();
      const r = findReq(ev.rid);
      if (r) { r.n_chars += (ev.content || "").length; renderDetailMeta(r); }
    }
  } else if (ev.type === "end" && ev.rid === selected) {
    const r = findReq(ev.rid);
    if (r) renderDetailMeta(r);
  }
};
es.onerror = () => $("conn").classList.remove("ok");

setInterval(poll, 3000);
setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString(); }, 1000);
renderTokens();
poll();
</script>
</body>
</html>
"""
