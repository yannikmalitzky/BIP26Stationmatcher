/* Manual review of uncertain plans, zoom viewer, decisions and folder export.
 * Loaded after the main page script; uses its globals: rows, running, $, esc, renderTable, reasoningHtml. */

const STATIONS = window.STATION_NAMES || [];

// ---------------------------------------------------------------------------------------------
// Which plans need a human look, and what is the final station?
// ---------------------------------------------------------------------------------------------
// extra answer fields: additional stations and the type of a facility that belongs to no station
function parseExtraFields(text) {
  const get = label => {
    const m = text.match(new RegExp("^[\\s*\\-]*\\*\\*" + label + ":\\*\\*[ \\t]*(.*)$", "mi"));
    return m ? m[1].replace(/\*/g, "").trim() : "";
  };
  const none = v => !v || /^(none|n\/a|na|not\b|-)/i.test(v);
  const extras = none(get("Additional stations")) ? [] : get("Additional stations").split(/[;,]/).map(x => x.trim()).filter(x => x && !none(x));
  let facility = get("Facility type");
  facility = none(facility) ? "" : facility.replace(/\s*\(.*?\)/g, "").trim();
  return {extras, facility};
}

function needsReview(r) {
  if (r.state === "skip" || r.state === "err") return !!r.plan_id;
  if (r.state !== "done") return false;
  return !r.station || r.level !== "high" || !!(r.corr && r.corr.length) || !!r.retried || !!(r.prep && r.prep.error)
    || !!(r.extras && r.extras.length);
}
const effectiveStation = r => r.final ? r.final.station : (r.state === "done" ? (r.station || "") : "");
// all stations a plan is sorted into (the main station first) and, without a station, the facility type
const effectiveStations = r => r.final ? [r.final.station].concat(r.final.extras || []).filter(Boolean)
  : (r.state === "done" && r.station ? [r.station].concat(r.extras || []) : []);
const effectiveFacility = r => r.final ? (r.final.facility || "") : (r.facility || "");
const decidedBy = r => r.final ? r.final.by : (r.state === "done" && r.station ? "model" : "none");

// hooks called by the results table of the main script
function reviewDecorate(r, st, cf) {
  if (r.final) {
    const name = r.final.station ? `<b>${esc(r.final.station)}</b>` : `<span class="none">no station</span>`;
    const tag = r.final.by === "manual" ? `<span class="tag manual">manual</span>` : `<span class="tag ok">confirmed</span>`;
    const more = r.final.station && (r.final.extras || []).length ? `<div class="hint">also: ${esc(r.final.extras.join(", "))}</div>`
      : !r.final.station && r.final.facility ? `<div class="hint">${esc(r.final.facility)}</div>` : "";
    return {st: name + tag + more, cf};
  }
  if (r.state === "done") {
    if (r.station && (r.extras || []).length) st += `<div class="hint">also: ${esc(r.extras.join(", "))}</div>`;
    else if (!r.station && r.facility) st += `<div class="hint">${esc(r.facility)}</div>`;
  }
  if (needsReview(r)) st += `<span class="tag review">review</span>`;
  return {st, cf};
}
function reviewDocExtra(r, i) {
  return r.plan_id && (r.state === "done" || r.state === "skip")
    ? `<button class="inspect" data-inspect="${i}" title="Inspect this plan in full resolution">🔍</button>` : "";
}
$("out").addEventListener("click", e => {
  const b = e.target.closest("[data-inspect]");
  if (!b) return;
  e.stopPropagation();  // do not also toggle the reasoning row
  openReview(Number(b.dataset.inspect));
}, true);

// ---------------------------------------------------------------------------------------------
// State (survives a page reload) and the info bar above the results
// ---------------------------------------------------------------------------------------------
let saveTimer = null;
function saveState() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const slim = rows.map(r => { const c = Object.assign({}, r); delete c.file; return c; });
    fetch("/state", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({rows: slim})})
      .catch(() => {});
  }, 300);
}

function updateReviewBar() {
  const bar = $("reviewbar");
  if (!rows.length || running) { bar.hidden = true; $("folders").hidden = true; return; }
  const flagged = rows.filter(needsReview), open = flagged.filter(r => !r.reviewed);
  $("folders").hidden = !rows.some(r => r.plan_id);
  if (!flagged.length) {
    bar.className = "rbar ok";
    bar.innerHTML = `<span>✓ All ${rows.length} plans were assigned with high confidence.</span>
      <div class="acts"><button id="rvstart" class="ghost">Inspect plans</button><button id="rvfolders">Create folders</button></div>`;
  } else if (open.length) {
    bar.className = "rbar";
    bar.innerHTML = `<span>⚠ <b>${open.length}</b> of ${rows.length} plans should be checked manually: the model was unsure,
      found no station, or an answer was corrected by a check.</span>
      <div class="acts"><button id="rvstart">Start review</button></div>`;
  } else {
    bar.className = "rbar ok";
    bar.innerHTML = `<span>✓ All ${flagged.length} uncertain plans were reviewed. You can now create the folder structure.</span>
      <div class="acts"><button id="rvfolders">Create folders</button><button id="rvstart" class="ghost">Review again</button></div>`;
  }
  bar.hidden = false;
}
$("reviewbar").addEventListener("click", e => {
  if (e.target.id === "rvstart") openReview();
  if (e.target.id === "rvfolders") openExport();
});
$("folders").onclick = () => openExport();

function afterRun() {
  updateReviewBar();
  saveState();
  // warm up the full-resolution images of the uncertain plans (one after the other), so the review opens instantly
  const ids = rows.filter(needsReview).map(r => r.plan_id).filter(Boolean);
  (async () => { for (const id of ids) { try { await fetch(`/plan_image/${id}`); } catch (e) { return; } } })();
}

// offer to restore the last session after a page reload
fetch("/state").then(r => r.json()).then(d => {
  if (!d.rows || !d.rows.length || rows.length) return;
  const bar = $("restorebar");
  bar.innerHTML = `<span>A previous session with <b>${d.rows.length}</b> plans was found.</span>
    <div class="acts"><button id="restorebtn" class="ghost">Restore</button></div>`;
  bar.hidden = false;
  $("restorebtn").onclick = () => {
    rows = d.rows.map(r => Object.assign({}, r, {file: null}));
    bar.hidden = true;
    $("csv").hidden = false;
    renderTable();
    updateReviewBar();
  };
}).catch(() => {});

// ---------------------------------------------------------------------------------------------
// Zoom viewer: wheel = zoom at the cursor, drag = pan, double click = zoom in
// ---------------------------------------------------------------------------------------------
class Zoom {
  constructor(box, inner, img, layer, info) {
    Object.assign(this, {box, inner, img, layer, info, s: 1, tx: 0, ty: 0, w: 1, h: 1, fitScale: 1, drag: null});
    box.addEventListener("wheel", e => {
      e.preventDefault();
      const rc = box.getBoundingClientRect();
      this.zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - rc.left, e.clientY - rc.top);
    }, {passive: false});
    box.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      this.drag = {x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty, moved: false};
      box.setPointerCapture(e.pointerId);
      box.classList.add("drag");
    });
    box.addEventListener("pointermove", e => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) this.drag.moved = true;
      this.tx = this.drag.tx + dx; this.ty = this.drag.ty + dy;
      this.apply();
    });
    box.addEventListener("pointerup", e => {
      const d = this.drag;
      this.drag = null;
      box.classList.remove("drag");
      if (d && !d.moved && this.onBoxClick) {  // a click (no drag) on a region frame opens that region
        const hit = document.elementFromPoint(e.clientX, e.clientY);
        const z = hit && hit.closest && hit.closest(".zbox");
        if (z) this.onBoxClick(Number(z.dataset.view));
      }
    });
    box.addEventListener("dblclick", e => {
      const rc = box.getBoundingClientRect();
      this.zoomAt(2, e.clientX - rc.left, e.clientY - rc.top);
    });
  }
  apply() {
    this.inner.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.s})`;
    this.inner.style.setProperty("--s", this.s);
    this.info.textContent = Math.round(this.s * 100) + " %  ·  " + this.w + " × " + this.h + " px";
  }
  fit() {
    const W = this.box.clientWidth, H = this.box.clientHeight;
    this.fitScale = Math.min(W / this.w, H / this.h);
    this.s = this.fitScale;
    this.tx = (W - this.w * this.s) / 2; this.ty = (H - this.h * this.s) / 2;
    this.apply();
  }
  zoomAt(factor, cx, cy) {
    const s2 = Math.min(Math.max(this.s * factor, this.fitScale * 0.8), 8);
    const k = s2 / this.s;
    this.tx = cx - (cx - this.tx) * k; this.ty = cy - (cy - this.ty) * k;
    this.s = s2;
    this.apply();
  }
  zoomCenter(factor) { this.zoomAt(factor, this.box.clientWidth / 2, this.box.clientHeight / 2); }
  actual() {  // 100 %: one image pixel = one screen pixel, around the current centre
    const cx = this.box.clientWidth / 2, cy = this.box.clientHeight / 2;
    this.zoomAt(1 / this.s, cx, cy);
  }
}

// ---------------------------------------------------------------------------------------------
// Review screen
// ---------------------------------------------------------------------------------------------
const RV = {built: false, queue: [], pos: 0, views: [], view: 0, cache: {}, z: null, token: 0};

function buildReview() {
  if (RV.built) return;
  RV.built = true;
  const dl = STATIONS.map(n => `<option value="${esc(n)}">`).join("");
  document.body.insertAdjacentHTML("beforeend", `
  <datalist id="stationlist">${dl}</datalist>
  <datalist id="factypes"><option value="Rectifier substation"><option value="Workshop"><option value="River tunnel">
    <option value="Track section"><option value="Unclear location"><option value="Other"></datalist>
  <div id="rv" class="rv" hidden>
    <div class="rv-top">
      <div><b id="rvtitle"></b><span class="hint" id="rvsub"></span></div>
      <div><label><input type="checkbox" id="rvall"> all plans</label><button id="rvclose" class="ghost">Close ✕</button></div>
    </div>
    <div class="rv-main">
      <div class="rv-left">
        <div class="rv-tabs" id="rvtabs"></div>
        <div class="zoom" id="zoom">
          <div class="zi" id="zi"><img id="zimg" draggable="false" alt="Plan"><div id="zlayer"></div></div>
          <div class="zload" id="zload">Loading plan in full resolution …</div>
        </div>
        <div class="zctl">
          <button data-z="in" title="Zoom in">+</button><button data-z="out" title="Zoom out">−</button>
          <button data-z="fit">Fit</button><button data-z="100">100 %</button>
          <label><input type="checkbox" id="rvregions" checked> show text regions</label>
          <span id="zinfo"></span><span>· wheel: zoom · drag: move · double click: zoom in</span>
        </div>
      </div>
      <aside class="rv-side">
        <div class="rv-card">
          <div class="label">Proposed station</div>
          <div class="rv-prop" id="rvprop"></div>
          <div id="rvconf"></div>
          <div class="rv-why" id="rvwhy"></div>
          <details><summary>Model reasoning</summary><div id="rvfull"></div></details>
        </div>
        <div class="rv-card">
          <div class="label">Your decision</div>
          <button id="rvuse" class="primary">Use proposed station</button>
          <div class="rv-own"><input id="rvown" list="stationlist" placeholder="Own station name …" autocomplete="off">
            <button id="rvownbtn">Use name</button></div>
          <div class="k" style="margin-top:10px">Also belongs to (optional, comma-separated)</div>
          <input id="rvextra" class="rv-in" placeholder="further stations …" autocomplete="off">
          <div class="k" style="margin-top:10px">No station</div>
          <div class="rv-own"><input id="rvfac" list="factypes" placeholder="Type, e.g. rectifier substation" autocomplete="off">
            <button id="rvnone" class="ghost">No station</button></div>
          <div class="rv-decision" id="rvdecision" hidden></div>
        </div>
        <div class="rv-nav"><button id="rvprev" class="ghost">← Previous</button><button id="rvnext" class="ghost">Next →</button></div>
      </aside>
    </div>
  </div>`);
  RV.z = new Zoom($("zoom"), $("zi"), $("zimg"), $("zlayer"), $("zinfo"));
  RV.z.onBoxClick = i => showView(i);
  $("zimg").onload = () => {
    const z = RV.z, im = $("zimg");
    z.w = im.naturalWidth; z.h = im.naturalHeight;
    $("zlayer").style.cssText = `position:absolute;left:0;top:0;width:${z.w}px;height:${z.h}px`;
    z.fit();
    $("zload").hidden = true;
  };
  $("zimg").onerror = () => { $("zload").textContent = "The plan image could not be loaded."; };
  document.querySelectorAll(".zctl button").forEach(b => b.onclick = () => {
    const k = b.dataset.z;
    if (k === "in") RV.z.zoomCenter(1.5); else if (k === "out") RV.z.zoomCenter(1 / 1.5);
    else if (k === "fit") RV.z.fit(); else RV.z.actual();
  });
  $("rvregions").onchange = () => drawOverlays();
  $("rvclose").onclick = closeReview;
  $("rvall").onchange = () => { RV.all = $("rvall").checked; rebuildQueue(true); };
  $("rvprev").onclick = () => go(-1);
  $("rvnext").onclick = () => go(1);
  $("rvuse").onclick = () => decide(curRow().station || "", "model");
  $("rvnone").onclick = () => decide("", "none");
  $("rvextra").addEventListener("keydown", e => { if (e.key === "Enter") $("rvuse").click(); });
  $("rvownbtn").onclick = () => { const v = $("rvown").value.trim(); if (v) decide(v, "manual"); };
  $("rvown").addEventListener("keydown", e => { if (e.key === "Enter") $("rvownbtn").click(); });
  document.addEventListener("keydown", e => {
    if ($("rv").hidden) return;
    if (e.key === "Escape") closeReview();
    if (document.activeElement && document.activeElement.tagName === "INPUT" && document.activeElement.type !== "checkbox") return;
    if (e.key === "ArrowRight") go(1);
    if (e.key === "ArrowLeft") go(-1);
  });
}

const curRow = () => rows[RV.queue[RV.pos]];
const reviewable = () => rows.map((r, i) => i).filter(i => rows[i].plan_id && (rows[i].state === "done" || rows[i].state === "skip"));

function rebuildQueue(keepCurrent) {
  const cur = RV.queue.length ? RV.queue[RV.pos] : null;
  RV.queue = RV.all ? reviewable() : reviewable().filter(i => needsReview(rows[i]) && !rows[i].reviewed);
  if (!RV.queue.length) { closeReview(); return; }
  RV.pos = keepCurrent && cur !== null && RV.queue.includes(cur) ? RV.queue.indexOf(cur) : 0;
  loadPlan();
}

function openReview(startIndex) {
  buildReview();
  RV.all = startIndex !== undefined;                 // inspecting one plan: browse all plans
  if (!RV.all && !rows.some(r => needsReview(r) && !r.reviewed)) RV.all = true;  // "Review again"
  $("rvall").checked = RV.all;
  RV.queue = RV.all ? reviewable() : reviewable().filter(i => needsReview(rows[i]) && !rows[i].reviewed);
  if (!RV.queue.length) return;
  RV.pos = startIndex !== undefined && RV.queue.includes(startIndex) ? RV.queue.indexOf(startIndex) : 0;
  $("rv").hidden = false;
  document.body.style.overflow = "hidden";
  loadPlan();
}

function closeReview() {
  if (!RV.built) return;
  $("rv").hidden = true;
  document.body.style.overflow = "";
  renderTable(); updateReviewBar(); saveState();
}

function go(d) {
  if (!RV.queue.length) return;
  RV.pos = (RV.pos + d + RV.queue.length) % RV.queue.length;
  loadPlan();
}

function shortReasons(r) {
  if (!r.answer) return r.state === "skip" ? `<div>${esc(r.error || "skipped")}: no proposal for this plan.</div>` : "";
  const get = label => {
    const m = r.answer.match(new RegExp("^[\\s*\\-]*\\*\\*" + label + ":\\*\\*\\s*([\\s\\S]*?)(?=^[\\s*\\-]*\\*\\*[A-Za-z ]+:\\*\\*|$(?![\\s\\S]))", "mi"));
    return m ? m[1].replace(/\*\*/g, "").trim() : "";
  };
  const rows2 = [["Title", get("Title")], ["Plan date", get("Plan date")], ["How found", get("Match type")],
                 ["Confidence", get("Confidence")], ["Check", get("Post-check")]].filter(x => x[1]);
  return rows2.map(([k, v]) => `<div style="margin-top:6px"><div class="k">${k}</div>${esc(v)}</div>`).join("");
}

function loadPlan() {
  const r = curRow(), token = ++RV.token;
  $("rvtitle").textContent = r.title;
  $("rvsub").textContent = `plan ${RV.pos + 1} of ${RV.queue.length}`;
  const prop = $("rvprop");
  prop.textContent = r.station || "no station found";
  prop.className = "rv-prop" + (r.station ? "" : " none");
  $("rvconf").innerHTML = r.level ? `<span class="badge ${r.level}">${r.level} confidence</span>` : "";
  $("rvwhy").innerHTML = shortReasons(r)
    + ((r.extras || []).length ? `<div style="margin-top:6px"><div class="k">Also</div>${esc(r.extras.join(", "))}</div>` : "")
    + (!r.station && r.facility ? `<div style="margin-top:6px"><div class="k">Type</div>${esc(r.facility)}</div>` : "");
  $("rvfull").innerHTML = r.answer ? reasoningHtml(r.answer) : "";
  $("rvuse").disabled = !r.station;
  $("rvown").value = "";
  $("rvextra").value = r.final ? (r.final.extras || []).join(", ") : (r.extras || []).join(", ");
  $("rvfac").value = r.final ? (r.final.facility || "") : (r.facility || "");
  const dec = $("rvdecision");
  if (r.final) {
    dec.hidden = false;
    dec.innerHTML = `Decision: <b>${esc([r.final.station || "no station"].concat(r.final.extras || []).join(" + "))}</b> (${r.final.by === "manual" ? "manual" : "confirmed"})
      <a href="#" id="rvreset" style="margin-left:8px">reset</a>`;
    $("rvreset").onclick = ev => { ev.preventDefault(); delete r.final; r.reviewed = false; loadPlan(); renderTable(); updateReviewBar(); saveState(); };
  } else dec.hidden = true;

  $("zload").hidden = false; $("zload").textContent = "Loading plan in full resolution …";
  $("zimg").removeAttribute("src");
  RV.views = [{name: "Whole plan", url: `/plan_image/${r.plan_id}`}];
  renderTabs();
  const cached = RV.cache[r.plan_id];
  (cached ? Promise.resolve(cached) : fetch(`/plan_regions/${r.plan_id}`).then(x => x.json()).then(d => (RV.cache[r.plan_id] = d)))
    .then(d => {
      if (token !== RV.token) return;
      RV.regions = d.regions || [];
      RV.views = [{name: "Whole plan", url: `/plan_image/${r.plan_id}`}].concat(RV.regions.map(g => ({
        name: g.name, box: g.box,
        url: `/plan_crop/${r.plan_id}?x0=${g.box[0]}&y0=${g.box[1]}&x1=${g.box[2]}&y1=${g.box[3]}`})));
      renderTabs();
    }).catch(() => {});
  showView(0);
  const next = RV.queue[(RV.pos + 1) % RV.queue.length], nr = rows[next];   // warm up the next plan
  if (nr && nr.plan_id && nr !== r) { const im = new Image(); im.src = `/plan_image/${nr.plan_id}`; }
}

function renderTabs() {
  $("rvtabs").innerHTML = RV.views.map((v, i) => `<button data-v="${i}" class="${i === RV.view ? "on" : ""}">${esc(v.name)}</button>`).join("");
  $("rvtabs").querySelectorAll("button").forEach(b => b.onclick = () => showView(Number(b.dataset.v)));
  drawOverlays();
}

function showView(i) {
  if (!RV.views[i]) return;
  RV.view = i;
  renderTabs();  // also (re)draws the region frames on the whole-plan view
  $("zload").hidden = false; $("zload").textContent = "Loading …";
  $("zimg").src = RV.views[i].url;
}

function drawOverlays() {
  const layer = $("zlayer");
  layer.innerHTML = "";
  if (RV.view !== 0 || !$("rvregions").checked) return;
  RV.views.slice(1).forEach((v, k) => {
    const [x0, y0, x1, y1] = v.box;
    const cls = v.name.startsWith("Title") ? "title" : v.name.startsWith("Corner") ? "corner" : "text";
    layer.insertAdjacentHTML("beforeend",
      `<div class="zbox ${cls}" data-view="${k + 1}" style="left:${x0}px;top:${y0}px;width:${x1 - x0}px;height:${y1 - y0}px"><span>${esc(v.name)}</span></div>`);
  });
}
function decide(station, by) {
  const r = curRow();
  const extras = station ? $("rvextra").value.split(",").map(x => x.trim()).filter(x => x && x !== station) : [];
  r.final = {station, by, extras, facility: station ? "" : $("rvfac").value.trim()};
  r.reviewed = true;
  renderTable(); updateReviewBar(); saveState();
  const open = RV.queue.map((q, k) => k).filter(k => !rows[RV.queue[k]].reviewed);
  if (!open.length) { closeReview(); return; }
  const after = open.find(k => k > RV.pos);
  RV.pos = after !== undefined ? after : open[0];
  loadPlan();
}

// ---------------------------------------------------------------------------------------------
// Folder export
// ---------------------------------------------------------------------------------------------
function openExport() {
  if (!$("dlg")) {
    document.body.insertAdjacentHTML("beforeend", `
    <div class="dlg" id="dlg" hidden><div class="dlg-box">
      <h3>Create folder structure</h3>
      <div class="msg" id="dlgsum"></div>
      <label for="dlgpath">Target folder on this computer</label>
      <input id="dlgpath">
      <div class="msg warn" id="dlgwarn"></div>
      <div class="msg" id="dlgres"></div>
      <div class="row2"><button class="ghost" id="dlgcancel">Close</button><button class="ghost" id="dlgopen" hidden>Open folder</button>
        <button id="dlgok">Create folders</button></div>
    </div></div>`);
    $("dlgcancel").onclick = () => { $("dlg").hidden = true; };
    $("dlgok").onclick = runExport;
    $("dlgopen").onclick = () => fetch("/reveal", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: $("dlgopen").dataset.path})});
  }
  const withPlan = rows.filter(r => r.plan_id);
  const stations = new Set(withPlan.flatMap(effectiveStations));
  const multi = withPlan.filter(r => effectiveStations(r).length > 1).length;
  const open = rows.filter(r => needsReview(r) && !r.reviewed).length;
  $("dlgsum").textContent = `${withPlan.length} plans will be copied into one folder per station (${stations.size} stations found so far${multi ? `; ${multi} plan${multi > 1 ? "s belong" : " belongs"} to several stations and ${multi > 1 ? "are" : "is"} copied into each` : ""}).
    Plans without a station go to "_no_station/<type>". Your original files are not changed.`;
  $("dlgwarn").textContent = open ? `${open} uncertain plan${open > 1 ? "s are" : " is"} not reviewed yet and will be sorted by the model's proposal.` : "";
  $("dlgres").textContent = ""; $("dlgres").className = "msg"; $("dlgopen").hidden = true;
  if (!$("dlgpath").value) $("dlgpath").value = window.DEFAULT_EXPORT_DIR || "";
  $("dlgok").disabled = false;
  $("dlg").hidden = false;
}

async function runExport() {
  const items = rows.filter(r => r.plan_id).map(r => ({
    plan_id: r.plan_id, name: r.title, stations: effectiveStations(r), facility: effectiveFacility(r),
    source: decidedBy(r), confidence: r.level || ""}));
  $("dlgok").disabled = true;
  $("dlgres").className = "msg"; $("dlgres").textContent = "Copying plans …";
  try {
    const res = await fetch("/export", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({out_dir: $("dlgpath").value.trim(), items})});
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || res.statusText);
    $("dlgres").className = "msg okc";
    $("dlgres").textContent = `✓ ${d.files} plans sorted into ${d.folders} folders in ${d.out_dir}` +
      (d.missing.length ? ` (${d.missing.length} could not be copied)` : "");
    $("dlgopen").dataset.path = d.out_dir; $("dlgopen").hidden = false;
  } catch (err) {
    $("dlgres").className = "msg warn"; $("dlgres").textContent = err.message;
    $("dlgok").disabled = false;
  }
}
