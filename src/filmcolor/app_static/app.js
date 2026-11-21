/* FilmColor Darkroom Studio — Frontend Application */
(() => {
"use strict";

const $ = id => document.getElementById(id);

const BASE_LABELS = {
  auto: "单帧自动 (B0)",
  manual: "手动框选片边",
  roll_nat: "整卷自然融合",
  roll_dq: "整卷解耦 (DQ)",
  roll_b1: "整卷片基 (B1)",
};

const HSL_META = [
  { id: "red",     label: "红", color: "#e5484d" },
  { id: "orange",  label: "橙", color: "#f08c3a" },
  { id: "yellow",  label: "黄", color: "#e8c547" },
  { id: "green",   label: "绿", color: "#52b768" },
  { id: "aqua",    label: "青", color: "#38b2c4" },
  { id: "blue",    label: "蓝", color: "#4c8bf5" },
  { id: "purple",  label: "紫", color: "#9b6ef3" },
  { id: "magenta", label: "品", color: "#e056a8" },
];

/* ======================== Application State ======================== */
let meta = null;
let engineSel = "t24";
let presetSel = "standard";
let filmSel = "color";
let fileList = [];
let currentRel = null;
let rollData = {};
let savedPerFile = {}; // rel -> { params, grade, geom, engine, preset, film, edited }
let historyPerFile = {}; // rel -> { undo: [], redo: [] }
let clipboardSnap = null;
let userPresets = [];

// Live editing state for open photo
let params = {};
let grade = defaultGrade();
let geom = defaultGeom();
let autoStartParams = {};
let autoStartGrade = defaultGrade();
let softness = 0.5;
let curves = defaultCurves();
let curveChan = "rgb";
let dragCurveIdx = -1;
let lastHistBins = null;

// UI & Canvas state
let viewMode = "pos";          // "pos" | "compare" | "neg"
let compareStyle = "wipe";     // "wipe" | "side"
let compareSource = "auto";    // "auto" | "neg"
let spaceComparing = false;
let wipePos = 0.5;
let activeTool = null;         // null | "wb" | "crop" | "loupe" | "pick_base"
let showClip = false;
let browserGrid = false;
let browserFilter = "all";
let cgZone = "cg_shadows";
let hslBand = "orange";
let cropRatio = "free";
let cropPortrait = false;
let cropDrag = null;
let baseDrag = null;

// Zoom / Pan state
let zoom = 1.0;
let panX = 0, panY = 0;
let panning = null;

// Async request orchestration
let renderTimer = null;
let renderAbort = null;
let beforeAbort = null;
let renderSeq = 0;
let beforeSeq = 0;
let lastRenderedUrl = null;
let lastBeforeUrl = null;
let busy = false;
let toastTimer = null;

// Export modal state
let exportScope = "single";
let batchTarget = "edited";
let exportFmt = "jpeg";

/* ======================== Defaults & Serialization ======================== */
function defaultCurves() {
  return { rgb: [[0, 0], [1, 1]], r: [[0, 0], [1, 1]], g: [[0, 0], [1, 1]], b: [[0, 0], [1, 1]] };
}
function defaultGrade() {
  return {
    highlights: 0, shadows: 0, whites: 0, blacks: 0, vibrance: 0, skin_protect: 0,
    tone_curve: {},
    cg_shadows: [0, 0, 0], cg_midtones: [0, 0, 0], cg_highlights: [0, 0, 0],
    cg_balance: 0, cg_blending: 0.5,
    hsl: {},
    dehaze: 0, clarity: 0, sharpen: 0, vignette: 0, grain: 0,
  };
}
function defaultGeom() {
  return { rot90: 0, angle: 0, flip_h: false, flip_v: false, crop: null };
}
function defaultV3Params() {
  return {
    ev: 0.0, temp: 0.0, tint: 0.0, sat: 1.0, contrast: 1.30, midtone: 0.23, shoulder: 0.15,
    highlights: 0.0, shadows: 0.0, whites: 0.0, blacks: 0.0, vibrance: 0.0, slope_r: 0.0, slope_b: 0.0,
    chroma_preserve: 0.40, bpc: 0.0, soft_clip: 0.0, skin_protect: 0.0, hd_curve: 0.0, log_unmix: 0.0,
    cam_mix: 0.40, tone_curve: {},
  };
}
function defaultT24Params(isT24 = true) {
  const base = {
    ev: 0.0, temp: 0.0, tint: 0.0, sat: 1.0, contrast: 1.15, shoulder: 0.5, toe: 0.0,
    cam_mix: 0.35,
  };
  if (isT24) {
    base.slope_r = 0.0;
    base.slope_b = 0.0;
    base.eps_rgb = [0.0, 0.0, 0.0];
    base.residual_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  }
  return base;
}
function isT19Engine(eng) { return (eng || engineSel) === "t24" || (eng || engineSel) === "t23"; }
function deepClone(x) { return JSON.parse(JSON.stringify(x)); }
function folderOf(rel) { const i = rel ? rel.lastIndexOf("/") : -1; return i >= 0 ? rel.slice(0, i) : "."; }

function camMatrix(mix) {
  const M = meta && meta.camera_matrix;
  if (!M) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return M.map((row, i) => {
    const r = row.map((v, j) => mix * v + (1 - mix) * (i === j ? 1 : 0));
    const d = (r[0] + r[1] + r[2] - 1) / 3;
    return r.map(v => v - d);
  });
}
function camMixOf(m) {
  const M = meta && meta.camera_matrix;
  if (!M || !m) return 0;
  let num = 0, den = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const a = M[i][j] - (i === j ? 1 : 0);
    num += (m[i][j] - (i === j ? 1 : 0)) * a; den += a * a;
  }
  return den > 0 ? Math.max(0, Math.min(1, num / den)) : 0;
}

function serializeCurves() {
  const tc = {};
  for (const ch of ["rgb", "r", "g", "b"]) {
    const pts = curves[ch];
    if (pts && !(pts.length === 2 && Math.abs(pts[0][0]) < 1e-4 && Math.abs(pts[0][1]) < 1e-4 && Math.abs(pts[1][0] - 1) < 1e-4 && Math.abs(pts[1][1] - 1) < 1e-4)) {
      tc[ch] = pts.map(pt => [Number(pt[0].toFixed(4)), Number(pt[1].toFixed(4))]);
    }
  }
  return tc;
}

function loadCurvesFromDict(tc) {
  curves = defaultCurves();
  if (tc) {
    for (const ch of ["rgb", "r", "g", "b"]) {
      if (tc[ch] && tc[ch].length >= 2) {
        curves[ch] = tc[ch].map(pt => [Number(pt[0]), Number(pt[1])]);
      }
    }
  }
  drawCurveCanvas();
}

function serializeParams(srcParams, eng) {
  const e = eng || engineSel;
  const p = deepClone(srcParams || params);
  if (p.cam_mix != null) p.matrix = camMatrix(p.cam_mix);
  delete p.cam_mix;
  if (isT19Engine(e)) {
    const allowed = e === "t24"
      ? ["ev", "temp", "tint", "sat", "contrast", "shoulder", "toe", "slope_r", "slope_b", "eps_rgb", "residual_matrix", "matrix"]
      : ["ev", "temp", "tint", "sat", "contrast", "shoulder", "toe", "matrix"];
    const out = {};
    for (const k of allowed) if (p[k] !== undefined) out[k] = p[k];
    return out;
  }
  p.tone_curve = serializeCurves();
  delete p.toe; delete p.eps_rgb; delete p.residual_matrix;
  return p;
}

function serializeGrade(srcGrade, eng) {
  const e = eng || engineSel;
  const g = deepClone(srcGrade || grade);
  const out = {};
  const scalarKeys = isT19Engine(e)
    ? ["highlights", "shadows", "whites", "blacks", "vibrance", "skin_protect", "dehaze", "clarity", "sharpen", "vignette", "grain"]
    : ["dehaze", "clarity", "sharpen", "vignette", "grain"];
  for (const k of scalarKeys) {
    if (g[k] && Math.abs(g[k]) > 1e-5) out[k] = Number(Number(g[k]).toFixed(4));
  }
  if (isT19Engine(e)) {
    const tc = srcGrade ? (srcGrade.tone_curve || {}) : serializeCurves();
    if (Object.keys(tc).length) out.tone_curve = tc;
  }
  const hasCg = [g.cg_shadows, g.cg_midtones, g.cg_highlights].some(w => w && (w[1] > 1e-4 || Math.abs(w[2]) > 1e-4));
  if (hasCg) {
    if (g.cg_shadows && (g.cg_shadows[1] > 1e-4 || Math.abs(g.cg_shadows[2]) > 1e-4)) out.cg_shadows = g.cg_shadows;
    if (g.cg_midtones && (g.cg_midtones[1] > 1e-4 || Math.abs(g.cg_midtones[2]) > 1e-4)) out.cg_midtones = g.cg_midtones;
    if (g.cg_highlights && (g.cg_highlights[1] > 1e-4 || Math.abs(g.cg_highlights[2]) > 1e-4)) out.cg_highlights = g.cg_highlights;
    if (g.cg_balance && Math.abs(g.cg_balance) > 1e-4) out.cg_balance = Number(g.cg_balance.toFixed(4));
    if (g.cg_blending != null && Math.abs(g.cg_blending - 0.5) > 1e-4) out.cg_blending = Number(g.cg_blending.toFixed(4));
  }
  if (g.hsl) {
    const cleanHsl = {};
    for (const [band, triple] of Object.entries(g.hsl)) {
      if (triple && (Math.abs(triple[0]) > 1e-4 || Math.abs(triple[1]) > 1e-4 || Math.abs(triple[2]) > 1e-4)) {
        cleanHsl[band] = triple.map(v => Number(Number(v).toFixed(4)));
      }
    }
    if (Object.keys(cleanHsl).length) out.hsl = cleanHsl;
  }
  return out;
}

function serializeGeom(srcGeom) {
  const g = srcGeom || geom;
  const out = {};
  if (g.rot90) out.rot90 = ((g.rot90 % 4) + 4) % 4;
  if (g.angle && Math.abs(g.angle) >= 0.01) out.angle = Number(g.angle.toFixed(2));
  if (g.flip_h) out.flip_h = true;
  if (g.flip_v) out.flip_v = true;
  if (g.crop) out.crop = g.crop.map(v => Number(v.toFixed(5)));
  return out;
}

/* ======================== Notifications & API ======================== */
function setStatus(msg, isWarn = false) {
  $("statusText").textContent = msg;
  $("statusText").className = isWarn ? "warn" : "";
}
function toast(msg, isWarn = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show" + (isWarn ? " warn" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2300);
}
function setLoading(on) { $("loadBar").classList.toggle("active", on); }

async function api(path, body, signal) {
  const opts = body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : {};
  if (signal) opts.signal = signal;
  const r = await fetch(path, opts);
  const ct = r.headers.get("Content-Type") || "";
  if (ct.startsWith("application/json")) {
    const j = await r.json();
    if (r.status === 409) throw new Error("照片状态已更新，请重新打开（" + j.error + "）");
    if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r;
}

/* ======================== History (Undo / Redo) ======================== */
function currentSnapshot() {
  return {
    params: deepClone(params),
    grade: deepClone(Object.assign({}, grade, { tone_curve: serializeCurves() })),
    geom: deepClone(geom),
    curves: deepClone(curves),
  };
}
function restoreSnapshot(snap) {
  params = deepClone(snap.params);
  grade = Object.assign(defaultGrade(), deepClone(snap.grade));
  geom = Object.assign(defaultGeom(), deepClone(snap.geom));
  curves = deepClone(snap.curves || defaultCurves());
  syncAllControls();
  persistCurrentFile(true);
  scheduleRender(0);
}
function pushHistory() {
  if (!currentRel) return;
  const h = historyPerFile[currentRel] || (historyPerFile[currentRel] = { undo: [], redo: [] });
  const snap = currentSnapshot();
  const sStr = JSON.stringify(snap);
  if (h.undo.length && JSON.stringify(h.undo[h.undo.length - 1]) === sStr) return;
  h.undo.push(snap);
  if (h.undo.length > 50) h.undo.shift();
  h.redo = [];
  updateUndoButtons();
}
function doUndo() {
  if (!currentRel || !meta) return;
  const h = historyPerFile[currentRel];
  if (!h || h.undo.length <= 1) return;
  const cur = h.undo.pop();
  h.redo.push(cur);
  restoreSnapshot(h.undo[h.undo.length - 1]);
  updateUndoButtons();
  toast("已撤销");
}
function doRedo() {
  if (!currentRel || !meta) return;
  const h = historyPerFile[currentRel];
  if (!h || !h.redo.length) return;
  const nxt = h.redo.pop();
  h.undo.push(nxt);
  restoreSnapshot(nxt);
  updateUndoButtons();
  toast("已重做");
}
function updateUndoButtons() {
  const h = currentRel ? historyPerFile[currentRel] : null;
  $("bUndo").disabled = !meta || !h || h.undo.length <= 1;
  $("bRedo").disabled = !meta || !h || h.redo.length === 0;
}

function persistCurrentFile(markEdited = true) {
  if (!currentRel || !meta) return;
  savedPerFile[currentRel] = {
    params: serializeParams(),
    grade: serializeGrade(),
    geom: serializeGeom(),
    rawParams: deepClone(params),
    rawGrade: deepClone(grade),
    rawGeom: deepClone(geom),
    curves: deepClone(curves),
    engine: engineSel,
    preset: presetSel,
    film: filmSel,
    edited: markEdited ? true : !!(savedPerFile[currentRel] && savedPerFile[currentRel].edited),
  };
  updateFileListUI();
}

/* ======================== Precision Slider Factory ======================== */
function updateTrackFill(input, min, max, bipolar) {
  if (input.classList.contains("wb-temp") || input.classList.contains("wb-tint")) return;
  const val = parseFloat(input.value);
  const pct = ((val - min) / (max - min)) * 100;
  if (bipolar && min < 0 && max > 0) {
    const zeroPct = ((0 - min) / (max - min)) * 100;
    const lo = Math.min(zeroPct, pct), hi = Math.max(zeroPct, pct);
    input.style.background = `linear-gradient(90deg, #2a2a33 ${lo}%, #dda14b ${lo}%, #dda14b ${hi}%, #2a2a33 ${hi}%)`;
  } else {
    input.style.background = `linear-gradient(90deg, #dda14b 0%, #dda14b ${pct}%, #2a2a33 ${pct}%, #2a2a33 100%)`;
  }
}

function createSliderRow(spec) {
  const { id, label, min, max, step, def, cls = "", bipolar = (min < 0 && max > 0), get, set, fmt, onCommit, disabled = false, title = "" } = spec;
  const uVal = get();
  const row = document.createElement("div");
  row.className = "srow" + (Math.abs(uVal - def) > step * 0.4 ? " modified" : "");
  row.dataset.sliderId = id;
  row.innerHTML = `
    <div class="srow-top">
      <span class="srow-lbl" title="${title || "双击归零 / 恢复默认"}">${label}</span>
      <span class="srow-val" title="点击精确输入数值">${fmt(uVal)}</span>
    </div>
    <div class="srow-track-wrap">
      ${bipolar ? `<span class="srow-zero" style="left:${((0 - min) / (max - min)) * 100}%"></span>` : ""}
      <input type="range" class="fc-range ${cls}" id="sl_${id}" min="${min}" max="${max}" step="${step}" value="${uVal}" ${disabled ? "disabled" : ""}>
    </div>
  `;
  const input = row.querySelector("input");
  const lbl = row.querySelector(".srow-lbl");
  const valEl = row.querySelector(".srow-val");
  updateTrackFill(input, min, max, bipolar);

  const applyVal = (raw, commit) => {
    const clamped = Math.max(min, Math.min(max, raw));
    input.value = clamped;
    valEl.textContent = fmt(clamped);
    row.classList.toggle("modified", Math.abs(clamped - def) > step * 0.4);
    updateTrackFill(input, min, max, bipolar);
    set(clamped);
    if (commit) {
      if (onCommit) onCommit(clamped);
      else { persistCurrentFile(true); pushHistory(); }
    }
    scheduleRender(commit ? 0 : 45);
  };

  input.oninput = () => applyVal(parseFloat(input.value), false);
  input.onchange = () => applyVal(parseFloat(input.value), true);
  const reset = () => { if (!input.disabled) applyVal(def, true); };
  lbl.ondblclick = reset;
  input.ondblclick = reset;

  // Click numeric value to type exact number
  valEl.onclick = () => {
    if (input.disabled) return;
    const cur = parseFloat(input.value);
    const box = document.createElement("input");
    box.type = "text";
    box.className = "srow-val-input";
    box.value = String(cur);
    valEl.replaceWith(box);
    box.focus();
    box.select();
    let done = false;
    const finish = (accept) => {
      if (done) return;
      done = true;
      box.replaceWith(valEl);
      if (accept) {
        const parsed = parseFloat(box.value);
        if (Number.isFinite(parsed)) applyVal(parsed, true);
      }
    };
    box.onkeydown = e => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    };
    box.onblur = () => finish(true);
  };

  return row;
}

/* ======================== Slider Definitions ======================== */
const fmtSigned = u => (u > 0 ? "+" + Math.round(u) : "" + Math.round(u));
const fmtInt = u => "" + Math.round(u);
const fmtEv = u => (u >= 0 ? "+" : "") + Number(u).toFixed(2);
const fmtDeg = u => (u > 0 ? "+" : "") + Number(u).toFixed(1) + "°";

function buildBasicSliders() {
  const t19 = isT19Engine();
  const isBw = filmSel === "bw";

  // 1. White Balance
  const wbBox = $("slWb"); wbBox.innerHTML = "";
  wbBox.appendChild(createSliderRow({
    id: "temp", label: "色温（冷 ↔ 暖）", cls: "wb-temp", min: -100, max: 100, step: 1, def: 0, disabled: isBw,
    get: () => Math.round((params.temp || 0) * 100),
    set: u => { params.temp = u / 100; },
    fmt: fmtSigned,
  }));
  wbBox.appendChild(createSliderRow({
    id: "tint", label: "色调（绿 ↔ 洋红）", cls: "wb-tint", min: -100, max: 100, step: 1, def: 0, disabled: isBw,
    get: () => Math.round((params.tint || 0) * 100),
    set: u => { params.tint = u / 100; },
    fmt: fmtSigned,
  }));

  // 2. Tone
  const toneBox = $("slTone"); toneBox.innerHTML = "";
  const evMax = t19 ? 2.5 : 3.0;
  toneBox.appendChild(createSliderRow({
    id: "ev", label: "曝光 Exposure", min: -evMax, max: evMax, step: 0.05, def: 0,
    get: () => Number((params.ev || 0).toFixed(2)),
    set: u => { params.ev = u; },
    fmt: fmtEv,
  }));
  // Contrast mapping: T19 [0.55..2.0] pivot 1.15; V3 [0.5..2.0] pivot 1.30
  const cPivot = t19 ? 1.15 : 1.30, cLo = t19 ? 0.60 : 0.80, cHi = t19 ? 0.85 : 0.70;
  toneBox.appendChild(createSliderRow({
    id: "contrast", label: "对比度 Contrast", min: -100, max: 100, step: 1, def: 0,
    get: () => {
      const v = params.contrast != null ? params.contrast : cPivot;
      return Math.round((v >= cPivot ? (v - cPivot) / cHi : (v - cPivot) / cLo) * 100);
    },
    set: u => { params.contrast = u >= 0 ? cPivot + (u / 100) * cHi : cPivot + (u / 100) * cLo; },
    fmt: fmtSigned,
  }));

  // Unified zonal sliders: route to grade in T24/T23, to params in V3
  const zonalTarget = () => t19 ? grade : params;
  for (const [k, lbl] of [
    ["highlights", "高光 Highlights"],
    ["shadows", "阴影 Shadows"],
    ["whites", "白色色阶 Whites"],
    ["blacks", "黑色色阶 Blacks"],
  ]) {
    toneBox.appendChild(createSliderRow({
      id: k, label: lbl, min: -100, max: 100, step: 1, def: 0,
      get: () => Math.round((zonalTarget()[k] || 0) * 100),
      set: u => { zonalTarget()[k] = u / 100; },
      fmt: fmtSigned,
    }));
  }

  // Shoulder & Toe (T24/T23) or Midtone & Shoulder (V3)
  toneBox.appendChild(createSliderRow({
    id: "shoulder", label: "高光柔滚 Shoulder", min: 0, max: 100, step: 1, def: t19 ? 50 : 15, bipolar: false,
    get: () => Math.round((params.shoulder != null ? params.shoulder : (t19 ? 0.5 : 0.15)) * 100),
    set: u => { params.shoulder = u / 100; },
    fmt: fmtInt,
  }));
  if (t19) {
    toneBox.appendChild(createSliderRow({
      id: "toe", label: "暗部趾部 Toe", min: -100, max: 100, step: 1, def: 0,
      get: () => Math.round((params.toe || 0) * 100),
      set: u => { params.toe = u / 100; },
      fmt: fmtSigned,
    }));
  } else {
    toneBox.appendChild(createSliderRow({
      id: "midtone", label: "中灰亮度 Midtone", min: -100, max: 100, step: 1, def: 0,
      get: () => {
        const v = params.midtone != null ? params.midtone : 0.23;
        return Math.round((v >= 0.23 ? (v - 0.23) / 0.22 : (v - 0.23) / 0.13) * 100);
      },
      set: u => { params.midtone = u >= 0 ? 0.23 + (u / 100) * 0.22 : 0.23 + (u / 100) * 0.13; },
      fmt: fmtSigned,
    }));
  }

  // 3. Basic Color
  const colBox = $("slBasicColor"); colBox.innerHTML = "";
  colBox.appendChild(createSliderRow({
    id: "vibrance", label: "自然饱和度 Vibrance", min: -100, max: 100, step: 1, def: 0, disabled: isBw,
    get: () => Math.round((zonalTarget().vibrance || 0) * 100),
    set: u => { zonalTarget().vibrance = u / 100; },
    fmt: fmtSigned,
  }));
  colBox.appendChild(createSliderRow({
    id: "sat", label: "饱和度 Saturation", min: -100, max: 100, step: 1, def: 0, disabled: isBw,
    get: () => Math.round(((params.sat != null ? params.sat : 1.0) - 1.0) * 100),
    set: u => { params.sat = 1.0 + u / 100; },
    fmt: fmtSigned,
  }));
}

function buildDetailSliders() {
  const det = $("slDetail"); det.innerHTML = "";
  det.appendChild(createSliderRow({
    id: "clarity", label: "清晰度 Clarity", min: -100, max: 100, step: 1, def: 0,
    get: () => Math.round((grade.clarity || 0) * 100),
    set: u => { grade.clarity = u / 100; },
    fmt: fmtSigned,
  }));
  det.appendChild(createSliderRow({
    id: "dehaze", label: "去朦胧 Dehaze", min: -100, max: 100, step: 1, def: 0,
    get: () => Math.round((grade.dehaze || 0) * 100),
    set: u => { grade.dehaze = u / 100; },
    fmt: fmtSigned,
  }));
  det.appendChild(createSliderRow({
    id: "sharpen", label: "锐化 Sharpen", min: 0, max: 100, step: 1, def: 0, bipolar: false,
    get: () => Math.round((grade.sharpen || 0) * 100),
    set: u => { grade.sharpen = u / 100; },
    fmt: fmtInt,
  }));
  det.appendChild(createSliderRow({
    id: "vignette", label: "暗角 Vignette", min: -100, max: 100, step: 1, def: 0,
    get: () => Math.round((grade.vignette || 0) * 100),
    set: u => { grade.vignette = u / 100; },
    fmt: fmtSigned,
  }));
  det.appendChild(createSliderRow({
    id: "grain", label: "胶片颗粒 Grain", min: 0, max: 100, step: 1, def: 0, bipolar: false,
    get: () => Math.round((grade.grain || 0) * 100),
    set: u => { grade.grain = u / 100; },
    fmt: fmtInt,
  }));

  const gmBox = $("slGeom"); gmBox.innerHTML = "";
  gmBox.appendChild(createSliderRow({
    id: "angle", label: "微调拉直 Straighten", min: -15, max: 15, step: 0.1, def: 0,
    get: () => Number((geom.angle || 0).toFixed(1)),
    set: u => { geom.angle = u; },
    fmt: fmtDeg,
  }));
  $("bFlipH").classList.toggle("on", !!geom.flip_h);
  $("bFlipV").classList.toggle("on", !!geom.flip_v);
}

function buildOpticsSliders() {
  const box = $("slOptics"); box.innerHTML = "";
  const t19 = isT19Engine();
  const isBw = filmSel === "bw";
  const hasCam = !!(meta && meta.camera_matrix);

  // Skin protect (works in BOTH T24 via grade and V3 via params)
  const skinTarget = () => t19 ? grade : params;
  box.appendChild(createSliderRow({
    id: "skin_protect", label: "肤色保护 Skin Protect", min: 0, max: 100, step: 1, def: 0, bipolar: false, disabled: isBw,
    get: () => Math.round((skinTarget().skin_protect || 0) * 100),
    set: u => { skinTarget().skin_protect = u / 100; },
    fmt: fmtInt,
  }));

  box.appendChild(createSliderRow({
    id: "cam_mix", label: "相机色彩校准 Matrix", min: 0, max: 100, step: 1, def: t19 ? 35 : 40, bipolar: false,
    disabled: !hasCam || isBw,
    title: hasCam ? "调节相机原生色彩矩阵的混合比例" : "扫描仪 DNG / TIFF 不含相机色彩矩阵",
    get: () => Math.round((params.cam_mix != null ? params.cam_mix : (t19 ? 0.35 : 0.40)) * 100),
    set: u => { params.cam_mix = u / 100; },
    fmt: fmtInt,
  }));

  if (engineSel === "t24" || engineSel === "v3") {
    const sMax = engineSel === "t24" ? 60 : 50;
    box.appendChild(createSliderRow({
      id: "slope_r", label: "红层乳剂斜率 Red Slope", min: -sMax, max: sMax, step: 1, def: 0, disabled: isBw,
      get: () => Math.round((params.slope_r || 0) * 100),
      set: u => { params.slope_r = u / 100; },
      fmt: fmtSigned,
    }));
    box.appendChild(createSliderRow({
      id: "slope_b", label: "蓝层乳剂斜率 Blue Slope", min: -sMax, max: sMax, step: 1, def: 0, disabled: isBw,
      get: () => Math.round((params.slope_b || 0) * 100),
      set: u => { params.slope_b = u / 100; },
      fmt: fmtSigned,
    }));
  }

  // V3-only optical controls
  if (!t19) {
    box.appendChild(createSliderRow({
      id: "chroma_preserve", label: "中间调保色 Chroma Preserve", min: 0, max: 100, step: 1, def: 40, bipolar: false, disabled: isBw,
      get: () => Math.round((params.chroma_preserve || 0) * 100),
      set: u => { params.chroma_preserve = u / 100; },
      fmt: fmtInt,
    }));
    box.appendChild(createSliderRow({
      id: "bpc", label: "黑点补偿 BPC", min: 0, max: 25, step: 1, def: 0, bipolar: false,
      get: () => Math.round((params.bpc || 0) * 100),
      set: u => { params.bpc = u / 100; },
      fmt: fmtInt,
    }));
    box.appendChild(createSliderRow({
      id: "soft_clip", label: "高光柔剪 Soft Clip", min: 0, max: 99, step: 1, def: 0, bipolar: false,
      get: () => Math.round((params.soft_clip || 0) * 100),
      set: u => { params.soft_clip = u / 100; },
      fmt: fmtInt,
    }));
    box.appendChild(createSliderRow({
      id: "log_unmix", label: "C-41 染料去混 Log Unmix", min: 0, max: 100, step: 1, def: 0, bipolar: false, disabled: isBw,
      get: () => Math.round((params.log_unmix || 0) * 100),
      set: u => { params.log_unmix = u / 100; },
      fmt: fmtInt,
    }));
    box.appendChild(createSliderRow({
      id: "hd_curve", label: "H&D 胶片特性曲线", min: 0, max: 100, step: 1, def: 0, bipolar: false,
      get: () => Math.round((params.hd_curve || 0) * 100),
      set: u => { params.hd_curve = u / 100; },
      fmt: fmtInt,
    }));
  }

  // Roll softness slider (shown when roll_nat active)
  const sw = $("softnessWrap"); sw.innerHTML = "";
  const isRollNat = meta && meta.base && meta.base.method === "roll_nat";
  sw.style.display = isRollNat ? "block" : "none";
  if (isRollNat) {
    sw.appendChild(createSliderRow({
      id: "softness", label: "整卷黑位柔和度 (DQ ↔ B1)", min: 0, max: 100, step: 5, def: 50, bipolar: false,
      get: () => Math.round(softness * 100),
      set: u => { softness = u / 100; },
      onCommit: () => switchBaseMode("roll_nat"),
      fmt: fmtInt,
    }));
  }
}

/* ======================== 3-Way Color Wheel ======================== */
let wheelBgCanvas = null;
function ensureWheelBg(size) {
  if (wheelBgCanvas && wheelBgCanvas.width === size) return wheelBgCanvas;
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
  const r = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - r) / (r - 4), dy = (y - r) / (r - 4);
      const dist = Math.hypot(dx, dy);
      const idx = (y * size + x) * 4;
      if (dist > 1.0) { img.data[idx + 3] = 0; continue; }
      const hue = ((Math.atan2(dy, dx) * 180 / Math.PI) + 360) % 360;
      const sat = Math.min(1, dist);
      const [rr, gg, bb] = hsvToRgb255(hue, sat * 0.72, 0.24 + 0.56 * sat);
      const alpha = dist > 0.96 ? Math.max(0, (1 - dist) / 0.04) * 255 : 255;
      img.data[idx] = rr; img.data[idx + 1] = gg; img.data[idx + 2] = bb; img.data[idx + 3] = alpha;
    }
  }
  ctx.putImageData(img, 0, 0);
  wheelBgCanvas = c;
  return c;
}

function hsvToRgb255(h, s, v) {
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

function drawCgWheel() {
  const cv = $("cgWheelCanvas"), ctx = cv.getContext("2d");
  const W = cv.width, R = W / 2, radMax = R - 8;
  ctx.clearRect(0, 0, W, W);
  ctx.drawImage(ensureWheelBg(W), 0, 0);

  // Subtle concentric rings + crosshair
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(R, R, radMax * 0.5, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(R, R, radMax, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(R - 5, R); ctx.lineTo(R + 5, R); ctx.moveTo(R, R - 5); ctx.lineTo(R, R + 5); ctx.stroke();

  const w = grade[cgZone] || [0, 0, 0];
  const rad = (w[0] * Math.PI) / 180;
  const px = R + Math.cos(rad) * w[1] * radMax;
  const py = R + Math.sin(rad) * w[1] * radMax;

  if (w[1] > 0.01) {
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(R, R); ctx.lineTo(px, py); ctx.stroke();
  }
  const [cr, cg, cb] = hsvToRgb255(w[0], Math.max(0.25, w[1]), 0.92);
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fillStyle = w[1] > 0.01 ? `rgb(${cr},${cg},${cb})` : "#d4d4dc";
  ctx.fill();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.8; ctx.stroke();

  // Update zone lit dots
  for (const z of ["cg_shadows", "cg_midtones", "cg_highlights"]) {
    const zw = grade[z] || [0, 0, 0];
    const dot = $("dot_" + z);
    const active = zw[1] > 0.005 || Math.abs(zw[2]) > 0.005;
    dot.classList.toggle("lit", active);
    if (active) {
      const [dr, dg, db] = hsvToRgb255(zw[0], Math.max(0.5, zw[1]), 0.95);
      dot.style.background = zw[1] > 0.005 ? `rgb(${dr},${dg},${db})` : "var(--acc)";
    }
  }
}

function buildCgSliders() {
  const box = $("slCg"); box.innerHTML = "";
  const w = grade[cgZone] || (grade[cgZone] = [0, 0, 0]);
  box.appendChild(createSliderRow({
    id: "cg_s", label: "染色强度 Saturation", min: 0, max: 100, step: 1, def: 0, bipolar: false,
    get: () => Math.round((w[1] || 0) * 100),
    set: u => { grade[cgZone][1] = u / 100; drawCgWheel(); },
    fmt: fmtInt,
  }));
  box.appendChild(createSliderRow({
    id: "cg_l", label: "明度 Luminance", min: -100, max: 100, step: 1, def: 0,
    get: () => Math.round((w[2] || 0) * 100),
    set: u => { grade[cgZone][2] = u / 100; drawCgWheel(); },
    fmt: fmtSigned,
  }));
  box.appendChild(createSliderRow({
    id: "cg_balance", label: "影调平衡 Balance", min: -100, max: 100, step: 1, def: 0,
    get: () => Math.round((grade.cg_balance || 0) * 100),
    set: u => { grade.cg_balance = u / 100; },
    fmt: fmtSigned,
  }));
  box.appendChild(createSliderRow({
    id: "cg_blending", label: "过渡混合 Blending", min: 0, max: 100, step: 1, def: 50, bipolar: false,
    get: () => Math.round((grade.cg_blending != null ? grade.cg_blending : 0.5) * 100),
    set: u => { grade.cg_blending = u / 100; },
    fmt: fmtInt,
  }));
  drawCgWheel();
}

let cgDragging = false;
function updateWheelFromEvent(ev) {
  const cv = $("cgWheelCanvas"), r = cv.getBoundingClientRect();
  const cx = r.width / 2, cy = r.height / 2, radMax = cx - 8;
  const dx = ev.clientX - r.left - cx, dy = ev.clientY - r.top - cy;
  const dist = Math.min(1, Math.hypot(dx, dy) / radMax);
  const hue = ((Math.atan2(dy, dx) * 180 / Math.PI) + 360) % 360;
  grade[cgZone][0] = Number(hue.toFixed(1));
  grade[cgZone][1] = dist < 0.03 ? 0 : Number(dist.toFixed(3));
  buildCgSliders();
  scheduleRender(40);
}
$("cgWheelCanvas").onmousedown = ev => { cgDragging = true; updateWheelFromEvent(ev); };
window.addEventListener("mousemove", ev => { if (cgDragging) updateWheelFromEvent(ev); });
window.addEventListener("mouseup", () => { if (cgDragging) { cgDragging = false; persistCurrentFile(true); pushHistory(); } });
$("cgWheelCanvas").ondblclick = () => {
  grade[cgZone] = [0, 0, 0];
  buildCgSliders(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
};
$("cgZoneTabs").onclick = ev => {
  const btn = ev.target.closest("button[data-zone]");
  if (!btn) return;
  cgZone = btn.dataset.zone;
  $("cgZoneTabs").querySelectorAll("button").forEach(b => b.classList.toggle("on", b === btn));
  buildCgSliders();
};
$("bCgResetZone").onclick = () => {
  grade[cgZone] = [0, 0, 0]; buildCgSliders(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
};
$("bCgResetAll").onclick = () => {
  grade.cg_shadows = [0, 0, 0]; grade.cg_midtones = [0, 0, 0]; grade.cg_highlights = [0, 0, 0];
  grade.cg_balance = 0; grade.cg_blending = 0.5;
  buildCgSliders(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
};

/* ======================== 8-Band HSL Color Mixer ======================== */
function buildHslUI() {
  const dots = $("hslDots"); dots.innerHTML = "";
  for (const b of HSL_META) {
    const t = (grade.hsl && grade.hsl[b.id]) || [0, 0, 0];
    const edited = Math.abs(t[0]) > 1e-3 || Math.abs(t[1]) > 1e-3 || Math.abs(t[2]) > 1e-3;
    const btn = document.createElement("button");
    btn.className = "hsl-dot-btn" + (b.id === hslBand ? " on" : "") + (edited ? " edited" : "");
    btn.title = b.label + "色通道";
    btn.innerHTML = `<span class="hsl-swatch" style="background:${b.color}"></span><span class="hsl-mark"></span>`;
    btn.onclick = () => { hslBand = b.id; buildHslUI(); };
    dots.appendChild(btn);
  }
  const sl = $("slHsl"); sl.innerHTML = "";
  const isBw = filmSel === "bw";
  const bMeta = HSL_META.find(x => x.id === hslBand);
  const ensureBand = () => {
    if (!grade.hsl) grade.hsl = {};
    if (!grade.hsl[hslBand]) grade.hsl[hslBand] = [0, 0, 0];
    return grade.hsl[hslBand];
  };
  for (const [idx, key, lbl] of [[0, "h", "色相 Hue"], [1, "s", "饱和度 Saturation"], [2, "l", "明度 Luminance"]]) {
    sl.appendChild(createSliderRow({
      id: `hsl_${hslBand}_${key}`, label: `${bMeta.label} · ${lbl}`, min: -100, max: 100, step: 1, def: 0, disabled: isBw,
      get: () => Math.round(((grade.hsl && grade.hsl[hslBand] && grade.hsl[hslBand][idx]) || 0) * 100),
      set: u => {
        ensureBand()[idx] = u / 100;
        // Update edited dot without rebuilding sliders mid-drag
        const t = grade.hsl[hslBand];
        const ed = Math.abs(t[0]) > 1e-3 || Math.abs(t[1]) > 1e-3 || Math.abs(t[2]) > 1e-3;
        const btn = $("hslDots").children[HSL_META.findIndex(x => x.id === hslBand)];
        if (btn) btn.classList.toggle("edited", ed);
      },
      fmt: fmtSigned,
    }));
  }
}
$("bHslResetBand").onclick = () => {
  if (grade.hsl) delete grade.hsl[hslBand];
  buildHslUI(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
};
$("bHslResetAll").onclick = () => {
  grade.hsl = {}; buildHslUI(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
};

/* ======================== Interactive Tone Curve ======================== */
function evalCurveJS(pts, x) {
  const n = pts.length;
  if (n < 2) return x;
  if (x <= pts[0][0]) return pts[0][1];
  if (x >= pts[n - 1][0]) return pts[n - 1][1];
  if (n === 2) {
    const t = (x - pts[0][0]) / Math.max(pts[1][0] - pts[0][0], 1e-6);
    return pts[0][1] + t * (pts[1][1] - pts[0][1]);
  }
  const dx = [], dy = [], delta = [], m = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) {
    dx.push(pts[i + 1][0] - pts[i][0]);
    dy.push(pts[i + 1][1] - pts[i][1]);
    delta.push(dy[i] / Math.max(dx[i], 1e-6));
  }
  m[0] = delta[0]; m[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) m[i] = 0;
    else {
      const w1 = 2 * dx[i] + dx[i - 1], w2 = dx[i] + 2 * dx[i - 1];
      m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }
  let k = 0;
  while (k < n - 2 && x > pts[k + 1][0]) k++;
  const h = Math.max(dx[k], 1e-6), t = (x - pts[k][0]) / h, t2 = t * t, t3 = t2 * t;
  const y = (2 * t3 - 3 * t2 + 1) * pts[k][1] + (t3 - 2 * t2 + t) * h * m[k] + (-2 * t3 + 3 * t2) * pts[k + 1][1] + (t3 - t2) * h * m[k + 1];
  return Math.max(0, Math.min(1, y));
}

function drawCurveCanvas() {
  const cv = $("curveCanvas"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (lastHistBins) {
    let pk = 1;
    for (let i = 2; i < 254; i++) pk = Math.max(pk, lastHistBins[i]);
    ctx.beginPath(); ctx.moveTo(0, H);
    for (let i = 0; i < 256; i++) ctx.lineTo((i / 255) * W, H - Math.min(1, lastHistBins[i] / pk) * H * 0.82);
    ctx.lineTo(W, H); ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,0.055)"; ctx.fill();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.065)"; ctx.lineWidth = 1;
  for (let i = 1; i <= 3; i++) {
    const p = (i / 4) * W;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(W, p); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.13)";
  ctx.beginPath(); ctx.moveTo(0, H); ctx.lineTo(W, 0); ctx.stroke();

  const colors = { rgb: "#ececf1", r: "#ff6b6b", g: "#51cf66", b: "#4dabf7" };
  for (const ch of ["r", "g", "b", "rgb"]) {
    const pts = curves[ch];
    if (ch !== curveChan && pts.length === 2 && pts[0][1] === 0 && pts[1][1] === 1) continue;
    ctx.strokeStyle = colors[ch];
    ctx.lineWidth = ch === curveChan ? 2.1 : 1.1;
    ctx.globalAlpha = ch === curveChan ? 1.0 : 0.4;
    ctx.beginPath();
    for (let px = 0; px <= W; px += 2) {
      const py = (1 - evalCurveJS(pts, px / W)) * H;
      if (px === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1.0;
  const activePts = curves[curveChan];
  for (let i = 0; i < activePts.length; i++) {
    const cx = activePts[i][0] * W, cy = (1 - activePts[i][1]) * H;
    ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = i === dragCurveIdx ? "#dda14b" : colors[curveChan];
    ctx.fill(); ctx.strokeStyle = "#0e0e11"; ctx.lineWidth = 1.6; ctx.stroke();
  }
}

function curveCoords(ev) {
  const r = $("curveCanvas").getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)),
    Math.max(0, Math.min(1, 1 - (ev.clientY - r.top) / r.height)),
  ];
}
$("curveCanvas").onmousedown = ev => {
  const [x, y] = curveCoords(ev), pts = curves[curveChan];
  let best = -1, bestD = 0.065;
  for (let i = 0; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - x, pts[i][1] - y);
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best >= 0) dragCurveIdx = best;
  else {
    pts.push([x, y]); pts.sort((a, b) => a[0] - b[0]);
    dragCurveIdx = pts.findIndex(pt => pt[0] === x && pt[1] === y);
  }
  drawCurveCanvas(); scheduleRender(45);
};
window.addEventListener("mousemove", ev => {
  if (dragCurveIdx < 0) return;
  const [x, y] = curveCoords(ev), pts = curves[curveChan];
  if (dragCurveIdx === 0) { pts[0][0] = 0; pts[0][1] = y; }
  else if (dragCurveIdx === pts.length - 1) { pts[pts.length - 1][0] = 1; pts[pts.length - 1][1] = y; }
  else {
    pts[dragCurveIdx][0] = Math.max(pts[dragCurveIdx - 1][0] + 0.02, Math.min(pts[dragCurveIdx + 1][0] - 0.02, x));
    pts[dragCurveIdx][1] = y;
  }
  drawCurveCanvas(); scheduleRender(45);
});
window.addEventListener("mouseup", () => {
  if (dragCurveIdx >= 0) { dragCurveIdx = -1; persistCurrentFile(true); pushHistory(); }
});
$("curveCanvas").ondblclick = ev => {
  const [x, y] = curveCoords(ev), pts = curves[curveChan];
  for (let i = 1; i < pts.length - 1; i++) {
    if (Math.hypot(pts[i][0] - x, pts[i][1] - y) < 0.075) {
      pts.splice(i, 1); drawCurveCanvas(); persistCurrentFile(true); pushHistory(); scheduleRender(0); return;
    }
  }
};
$("curveChanSeg").onclick = ev => {
  const b = ev.target.closest("button[data-ch]");
  if (!b) return;
  curveChan = b.dataset.ch;
  $("curveChanSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  drawCurveCanvas();
};
document.querySelectorAll("button[data-cpreset]").forEach(btn => {
  btn.onclick = () => {
    const name = btn.dataset.cpreset;
    const cp = meta && meta.curve_presets && meta.curve_presets[name];
    if (!cp) return;
    curves = defaultCurves();
    for (const ch in cp) curves[ch] = cp[ch].map(pt => [pt[0], pt[1]]);
    drawCurveCanvas(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
  };
});
$("bCurveReset").onclick = () => {
  curves = defaultCurves(); drawCurveCanvas(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
};

/* ======================== Engine & Preset Builders ======================== */
function buildPresetsUI() {
  const grid = $("presetGrid"); grid.innerHTML = "";
  const list = (meta && meta.presets) || [
    { id: "frontier", label: "富士 Frontier" }, { id: "leica", label: "德味徕卡 Leica" },
    { id: "noritsu", label: "诺日士 Noritsu" }, { id: "standard", label: "标准 Standard" }, { id: "linear", label: "线性 Linear" }, { id: "natural", label: "自然 Natural" },
    { id: "portrait", label: "人像 Portrait" }, { id: "vivid", label: "鲜明 Vivid" },
    { id: "print_classic", label: "经典印相 Print" }, { id: "cinema", label: "电影质感 Cinema" },
  ];
  for (const p of list) {
    const b = document.createElement("button");
    b.className = "preset-pill" + (p.id === presetSel ? " on" : "");
    b.textContent = p.label.split(" ")[0];
    b.title = p.desc ? `${p.label} — ${p.desc}` : p.label;
    b.onclick = () => switchPreset(p.id);
    grid.appendChild(b);
  }
  const upBox = $("userPresetList"); upBox.innerHTML = "";
  for (const up of userPresets) {
    const chip = document.createElement("span");
    chip.className = "up-chip";
    chip.innerHTML = `<span class="up-name">${escapeHtml(up.name)}</span><span class="up-del" title="删除预设">×</span>`;
    chip.querySelector(".up-name").onclick = () => applyUserPreset(up);
    chip.querySelector(".up-del").onclick = async (e) => {
      e.stopPropagation();
      try {
        const res = await api("/api/user_presets/delete", { name: up.name });
        userPresets = res.presets || [];
        buildPresetsUI();
        toast("已删除预设：" + up.name);
      } catch (err) { toast(err.message, true); }
    };
    upBox.appendChild(chip);
  }
}

function buildEngineCards() {
  const box = $("engineCards"); box.innerHTML = "";
  const list = (meta && meta.engines) || [
    { id: "t24", label: "T24 AI 残差（推荐）", desc: "最新物理+残差树联合模型，精度最高" },
    { id: "t23", label: "T23 物理基准", desc: "纯物理参数估计 + T19 正交相纸曲线" },
    { id: "v3",  label: "V3 经典引擎", desc: "经典暗房规则引擎（兼容旧预设）" },
  ];
  for (const e of list) {
    const card = document.createElement("button");
    card.className = "eng-card" + (e.id === engineSel ? " on" : "");
    card.innerHTML = `<div class="eng-card-title">${e.label}</div><div class="eng-card-desc">${e.desc}</div>`;
    card.onclick = () => switchEngine(e.id);
    box.appendChild(card);
  }
  $("engineBadge").textContent = engineSel.toUpperCase();
}

function syncAllControls() {
  buildPresetsUI();
  buildEngineCards();
  buildBasicSliders();
  buildCgSliders();
  buildHslUI();
  drawCurveCanvas();
  buildDetailSliders();
  buildOpticsSliders();
  syncBaseSelect();
  updateDiagnostics();
  updateUndoButtons();
}

function syncBaseSelect() {
  if (!meta || !meta.base) return;
  const m = meta.base.method;
  const sel = $("baseSelect");
  if ([...sel.options].some(o => o.value === m)) sel.value = m;
  $("bPickBase").classList.toggle("on", activeTool === "pick_base" || m === "manual");
}

function updateDiagnostics() {
  if (!meta) { $("diagInfo").textContent = "未打开照片"; return; }
  const b = meta.base || {}, w = b.warnings || [];
  const fd = rollData[folderOf(meta.path)];
  const fr = fd && fd.frames ? fd.frames[meta.path] : null;
  const dc = b.chroma_part != null ? b.chroma_part : (fr ? fr.chroma_part : null);
  const dl = b.level_diff != null ? b.level_diff : (fr ? fr.level_diff : null);
  const iw = (meta.input_warnings || []).map(x => x.message);
  const rows = [
    ["文件", meta.path],
    ["全分辨率", `${meta.full_shape[1]} × ${meta.full_shape[0]}`],
    ["预览缩放", `${meta.preview_shape[1]} × ${meta.preview_shape[0]} (1/${meta.factor})`],
    ["当前引擎", `${(meta.engine || engineSel).toUpperCase()}${meta.auto && meta.auto.branch ? " · " + meta.auto.branch : ""}`],
    ["片基策略", BASE_LABELS[b.method] || b.method],
  ];
  if (dc != null && dl != null) rows.push(["片基偏差", `ΔC ${dc.toFixed(3)} D · Δℓ ${dl >= 0 ? "+" : ""}${dl.toFixed(3)} D`]);
  if (b.d_lo_shared != null) rows.push(["锚定黑点", `d_lo ${b.d_lo_shared.toFixed(3)} D`]);
  let html = rows.map(([k, v]) => `<div class="diag-kv"><span class="diag-k">${k}</span><span class="diag-v">${escapeHtml(v)}</span></div>`).join("");
  if (w.length) html += `<div class="warn" style="margin-top:6px">片基提示：${escapeHtml(w.join("；"))}</div>`;
  if (iw.length) html += `<div class="warn" style="margin-top:4px">输入告警：${escapeHtml(iw.join("；"))}</div>`;
  $("diagInfo").innerHTML = html;
  $("dimText").textContent = `${meta.full_shape[1]}×${meta.full_shape[0]}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
}

/* ======================== File Browser & Roll Scan ======================== */
async function loadFiles() {
  const [fj, pj] = await Promise.all([api("/api/files"), api("/api/user_presets").catch(() => ({ presets: [] }))]);
  fileList = fj.files || [];
  userPresets = pj.presets || [];
  if (fj.default_engine) engineSel = fj.default_engine;
  $("engineBadge").textContent = engineSel.toUpperCase();
  buildEngineCards();
  buildPresetsUI();
  renderFileList();
  const folders = new Set(fileList.map(folderOf));
  if (folders.size === 1 && fileList.length >= 2 && fileList.length <= 60) {
    await scanRoll([...folders][0]);
  } else {
    hydrateThumbnails();
  }
  if (fileList.length) openFile(fileList[0]);
}

function renderFileList() {
  const list = $("fileList");
  list.className = browserGrid ? "grid-view" : "";
  const q = ($("fileSearch").value || "").trim().toLowerCase();
  list.innerHTML = "";
  let visible = 0;
  fileList.forEach((rel, idx) => {
    const fd = rollData[folderOf(rel)];
    const fr = fd && fd.frames ? fd.frames[rel] : null;
    const isEd = !!(savedPerFile[rel] && savedPerFile[rel].edited);
    const isWarn = !!(fr && (fr.flag || fr.own_error));
    if (q && !rel.toLowerCase().includes(q)) return;
    if (browserFilter === "edited" && !isEd) return;
    if (browserFilter === "warn" && !isWarn) return;
    visible++;
    const shortName = rel.split("/").pop();
    const card = document.createElement("div");
    card.className = "fcard" + (rel === currentRel ? " on" : "");
    card.dataset.rel = rel;
    card.title = rel + (fr && fr.chroma_part != null ? `\nΔC = ${fr.chroma_part.toFixed(3)} D` : "");
    const dots = `${isWarn ? '<span class="fdot warn" title="片基偏离较大，将自动启用整卷自然片基"></span>' : ""}${isEd ? '<span class="fdot edited" title="已手动调整"></span>' : ""}`;
    card.innerHTML = `
      <div class="fthumb-wrap">
        <img class="fthumb" data-rel="${escapeHtml(rel)}" loading="lazy" alt="">
        <span class="fidx">${String(idx + 1).padStart(2, "0")}</span>
      </div>
      <div class="fmeta">
        <div class="fn">${escapeHtml(shortName)}</div>
        <div class="fsub"><span>${isEd ? "已调" : isWarn ? "已校正" : "自动"}</span><span style="display:flex;gap:4px">${dots}</span></div>
      </div>
    `;
    card.onclick = () => openFile(rel);
    list.appendChild(card);
  });
  $("fileCount").textContent = `${visible} / ${fileList.length} 帧`;
  hydrateThumbnails();
}

function hydrateThumbnails() {
  document.querySelectorAll("#fileList img.fthumb").forEach(img => {
    const rel = img.dataset.rel;
    const target = `/api/thumb?path=${encodeURIComponent(rel)}&film=${filmSel}`;
    if (img.dataset.loaded !== target) {
      img.src = target;
      img.dataset.loaded = target;
    }
  });
}

function updateFileListUI() {
  document.querySelectorAll("#fileList .fcard").forEach(card => {
    const rel = card.dataset.rel;
    card.classList.toggle("on", rel === currentRel);
    const isEd = !!(savedPerFile[rel] && savedPerFile[rel].edited);
    const fd = rollData[folderOf(rel)];
    const fr = fd && fd.frames ? fd.frames[rel] : null;
    const isWarn = !!(fr && (fr.flag || fr.own_error));
    const sub = card.querySelector(".fsub");
    if (sub) {
      const dots = `${isWarn ? '<span class="fdot warn"></span>' : ""}${isEd ? '<span class="fdot edited"></span>' : ""}`;
      sub.innerHTML = `<span>${isEd ? "已调" : isWarn ? "已校正" : "自动"}</span><span style="display:flex;gap:4px">${dots}</span>`;
    }
  });
}

async function scanRoll(folder) {
  folder = folder || (currentRel ? folderOf(currentRel) : ".");
  $("bScanRoll").disabled = true;
  $("rollStatus").textContent = "正在分析整卷片基…";
  try {
    const j = await api("/api/roll?folder=" + encodeURIComponent(folder));
    rollData[folder] = j;
    if (j.mixed_rolls) {
      const ns = Object.keys(j.sub_rolls || {}).length;
      $("rollStatus").textContent = `多卷混合 · ${ns} 个子卷 (${j.n_files} 帧)`;
    } else if (j.roll) {
      const nWarn = Object.values(j.frames).filter(x => x.flag).length;
      $("rollStatus").textContent = `整卷就绪 · ${j.n_files} 帧` + (nWarn ? ` (${nWarn} 帧自动整卷)` : "");
    } else {
      $("rollStatus").textContent = "单帧模式";
    }
    renderFileList();
    if (meta) updateDiagnostics();
  } catch (e) {
    $("rollStatus").textContent = "整卷扫描失败";
  } finally {
    $("bScanRoll").disabled = false;
  }
}

/* ======================== Open Photo / Switch Engine / Preset / Base ======================== */
let presetStartParams = {};
let presetStartGrade = defaultGrade();
let openSeq = 0;
let openAbort = null;
let lastBb = null, lastCh = null;
let lastLoupeUrl = null;

function hydrateFromAutoMeta(m) {
  const eng = m.engine || engineSel;
  engineSel = eng;
  if (m.preset) presetSel = m.preset;
  const defP = isT19Engine(eng) ? defaultT24Params(eng === "t24") : defaultV3Params();
  const baseRaw = (m.auto && (m.auto.base_params || m.auto.params)) || {};
  const baseP = Object.assign({}, defP, baseRaw);
  baseP.cam_mix = baseP.matrix ? camMixOf(baseP.matrix) : (isT19Engine(eng) ? 0.35 : 0.40);
  autoStartParams = deepClone(baseP);
  autoStartGrade = defaultGrade();

  const effRaw = (m.auto && m.auto.params) || {};
  const effP = Object.assign({}, defP, effRaw);
  effP.cam_mix = effP.matrix ? camMixOf(effP.matrix) : (isT19Engine(eng) ? 0.35 : 0.40);
  presetStartParams = deepClone(effP);
  presetStartGrade = Object.assign(defaultGrade(), deepClone(m.auto_grade || {}));
}

async function openFile(rel, explicitBaseMode) {
  if (currentRel && meta) persistCurrentFile(false);
  const myOpen = ++openSeq;
  if (openAbort) openAbort.abort();
  openAbort = new AbortController();
  currentRel = rel;
  activeTool = null;
  syncToolButtons();
  updateFileListUI();
  $("frameTitle").textContent = rel.split("/").pop();
  setStatus(`正在解码与校色 ${rel.split("/").pop()}…`);
  setLoading(true);
  busy = true;
  let newMeta = null;
  try {
    const prev = savedPerFile[rel];
    const reqEng = prev ? prev.engine : engineSel;
    const reqPreset = prev ? prev.preset : presetSel;
    const reqFilm = prev ? prev.film : filmSel;
    const body = { path: rel, film: reqFilm, softness, preset: reqPreset, engine: reqEng };
    if (explicitBaseMode) body.base_mode = explicitBaseMode;
    newMeta = await api("/api/open", body, openAbort.signal);
  } catch (e) {
    if (e.name === "AbortError" || myOpen !== openSeq) return;
    meta = null; busy = false; setLoading(false);
    setStatus("打开失败：" + e.message, true);
    toast("打开失败：" + e.message, true);
    return;
  }
  if (myOpen !== openSeq) return;
  meta = newMeta;
  busy = false;
  filmSel = meta.film || filmSel;
  $("filmSeg").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.film === filmSel));
  softness = meta.base && meta.base.softness != null ? meta.base.softness : 0.5;
  hydrateFromAutoMeta(meta);

  const saved = savedPerFile[rel];
  if (saved && saved.rawParams) {
    params = deepClone(saved.rawParams);
    grade = Object.assign(defaultGrade(), deepClone(saved.rawGrade || {}));
    geom = Object.assign(defaultGeom(), deepClone(saved.rawGeom || { rot90: meta.rot90 || 0 }));
    curves = deepClone(saved.curves || defaultCurves());
  } else {
    params = deepClone(presetStartParams);
    grade = deepClone(presetStartGrade);
    geom = Object.assign(defaultGeom(), { rot90: meta.rot90 || 0 });
    loadCurvesFromDict(isT19Engine() ? grade.tone_curve : params.tone_curve);
  }

  // Initialize history if first open
  if (!historyPerFile[rel]) {
    historyPerFile[rel] = { undo: [currentSnapshot()], redo: [] };
  }

  $("emptyState").style.display = "none";
  $("viewport").style.display = viewMode === "compare" && compareStyle === "side" ? "none" : "block";
  resetZoom(false);
  syncAllControls();
  await Promise.all([renderMain(), viewMode === "compare" ? renderBefore() : Promise.resolve()]);
  setLoading(false);

  const engTag = (meta.engine || engineSel).toUpperCase();
  setStatus(`${meta.path} · ${engTag} + ${BASE_LABELS[meta.base.method] || meta.base.method}`);
  if (meta.auto && meta.auto.status === "manual_review_required") {
    toast("该画格密度跨度异常，已保持保守自动参数", true);
  }
}

async function switchEngine(newEng) {
  if (newEng === engineSel && meta) return;
  engineSel = newEng;
  $("engineBadge").textContent = engineSel.toUpperCase();
  buildEngineCards();
  if (!meta) return;
  setLoading(true);
  try {
    meta = await api("/api/engine", { token: meta.token, engine: newEng, preset: presetSel });
    hydrateFromAutoMeta(meta);
    params = deepClone(presetStartParams);
    grade = deepClone(presetStartGrade);
    loadCurvesFromDict(isT19Engine() ? grade.tone_curve : params.tone_curve);
    syncAllControls();
    persistCurrentFile(false);
    pushHistory();
    await Promise.all([renderMain(), viewMode === "compare" ? renderBefore() : Promise.resolve()]);
    toast(`已切换至 ${newEng.toUpperCase()} 引擎`);
  } catch (e) {
    toast("切换引擎失败：" + e.message, true);
  } finally {
    setLoading(false);
  }
}

async function switchPreset(id) {
  presetSel = id;
  buildPresetsUI();
  if (!meta) return;
  setLoading(true);
  try {
    meta = await api("/api/preset", { token: meta.token, preset: id });
    hydrateFromAutoMeta(meta);
    params = deepClone(presetStartParams);
    grade = deepClone(presetStartGrade);
    loadCurvesFromDict(isT19Engine() ? grade.tone_curve : params.tone_curve);
    syncAllControls();
    persistCurrentFile(true);
    pushHistory();
    await Promise.all([renderMain(), viewMode === "compare" ? renderBefore() : Promise.resolve()]);
    const pObj = (meta && meta.presets || []).find(x => x.id === id);
    toast("已应用风格：" + (pObj ? pObj.label.split(" ")[0] : id));
  } catch (e) {
    toast("风格切换失败：" + e.message, true);
  } finally {
    setLoading(false);
  }
}

async function switchBaseMode(bm) {
  if (!meta) return;
  setLoading(true);
  try {
    meta = await api("/api/base", { mode: bm, softness, token: meta.token });
    hydrateFromAutoMeta(meta);
    params = deepClone(presetStartParams);
    grade = deepClone(presetStartGrade);
    loadCurvesFromDict(isT19Engine() ? grade.tone_curve : params.tone_curve);
    syncAllControls();
    persistCurrentFile(true);
    pushHistory();
    await Promise.all([renderMain(), viewMode === "compare" ? renderBefore() : Promise.resolve()]);
    toast("片基已切换：" + (BASE_LABELS[bm] || bm));
  } catch (e) {
    toast("片基切换失败：" + e.message, true);
  } finally {
    setLoading(false);
  }
}

function applyUserPreset(up) {
  if (!meta) return;
  if (up.grade) {
    grade = Object.assign(defaultGrade(), deepClone(up.grade));
    if (isT19Engine()) loadCurvesFromDict(grade.tone_curve);
  }
  if (up.params) {
    // Apply tonal/color look parameters while keeping photo's camera matrix
    const keepMatrix = params.cam_mix;
    for (const [k, v] of Object.entries(up.params)) {
      if (k === "matrix" || k === "eps_rgb" || k === "residual_matrix") continue;
      if (k === "tone_curve" && !isT19Engine()) loadCurvesFromDict(v);
      else if (k in params) params[k] = v;
    }
    params.cam_mix = keepMatrix;
  }
  syncAllControls();
  persistCurrentFile(true);
  pushHistory();
  scheduleRender(0);
  toast("已套用自定义预设：" + up.name);
}

/* ======================== Rendering & Histogram & Clipping ======================== */
function scheduleRender(delay = 45) {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => { renderMain(); }, delay);
}

function buildRenderQuery(overrideP, overrideG, cropApply = true) {
  const p = serializeParams(overrideP);
  const g = serializeGrade(overrideG);
  const gm = serializeGeom();
  let qs = `token=${meta.token}&fmt=jpeg&p=${encodeURIComponent(JSON.stringify(p))}`;
  if (Object.keys(g).length) qs += `&g=${encodeURIComponent(JSON.stringify(g))}`;
  if (Object.keys(gm).length) qs += `&geom=${encodeURIComponent(JSON.stringify(gm))}`;
  if (!cropApply) qs += "&crop_apply=0";
  return qs;
}

async function renderMain() {
  if (!meta) return;
  const my = ++renderSeq;
  if (renderAbort) renderAbort.abort();
  renderAbort = new AbortController();
  const isCropEdit = activeTool === "crop" || activeTool === "pick_base";
  const showNeg = viewMode === "neg";
  const useAuto = spaceComparing && !showNeg;
  const url = showNeg
    ? `/api/negative?token=${meta.token}`
    : `/api/render?${buildRenderQuery(useAuto ? autoStartParams : null, useAuto ? autoStartGrade : null, !isCropEdit)}`;

  try {
    const r = await api(url, null, renderAbort.signal);
    if (my !== renderSeq) return;
    const bb = !showNeg ? parseFloat(r.headers.get("X-Below-Black")) : null;
    const ch = !showNeg ? parseFloat(r.headers.get("X-Clip-High")) : null;
    lastBb = bb; lastCh = ch;
    const blob = await r.blob();
    if (my !== renderSeq) return;
    const objUrl = URL.createObjectURL(blob);
    const img = $("imgMain");
    img.onload = () => {
      if (lastRenderedUrl && lastRenderedUrl !== objUrl) URL.revokeObjectURL(lastRenderedUrl);
      lastRenderedUrl = objUrl;
      $("sbsAfter").src = objUrl;
      updateHistogramAndClip(img, bb, ch);
      drawOverlay();
    };
    img.src = objUrl;
  } catch (e) {
    if (e.name !== "AbortError" && my === renderSeq) setStatus("渲染失败：" + e.message, true);
  }
}

async function renderBefore() {
  if (!meta) return;
  const my = ++beforeSeq;
  if (beforeAbort) beforeAbort.abort();
  beforeAbort = new AbortController();
  const url = compareSource === "neg"
    ? `/api/negative?token=${meta.token}`
    : `/api/render?${buildRenderQuery(autoStartParams, autoStartGrade, activeTool !== "crop")}`;
  try {
    const r = await api(url, null, beforeAbort.signal);
    if (my !== beforeSeq) return;
    const blob = await r.blob();
    if (my !== beforeSeq) return;
    const objUrl = URL.createObjectURL(blob);
    const old = lastBeforeUrl;
    lastBeforeUrl = objUrl;
    $("imgBefore").src = objUrl;
    $("sbsBefore").src = objUrl;
    if (old && old !== objUrl) setTimeout(() => URL.revokeObjectURL(old), 300);
  } catch (e) {
    if (e.name !== "AbortError") console.warn(e);
  }
}

const _histOff = document.createElement("canvas");
const _histOffCtx = _histOff.getContext("2d", { willReadFrequently: true });

function updateHistogramAndClip(imgEl, bb, ch) {
  const hc = $("histCanvas"), ctx = hc.getContext("2d");
  const W = hc.width, H = hc.height;
  ctx.clearRect(0, 0, W, H);
  if (!imgEl || !imgEl.naturalWidth) return;

  const sw = Math.min(360, imgEl.naturalWidth), sh = Math.min(260, imgEl.naturalHeight);
  if (_histOff.width !== sw) _histOff.width = sw;
  if (_histOff.height !== sh) _histOff.height = sh;
  const octx = _histOffCtx;
  octx.drawImage(imgEl, 0, 0, sw, sh);
  const data = octx.getImageData(0, 0, sw, sh).data;

  const rB = new Uint32Array(256), gB = new Uint32Array(256), bB = new Uint32Array(256), yB = new Uint32Array(256);
  let ySum = 0, nPx = sw * sh, nBlk = 0, nWht = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const y = (r * 54 + g * 183 + b * 19) >> 8;
    rB[r]++; gB[g]++; bB[b]++; yB[y]++;
    ySum += y;
    if (r <= 1 && g <= 1 && b <= 1) nBlk++;
    if (r >= 254 || g >= 254 || b >= 254) nWht++;
  }
  lastHistBins = yB;
  if ($("tab-color").classList.contains("on")) drawCurveCanvas();

  let peak = 1;
  for (let i = 2; i < 254; i++) peak = Math.max(peak, rB[i], gB[i], bB[i], yB[i]);
  const drawCh = (bins, fill, stroke) => {
    ctx.beginPath(); ctx.moveTo(0, H);
    for (let i = 0; i < 256; i++) {
      const sm = (bins[Math.max(0, i - 1)] + 2 * bins[i] + bins[Math.min(255, i + 1)]) * 0.25;
      ctx.lineTo((i / 255) * W, H - Math.min(1, sm / peak) * (H - 5));
    }
    ctx.lineTo(W, H); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = stroke; ctx.lineWidth = 1.1; ctx.stroke();
  };
  ctx.globalCompositeOperation = "screen";
  drawCh(rB, "rgba(235, 75, 75, 0.34)", "rgba(255, 96, 96, 0.85)");
  drawCh(gB, "rgba(75, 215, 105, 0.34)", "rgba(95, 235, 125, 0.85)");
  drawCh(bB, "rgba(75, 145, 245, 0.34)", "rgba(105, 170, 255, 0.85)");
  ctx.globalCompositeOperation = "source-over";
  drawCh(yB, "rgba(225, 225, 235, 0.10)", "rgba(230, 230, 240, 0.50)");

  const blkShare = bb != null && !Number.isNaN(bb) ? bb : nBlk / Math.max(nPx, 1);
  const whtShare = ch != null && !Number.isNaN(ch) ? ch : nWht / Math.max(nPx, 1);
  $("lumaMed").textContent = `均亮 ${(ySum / Math.max(nPx, 1) / 255 * 100).toFixed(0)}%`;
  $("clipBlack").innerHTML = `黑位 <span class="${blkShare > 0.05 ? "warn" : ""}">${(blkShare * 100).toFixed(2)}%</span>`;
  $("clipWhite").innerHTML = `高光 <span class="${whtShare > 0.005 ? "warn" : ""}">${(whtShare * 100).toFixed(2)}%</span>`;
  $("triBlack").classList.toggle("lit-blue", blkShare > 0.01);
  $("triWhite").classList.toggle("lit-red", whtShare > 0.003);

  // Client-side Clipping Warning Overlay
  const cc = $("clipCanvas");
  cc.style.display = showClip && viewMode !== "neg" ? "block" : "none";
  if (showClip && viewMode !== "neg") {
    cc.width = sw; cc.height = sh;
    const cctx = cc.getContext("2d");
    const out = cctx.createImageData(sw, sh);
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      if (r >= 254 || g >= 254 || b >= 254) {
        out.data[i] = 245; out.data[i + 1] = 56; out.data[i + 2] = 56; out.data[i + 3] = 200;
      } else if (r <= 1 && g <= 1 && b <= 1) {
        out.data[i] = 48; out.data[i + 1] = 132; out.data[i + 2] = 255; out.data[i + 3] = 200;
      }
    }
    cctx.putImageData(out, 0, 0);
  }
}

/* ======================== Zoom, Pan & View / Compare Modes ======================== */
function applyZoomTransform() {
  $("viewport").style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  $("zoomVal").textContent = Math.abs(zoom - 1.0) < 0.02 ? "适应" : Math.round(zoom * 100) + "%";
  updateOverlayCursor();
}
function resetZoom(announce = true) {
  zoom = 1.0; panX = 0; panY = 0;
  applyZoomTransform();
  if (announce) setStatus("画布视图：适应窗口");
}
function setZoom(newZ) {
  zoom = Math.max(0.25, Math.min(4.0, newZ));
  if (zoom <= 1.02) { panX = 0; panY = 0; }
  applyZoomTransform();
}

$("stage").addEventListener("wheel", ev => {
  if (!meta || (viewMode === "compare" && compareStyle === "side")) return;
  ev.preventDefault();
  const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
  setZoom(zoom * factor);
}, { passive: false });

$("bZoomIn").onclick = () => setZoom(zoom * 1.25);
$("bZoomOut").onclick = () => setZoom(zoom / 1.25);
$("zoomVal").onclick = () => { if (Math.abs(zoom - 1.0) < 0.05) setZoom(2.0); else resetZoom(false); };

function updateCompareUI() {
  const isCmp = viewMode === "compare";
  const isSide = isCmp && compareStyle === "side";
  const isWipe = isCmp && compareStyle === "wipe";
  $("cmpHudGroup").style.display = isCmp ? "inline-flex" : "none";
  $("bCmpWipe").classList.toggle("on", isWipe);
  $("bCmpSide").classList.toggle("on", isSide);
  $("bCmpSrc").textContent = compareSource === "auto" ? "基准:自动" : "基准:负片";
  const lbl = compareSource === "auto" ? "自动起点" : "原始负片";
  $("cmpTagL").textContent = lbl;
  $("sbsLabelL").textContent = lbl;

  if (!meta) return;
  $("viewport").style.display = isSide ? "none" : "block";
  $("sideBySide").style.display = isSide ? "flex" : "none";
  $("imgBeforeWrap").style.display = isWipe ? "block" : "none";
  $("wipeHandle").style.display = isWipe ? "block" : "none";
  $("cmpTagL").style.display = isWipe ? "block" : "none";
  $("cmpTagR").style.display = isWipe ? "block" : "none";
  if (isWipe) applyWipePosition();
}

function applyWipePosition() {
  const pct = Math.max(2, Math.min(98, wipePos * 100));
  $("imgBeforeWrap").style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  $("wipeHandle").style.left = pct + "%";
}

let wipeDragging = false;
$("wipeHandle").onmousedown = ev => { ev.stopPropagation(); wipeDragging = true; };
window.addEventListener("mousemove", ev => {
  if (!wipeDragging) return;
  const r = $("imgWrap").getBoundingClientRect();
  wipePos = Math.max(0.02, Math.min(0.98, (ev.clientX - r.left) / r.width));
  applyWipePosition();
});
window.addEventListener("mouseup", () => { wipeDragging = false; });

$("viewSeg").onclick = ev => {
  const b = ev.target.closest("button[data-view]");
  if (!b) return;
  viewMode = b.dataset.view;
  $("viewSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  updateCompareUI();
  if (meta) {
    renderMain();
    if (viewMode === "compare") renderBefore();
  }
};
$("bCmpWipe").onclick = () => { compareStyle = "wipe"; updateCompareUI(); };
$("bCmpSide").onclick = () => { compareStyle = "side"; updateCompareUI(); renderBefore(); };
$("bCmpSrc").onclick = () => {
  compareSource = compareSource === "auto" ? "neg" : "auto";
  updateCompareUI();
  if (meta) renderBefore();
};

/* ======================== Canvas Tools: WB, Crop, Loupe, Pick Base ======================== */
function updateOverlayCursor() {
  const ov = $("overlayCanvas");
  ov.className = "";
  if (activeTool === "wb" || activeTool === "pick_base" || activeTool === "loupe") ov.classList.add("tool-crosshair");
  else if (!activeTool && zoom > 1.02) ov.classList.add(panning ? "tool-grabbing" : "tool-grab");
}

function syncToolButtons() {
  $("bToolWb").classList.toggle("on", activeTool === "wb");
  $("bWbPickInline").classList.toggle("on", activeTool === "wb");
  $("bToolCrop").classList.toggle("on", activeTool === "crop");
  $("bEnterCrop").classList.toggle("on", activeTool === "crop");
  $("bToolLoupe").classList.toggle("on", activeTool === "loupe");
  $("bToolClip").classList.toggle("on", showClip);
  $("cropBar").style.display = activeTool === "crop" ? "flex" : "none";
  $("canvasHud").style.display = activeTool === "crop" ? "none" : "flex";
  syncBaseSelect();
  updateOverlayCursor();
}

function setTool(t) {
  const wasCrop = activeTool === "crop";
  activeTool = activeTool === t ? null : t;
  if (activeTool === "crop" && !geom.crop) {
    geom.crop = [0.04, 0.04, 0.96, 0.96];
  }
  syncToolButtons();
  if (activeTool === "crop" || wasCrop) {
    if (wasCrop && activeTool !== "crop") { persistCurrentFile(true); pushHistory(); }
    resetZoom(false);
    renderMain();
  } else {
    drawOverlay();
  }
  if (activeTool === "wb") setStatus("白平衡吸管：请点击画面中的中性灰或白色区域");
  else if (activeTool === "crop") setStatus("裁切模式：拖动边缘或角落调整画幅，按 Enter 完成，Esc 取消");
  else if (activeTool === "pick_base") setStatus("片基框选：在画面未曝光片边上拖拽框选区域");
  else if (activeTool === "loupe") setStatus("1:1 检视：点击画面任意位置查看全分辨率传感器原图");
  else if (meta) setStatus(`${meta.path} · ${(meta.engine || engineSel).toUpperCase()} + ${BASE_LABELS[meta.base.method] || meta.base.method}`);
}

function autoBorderCrop() {
  if (!meta) return;
  if (!meta.film_box) {
    toast("当前画面未检测到明显片夹黑边，已使用 3% 轻度裁边");
    geom.crop = [0.03, 0.03, 0.97, 0.97];
  } else {
    // Convert sensor-oriented film_box [t, l, h, w] -> display normalized [t, l, b, r]
    const [ft, fl, fh, fw] = meta.film_box;
    const [H, W] = meta.preview_shape;
    const pts = [[fl / W, ft / H], [(fl + fw) / W, (ft + fh) / H]].map(([x, y]) => {
      let [u, v] = [x, y];
      const k = ((geom.rot90 || 0) % 4 + 4) % 4;
      if (k === 1) [u, v] = [v, 1 - u];
      else if (k === 2) [u, v] = [1 - u, 1 - v];
      else if (k === 3) [u, v] = [1 - v, u];
      if (geom.flip_h) u = 1 - u;
      if (geom.flip_v) v = 1 - v;
      return [u, v];
    });
    const pad = 0.008;
    const l = Math.max(0, Math.min(pts[0][0], pts[1][0]) + pad);
    const r = Math.min(1, Math.max(pts[0][0], pts[1][0]) - pad);
    const t = Math.max(0, Math.min(pts[0][1], pts[1][1]) + pad);
    const b = Math.min(1, Math.max(pts[0][1], pts[1][1]) - pad);
    geom.crop = [t, l, b, r];
    toast("已按片夹边界自动去片边");
  }
  persistCurrentFile(true); pushHistory();
  drawOverlay();
  if (activeTool !== "crop") scheduleRender(0);
}

function applyCropAspectRatio(ratioStr) {
  cropRatio = ratioStr;
  document.querySelectorAll("#cropBar .ratio-btn").forEach(b => b.classList.toggle("on", b.dataset.ratio === ratioStr));
  if (ratioStr === "free" || !meta) return;
  const img = $("imgMain");
  const iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
  let targetR = ratioStr === "orig" ? iw / ih : parseFloat(ratioStr);
  if (cropPortrait && targetR !== 1) targetR = 1 / targetR;
  const c = geom.crop || [0.05, 0.05, 0.95, 0.95];
  const cy = (c[0] + c[2]) / 2, cx = (c[1] + c[3]) / 2;
  let wPx = (c[3] - c[1]) * iw, hPx = wPx / targetR;
  if (hPx > ih * 0.96) { hPx = ih * 0.96; wPx = hPx * targetR; }
  if (wPx > iw * 0.96) { wPx = iw * 0.96; hPx = wPx / targetR; }
  const nw = wPx / iw, nh = hPx / ih;
  const l = Math.max(0, Math.min(1 - nw, cx - nw / 2));
  const t = Math.max(0, Math.min(1 - nh, cy - nh / 2));
  geom.crop = [t, l, t + nh, l + nw];
  drawOverlay();
}

function drawOverlay() {
  const cv = $("overlayCanvas"), im = $("imgMain");
  const W = im.clientWidth, H = im.clientHeight;
  if (!W || !H) return;
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, W, H);
  if (!meta) return;

  // Draw manual base pick drag box
  if (baseDrag) {
    g.strokeStyle = "#dda14b"; g.lineWidth = 2;
    g.fillStyle = "rgba(221, 161, 75, 0.18)";
    const x = Math.min(baseDrag.x0, baseDrag.x1), y = Math.min(baseDrag.y0, baseDrag.y1);
    const w = Math.abs(baseDrag.x1 - baseDrag.x0), h = Math.abs(baseDrag.y1 - baseDrag.y0);
    g.fillRect(x, y, w, h); g.strokeRect(x, y, w, h);
  }

  // Draw Crop overlay
  if (activeTool === "crop") {
    const c = geom.crop || [0, 0, 1, 1];
    const x0 = c[1] * W, y0 = c[0] * H, x1 = c[3] * W, y1 = c[2] * H;
    const cw = x1 - x0, ch = y1 - y0;
    // Darkened mask outside crop
    g.fillStyle = "rgba(0, 0, 0, 0.62)";
    g.fillRect(0, 0, W, y0);
    g.fillRect(0, y1, W, H - y1);
    g.fillRect(0, y0, x0, ch);
    g.fillRect(x1, y0, W - x1, ch);

    // Rule of thirds
    g.strokeStyle = "rgba(255,255,255,0.25)"; g.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      g.beginPath(); g.moveTo(x0 + cw * i / 3, y0); g.lineTo(x0 + cw * i / 3, y1); g.stroke();
      g.beginPath(); g.moveTo(x0, y0 + ch * i / 3); g.lineTo(x1, y0 + ch * i / 3); g.stroke();
    }

    // Crisp outer border + corner L-handles
    g.strokeStyle = "#fff"; g.lineWidth = 1.5;
    g.strokeRect(x0, y0, cw, ch);
    const L = Math.min(18, cw * 0.25, ch * 0.25);
    g.strokeStyle = "#dda14b"; g.lineWidth = 3;
    const corners = [
      [x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1],
    ];
    for (const [cx, cy, sx, sy] of corners) {
      g.beginPath(); g.moveTo(cx + sx * L, cy); g.lineTo(cx, cy); g.lineTo(cx, cy + sy * L); g.stroke();
    }
  }
}

function cropHitTest(nx, ny) {
  const c = geom.crop || [0, 0, 1, 1];
  const [t, l, b, r] = c, thr = 0.035;
  const nearL = Math.abs(nx - l) < thr, nearR = Math.abs(nx - r) < thr;
  const nearT = Math.abs(ny - t) < thr, nearB = Math.abs(ny - b) < thr;
  if (nearT && nearL) return "tl";
  if (nearT && nearR) return "tr";
  if (nearB && nearL) return "bl";
  if (nearB && nearR) return "br";
  if (nearT && nx > l && nx < r) return "t";
  if (nearB && nx > l && nx < r) return "b";
  if (nearL && ny > t && ny < b) return "l";
  if (nearR && ny > t && ny < b) return "r";
  if (nx > l && nx < r && ny > t && ny < b) return "move";
  return null;
}

$("overlayCanvas").onmousedown = ev => {
  if (!meta) return;
  const r = $("overlayCanvas").getBoundingClientRect();
  const lx = ev.clientX - r.left, ly = ev.clientY - r.top;
  const nx = lx / r.width, ny = ly / r.height;

  if (activeTool === "crop") {
    const mode = cropHitTest(nx, ny);
    if (mode) cropDrag = { mode, sx: nx, sy: ny, orig: [...(geom.crop || [0, 0, 1, 1])] };
    return;
  }
  if (activeTool === "pick_base") {
    baseDrag = { x0: lx * ($("overlayCanvas").width / r.width), y0: ly * ($("overlayCanvas").height / r.height),
                 x1: lx * ($("overlayCanvas").width / r.width), y1: ly * ($("overlayCanvas").height / r.height) };
    return;
  }
  if (!activeTool && zoom > 1.02) {
    panning = { sx: ev.clientX, sy: ev.clientY, px: panX, py: panY };
    updateOverlayCursor();
  }
};

window.addEventListener("mousemove", ev => {
  if (panning) {
    panX = panning.px + (ev.clientX - panning.sx);
    panY = panning.py + (ev.clientY - panning.sy);
    applyZoomTransform();
    return;
  }
  const ov = $("overlayCanvas"), r = ov.getBoundingClientRect();
  if (baseDrag) {
    baseDrag.x1 = (ev.clientX - r.left) * (ov.width / r.width);
    baseDrag.y1 = (ev.clientY - r.top) * (ov.height / r.height);
    drawOverlay();
    return;
  }
  if (cropDrag) {
    const nx = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    const ny = Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height));
    const dx = nx - cropDrag.sx, dy = ny - cropDrag.sy;
    let [t, l, b, ri] = cropDrag.orig;
    const minS = 0.06;
    if (cropDrag.mode === "move") {
      const h = b - t, w = ri - l;
      t = Math.max(0, Math.min(1 - h, t + dy)); l = Math.max(0, Math.min(1 - w, l + dx));
      b = t + h; ri = l + w;
    } else {
      if (cropDrag.mode.includes("t")) t = Math.max(0, Math.min(b - minS, t + dy));
      if (cropDrag.mode.includes("b")) b = Math.min(1, Math.max(t + minS, b + dy));
      if (cropDrag.mode.includes("l")) l = Math.max(0, Math.min(ri - minS, l + dx));
      if (cropDrag.mode.includes("r")) ri = Math.min(1, Math.max(l + minS, ri + dx));
      // Enforce aspect ratio lock on corner drag
      if (cropRatio !== "free" && cropDrag.mode.length === 2) {
        const im = $("imgMain"), iw = im.naturalWidth || 1, ih = im.naturalHeight || 1;
        let ar = cropRatio === "orig" ? iw / ih : parseFloat(cropRatio);
        if (cropPortrait && ar !== 1) ar = 1 / ar;
        const wPx = (ri - l) * iw;
        const hNorm = (wPx / ar) / ih;
        if (cropDrag.mode.includes("b")) b = Math.min(1, t + hNorm);
        else t = Math.max(0, b - hNorm);
      }
    }
    geom.crop = [t, l, b, ri];
    drawOverlay();
  }
});

window.addEventListener("mouseup", async ev => {
  if (panning) { panning = null; updateOverlayCursor(); return; }
  if (cropDrag) { cropDrag = null; return; }
  if (!meta) return;

  const ov = $("overlayCanvas"), r = ov.getBoundingClientRect();
  const inBounds = ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;

  if (baseDrag) {
    const bd = baseDrag; baseDrag = null;
    const [H, W] = meta.preview_shape;
    // Map overlay box to sensor preview coords
    const toSensorPt = (ox, oy) => {
      let u = ox / ov.width, v = oy / ov.height;
      if (geom.flip_h) u = 1 - u;
      if (geom.flip_v) v = 1 - v;
      const k = ((geom.rot90 || 0) % 4 + 4) % 4;
      let sx = u, sy = v;
      if (k === 1) [sx, sy] = [1 - v, u];
      else if (k === 2) [sx, sy] = [1 - u, 1 - v];
      else if (k === 3) [sx, sy] = [v, 1 - u];
      return [sx * W, sy * H];
    };
    const [xa, ya] = toSensorPt(bd.x0, bd.y0), [xb, yb] = toSensorPt(bd.x1, bd.y1);
    const l = Math.round(Math.min(xa, xb)), t = Math.round(Math.min(ya, yb));
    const w = Math.round(Math.abs(xb - xa)), h = Math.round(Math.abs(yb - ya));
    if (w < 2 || h < 2) { drawOverlay(); return; }
    setLoading(true);
    try {
      meta = await api("/api/base", { region: [t, l, h, w], token: meta.token });
      activeTool = null; syncToolButtons();
      hydrateFromAutoMeta(meta);
      params = deepClone(presetStartParams); grade = deepClone(presetStartGrade);
      loadCurvesFromDict(isT19Engine() ? grade.tone_curve : params.tone_curve);
      syncAllControls(); persistCurrentFile(true); pushHistory();
      await renderMain();
      toast("已使用手动框选片边重新反相");
    } catch (e) { toast("框选片基失败：" + e.message, true); }
    finally { setLoading(false); drawOverlay(); }
    return;
  }

  if (!inBounds) return;
  const nx = (ev.clientX - r.left) / r.width, ny = (ev.clientY - r.top) / r.height;

  if (activeTool === "wb") {
    const im = $("imgMain");
    const px = Math.round(nx * im.naturalWidth), py = Math.round(ny * im.naturalHeight);
    setLoading(true);
    try {
      const res = await api("/api/wb_pick", {
        token: meta.token, x: px, y: py,
        params: serializeParams(),
        geom: serializeGeom(),
        crop_apply: true,
      });
      params.temp = res.temp; params.tint = res.tint;
      activeTool = null; syncToolButtons();
      buildBasicSliders(); persistCurrentFile(true); pushHistory();
      await renderMain();
      toast(`白平衡已吸取：色温 ${Math.round(res.temp * 100)} · 色调 ${Math.round(res.tint * 100)}`);
    } catch (e) { toast("吸取失败：" + e.message, true); }
    finally { setLoading(false); }
    return;
  }

  if (activeTool === "loupe") {
    const [H, W] = meta.preview_shape;
    let u = nx, v = ny;
    if (geom.crop) { u = geom.crop[1] + u * (geom.crop[3] - geom.crop[1]); v = geom.crop[0] + v * (geom.crop[2] - geom.crop[0]); }
    if (geom.flip_h) u = 1 - u;
    if (geom.flip_v) v = 1 - v;
    const k = ((geom.rot90 || 0) % 4 + 4) % 4;
    let sx = u, sy = v;
    if (k === 1) [sx, sy] = [1 - v, u];
    else if (k === 2) [sx, sy] = [1 - u, 1 - v];
    else if (k === 3) [sx, sy] = [v, 1 - u];
    const size = 600, f = meta.factor;
    const ct = Math.max(0, Math.round(sy * H * f - size / 2));
    const cl = Math.max(0, Math.round(sx * W * f - size / 2));
    setLoading(true);
    try {
      const p = serializeParams(), g = serializeGrade();
      let url = `/api/crop?token=${meta.token}&t=${ct}&l=${cl}&size=${size}&p=${encodeURIComponent(JSON.stringify(p))}`;
      if (Object.keys(g).length) url += `&g=${encodeURIComponent(JSON.stringify(g))}`;
      const res = await api(url);
      const lUrl = URL.createObjectURL(await res.blob());
      if (lastLoupeUrl) URL.revokeObjectURL(lastLoupeUrl);
      lastLoupeUrl = lUrl;
      $("loupeImg").src = lUrl;
      $("loupeInfo").textContent = `传感器原始区域 (${ct}, ${cl}) · ${size}×${size} px · 与导出像素一致`;
      $("loupeModal").classList.add("open");
    } catch (e) { toast("1:1 检视失败：" + e.message, true); }
    finally { setLoading(false); }
  }
});

/* ======================== Rotation, Flip & Wiring ======================== */
async function rotateBy(dk) {
  if (!meta) return;
  geom.rot90 = (((geom.rot90 || 0) + dk) % 4 + 4) % 4;
  try { await api("/api/orient", { path: meta.path, rot90: geom.rot90 }); } catch (_) {}
  persistCurrentFile(true); pushHistory();
  await Promise.all([renderMain(), viewMode === "compare" ? renderBefore() : Promise.resolve()]);
  toast("方向：" + ["0°", "左转 90°", "180°", "右转 90°"][geom.rot90]);
}

$("bRotL").onclick = () => rotateBy(1);
$("bRotR").onclick = () => rotateBy(-1);
$("bFlipH").onclick = () => { if (!meta) return; geom.flip_h = !geom.flip_h; buildDetailSliders(); persistCurrentFile(true); pushHistory(); renderMain(); };
$("bFlipV").onclick = () => { if (!meta) return; geom.flip_v = !geom.flip_v; buildDetailSliders(); persistCurrentFile(true); pushHistory(); renderMain(); };
$("bGeomReset").onclick = () => {
  if (!meta) return;
  geom = Object.assign(defaultGeom(), { rot90: geom.rot90 });
  buildDetailSliders(); persistCurrentFile(true); pushHistory(); renderMain(); toast("已重置裁切与拉直");
};
$("bDetailReset").onclick = () => {
  if (!meta) return;
  grade.clarity = 0; grade.dehaze = 0; grade.sharpen = 0; grade.vignette = 0; grade.grain = 0;
  buildDetailSliders(); persistCurrentFile(true); pushHistory(); renderMain();
};

$("bToolWb").onclick = () => setTool("wb");
$("bWbPickInline").onclick = () => setTool("wb");
$("bToolCrop").onclick = () => setTool("crop");
$("bEnterCrop").onclick = () => setTool("crop");
$("bToolLoupe").onclick = () => setTool("loupe");
$("bPickBase").onclick = () => setTool("pick_base");
$("bAutoBorder").onclick = autoBorderCrop;
$("bCropAuto").onclick = autoBorderCrop;
$("bCropSwap").onclick = () => { cropPortrait = !cropPortrait; applyCropAspectRatio(cropRatio); };
$("bCropReset").onclick = () => { geom.crop = null; cropRatio = "free"; drawOverlay(); toast("已清除裁切框"); };
$("bCropDone").onclick = () => { if (activeTool === "crop") setTool("crop"); };
document.querySelectorAll("#cropBar .ratio-btn").forEach(b => { b.onclick = () => applyCropAspectRatio(b.dataset.ratio); });

const toggleClip = () => {
  showClip = !showClip;
  syncToolButtons();
  updateHistogramAndClip($("imgMain"), lastBb, lastCh);
  toast(showClip ? "已开启溢出警示（红=高光过曝 蓝=暗部死黑）" : "已关闭溢出警示");
};
$("bToolClip").onclick = toggleClip;
$("triBlack").onclick = toggleClip;
$("triWhite").onclick = toggleClip;

$("bAuto").onclick = () => {
  if (!meta) return;
  params = deepClone(presetStartParams);
  grade = deepClone(presetStartGrade);
  loadCurvesFromDict(isT19Engine() ? grade.tone_curve : params.tone_curve);
  syncAllControls(); persistCurrentFile(false); pushHistory(); scheduleRender(0);
  toast("已恢复当前预设自动起点");
};
$("bResetAll").onclick = () => {
  if (!meta) return;
  params = isT19Engine() ? defaultT24Params(engineSel === "t24") : defaultV3Params();
  if (meta.camera_matrix) params.cam_mix = isT19Engine() ? 0.35 : 0.40;
  grade = defaultGrade();
  curves = defaultCurves();
  syncAllControls(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
  toast("已重置全部调色参数");
};

$("bUndo").onclick = doUndo;
$("bRedo").onclick = doRedo;
$("bToggleBrowser").onclick = () => { $("workspace").classList.toggle("left-collapsed"); setTimeout(drawOverlay, 200); };
$("bBrowserView").onclick = () => { browserGrid = !browserGrid; renderFileList(); };
$("fileSearch").oninput = renderFileList;
$("filterSeg").onclick = ev => {
  const b = ev.target.closest("button[data-filter]");
  if (!b) return;
  browserFilter = b.dataset.filter;
  $("filterSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  renderFileList();
};
$("bScanRoll").onclick = () => scanRoll();

$("filmSeg").onclick = ev => {
  const b = ev.target.closest("button[data-film]");
  if (!b || b.dataset.film === filmSel) return;
  filmSel = b.dataset.film;
  $("filmSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  hydrateThumbnails();
  if (currentRel) {
    delete savedPerFile[currentRel];
    openFile(currentRel);
  }
};

$("baseSelect").onchange = () => { if (meta) switchBaseMode($("baseSelect").value); };

// Tabs & Accordions
function activateTab(tabName) {
  $("inspTabs").querySelectorAll(".insp-tab").forEach(b => b.classList.toggle("on", b.dataset.tab === tabName));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("on", p.id === "tab-" + tabName));
  if (tabName === "color") { drawCgWheel(); drawCurveCanvas(); }
}
$("inspTabs").onclick = ev => {
  const b = ev.target.closest(".insp-tab");
  if (b) activateTab(b.dataset.tab);
};
$("engineBadge").onclick = () => activateTab("engine");

document.querySelectorAll(".acc-head").forEach(head => {
  head.onclick = ev => {
    if (ev.target.closest("button")) return;
    head.closest(".acc-sec").classList.toggle("collapsed");
  };
});

// Copy / Paste / Sync 3
$("bCopy").onclick = () => {
  if (!meta) return;
  clipboardSnap = {
    params: deepClone(params), grade: deepClone(grade), curves: deepClone(curves), engine: engineSel,
  };
  $("bPaste").disabled = false;
  toast("已复制当前调色参数");
};
$("bPaste").onclick = () => {
  if (!meta || !clipboardSnap) return;
  const keepCam = params.cam_mix;
  params = deepClone(clipboardSnap.params);
  params.cam_mix = keepCam;
  grade = Object.assign(defaultGrade(), deepClone(clipboardSnap.grade));
  curves = deepClone(clipboardSnap.curves);
  syncAllControls(); persistCurrentFile(true); pushHistory(); scheduleRender(0);
  toast("已粘贴调色参数");
};
$("bSync3").onclick = () => {
  if (!meta || !currentRel) return;
  const idx = fileList.indexOf(currentRel);
  if (idx < 0) return;
  persistCurrentFile(true);
  const src = savedPerFile[currentRel];
  const targets = fileList.slice(idx + 1, idx + 4);
  for (const t of targets) {
    const clone = deepClone(src);
    clone.geom = defaultGeom(); // don't overwrite target's crop/rotation
    clone.rawGeom = defaultGeom();
    savedPerFile[t] = clone;
  }
  updateFileListUI();
  toast(`已将当前调色同步至后续 ${targets.length} 帧`);
};

/* ======================== Save User Preset Modal ======================== */
$("bSavePreset").onclick = () => {
  if (!meta) return toast("请先打开一张照片", true);
  $("presetNameInput").value = "";
  $("presetModal").classList.add("open");
  setTimeout(() => $("presetNameInput").focus(), 40);
};
const closePresetModal = () => $("presetModal").classList.remove("open");
$("bPresetClose").onclick = closePresetModal;
$("bPresetCancel").onclick = closePresetModal;
$("bPresetConfirm").onclick = async () => {
  const name = $("presetNameInput").value.trim();
  if (!name) return toast("请输入预设名称", true);
  try {
    const res = await api("/api/user_presets", {
      name, engine: engineSel, params: serializeParams(), grade: serializeGrade(),
    });
    userPresets = res.presets || [];
    buildPresetsUI();
    closePresetModal();
    toast("已保存自定义预设：" + name);
  } catch (e) { toast("保存失败：" + e.message, true); }
};

/* ======================== Export Modal (Single & Batch) ======================== */
function openExportModal() {
  if (!fileList.length) return toast("当前目录没有可导出的照片", true);
  if (currentRel && meta) persistCurrentFile(false);
  const nEd = Object.values(savedPerFile).filter(x => x && x.edited).length;
  $("nEdited").textContent = String(nEd);
  $("nAll").textContent = String(fileList.length);
  $("expProgressBox").style.display = "none";
  $("expResultBox").style.display = "none";
  $("bDoExport").disabled = false;
  $("exportModal").classList.add("open");
}
const closeExportModal = () => $("exportModal").classList.remove("open");
$("bOpenExport").onclick = openExportModal;
$("bExportClose").onclick = closeExportModal;
$("bExportCancel").onclick = closeExportModal;

$("exportScopeSeg").onclick = ev => {
  const b = ev.target.closest("button[data-scope]");
  if (!b) return;
  exportScope = b.dataset.scope;
  $("exportScopeSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  $("batchScopeBox").style.display = exportScope === "batch" ? "block" : "none";
};
$("batchTargetSeg").onclick = ev => {
  const b = ev.target.closest("button[data-target]");
  if (!b) return;
  batchTarget = b.dataset.target;
  $("batchTargetSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
};
$("exportFmtCards").onclick = ev => {
  const card = ev.target.closest(".fmt-card");
  if (!card) return;
  exportFmt = card.dataset.fmt;
  $("exportFmtCards").querySelectorAll(".fmt-card").forEach(x => x.classList.toggle("on", x === card));
  $("expJpegOpt").style.display = exportFmt === "jpeg" ? "block" : "none";
  $("expTiffOpt").style.display = exportFmt === "tiff16" ? "block" : "none";
  $("expSidecar").checked = exportFmt === "tiff16";
};
const syncExpQualFill = () => {
  $("vExpQual").textContent = $("sExpQual").value;
  updateTrackFill($("sExpQual"), 60, 100, false);
};
$("sExpQual").oninput = syncExpQualFill;
syncExpQualFill();

$("bDoExport").onclick = async () => {
  const quality = parseInt($("sExpQual").value, 10);
  const encoding = $("expEncoding").value;
  const maxEdge = $("expMaxEdge").value ? parseInt($("expMaxEdge").value, 10) : null;
  const writeSidecar = $("expSidecar").checked;
  $("bDoExport").disabled = true;
  $("expResultBox").style.display = "none";

  if (exportScope === "single") {
    if (!meta) { $("bDoExport").disabled = false; return toast("请先打开一张照片", true); }
    $("expProgressBox").style.display = "block";
    $("expProgressTxt").textContent = "正在全分辨率渲染并导出…";
    $("expProgressPct").textContent = "50%";
    $("expProgressFill").style.width = "55%";
    try {
      const j = await api("/api/export", {
        token: meta.token,
        params: serializeParams(),
        grade: serializeGrade(),
        geometry: serializeGeom(),
        rot90: geom.rot90 || 0,
        format: exportFmt,
        encoding, quality, max_edge: maxEdge, write_sidecar: writeSidecar,
      });
      $("expProgressPct").textContent = "100%";
      $("expProgressFill").style.width = "100%";
      $("expProgressTxt").textContent = "导出完成";
      $("expResultBox").style.display = "block";
      $("expResultBox").innerHTML =
        `<div><b>已导出：</b>${escapeHtml(j.path)}</div>` +
        (j.sidecar ? `<div><b>参数记录：</b>${escapeHtml(j.sidecar)}</div>` : "") +
        `<div><b>质检：</b><span class="${j.qc.status === "pass" ? "" : "warn"}">${j.qc.status}</span></div>`;
      toast("导出完成");
    } catch (e) {
      $("expProgressTxt").textContent = "导出失败";
      $("expResultBox").style.display = "block";
      $("expResultBox").innerHTML = `<span class="warn">${escapeHtml(e.message)}</span>`;
      toast("导出失败：" + e.message, true);
    } finally {
      $("bDoExport").disabled = false;
    }
    return;
  }

  // Batch export
  const targets = batchTarget === "edited"
    ? fileList.filter(rel => savedPerFile[rel] && savedPerFile[rel].edited)
    : [...fileList];
  if (!targets.length) {
    $("bDoExport").disabled = false;
    return toast("没有符合条件的画格", true);
  }
  const items = targets.map(rel => {
    const s = savedPerFile[rel];
    if (s) {
      return {
        path: rel, film: s.film || filmSel, engine: s.engine || engineSel, preset: s.preset || presetSel,
        params: s.params, grade: s.grade, geometry: s.geom, rot90: (s.geom && s.geom.rot90) || 0,
      };
    }
    return { path: rel, film: filmSel, engine: engineSel, preset: presetSel };
  });

  $("expProgressBox").style.display = "block";
  $("expProgressTxt").textContent = `正在启动批量导出（共 ${items.length} 帧）…`;
  $("expProgressPct").textContent = "0%";
  $("expProgressFill").style.width = "4%";
  try {
    const start = await api("/api/batch_export", {
      items, format: exportFmt, encoding, quality, max_edge: maxEdge, write_sidecar: writeSidecar,
    });
    const poll = async () => {
      const st = await api("/api/batch_export?job_id=" + encodeURIComponent(start.job_id));
      const pct = Math.round((st.done / Math.max(st.total, 1)) * 100);
      $("expProgressPct").textContent = pct + "%";
      $("expProgressFill").style.width = Math.max(4, pct) + "%";
      $("expProgressTxt").textContent = st.status === "done"
        ? `批量导出完成：成功 ${st.results.length} 帧` + (st.errors.length ? `，失败 ${st.errors.length} 帧` : "")
        : `正在导出 (${st.done}/${st.total})：${st.current || ""}`;
      if (st.status === "done") {
        $("expResultBox").style.display = "block";
        const okLines = st.results.slice(0, 6).map(r => `<div>✓ ${escapeHtml(r.path)}</div>`).join("");
        const more = st.results.length > 6 ? `<div>… 等共 ${st.results.length} 个文件</div>` : "";
        const errLines = st.errors.map(e => `<div class="warn">✕ ${escapeHtml(e.source)}：${escapeHtml(e.error)}</div>`).join("");
        $("expResultBox").innerHTML = okLines + more + errLines;
        $("bDoExport").disabled = false;
        toast(`批量导出完成（${st.results.length}/${st.total}）`);
      } else {
        setTimeout(poll, 450);
      }
    };
    setTimeout(poll, 350);
  } catch (e) {
    $("bDoExport").disabled = false;
    toast("批量导出失败：" + e.message, true);
  }
};

/* ======================== Other Modals & Keyboard Shortcuts ======================== */
$("bLoupeClose").onclick = () => $("loupeModal").classList.remove("open");
$("bHelp").onclick = () => $("helpModal").classList.add("open");
$("bHelpClose").onclick = () => $("helpModal").classList.remove("open");
document.querySelectorAll(".modal-backdrop").forEach(m => {
  m.addEventListener("click", ev => { if (ev.target === m) m.classList.remove("open"); });
});

window.addEventListener("keydown", ev => {
  const tag = document.activeElement && document.activeElement.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
  const cmd = ev.metaKey || ev.ctrlKey;

  if (cmd && ev.key.toLowerCase() === "z") {
    ev.preventDefault();
    if (ev.shiftKey) doRedo(); else doUndo();
    return;
  }
  if (cmd && ev.key.toLowerCase() === "e") { ev.preventDefault(); openExportModal(); return; }
  if (cmd || ev.altKey) return;

  const modalOpen = !!document.querySelector(".modal-backdrop.open");
  if (ev.key === "Escape") {
    document.querySelectorAll(".modal-backdrop.open").forEach(m => m.classList.remove("open"));
    if (activeTool) setTool(activeTool);
    return;
  }
  if (modalOpen) return;
  if (ev.key === "Enter" && activeTool === "crop") { ev.preventDefault(); setTool("crop"); return; }
  if (ev.code === "Space" && !ev.repeat) {
    ev.preventDefault();
    if (meta && !spaceComparing && viewMode === "pos") { spaceComparing = true; renderMain(); }
    return;
  }
  if (ev.key === "ArrowRight" || ev.key === "ArrowDown") {
    ev.preventDefault();
    const i = fileList.indexOf(currentRel);
    if (i >= 0 && i < fileList.length - 1) openFile(fileList[i + 1]);
    return;
  }
  if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") {
    ev.preventDefault();
    const i = fileList.indexOf(currentRel);
    if (i > 0) openFile(fileList[i - 1]);
    return;
  }
  const k = ev.key.toLowerCase();
  if (k === "w") { ev.preventDefault(); setTool("wb"); }
  else if (k === "c") { ev.preventDefault(); setTool("crop"); }
  else if (k === "j") { ev.preventDefault(); toggleClip(); }
  else if (k === "r") { ev.preventDefault(); rotateBy(ev.shiftKey ? -1 : 1); }
  else if (k === "f") { ev.preventDefault(); $("bToggleBrowser").click(); }
  else if (ev.key === "\\") {
    ev.preventDefault();
    viewMode = viewMode === "compare" ? "pos" : "compare";
    $("viewSeg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.view === viewMode));
    updateCompareUI();
    if (meta) { renderMain(); if (viewMode === "compare") renderBefore(); }
  }
  else if (ev.key === "0") { ev.preventDefault(); resetZoom(); }
  else if (ev.key === "1") { ev.preventDefault(); setZoom(2.0); }
  else if (ev.key === "+" || ev.key === "=") { ev.preventDefault(); setZoom(zoom * 1.25); }
  else if (ev.key === "-") { ev.preventDefault(); setZoom(zoom / 1.25); }
  else if (ev.key === "?") { ev.preventDefault(); $("helpModal").classList.add("open"); }
});

window.addEventListener("keyup", ev => {
  if (ev.code === "Space" && spaceComparing) {
    spaceComparing = false;
    renderMain();
  }
});

window.addEventListener("resize", () => { drawOverlay(); });

/* ======================== Bootstrap ======================== */
drawCgWheel();
drawCurveCanvas();
buildEngineCards();
buildPresetsUI();
loadFiles().catch(e => { setStatus("加载目录失败：" + e.message, true); toast(e.message, true); });
})();
