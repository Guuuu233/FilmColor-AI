/* FilmColor v2.0 — Adobe Photoshop UXP v5 Plugin */
const BRIDGE_URL = "http://127.0.0.1:18765";
const ps = require("photoshop");
const { app, core, action } = ps;
const uxp = require("uxp");
const fs = uxp.storage.localFileSystem;
const shell = uxp.shell;

const $ = id => document.getElementById(id);
const T = (...args) => (window.FC_I18N ? window.FC_I18N.t(...args) : args[0]);

let bridgeVersion = "2.0.0";

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.getAttribute("data-i18n");
    if (k === "ps_no_src" && sourcePath) return;
    el.textContent = T(k);
  });
  if (bridgeOk) {
    $("connText").textContent = T("ps_connected", bridgeVersion);
  } else {
    $("connText").textContent = T("ps_disconnected");
  }
}

const SLIDER_KEYS = ["ev", "temp", "tint", "contrast", "highlights", "shadows", "whites", "blacks", "vibrance", "sat", "skin_protect"];

let sourcePath = null;
let sourceIsTemp = false;
let lastTempEntry = null;
let viewMode = "after";
let rotDelta = 0;
let previewTimer = null;
let previewSeq = 0;
let previewAbort = null;
let lastPreviewPayload = null;
let isConverting = false;
let bridgeOk = false;
let healthCheckInFlight = false;

async function cleanupTempEntry() {
  if (lastTempEntry) {
    const entry = lastTempEntry;
    lastTempEntry = null;
    try { await entry.delete(); } catch (_) {}
  }
}

async function safeReadJson(r) {
  const txt = await r.text();
  if (!txt) return {};
  try {
    return JSON.parse(txt);
  } catch (_) {
    if (!r.ok) throw new Error(txt.slice(0, 200) || `HTTP ${r.status}`);
    throw new Error(`Invalid JSON response (HTTP ${r.status})`);
  }
}
const COLOR_ONLY_SLIDERS = ["temp", "tint", "vibrance", "sat", "skin_protect"];

function encodeFileUrl(nativePath) {
  const norm = String(nativePath).replace(/\\/g, "/");
  const parts = norm.split("/");
  const encoded = parts.map((seg, i) => {
    if (i === 0 && (seg === "" || /^[A-Za-z]:$/.test(seg))) return seg;
    return encodeURIComponent(seg);
  }).join("/");
  return encoded.startsWith("/") ? "file://" + encoded : "file:///" + encoded;
}

function syncFilmSliderState() {
  const isBw = ($("pFilm").value || "color") === "bw";
  for (const k of COLOR_ONLY_SLIDERS) {
    const el = $("sl_" + k);
    if (el) el.disabled = isBw;
  }
}

function setStatus(msg) {
  $("statusMsg").textContent = msg;
}

async function checkBridge() {
  if (healthCheckInFlight) return bridgeOk;
  healthCheckInFlight = true;
  let timer = null;
  try {
    const opts = {};
    if (typeof AbortController !== "undefined") {
      const ctrl = new AbortController();
      opts.signal = ctrl.signal;
      timer = setTimeout(() => { try { ctrl.abort(); } catch (_) {} }, 2500);
    }
    const r = await fetch(`${BRIDGE_URL}/health`, opts);
    const j = await safeReadJson(r);
    if (r.ok && j && j.ok) {
      bridgeOk = true;
      bridgeVersion = j.version || bridgeVersion;
      $("connDot").className = "dot ok";
      $("connText").textContent = T("ps_connected", bridgeVersion);
      if ($("bStartBridge")) $("bStartBridge").style.display = "none";
      return true;
    }
  } catch (_) {}
  finally {
    if (timer) clearTimeout(timer);
    healthCheckInFlight = false;
  }
  bridgeOk = false;
  $("connDot").className = "dot err";
  $("connText").textContent = T("ps_disconnected");
  if ($("bStartBridge")) $("bStartBridge").style.display = "inline-flex";
  setStatus(T("ps_offline_hint"));
  return false;
}

function readAdjustments() {
  const adj = {};
  for (const k of SLIDER_KEYS) {
    adj[k] = Number($("sl_" + k).value || 0);
  }
  return adj;
}

function schedulePreview(delay = 120) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updatePreview, delay);
}

function engineLabel(eng) {
  const key = ({ t25: "ps_eng_t25", t24: "ps_eng_t25", t23: "ps_eng_t23", v3: "ps_eng_v3" })[eng || "t25"] || "ps_eng_t25";
  return T(key);
}

async function updatePreview(force = false) {
  if (!sourcePath) return;
  if (!bridgeOk && !(await checkBridge())) return;
  const payloadStr = JSON.stringify({
    path: sourcePath,
    film: $("pFilm").value || "color",
    engine: $("pEngine").value || "t25",
    preset: $("pPreset").value || "frontier",
    auto_crop: $("chkAutoCrop").checked,
    rot_delta: rotDelta,
    view: viewMode,
    max_width: 640,
    max_height: 400,
    return_base64: true,
    write_preview_file: false,
    adjustments: readAdjustments(),
  });
  if (!force && payloadStr === lastPreviewPayload) return;
  lastPreviewPayload = payloadStr;
  const mySeq = ++previewSeq;
  if (previewAbort) {
    try { previewAbort.abort(); } catch (_) {}
  }
  previewAbort = typeof AbortController !== "undefined" ? new AbortController() : null;
  setStatus(T("ps_rendering"));
  try {
    const fetchOpts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payloadStr,
    };
    if (previewAbort) fetchOpts.signal = previewAbort.signal;
    const r = await fetch(`${BRIDGE_URL}/plugin/preview`, fetchOpts);
    const j = await safeReadJson(r);
    if (mySeq !== previewSeq) return;
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
    if (j.image_data_url) {
      $("previewImg").src = j.image_data_url;
      $("previewImg").style.display = "block";
      $("previewEmpty").style.display = "none";
    }
    const clip = j.above_white != null ? T("ps_clip_pct", j.above_white * 100) : "";
    setStatus(T("ps_preview_ready", engineLabel(j.engine), clip));
  } catch (e) {
    if (e && e.name === "AbortError") return;
    lastPreviewPayload = null;
    if (mySeq === previewSeq) setStatus(T("ps_preview_failed", e.message));
  }
}

async function pickNegativeFile() {
  try {
    const entry = await fs.getFileForOpening({
      types: ["nef", "nrw", "raf", "cr2", "cr3", "arw", "rw2", "orf", "pef", "srw", "dng", "tif", "tiff"],
    });
    if (!entry) return;
    await cleanupTempEntry();
    sourcePath = entry.nativePath;
    sourceIsTemp = false;
    lastPreviewPayload = null;
    $("srcPathText").textContent = sourcePath;
    await updatePreview(true);
  } catch (e) {
    setStatus(T("ps_pick_failed", e.message));
  }
}

async function useActiveDocument() {
  if (!app.activeDocument) {
    return setStatus(T("ps_no_doc"));
  }
  try {
    // Use direct file path only if active doc is an unmodified single-layer negative on disk
    const doc = app.activeDocument;
    const isUnmodifiedDiskNegative = (
      doc.path
      && /\.(nef|nrw|raf|cr2|cr3|arw|rw2|orf|pef|srw|dng|tif|tiff)$/i.test(doc.path)
      && doc.saved !== false
      && (!doc.layers || doc.layers.length <= 1)
    );
    if (isUnmodifiedDiskNegative) {
      await cleanupTempEntry();
      sourcePath = doc.path;
      sourceIsTemp = false;
      lastPreviewPayload = null;
      $("srcPathText").textContent = sourcePath;
      await updatePreview(true);
      return;
    }
    // Otherwise save a temporary 16-bit TIFF copy capturing current edits/crops/layers
    await cleanupTempEntry();
    const tmpDir = await fs.getTemporaryFolder();
    const tmpFile = await tmpDir.createFile(`fc_ps_src_${Date.now()}.tif`, { overwrite: true });
    await core.executeAsModal(async () => {
      await doc.saveAs.tiff(tmpFile, { byteOrder: "IBM", imageCompression: "none" }, true);
    }, { commandName: T("ps_cmd_read_doc") });
    lastTempEntry = tmpFile;
    sourcePath = tmpFile.nativePath;
    sourceIsTemp = true;
    lastPreviewPayload = null;
    $("srcPathText").textContent = T("ps_temp_copy", doc.title);
    await updatePreview(true);
  } catch (e) {
    setStatus(T("ps_read_doc_failed", e.message));
  }
}

async function runConvert(asNewDocument) {
  if (isConverting) return;
  if (!sourcePath) return setStatus(T("ps_pick_src_first"));
  if (!bridgeOk && !(await checkBridge())) return;
  isConverting = true;
  $("bApplyLayer").disabled = true;
  $("bOpenDoc").disabled = true;
  setStatus(T("ps_converting"));
  let tempOutEntry = null;
  try {
    const preset = $("pPreset").value || "frontier";
    const engine = $("pEngine").value || "t25";
    const r = await fetch(`${BRIDGE_URL}/plugin/convert`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: sourcePath,
        film: $("pFilm").value || "color",
        engine,
        preset,
        auto_crop: $("chkAutoCrop").checked,
        rot_delta: rotDelta,
        format: "tiff16",
        encoding: "srgb",
        write_sidecar: false,
        copy_exif: true,
        adjustments: readAdjustments(),
      }),
    });
    const j = await safeReadJson(r);
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);

    const outEntry = await fs.getEntryWithUrl(encodeFileUrl(j.output_path));
    const placingIntoDoc = !asNewDocument && !!app.activeDocument;
    if (sourceIsTemp && placingIntoDoc) {
      tempOutEntry = outEntry;
    }
    await core.executeAsModal(async () => {
      if (!placingIntoDoc) {
        await app.open(outEntry);
      } else {
        const token = fs.createSessionToken(outEntry);
        const bpRes = await action.batchPlay([
          {
            _obj: "placeEvent",
            null: { _path: token, _kind: "local" },
            freeTransformCenterState: { _enum: "quadCenterState", _value: "QCSAverage" },
            _options: { dialogOptions: "dontDisplay" },
          },
        ], {});
        const bpErr = Array.isArray(bpRes) ? bpRes.find(item => item && item._obj === "error") : null;
        if (bpErr) {
          throw new Error(bpErr.message || "Photoshop placeEvent failed");
        }
        if (app.activeDocument && app.activeDocument.activeLayers.length) {
          app.activeDocument.activeLayers[0].name = `${engineLabel(engine)} [${preset}]`;
        }
      }
    }, { commandName: T("ps_cmd_convert") });
    setStatus(T("ps_done", j.output_path));
  } catch (e) {
    setStatus(T("ps_convert_failed", e.message));
  } finally {
    if (tempOutEntry) {
      try { await tempOutEntry.delete(); } catch (_) {}
    }
    isConverting = false;
    $("bApplyLayer").disabled = false;
    $("bOpenDoc").disabled = false;
  }
}

// Wire up UI events

let autoStartTried = false;

async function findBridgeLauncher() {
  try {
    const df = await fs.getDataFolder();
    const p = String(df.nativePath || "");
    const cands = [];
    const macMatch = p.match(/^(\/Users\/[^/]+)/);
    const winMatch = p.match(/^([A-Za-z]:\\Users\\[^\\]+)/i);
    if (macMatch) {
      const home = macMatch[1];
      cands.push(`${home}/Library/Application Support/FilmColor/start_bridge.command`);
    } else if (winMatch) {
      const home = winMatch[1];
      cands.push(`${home}\\AppData\\Roaming\\FilmColor\\start_bridge.bat`);
      cands.push(`${home}\\AppData\\Roaming\\Adobe\\Lightroom\\Modules\\FilmColor.lrplugin\\bin\\win\\filmcolor-bridge\\filmcolor-bridge.exe`);
    }
    for (const c of cands) {
      try {
        const entry = await fs.getEntryWithUrl(encodeFileUrl(c));
        if (entry) return c;
      } catch (_) {}
    }
  } catch (_) {}
  return null;
}

async function launchBridgeAndWait(silent = false) {
  if (!shell || typeof shell.openPath !== "function") {
    if (!silent) setStatus(T("ps_auto_launch_unsupported"));
    return false;
  }
  const launcher = await findBridgeLauncher();
  if (!launcher) {
    if (!silent) setStatus(T("ps_launcher_missing"));
    return false;
  }
  setStatus(T("ps_starting"));
  try {
    await shell.openPath(launcher);
  } catch (e) {
    if (!silent) setStatus(T("ps_launch_failed", e.message));
    return false;
  }
  for (let i = 0; i < 24; i++) {
    await new Promise(res => setTimeout(res, 500));
    if (await checkBridge()) {
      setStatus(T("ps_started"));
      if (sourcePath) updatePreview(true);
      return true;
    }
  }
  if (!silent) setStatus(T("ps_launch_timeout"));
  return false;
}

$("bRefreshConn").addEventListener("click", checkBridge);
if ($("bStartBridge")) $("bStartBridge").addEventListener("click", () => launchBridgeAndWait(false));
$("bPickFile").addEventListener("click", pickNegativeFile);
$("bUseActive").addEventListener("click", useActiveDocument);
$("bApplyLayer").addEventListener("click", () => runConvert(false));
$("bOpenDoc").addEventListener("click", () => runConvert(true));

for (const [id, mode] of [["vAfter", "after"], ["vAuto", "auto"], ["vNeg", "negative"]]) {
  $(id).addEventListener("click", () => {
    viewMode = mode;
    for (const [oid] of [["vAfter"], ["vAuto"], ["vNeg"]]) $(oid).selected = (oid === id);
    updatePreview();
  });
}

$("bRotL").addEventListener("click", () => { rotDelta = (rotDelta + 1) % 4; updatePreview(); });
$("bRotR").addEventListener("click", () => { rotDelta = (rotDelta + 3) % 4; updatePreview(); });

$("bResetAdj").addEventListener("click", () => {
  for (const k of SLIDER_KEYS) $("sl_" + k).value = 0;
  rotDelta = 0;
  updatePreview();
});

for (const id of ["pFilm", "pEngine", "pPreset", "chkAutoCrop"]) {
  $(id).addEventListener("change", () => {
    if (id === "pFilm") syncFilmSliderState();
    schedulePreview(30);
  });
}
syncFilmSliderState();
for (const k of SLIDER_KEYS) {
  $("sl_" + k).addEventListener("input", () => schedulePreview(120));
  $("sl_" + k).addEventListener("change", () => schedulePreview(0));
}

let reconnectTimer = null;
let reconnectBusy = false;
async function bootConnect() {
  let ok = await checkBridge();
  if (!ok && !autoStartTried) {
    autoStartTried = true;
    ok = await launchBridgeAndWait(true);
  }
  if (!ok && !reconnectTimer) {
    reconnectTimer = setInterval(async () => {
      if (reconnectBusy) return;
      reconnectBusy = true;
      try {
        if (await checkBridge()) {
          clearInterval(reconnectTimer);
          reconnectTimer = null;
          if (sourcePath) updatePreview(true);
        }
      } finally {
        reconnectBusy = false;
      }
    }, 4000);
  }
}
if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("unload", () => {
    if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
    cleanupTempEntry();
  });
}
if ($("pLang")) {
  $("pLang").value = window.FC_I18N ? window.FC_I18N.getSavedLang() : "auto";
  $("pLang").addEventListener("change", () => {
    if (window.FC_I18N) window.FC_I18N.setLang($("pLang").value || "auto");
    applyI18n();
  });
}
applyI18n();
bootConnect();
