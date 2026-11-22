/* FilmColor v1.0 — Adobe Photoshop UXP v5 Plugin */
const BRIDGE_URL = "http://127.0.0.1:18765";
const ps = require("photoshop");
const { app, core, action } = ps;
const fs = require("uxp").storage.localFileSystem;

const $ = id => document.getElementById(id);
const SLIDER_KEYS = ["ev", "temp", "tint", "contrast", "highlights", "shadows", "whites", "blacks", "vibrance", "sat", "skin_protect"];

let sourcePath = null;
let lastTempEntry = null;
let viewMode = "after";
let rotDelta = 0;
let previewTimer = null;
let previewSeq = 0;
let previewAbort = null;
let isConverting = false;
let bridgeOk = false;
const COLOR_ONLY_SLIDERS = ["temp", "tint", "vibrance", "sat", "skin_protect"];

function encodeFileUrl(nativePath) {
  const parts = String(nativePath).split("/");
  return "file://" + parts.map((seg, i) => (i === 0 && seg === "" ? "" : encodeURIComponent(seg))).join("/");
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
  try {
    const r = await fetch(`${BRIDGE_URL}/health`);
    const j = await r.json();
    if (r.ok && j.ok) {
      bridgeOk = true;
      $("connDot").className = "dot ok";
      $("connText").textContent = `FilmColor 引擎已连接 (v${j.version})`;
      return true;
    }
  } catch (_) {}
  bridgeOk = false;
  $("connDot").className = "dot err";
  $("connText").textContent = "未连接本地引擎（端口 18765）";
  setStatus("请先运行 scripts/start_filmcolor_bridge.command 启动本地引擎");
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

async function updatePreview() {
  if (!sourcePath) return;
  if (!bridgeOk && !(await checkBridge())) return;
  const mySeq = ++previewSeq;
  if (previewAbort) {
    try { previewAbort.abort(); } catch (_) {}
  }
  previewAbort = typeof AbortController !== "undefined" ? new AbortController() : null;
  setStatus("正在渲染 FilmColor v1.0 预览…");
  try {
    const fetchOpts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: sourcePath,
        film: $("pFilm").value || "color",
        engine: $("pEngine").value || "t24",
        preset: $("pPreset").value || "frontier",
        auto_crop: $("chkAutoCrop").checked,
        rot_delta: rotDelta,
        view: viewMode,
        return_base64: true,
        adjustments: readAdjustments(),
      }),
    };
    if (previewAbort) fetchOpts.signal = previewAbort.signal;
    const r = await fetch(`${BRIDGE_URL}/plugin/preview`, fetchOpts);
    const j = await r.json();
    if (mySeq !== previewSeq) return;
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
    if (j.image_data_url) {
      $("previewImg").src = j.image_data_url;
      $("previewImg").style.display = "block";
      $("previewEmpty").style.display = "none";
    }
    const clip = j.above_white != null ? ` · 高光 ${(j.above_white * 100).toFixed(2)}%` : "";
    const _eLbl = ({ t24: "FilmColor v1.0", t23: "物理直出", v3: "经典引擎" })[j.engine || "t24"] || "FilmColor v1.0";
    setStatus(`预览就绪 · ${_eLbl}${clip}`);
  } catch (e) {
    if (e && e.name === "AbortError") return;
    if (mySeq === previewSeq) setStatus("预览失败：" + e.message);
  }
}

async function pickNegativeFile() {
  try {
    const entry = await fs.getFileForOpening({
      types: ["nef", "nrw", "raf", "cr2", "cr3", "arw", "rw2", "orf", "pef", "srw", "dng", "tif", "tiff"],
    });
    if (!entry) return;
    sourcePath = entry.nativePath;
    $("srcPathText").textContent = sourcePath;
    await updatePreview();
  } catch (e) {
    setStatus("选择文件取消或失败：" + e.message);
  }
}

async function useActiveDocument() {
  if (!app.activeDocument) {
    return setStatus("Photoshop 当前没有打开任何文档");
  }
  try {
    // Try direct file path if active doc is a supported negative on disk
    const doc = app.activeDocument;
    if (doc.path && /\.(nef|nrw|raf|cr2|cr3|arw|rw2|orf|pef|srw|dng|tif|tiff)$/i.test(doc.path)) {
      sourcePath = doc.path;
      $("srcPathText").textContent = sourcePath;
      await updatePreview();
      return;
    }
    // Otherwise save a temporary 16-bit TIFF copy for conversion (cleaning up prior temp file)
    if (lastTempEntry) {
      try { await lastTempEntry.delete(); } catch (_) {}
      lastTempEntry = null;
    }
    const tmpDir = await fs.getTemporaryFolder();
    const tmpFile = await tmpDir.createFile(`fc_ps_src_${Date.now()}.tif`, { overwrite: true });
    await core.executeAsModal(async () => {
      await doc.saveAs.tiff(tmpFile, { byteOrder: "IBM", imageCompression: "none" }, true);
    }, { commandName: "FilmColor 读取当前文档" });
    lastTempEntry = tmpFile;
    sourcePath = tmpFile.nativePath;
    $("srcPathText").textContent = doc.title + "（临时 16 位副本）";
    await updatePreview();
  } catch (e) {
    setStatus("读取当前文档失败：" + e.message);
  }
}

async function runConvert(asNewDocument) {
  if (isConverting) return;
  if (!sourcePath) return setStatus("请先选择负片文件或读取当前文档");
  if (!bridgeOk && !(await checkBridge())) return;
  isConverting = true;
  $("bApplyLayer").disabled = true;
  $("bOpenDoc").disabled = true;
  setStatus("正在全分辨率运行 FilmColor v1.0 转换（16 位）…");
  try {
    const preset = $("pPreset").value || "frontier";
    const engine = $("pEngine").value || "t24";
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
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);

    const outEntry = await fs.getEntryWithUrl(encodeFileUrl(j.output_path));
    await core.executeAsModal(async () => {
      if (asNewDocument || !app.activeDocument) {
        await app.open(outEntry);
      } else {
        const token = fs.createSessionToken(outEntry);
        await action.batchPlay([
          {
            _obj: "placeEvent",
            null: { _path: token, _kind: "local" },
            freeTransformCenterState: { _enum: "quadCenterState", _value: "QCSAverage" },
            _options: { dialogOptions: "dontDisplay" },
          },
        ], { synchronousExecution: true });
        if (app.activeDocument && app.activeDocument.activeLayers.length) {
          app.activeDocument.activeLayers[0].name = `FilmColor v1.0 [${preset}]`;
        }
      }
    }, { commandName: "FilmColor v1.0 生成正片" });
    setStatus("转换完成：" + j.output_path);
  } catch (e) {
    setStatus("转换失败：" + e.message);
  } finally {
    isConverting = false;
    $("bApplyLayer").disabled = false;
    $("bOpenDoc").disabled = false;
  }
}

// Wire up UI events
$("bRefreshConn").addEventListener("click", checkBridge);
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

checkBridge();
