<div align="center">

# FilmColor v2.0

**Physics-First Film Negative Inversion for Adobe Lightroom Classic & Adobe Photoshop**

**English** · [简体中文](README.zh-CN.md) · [Download v2.0.0](https://github.com/Guuuu233/FilmColor-AI/releases/tag/v2.0.0)

[![Release](https://img.shields.io/github/v/release/Guuuu233/FilmColor-AI?color=0f766e&label=Release)](https://github.com/Guuuu233/FilmColor-AI/releases)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-1e293b)](#installation)
[![Adobe](https://img.shields.io/badge/Plugins-Lightroom%20Classic%20%2B%20Photoshop%20UXP-0f766e)](#native-plugin-experience)
[![Languages](https://img.shields.io/badge/UI%20Languages-8%20Built--in-334155)](#8-built-in-languages)

*True 16-bit forward positive output · Median ΔE2000 8.14 vs NegPy 15.45 on 179 blind frames · Beats NegPy on 91% of them · Zero-dependency offline bundles*

</div>

---

![FilmColor v2.0 Before and After Negative Conversion](docs/assets/showcase_before_after.jpg)

> **Real camera-scanned 35mm color negatives** converted directly to 16-bit positives by **FilmColor v2.0** with no manual per-channel curve work. Left: original orange-masked RAW negative. Center: **Frontier** preset. Right: **Leica Look** preset.

---

## What's New in v2.0

- **New default color engine.** The learned residual stage is retrained with a perceptual objective that protects tonal range, chroma and clean blacks, on **186 paired lab-print frames plus 177 visually verified frames**. On the frozen blind holdout it lowers median color error from 8.19 to **8.14**, mean from 8.95 to **8.75** and P90 from 13.46 to **13.14** ΔE2000, improving 73 frames and worsening 32 (Wilcoxon *p* = 3.0 × 10⁻⁶; details below).
- **10 scene-adaptive style presets.** `Frontier`, `Noritsu`, `Leica Look` and the rest now adapt to each frame's tonal and color content instead of applying a fixed look.
- **Exact Lightroom crop & straighten.** Crops and rotation angles set in Lightroom Classic are reproduced pixel-exactly in the converted positive.
- **Faster and leaner.** Lower peak memory on full-resolution 16-bit export (chunked writing), 4.4× faster LZW TIFF decoding, and a bridge that renders, decodes and encodes outside its cache lock so slider drags stay responsive.
- **Hardening.** More than 20 robustness fixes across scanner TIFF parsing, inversion, export, the local bridge, and both plugins (including clean shutdown during long batch exports and NaN-safe responses to Lightroom and Photoshop).
- **Upgrade in place.** Plugin identifiers are unchanged, so installing v2.0 over v1.0 keeps your Lightroom and Photoshop settings.

---

## Why FilmColor?

Most film conversion tools force a trade-off between **color accuracy**, **editing ergonomics**, and **ecosystem lock-in**. FilmColor runs a calibrated physical density-space inversion engine locally, inside both **Adobe Lightroom Classic** and **Adobe Photoshop**:

| Capability | **FilmColor v2.0** | Negative Lab Pro (NLP) | NegPy v0.69.0 |
| :--- | :--- | :--- | :--- |
| **Adobe Lightroom Classic Native Plugin** | **Yes** (Darkroom dialog + auto-stack) | Yes (Dialog) | No (standalone app) |
| **Adobe Photoshop Native Plugin** | **Yes** (UXP v5 panel, 16-bit layer) | No | No |
| **Editing Direction in Lightroom** | **True 16-bit forward positive** — native sliders, tone curves & AI masks work normally | Inverted tone curves — sliders run backwards until you export a positive copy | External export required |
| **Blind Color Error (ΔE2000, lower is better)** | **8.14 median** / 13.14 P90 (`n=179`) | Not benchmarked (closed source) | 15.45 median / 22.93 P90 (`n=179`) |
| **Multi-Frame Roll Sync** | **Built-in** (cross-frame film-base & cast pooling) | Built-in (Roll Analysis) | Manual / per-roll |
| **Sprocket & Backlight Guard** | **Automatic** (rejects film borders & backlight bleed) | Requires manual border crop % | Manual border crop |
| **Zero-Dependency Offline Installer** | **Yes** (macOS & Windows) | Commercial installer | Python / standalone |
| **Built-in UI Languages** | **8 languages** (EN, ZH-CN, ZH-TW, JA, KO, ES, FR, DE) | English only | English |
| **License & Cost** | **Free & Open Source** | Paid ($99) | Open Source |

---

## Blind Color Accuracy Benchmark

FilmColor v2.0, FilmColor v1.0 and **NegPy v0.69.0** were scored on the same **179-frame frozen blind holdout**: 69 orange-mask negatives (camera-scanned RAW and scanner) and 110 blue-base scanner negatives. None of these frames was used to train or tune any FilmColor version. Each conversion is compared with the lab print of the same frame using the perceptual color difference **CIEDE2000 (ΔE2000, lower is better)**, after automatic registration. FilmColor is scored engine-only (no style preset), with roll pooling, exactly as for the published v1.0 numbers.

![FilmColor v2.0 vs v1.0 vs NegPy on the 179-frame blind holdout](docs/assets/benchmark_accuracy.png)

| Blind holdout subset | Frames | **v2.0** median / mean / P90 | v1.0 median / mean / P90 | NegPy v0.69.0 median / mean / P90 | **v2.0 beats NegPy** | **v2.0 vs v1.0** (better / worse) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All frozen holdout** | **179** | **8.14 / 8.75 / 13.14** | 8.19 / 8.95 / 13.46 | 15.45 / 16.10 / 22.93 | **163 / 179 (91.1%)** | **73 / 32** (*p* = 3.0 × 10⁻⁶) |
| **Orange-mask camera & scanner** | **69** | **6.61 / 7.94** / 14.20 | 6.68 / 8.12 / **13.48** | 12.08 / 12.59 / 18.22 | **54 / 69 (78.3%)** | **47 / 22** (*p* = 8.8 × 10⁻⁴) |
| **Blue-base scanner negatives** | **110** | **8.73 / 9.26 / 12.65** | 8.77 / 9.47 / 13.45 | 17.65 / 18.30 / 23.99 | **109 / 110 (99.1%)** | **26 / 10** (*p* = 9.2 × 10⁻⁴) |

*p*-values: two-sided Wilcoxon signed-rank test on per-frame ΔE2000, v2.0 vs v1.0. Without roll pooling (single-frame mode), median ΔE2000 is 8.10 for v2.0 and 8.35 for v1.0.

**Notes on the benchmark**

- The v1.0 README also reported 46 extra frames (`n = 225`). Four of them are now in the v2.0 training set, so v2.0 cannot be scored blind on that group and all three tools are compared on the 179 frozen frames only.
- On this holdout the orange-mask P90 is higher for v2.0 (14.20 vs 13.48). It is driven by two snow scenes (A184, A189) whose cached film-base densities were recorded by an older estimator at about half the value the shipping pipeline measures. The v2.0 plugin, which measures the film base itself, renders both frames correctly.
- Reproduce with `scripts/eval_ho179.py` (frozen holdout data is not distributed).

### Runtime Speed & Memory Footprint

Measured on macOS Apple Silicon (18 GiB RAM, CPU only). Interactive figures are from the v1.0 release build; v2.0 adds further export and bridge optimizations.

| Operation | Measured Latency / Footprint |
| :--- | :--- |
| **Interactive 3D LUT Darkroom Preview (1.7 MP)** | **130.7 ms** (warm LUT cache) / **165.2 ms** (full grade) |
| **20 MP Camera RAW Preview (`half_size` single-pass)** | **3.30 s** |
| **20 MP Camera RAW Full-Resolution 16-bit TIFF Export** | **12.95 s** |
| **Idle Resident RAM (local bridge daemon)** | **~61 MB** (auto-trims caches at 5 min, auto-exits at 30 min) |
| **Offline Standalone Bundle Size (v2.0)** | **34.0 MB** zip (macOS arm64) |

---

## Native Plugin Experience

FilmColor integrates directly into the tools film photographers already use — no browser tabs, no cloud uploads, and no Python setup required.

![FilmColor v2.0 Lightroom Classic and Photoshop UXP Interfaces](docs/assets/ui_preview.png)

### Adobe Lightroom Classic (`FilmColor.lrplugin`)
- **True 16-bit Forward Positive Workflow**: Outputs a calibrated 16-bit linear TIFF positive automatically imported and stacked above your original negative — every Lightroom slider, tone curve, HSL wheel, and AI mask works in the natural forward direction.
- **Exact Crop, Straighten & Orientation Inheritance**: Reproduces your Lightroom crop, straighten angle and rotation, and samples the film base inside the crop, automatically rejecting sprocket holes and backlight borders.
- **Roll Analysis & Batch Sync**: Pool film-base and color-cast statistics across an entire roll for consistent frame-to-frame color.
- **Interactive Darkroom Dialog**: Live Positive / Split A-B / Original Negative / Clip Overlay views with exposure, contrast, highlights, shadows, white balance, tint, and saturation sliders.

### Adobe Photoshop (`FilmColor-v2.0-Photoshop.ccx`)
- **Native UXP v5 Panel**: Matches the Adobe Spectrum dark UI with one-click preset chips, live preview, and fine-tuning sliders.
- **Auto-Starting Local Bridge**: Automatically launches the bundled local engine on demand — no terminal commands.
- **Non-Destructive 16-bit Positive Layer**: Inserts the converted 16-bit positive directly into your active document for retouching and printing.

---

## 10 Scene-Adaptive Style Presets

Each preset adapts to the frame's own tonal range and color content, and switches in real time (~130 ms preview via a 33³ float32 trilinear 3D LUT):

![FilmColor v2.0 10 Adaptive Style Presets](docs/assets/showcase_presets.jpg)

`Frontier` · `Noritsu` · `Leica Look` · `Standard` · `Natural` · `Portrait` · `Vivid` · `Print Classic` · `Cinema` · `Linear`

---

## 8 Built-in Languages

Switch interface language live inside both Lightroom Classic and Photoshop with no restart:

- **English** (`en`)
- **简体中文** (`zh-CN`)
- **繁體中文** (`zh-TW`)
- **日本語** (`ja`)
- **한국어** (`ko`)
- **Español** (`es`)
- **Français** (`fr`)
- **Deutsch** (`de`)

---

## Installation

### Option 1 — Zero-Dependency Offline Bundle (Recommended)

Download the pre-built archive for your platform from **[Releases v2.0.0](https://github.com/Guuuu233/FilmColor-AI/releases/tag/v2.0.0)** — no Python, Conda, or command-line tools required:

- **macOS (Apple Silicon)**: `FilmColor-v2.0-macOS-arm64.zip` (34.0 MB)
- **Windows (64-bit)**: `FilmColor-v2.0-Windows-x64.zip` (built automatically by GitHub Actions and attached to the release)

#### Lightroom Classic Setup
1. Unzip the downloaded archive to a permanent folder.
2. macOS: double-click `Install_FilmColor_macOS.command` (right-click → **Open** if Gatekeeper asks). Windows: run `Install_FilmColor_Windows.bat`. Or add it manually: **File** → **Plug-in Manager...** → **Add** → select `FilmColor.lrplugin`.
3. Select any scanned negative(s) → **File** → **Plug-in Extras** → **FilmColor Negative Conversion...** (or **FilmColor Quick Convert** for one-click batch conversion).

#### Photoshop Setup
1. Double-click `FilmColor-v2.0-Photoshop.ccx` (installed via Creative Cloud Desktop), **or** load `plugins/photoshop-uxp/manifest.json` via the Adobe UXP Developer Tool.
2. Open **Plugins** → **FilmColor v2.0** in Photoshop.

Upgrading from v1.0: install over the existing copy; settings are kept.

---

### Option 2 — Install from Source (Developers)

Requires Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/) (or `pip`):

```bash
git clone https://github.com/Guuuu233/FilmColor-AI.git
cd FilmColor-AI
uv sync --no-group ml
```

Run the test suite:

```bash
uv run pytest
```

Build a standalone zero-dependency bundle locally:

```bash
uv run --no-group ml --with pyinstaller scripts/build_standalone.py --out dist
```

---

## Architecture Highlights

- **100% Local & Private**: All RAW decoding (`rawpy` / LibRaw) and color inversion run on your machine via `127.0.0.1:18765` — no telemetry, no cloud calls.
- **3 Selectable Engines**:
  - **FilmColor v2.0 (Recommended)** — Default calibrated physical inversion + perceptual residual model with adaptive style presets.
  - **Physical Direct Engine** — Pure analytical density-space inversion.
  - **Classic Engine** — Legacy baseline engine retained for backward compatibility.
- **Lean Runtime**: Inference uses only `numpy`, `scipy`, `Pillow`, `rawpy`, `tifffile`, and `exifread` — no PyTorch or other deep-learning runtime.
