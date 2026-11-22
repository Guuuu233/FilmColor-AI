# FilmColor v1.0

<p align="center">
  <strong>Native Film Negative Inversion &amp; Color Grading Plugins for Adobe Lightroom Classic &amp; Adobe Photoshop</strong><br>
  macOS (Apple Silicon) &amp; Windows (x64) · Out-of-the-Box Standalone Engine · 8 Languages
</p>

<p align="center">
  <a href="#english">English</a> · <a href="#简体中文">简体中文</a> · <a href="../../releases">Releases / 下载发布包</a>
</p>

---

<a id="english"></a>

## English

### Overview

**FilmColor v1.0** brings physics-based film negative inversion and adaptive darkroom color grading directly into **Adobe Lightroom Classic** and **Adobe Photoshop** — exporting true 16-bit positive TIFFs so every native Lightroom slider, tone curve, and AI mask works forward naturally.

### Key Features

- **Out-of-the-Box Standalone Engine (No Python Required)**: Release bundles include a self-contained engine (~33 MB zip) that launches automatically on demand, trims idle memory after 5 minutes, and shuts down cleanly after 30 minutes of inactivity.
- **Lightroom Classic Plugin (`FilmColor.lrplugin`)**:
  - Native Lightroom darkroom dialog with live preview (*Current / Auto Base / Negative*) and 10 adaptive film print presets (*Fuji Frontier, Leica Look, Noritsu, Standard, Linear, Natural, Portrait, Vivid, Classic Print, Cinema*).
  - Whole-roll film base & color consistency across multi-frame selections.
  - Full Lightroom crop & rotation inheritance, auto film-holder border crop, automatic 16-bit positive TIFF import & stacking above the source negative, plus EXIF/rating/keyword preservation.
- **Photoshop UXP Panel (`FilmColor-v1.0-Photoshop.ccx`)**:
  - One-click installation via Adobe Creative Cloud (`.ccx`), automatic background engine launch, live preview, and direct 16-bit positive layer creation.
- **8 Built-in Languages**: Auto-detects host language with manual override — **English, 简体中文, 繁體中文, 日本語, 한국어, Español, Français, Deutsch**.

### Quick Installation (Recommended)

Download the standalone archive for your OS from **[GitHub Releases](../../releases)**:

#### macOS (Apple Silicon)
1. Unzip `FilmColor-v1.0-macOS-arm64.zip`.
2. Double-click **`Install_FilmColor_macOS.command`** *(if macOS Gatekeeper prompts on first run, right-click → Open)*. This installs the Lightroom plugin and registers the Photoshop auto-start launcher.
3. **Lightroom Classic**: Restart Lightroom, select one or more negatives, and choose `Library → Plug-in Extras → FilmColor Negative Conversion...`.
4. **Photoshop**: Double-click **`FilmColor-v1.0-Photoshop.ccx`** to install via Creative Cloud. Opening the panel automatically starts the local engine.

#### Windows (x64)
1. Unzip `FilmColor-v1.0-Windows-x64.zip`.
2. Double-click **`Install_FilmColor_Windows.bat`** to install the Lightroom plugin and register the Photoshop auto-start launcher.
3. **Lightroom Classic**: Restart Lightroom, select negatives, and choose `Library → Plug-in Extras → FilmColor Negative Conversion...`.
4. **Photoshop**: Double-click **`FilmColor-v1.0-Photoshop.ccx`** to install via Creative Cloud.

### Installing from Source (Developers)

- **macOS**: Double-click `scripts/install_macos.command`
- **Windows**: Double-click `scripts/install_windows.bat`
- See [plugins/README.md](plugins/README.md) for detailed plugin notes.

---

<a id="简体中文"></a>

## 简体中文

### 简介

**FilmColor v1.0** 是面向 **Adobe Lightroom Classic** 与 **Adobe Photoshop** 的原生胶片负片反相与校色插件。转换结果以 **16 位正像 TIFF** 自动导入并堆叠在原负片上方，让 Lightroom 原生滑块、曲线与 AI 蒙版全部正向自然工作。

### 核心特性

- **开箱即用独立内置引擎（免装 Python）**：Release 离线包内置完整独立引擎（压缩包仅 ~33 MB），按需自动拉起；空闲 5 分钟自动释放缓存、30 分钟自动退出归零。
- **Lightroom Classic 插件（`FilmColor.lrplugin`）**：
  - 原生暗房弹窗与实时预览（当前效果 / 自动基准 / 原始负片），内置 10 套自适应胶片冲印风格预设（富士 Frontier、德味徕卡 Leica、诺日士 Noritsu、标准 Standard、线性 Linear、自然、人像、鲜明、经典印相、电影质感）。
  - 支持多帧整卷片基与色彩协同、继承 Lr 裁切与旋转、自动裁去片夹黑边、自动导入堆叠并保留星标/关键词与 EXIF 信息。
- **Photoshop UXP 面板（`FilmColor-v1.0-Photoshop.ccx`）**：
  - 双击 `.ccx` 经 Creative Cloud 一键安装，打开面板自动启动本地引擎，支持实时预览并一键生成 16 位无损正片图层。
- **内置 8 种主流语言**：自动跟随系统/宿主语言并支持手动切换——**简体中文、繁體中文、English、日本語、한국어、Español、Français、Deutsch**。

### 快速安装（推荐直接下载 Release 离线包）

前往 **[GitHub Releases](../../releases)** 页面下载对应系统的离线包：

#### macOS（Apple Silicon）
1. 解压 `FilmColor-v1.0-macOS-arm64.zip`。
2. 双击 **`Install_FilmColor_macOS.command`**（首次若系统提示未识别开发者，右键 → 打开即可），自动安装 Lightroom 插件并注册 Photoshop 自动启动器。
3. **Lightroom Classic**：重启后选中负片，点击 `图库 → 增效工具额外信息 → FilmColor 负片转换…`。
4. **Photoshop**：双击包内的 **`FilmColor-v1.0-Photoshop.ccx`** 经 Creative Cloud 一键安装。

#### Windows（x64）
1. 解压 `FilmColor-v1.0-Windows-x64.zip`。
2. 双击 **`Install_FilmColor_Windows.bat`** 自动安装 Lightroom 插件并注册 Photoshop 自动启动器。
3. **Lightroom Classic**：重启后选中负片，点击 `图库 → 增效工具额外信息 → FilmColor 负片转换…`。
4. **Photoshop**：双击包内的 **`FilmColor-v1.0-Photoshop.ccx`** 安装面板。

### 从源码安装（开发者）

- **macOS**：双击 `scripts/install_macos.command`
- **Windows**：双击 `scripts/install_windows.bat`
- 更多细节见 [plugins/README.md](plugins/README.md)。
