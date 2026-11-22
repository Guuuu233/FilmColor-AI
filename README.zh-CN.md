<div align="center">

# FilmColor v1.0

**面向 Adobe Lightroom Classic 与 Adobe Photoshop 的物理优先胶片负片反相引擎**

[English](README.md) · **简体中文** · [下载 v1.0.0 正式版](https://github.com/Guuuu233/FilmColor-AI/releases/tag/v1.0.0)

[![Release](https://img.shields.io/github/v/release/Guuuu233/FilmColor-AI?color=0f766e&label=版本)](https://github.com/Guuuu233/FilmColor-AI/releases)
[![Platform](https://img.shields.io/badge/支持平台-macOS%20%7C%20Windows-1e293b)](#安装指南)
[![Adobe](https://img.shields.io/badge/原生插件-Lightroom%20Classic%20%2B%20Photoshop%20UXP-0f766e)](#原生插件界面与工作流)
[![Languages](https://img.shields.io/badge/界面语言-内置%208%20国语言-334155)](#内置-8-国语言)

*真 16-bit 正向正片输出 · 盲测色彩准确度对 NegPy 胜率高达 89.8% · 零依赖离线安装包（macOS 32.9 MB / Windows 69.5 MB）*

</div>

---

![FilmColor v1.0 胶片负片反相前后效果对比](docs/assets/showcase_before_after.jpg)

> **真实相机翻拍 135 彩色负片**经 **FilmColor v1.0** 一键直出 16-bit 正片效果（零手动分通道拉曲线）。左：原始带橙色色罩 RAW 负片；中：**Frontier** 富士冲印预设；右：**Leica Look** 徕卡影调预设。

---

## 为什么选择 FilmColor v1.0？

市面上大多数负片转换工具往往不得不在**色彩准确度**、**后期调色手感**与**软件生态**之间妥协。FilmColor v1.0 基于标定物理密度空间反相模型，原生运行在 **Adobe Lightroom Classic** 与 **Adobe Photoshop** 中，彻底打破这些限制：

| 核心维度 | **FilmColor v1.0** | Negative Lab Pro (NLP) | NegPy v0.69.0 |
| :--- | :--- | :--- | :--- |
| **Lightroom Classic 原生插件** | **支持**（原生暗房弹窗 + 自动堆叠） | 支持（弹窗） | 不支持（独立软件） |
| **Photoshop 原生插件** | **支持**（UXP v5 面板，直出 16-bit 图层） | 不支持 | 不支持 |
| **Lightroom 后期调色方向** | **真 16-bit 正向正片** — 所有原生滑块、曲线与 AI 蒙版均正向生效 | 反转色调曲线 — 曝光/高光等滑块方向相反，需导出副本才恢复正向 | 需在外部转换后导入 |
| **盲测色彩误差（ΔE2000，↓ 越低越准）** | **中位 8.20** / P90 13.74（`n=225`） | 闭源 / 依赖逐张手动白平衡 | 中位 15.14 / P90 22.39（`n=225`） |
| **整卷统一分析（Roll Sync）** | **内置**（跨帧片基与色偏联合统计） | 内置（Roll Analysis） | 手动 / 按卷调整 |
| **齿孔与背光边缘自动过滤** | **全自动防护**（自动剔除片边与漏光干扰） | 需手动设置裁边百分比 | 需手动裁边 |
| **实时 3D LUT 预览速度（1.7 MP）** | **130.7 ms**（热缓存）/ **165.2 ms**（全链路） | 实时（LR 代理预览） | 交互式 |
| **零依赖独立离线安装包** | **32.9 MB**（macOS）/ **69.5 MB**（Windows） | 商业安装包 | Python / 独立包 |
| **内置界面语言** | **8 国语言**（简中、繁中、英、日、韩、西、法、德） | 仅英语 | 仅英语 |
| **授权与价格** | **免费开源** | 付费（$99） | 开源 |

---

## 盲测色彩准确度对比（真实数据）

为了杜绝挑片与过拟合，我们在 **225 张完全未参与训练的跨设备盲测负片**（覆盖相机翻拍橙色色罩 RAW 与扫描仪蓝色片基负片）上，将 **FilmColor v1.0** 与 **NegPy v0.69.0** 进行逐帧盲测对比。指标采用国际通用的感知色差 **CIEDE2000（ΔE2000，越低代表越接近冲印参考正片）**：

![FilmColor v1.0 与 NegPy 盲测色彩准确度对比图](docs/assets/benchmark_accuracy.png)

| 盲测数据集子集 | 帧数 (`n`) | **FilmColor v1.0** 中位 ΔE2000 | **FilmColor v1.0** P90 ΔE2000 | NegPy v0.69.0 中位 ΔE2000 | NegPy v0.69.0 P90 ΔE2000 | **FilmColor v1.0 胜率** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **全部匹配盲测集（All Blind OOS）** | **225** | **8.20** | **13.74** | 15.14 | 22.39 | **89.8% (202 / 225)** |
| **冻结盲测留出集（Frozen Holdout）** | **179** | **8.18** | **13.41** | 15.45 | 22.93 | **91.1% (163 / 179)** |
| **相机翻拍橙色色罩 RAW** | **73** | **6.69** | **15.58** | 12.19 | 19.01 | **78.1% (57 / 73)** |
| **扫描仪蓝色片基负片** | **152** | **8.78** | **13.12** | 16.53 | 22.82 | **95.4% (145 / 152)** |

> 在完整的 293 帧盲测评估集（`n = 293`）中，FilmColor v1.0 整体中位 ΔE2000 低至 **8.33**（其中相机 RAW 负片中位仅 **6.93**，高光/暗部交叉色偏仅 **2.01**），显著领先内置的物理直出引擎与经典引擎。

### 实测运行速度与内存占用

测试平台：macOS Apple Silicon（18 GiB 内存，纯 CPU 运行，无需独立显卡）：

| 操作场景 | 实测耗时 / 资源占用 |
| :--- | :--- |
| **实时 3D LUT 暗房调色预览（1.7 MP）** | **130.7 ms**（热缓存）/ **165.2 ms**（完整重算） |
| **2000 万像素相机 RAW 首次预览（单通道半尺寸解码）** | **3.30 s** |
| **2000 万像素相机 RAW 全分辨率 16-bit TIFF 导出** | **12.95 s** |
| **后台本地引擎常驻内存** | **约 61 MB**（空闲 5 分钟自动清理缓存，30 分钟自动退出释放内存） |
| **零依赖离线发布包体积** | **32.9 MB**（macOS arm64）· **69.5 MB**（Windows x64） |

---

## 原生插件界面与工作流

FilmColor v1.0 直接嵌入摄影师最常用的后期宿主软件——无需打开浏览器网页、无需上传云端、无需配置任何 Python 环境。

![FilmColor v1.0 Lightroom Classic 与 Photoshop UXP 插件界面展示](docs/assets/ui_preview.png)

### Adobe Lightroom Classic（`FilmColor.lrplugin`）
- **真 16-bit 正向正片工作流**：输出标定后的 16-bit 线性正片 TIFF，并自动导入图库、堆叠在原始负片上方——所有 Lightroom 原生滑块、曲线、HSL 和 AI 蒙版完全正向工作，告别反向曲线折磨。
- **自动继承裁剪与旋转**：在片基采样前自动继承你在 Lightroom 中的传感器空间裁剪与旋转角度，并智能屏蔽齿孔与背光板漏光。
- **整卷统一分析（Roll Analysis）**：跨帧聚合同一卷胶片的片基与色偏特征，保证整卷色调连贯一致。
- **实时暗房弹窗**：支持正片视图 / 左右分屏对比 / 原始负片 / 高光暗部溢出警告，内置曝光、对比度、高光、阴影、色温、色调、饱和度精细调节。

### Adobe Photoshop（`FilmColor-UXP.ccx`）
- **原生 UXP v5 面板**：深度融合 Adobe Spectrum 深色界面，提供一键风格标签、实时预览与暗房滑块。
- **全自动拉起本地引擎**：点击面板自动启动内置本地计算服务，零命令行操作。
- **非破坏性 16-bit 正片图层**：一键将反相后的 16-bit 正片写入当前文档新图层，方便衔接精修与输出。

---

## 内置 10 套暗房冲印风格预设

基于 33³ float32 三线性 3D LUT 实时渲染（预览切换 ~130 ms），一键切换经典彩扩机、编辑部与电影风格：

![FilmColor v1.0 内置风格预设效果总览](docs/assets/showcase_presets.jpg)

`Frontier（富士冲印）` · `Noritsu（诺日士）` · `Leica Look（徕卡影调）` · `Standard（标准平衡）` · `Natural（自然纪实）` · `Portrait（人像肤色）` · `Vivid（鲜艳反转）` · `Print Classic（经典相纸）` · `Cinema（电影青橙）` · `Linear（线性直出）`

---

## 内置 8 国语言

在 Lightroom Classic 与 Photoshop 面板内均可实时切换界面语言，无需重启软件：

- **简体中文**（`zh-CN`）
- **繁體中文**（`zh-TW`）
- **English**（`en`）
- **日本語**（`ja`）
- **한국어**（`ko`）
- **Español**（`es`）
- **Français**（`fr`）
- **Deutsch**（`de`）

---

## 安装指南

### 方式一：下载零依赖离线安装包（推荐，开箱即用）

前往 **[Releases v1.0.0](https://github.com/Guuuu233/FilmColor-AI/releases/tag/v1.0.0)** 下载对应系统的压缩包，无需安装 Python 或 Conda：

- **macOS（Apple Silicon）**：`FilmColor-v1.0-macOS-arm64.zip`（32.9 MB）
- **Windows（64 位）**：`FilmColor-v1.0-Windows-x64.zip`（69.5 MB）

#### Lightroom Classic 安装步骤
1. 将下载的压缩包解压到固定目录。
2. 打开 **Adobe Lightroom Classic** → **文件（File）** → **增效工具管理器（Plug-in Manager...）**。
3. 点击 **添加（Add）**，选中解压目录中的 `FilmColor.lrplugin`，点击 **完成**。
4. 选中任意负片 RAW/TIFF → **文件** → **增效工具额外功能（Plug-in Extras）** → **FilmColor v1.0 — 暗房转换**。

#### Photoshop 安装步骤
1. 双击 `FilmColor-UXP.ccx` 通过 Creative Cloud 自动安装；或在 Adobe UXP Developer Tool 中加载 `plugins/photoshop-uxp/manifest.json`。
2. 在 Photoshop 顶部菜单打开 **增效工具（Plugins）** → **FilmColor v1.0**。

---

### 方式二：源码安装（开发者）

需要 Python 3.10+ 与 [`uv`](https://docs.astral.sh/uv/)（或 `pip`）：

```bash
git clone https://github.com/Guuuu233/FilmColor-AI.git
cd FilmColor-AI
uv sync --no-group ml
```

运行完整单元测试：

```bash
uv run pytest
```

本地重新构建零依赖独立发布包：

```bash
uv run python scripts/build_standalone.py
```

---

## 架构特点

- **100% 本地离线与隐私安全**：全部 RAW 解码（`rawpy` / LibRaw）与色彩反相均在本地 `127.0.0.1:18765` 完成，无任何联网上传或遥测。
- **三大可选核心引擎**：
  - **FilmColor v1.0（智能推荐）** — 默认标定物理模型 + 轻量级残差校准，支持全套冲印预设。
  - **物理直出引擎** — 纯解析物理密度空间反相。
  - **经典引擎** — 保留历史版本算法以兼容既有工作流。
- **极致轻量化运行时**：核心推理仅依赖 `numpy`, `scipy`, `Pillow`, `rawpy`, `tifffile`, `exifread`，无需 PyTorch 或庞大深度学习框架。
