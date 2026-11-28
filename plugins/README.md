# FilmColor 插件（Lightroom Classic & Photoshop）

将 **FilmColor 负片反相与校色引擎** 直接嵌入 Lightroom Classic 与 Photoshop 原生工作流，无需独立前端软件。

## 1. Lightroom Classic 插件（`FilmColor.lrplugin`）

### 工作流特点
- **100% 原生 Lr 界面弹窗**：与 Lightroom 深色界面完全融为一体，支持实时预览、对比「当前效果 / 自动基准 / 原始负片」。
- **100% 保留 FilmColor 引擎精度**：后台直接调用完整物理反相与智能残差校色引擎，输出 16 位正像 TIFF 并**自动导入当前目录、堆叠在原负片上方**（自动继承星标、旗标、色标、关键词与 EXIF 拍摄信息）。
- **回导后全部 Lr 工具正向工作**：不像曲线反转插件那样滑块反向、破坏 AI 蒙版；导入后的 16 位正片可直接使用 Lightroom 原生的曝光/色轮/HSL/AI 天空与人物蒙版/修复画笔/导出预设。
- **整卷批量协同**：同时选中多帧负片时，自动启用整卷片基与**整卷色彩一致性平滑**，并严格单帧串行释放内存。
- **零常驻残留**：首次打开弹窗自动按需启动本地引擎，Lightroom 退出时自动关闭引擎（闲置 30 分钟也会自动退出）。

### 安装与使用
1. 运行安装脚本（已默认软链到 `~/Library/Application Support/Adobe/Lightroom/Modules/FilmColor.lrplugin`）：
   ```bash
   .venv/bin/python scripts/install_plugins.py
   ```
2. 重启 **Adobe Lightroom Classic**（或在 `文件 → 增效工具管理器` 中确认已启用 **FilmColor**）。
3. 选中 1 张或整卷负片，点击菜单：
   - `图库 → 增效工具额外信息 → FilmColor 负片转换…`（带实时预览与预设/微调弹窗）
   - `图库 → 增效工具额外信息 → FilmColor 快速一键转换`（免弹窗按上次偏好直接批量转换）
4. **绑定类似 NLP 的一键快捷键**：打开 macOS `系统设置 → 键盘 → 键盘快捷键 → App 快捷键`，添加 `Adobe Lightroom Classic`，菜单标题填 `FilmColor 负片转换…`，绑定 `⌃⌥F`（Ctrl+Option+F）即可。

---

## 2. Photoshop 原生 UXP 插件（`plugins/photoshop-uxp/`）

### 特点
- 基于 Adobe 官方 **UXP v5 + Spectrum Web Components**，原生适配 Photoshop 深色/浅色主题并可自由停靠在工作区侧栏。
- 支持直接选择磁盘上的 RAW/DNG/16 位 TIFF 负片或读取当前打开文档，在面板内实时预览 FilmColor 效果后，一键以 **16 位无损正片图层** 置入当前文档上方（或新建 16 位文档）。

### 安装与使用
1. 双击 `scripts/start_filmcolor_bridge.command`（或在终端运行 `.venv/bin/python -m filmcolor.plugin_bridge`）启动本地引擎（如果 Lightroom 已唤起引擎则无需重复启动）。
2. 在 **Adobe UXP Developer Tool** 中点击 `Add Plugin`，选择 `plugins/photoshop-uxp/manifest.json`，点击 `Load` 即可在 Photoshop `增效工具 → FilmColor` 中打开面板。

### Windows 安装与使用
1. 双击运行 **`scripts\install_windows.bat`**（自动创建精简 `.venv` 运行环境并将插件安装到 `%APPDATA%\Adobe\Lightroom\Modules\FilmColor.lrplugin`）。
2. 重启 **Adobe Lightroom Classic**，选中负片后点击 `图库 → 增效工具额外信息 → FilmColor 负片转换…`（插件将通过 `pythonw.exe` 静默自动拉起后台引擎，无黑框）。
3. **Photoshop UXP**：双击 **`scripts\start_filmcolor_bridge.bat`** 启动本地桥接服务，再在 Adobe UXP Developer Tool 中加载 `plugins\photoshop-uxp\manifest.json`。
