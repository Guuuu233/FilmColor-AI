# FilmColor v1.0

面向 **Adobe Lightroom Classic** 与 **Adobe Photoshop** 的原生胶片负片反相与校色插件（支持 macOS 与 Windows，开箱即用，无需预装 Python）。

## 核心特性

- **原生 Lightroom Classic 插件（`FilmColor.lrplugin`）**：实时暗房预览、10 套自适应胶片冲印风格预设（富士 Frontier、德味徕卡 Leica、诺日士 Noritsu、标准 Standard、线性 Linear、自然、人像、鲜明、经典印相、电影质感）、整卷片基与色彩协同、继承 Lr 裁切与旋转、一键输出 16 位正像 TIFF 并自动导入堆叠与保留 EXIF。
- **原生 Photoshop UXP 面板（`.ccx`）**：双击安装，打开面板自动拉起本地引擎，支持实时预览、直接生成 16 位无损正片图层或新建文档。
- **离线独立内置引擎**：Release 离线包内置完整独立引擎（约 33 MB 压缩包），无需安装 Python 或联网依赖；空闲 5 分钟自动回收缓存、30 分钟自动退出。

## 快速安装（推荐直接下载 Release 离线包）

前往本仓库 **[Releases](../../releases)** 页面下载对应系统的离线包：

### macOS（Apple Silicon）
1. 解压 `FilmColor-v1.0-macOS-arm64.zip`。
2. 双击 **`Install_FilmColor_macOS.command`**（首次若提示未识别开发者，右键 → 打开即可）——自动安装 Lightroom 插件并注册 Photoshop 自动启动器。
3. **Lightroom Classic**：重启后选中负片，点击 `图库 → 增效工具额外信息 → FilmColor 负片转换…`。
4. **Photoshop**：双击包内的 **`FilmColor-v1.0-Photoshop.ccx`** 通过 Creative Cloud 安装面板。

### Windows（x64）
1. 解压 `FilmColor-v1.0-Windows-x64.zip`。
2. 双击 **`Install_FilmColor_Windows.bat`** 自动安装 Lightroom 插件并注册 Photoshop 自动启动器。
3. **Lightroom Classic**：重启后选中负片，点击 `图库 → 增效工具额外信息 → FilmColor 负片转换…`。
4. **Photoshop**：双击包内的 **`FilmColor-v1.0-Photoshop.ccx`** 安装面板。

## 从源码安装（开发者）

- **macOS**：双击 `scripts/install_macos.command`
- **Windows**：双击 `scripts/install_windows.bat`
- 详细插件说明见 [plugins/README.md](plugins/README.md)。
