# FilmColor AI T5c 实验与验证报告（2026-11-06）

## 0. 分支合并与工作区状态

1. **分支合并与基准核验**：
   - `main` 从 `1c4c9aa10e843b742513dbbb5d84cd312494572f`（T5a）经 `git merge --ff-only t5b` 快进至 `06e3baaac0b068c8eb8e61af9ae24627bf74081e`（T5b），并基于该提交创建 `t5c` 分支。
   - 用户未提交的 `src/filmcolor/paths.py` 全程未改动，SHA-256 保持 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`。
   - 所有真图评测均在独立的 `FILMCOLOR_DATA_ROOT` 下运行，且对原始图像调用 `filmcolor.frozen.check_source(path, allow_frozen=False)`。

---

## 1. 本轮三项确定性改动

### 1.1 改动 A：显示端断开片边条带涂灰（`holder.py` / `preview.py` / `app.py`）
- **问题根因**：`A1298.ARW` 在 T5b 恢复真实橙罩片基后，左侧高密度片边带（900 px 预览下 `x = 122..156, y = 239..457`，共 `3,663` 个 `hot` 像素）与外框不透光片夹（`x <= 119`）之间隔着 1–2 px 窄缝（`s1` 中位数 `1.0–1.1 < 1.25`），不再经 4-连通触达画幅外边界。未被涂灰时，正片 `P_R ≈ 18.5, P_G ≈ 23.9` 白点截断，而传感器黑电平底噪处的蓝通道尖峰 `P_B ≈ 7.3 × 10⁴ ~ 2.2 × 10⁵` 经相机色彩矩阵负非对角项（`-0.0735 B, -0.1673 B`）将 R、G 推为负值并截断为 `0`，渲染成纯蓝斑点 `[0, 0, 255]`。
- **实现**：
  - 在 `src/filmcolor/holder.py` 新增常量 `REBATE_STRIP_MAX_GAP = 0.01`、`REBATE_STRIP_MIN_SPAN = 0.35` 及函数 `rebate_strip_components(P, holder)`：当 `float(holder.mean()) >= 0.15` 且 `film_box(holder)` 存在时，在 `film_box` 内对 `hot & ~holder` 做连通域标记，将距框边 `<= 0.01 * short`、向内延伸深度 `<= 0.08 * short`、沿该边跨度 `>= 0.35` 的高密度片边连通块标记为片边条带。
  - 在 `src/filmcolor/preview.py` 的 `Preview._paint_holder` 与 `src/filmcolor/app.py` 的 `State.thumbnail_png` 中，将涂灰掩膜扩展为 `(holder & ~scene_highlight_components(holder)) | rebate_strip_components(positive, holder)`。`Preview.holder`、`holder_share`、`film_box` 与测光掩膜 `noref._metering_mask` 完全不变。

### 1.2 改动 B：扫描 TIFF 单通道红层削顶门限扩展（`scanner.py`）
- **问题根因**：`A635.tif`（Nikon D750 翻拍导出的 8-bit 负片 TIFF）红通道在全幅 `15.67%` 像素处削顶于 `R == 255`，而 G、B 通道饱和占比均为 `0.0%` 且保留完整细节。因未达 T4u 的 `r_sat > 0.30` 门限，落入 `sat = np.any(sat_ch, -1)`，导致全图 `78.4%`（`8,699 / 11,100`）的 24×24 块被判为饱和剔除，片基落入画面高光 `[0.8963, 0.3613, 0.1195]`，整幅渲染成深电光蓝死黑。
- **实现**：在 `src/filmcolor/scanner.py` 的 `load_scanner` 中，将 `single_channel_prestretched = "R"` 判据扩展为：
  ```python
  r_sat, g_sat, b_sat = (float(sat_ch[:, :, i].mean()) for i in range(3))
  if (allow_tiff and n_samples >= 3
          and ((r_sat > 0.30 and g_sat < 0.05 and b_sat < 0.05)
               or (r_sat > 0.15 and g_sat < 0.002 and b_sat < 0.002))):
  ```

### 1.3 改动 C：扫描仪三通道近白裸背光片基剔除（`base.py`）
- **问题根因**：`A1086.TIF`（16-bit 扫描 TIFF）齿孔裸背光有 `2,580` 个饱和块（`65535`）及 `213` 个三通道均为 `65534 / 65535 = 0.999985` 的均匀块。因扫描仪路径 `exclude_near_saturated=False`，`near_sat_blocks == 0`，T5b 的 `clipped_base` 不触发，片基取到 `[1.0, 1.0, 1.0]` 导致整幅雪景出现青绿雾。
- **实现**：在 `src/filmcolor/base.py` 的 `estimate_base_blocks` 中新增 `white_ceiling = bool(sp_sat.any() and min(rgb) >= 0.995)` 并入 `clipped_base`，当 `white_ceiling` 为真时令 `valid_film = valid & (sp.min(-1) < CLIPPED_BASE_MAX)` 剔除三通道近白块并重估橙罩片基。

---

## 2. 单元测试与目标样本验证

### 2.1 全量单元测试
- 新增合成单元测试：
  - `tests/test_preview.py::test_t5c_rebate_strip_components_and_paint`
  - `tests/test_scanner.py::test_t5c_scanner_tiff8_moderate_single_channel_r_clip_exempt`
  - `tests/test_base.py::test_t5c_scanner_white_ceiling_excluded`
- 全量 `pytest` 结果：**`447 passed in 473.42s`**（峰值内存 `3.22 GiB`，退出码 `0`）。

### 2.2 三帧目标样本前后对比（900 px 预览，独立 `FILMCOLOR_DATA_ROOT`）

| 样本 | 指标 | T5b (`06e3baa`) | T5c 实测 |
|---|---|---|---|
| `A1298.ARW` | `base` / `d_hi` / `params` / `film_box` | `[0.1707, 0.2118, 0.0847]` / `[0.7832, 1.0005, 1.1626]` / `ev=1.227` / `[8, 121, 456, 656]` | **完全一致**（`holder_share = 0.3869`） |
| `A1298.ARW` | 左侧片边蓝斑 (`edge_band_blue_px` / 执行窗口 / 全幅 `blue_px`) | `1268` / `996` / `1678` | **`212` / `196` / `622`**（`display_sha256 = 1143c30c...`） |
| `A635.tif` | `base` / `d_hi` | `[0.8963, 0.3613, 0.1195]` / `[0.1245, 0.3227, 1.1546]` | **`[1.0000, 0.8388, 0.4508]` / `[0.1649, 0.6760, 1.6546]`** |
| `A635.tif` | `core_q50`（`film_box` / 中心 20%–80%） | `[0.0000, 0.0200, 0.3391]` / `[0.0000, 0.0200, 0.3548]` | **`[0.3838, 0.5709, 0.5447]` / `[0.3589, 0.5407, 0.5592]`**（`display_sha256 = 3a0d5149...`） |
| `A1086.TIF` | `base` / `d_hi` / `clipped_base_excluded` | `[1.0000, 1.0000, 1.0000]` / `[0.1844, 0.5774, 1.3667]` / `False` | **`[0.9357, 0.5186, 0.3180]` / `[0.1556, 0.2923, 0.8691]` / `True`** |
| `A1086.TIF` | `core_q50`（`film_box` / 中心 20%–80%） | `[0.6120, 0.6767, 0.5896]` / `[0.6547, 0.7024, 0.6246]` | **`[0.6740, 0.6184, 0.5307]` / `[0.7108, 0.6681, 0.5906]`**（`display_sha256 = 36dbf748...`） |

---

## 3. 基准集（315 条记录）与设备集（61 条记录）零回归核验

- **315 条基准记录**（`s3_dev_frames.json` 191 帧 × 5 预设 + `bn_rb_frames.json` 124 帧 `standard`）：在独立 `FILMCOLOR_DATA_ROOT` 下相对 T5b（`06e3baa`）除耗时字段（`seconds` / `estimate_seconds`）外**逐位相同**（`non_timing_diffs = []`）。
- **61 条设备记录**（`a1296`、`targets`、`new_targets`、`acros`、`tiff`、`new_tiff`、`special`）：在独立 `FILMCOLOR_DATA_ROOT` 下相对 T5b（`06e3baa`）全部字段（含 `params`、`inversion`、`status`、`flags`、`base`、`base_diagnostics`、`holder_share`、`film_box`、`display_sha256`、`preview_image_sha256`）**逐位相同**（`device_diffs = []`）。
