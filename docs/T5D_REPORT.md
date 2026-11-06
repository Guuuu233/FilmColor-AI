# FilmColor AI T5d 实验与验证报告（2026-11-06）

## 0. 分支合并与工作区状态

1. **分支合并与基准核验**：
   - 核对 `main` = `06e3baaac0b068c8eb8e61af9ae24627bf74081e`（T5b）、`t5c` = `fa0780dd56489dc644dbec62a7d2c549a61689d3`（T5c）后，经 `git checkout main && git merge --ff-only t5c` 将 `main` 快进至 `fa0780d`，并基于该提交创建 `t5d` 分支。
   - 用户未提交的 `src/filmcolor/paths.py` 全程未改动，SHA-256 保持 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`。
   - 所有真图评测均在独立的 `FILMCOLOR_DATA_ROOT` 下运行，且对原始图像调用 `filmcolor.frozen.check_source(path, allow_frozen=False)`。

---

## 1. 本轮两项确定性改动

### 1.1 改动 A：眩光 / 雾翳抬升阴影底座恢复（`noref.py`）
- **问题根因**：`A1125.TIF`（逆光树枝挂灯泡，下部带彩虹镜头眩光环）在画芯区域叠加了 `0.30..0.41 D` 的均匀雾翳底座，而受相机片门遮挡的边缘片基无眩光，使反相后最暗 2% 纹理格密度仅 `tc["p98_dark_d"] = 0.5139 D`（`ev = -0.25` 时编码 2% 分位 `encoded_p2 = 0.5027` 仍高于中灰）；同时 3 颗灯泡灯丝将 `span_d_g` 推至 `0.7949 > 0.72`、天空高光使 `iqr_25_90 = 0.2461 > 0.20`，漏过 `thin_flat_check` 与 `ultra_flat_check`。
- **实现**：
  - 在 `src/filmcolor/noref.py` 的 `NOREF_V3` 中新增配置项 `lifted_floor_check=True`、`lifted_floor_span_max=0.84`、`lifted_floor_ev_max=-0.22`、`lifted_floor_p98_dark_d_max=0.56`、`lifted_floor_p98_dark_d_pivot=0.58`、`lifted_floor_ev_gain=11.0`、`lifted_floor_max_pull=0.70`、`lifted_floor_blacks_gain=0.45`、`lifted_floor_blacks_max=0.30`、`lifted_floor_contrast_gain=0.65`、`lifted_floor_contrast_cap=1.75`。
  - 在 `estimate_params_v3` 的 Step 4b `ultra_flat_check` 之后新增 `elif` 分支 `lifted_floor_check`：当 `span_d_g < 0.84`、`st["range_stops"] >= 1.20`、`base_p["ev"] <= -0.22` 且 `tc["p98_dark_d"] <= 0.56` 时，按 `(0.58 - tc["p98_dark_d"])` 比例下调 `ev`（最多 `-0.70`）与 `blacks`（最多 `-0.30`）并提升 `contrast`（上限 `1.75`），记录于 `diagnostics["lifted_floor"]`。

### 1.2 改动 B：单通道极窄密度跨度输入提示（`preview.py`）
- **问题根因**：Batch 24 中的 `A824.tif`、`A828.tif`、`A1187.tif` 因蓝通道叠加高底座而压缩在极窄区间，白点密度 `d_hi_B` 仅有 `0.0262..0.0348 D`（而 R/G 达 `0.415..0.587 D`），但因其中心线性输入 `p99_B` 为 `0.074..0.141 >= 0.02`，未触发 `input_channel_empty`。
- **实现**：在 `src/filmcolor/preview.py` 新增常量 `INPUT_FLAT_CHANNEL_DHI = (0.04, 0.30)`，并在 `input_warnings(d_hi, center_p99)` 中对未触发 `input_channel_empty` 但满足 `d_hi[c] < 0.04 D` 且另一通道 `d_hi >= 0.30 D` 的通道追加 `code="input_channel_flat"` 非侵入式提示（不改变 `status`、`params` 与反相结果）。

---

## 2. 单元测试与目标样本验证

### 2.1 全量单元测试
- 新增合成单元测试：
  - `tests/test_noref.py::test_t5d_lifted_floor_recovers_fogged_pedestal_frame`
  - `tests/test_preview.py::test_t5d_input_flat_channel_warning`
- 全量 `pytest` 结果：**`449 passed in 361.09s`**（峰值内存 `2.99 GiB`，退出码 `0`）。

### 2.2 目标帧与保护帧实测核验（900 px 预览，独立 `FILMCOLOR_DATA_ROOT`）

| 样本 | 指标 | T5c (`fa0780d`) | T5d 实测 |
|---|---|---|---|
| `A1125.TIF` | `params` (`ev` / `blacks` / `contrast`) | `-0.25` / `0.0` / `1.30` | **`-0.95` / `-0.30` / `1.75`** (`lifted_floor.applied = True`) |
| `A1125.TIF` | `core_q50` / `core_q05` | `[0.5647, 0.5575, 0.5118]` / `[0.1404, 0.1160, 0.0847]` | **`[0.4353, 0.4269, 0.3763]` / `[0.0581, 0.0443, 0.0284]`** (`display_sha256 = ee3723cb...`) |
| `A824.tif` | `input_warnings` / `display_sha256` | `[]` / `0ae171a3...` | **`input_channel_flat` (`B`, `0.0262 D`)** / **`0ae171a3...`（渲染不变）** |
| `A828.tif` | `input_warnings` / `display_sha256` | `[]` / `41ce8302...` | **`input_channel_flat` (`B`, `0.0302 D`)** / **`41ce8302...`（渲染不变）** |
| `A1187.tif` | `input_warnings` / `display_sha256` | `[]` / `9a9045c8...` | **`input_channel_flat` (`B`, `0.0348 D`)** / **`9a9045c8...`（渲染不变）** |
| `A229.TIF` | `input_warnings` / `display_sha256` | `input_channel_empty` (`B`) / `42bdcbea...` | **仅 `input_channel_empty` (`B`)，不重复提示** / **`42bdcbea...`** |
| `A1004.DNG` / `A1033.CR2` / `A1007.CR2` | `params` / `core_q50` / `display_sha256` | `7d239628...` / `adc6ce58...` / `c0630031...` | **完全一致**（`lifted_floor = None`） |

---

## 3. 基准集（315 条记录）与设备集（61 条记录）零回归核验

- **315 条基准记录**（`s3_dev_frames.json` 191 帧 × 5 预设 + `bn_rb_frames.json` 124 帧 `standard`）：在独立 `FILMCOLOR_DATA_ROOT`（`/tmp/fcroot_t5d_bench_cand`）下相对 T5c（`fa0780d`）除耗时字段（`seconds` / `estimate_seconds`）外**逐位相同**（`benchmark_non_timing_diffs = {"a": [], "b": [], "cross": []}`）。
- **61 条设备记录**（`a1296`、`targets`、`new_targets`、`acros`、`tiff`、`new_tiff`、`special`）：在独立 `FILMCOLOR_DATA_ROOT`（`/tmp/fcroot_t5d_devices_cand`）下相对 T5c（`fa0780d`）全部字段（含 `params`、`inversion`、`status`、`flags`、`input_warnings`、`base`、`base_diagnostics`、`holder_share`、`film_box`、`display_sha256`、`preview_image_sha256`）**逐位相同**（`device_diffs = []`）。
