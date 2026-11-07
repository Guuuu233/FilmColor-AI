# FilmColor AI T5e 实验与验证报告（2026-11-07）

## 0. 分支合并与工作区状态

1. **分支合并与基准核验**：
   - 核对 `main` = `fa0780dd56489dc644dbec62a7d2c549a61689d3`（T5c）、`t5d` = `ad88466426060d08bd9f0627cd13e3b18b5ff8f8`（T5d）后，经 `git checkout main && git merge --ff-only t5d` 将 `main` 快进至 `ad88466`，并基于该提交创建 `t5e` 分支。
   - 用户未提交的 `src/filmcolor/paths.py` 全程未改动，SHA-256 保持 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`。
   - 所有真图评测均在独立的 `FILMCOLOR_DATA_ROOT` 下运行，且对原始图像调用 `filmcolor.frozen.check_source(path, allow_frozen=False)`。

---

## 1. 本轮两项确定性改动

### 1.1 改动 A：无边框扫描负片触边晴空高光涂灰豁免（`src/filmcolor/holder.py`）
- **问题根因**：`A342.tif`、`A502.tif`、`A495.tif` 为无外框扫描 TIFF（`film_box(holder) == [0, 0, H, W]`，`holder_share` 为 `0.0408 / 0.0128 / 0.0087 < 0.15`）。这三帧中央 50% 白点估计区位于树荫或凉棚下，而顶边 25% 区域有树冠缝隙透出的明亮晴空紧贴图像上缘，反相后正片达到 `s1 > 1.25, s2 > 8.0` 并连通顶边进入 `holder_mask(P)`。T5a 的 `scene_highlight_components(holder)` 在 `float(holder.mean()) < 0.15` 时直接返回全 `False`，导致白色天空中心被涂成 `HOLDER_GREY = 0.02` 黑斑。
- **实现**：扩展 `src/filmcolor/holder.py` 的 `scene_highlight_components(holder)`：当 `low_share = float(holder.mean()) < 0.15` 时，若 `(t, l, h, w) == (0, 0, *holder.shape)` 且四边无跨度 `>= max_edge_span`（`0.60`）的片夹条带，则将仅接触单条边（`sum(touch) == 1`）且满足 `(d_ratio >= 0.10 and sp >= 0.18)` 或 `(d_ratio >= 0.04 and sp >= 0.30)` 的连通分量及同一天空边上的单边碎块标记为场景高光予以释放，不涂 `HOLDER_GREY`。

### 1.2 改动 B：蓝通道异常高白点比例钳制（`src/filmcolor/invert.py`）
- **问题根因**：`A831.tif` 与 `A832.tif`（Sony ILCE-6700 8-bit sRGB 负片 TIFF）在树枝间亮天空高光处蓝通道逼近 8-bit 低量化阶，将 `16×16` 块均值 99.8% 白点 `d_hi_B` 推高至 `1.8028 D`（`1.9921 * d_hi_G`）与 `1.3373 D`（`2.1182 * d_hi_G`），漏过 T4t 的 `max(2.5, 2.2 * d_hi[1])` 门限，导致正片全图蓝通道被压低偏黄绿。
- **实现**：扩展 `src/filmcolor/invert.py` 中 `block_white_point` 的 `channel_cap`，当 `d_hi[1] >= 0.55 and d_hi[2] > 1.95 * d_hi[1]` 时同样触发 `channel_cap["B"]` 并将 `d_hi[2]` 钳制为 `1.45 * d_hi[1]`。

---

## 2. 单元测试与目标样本验证

### 2.1 全量单元测试
- 新增合成单元测试：
  - `tests/test_preview.py::test_t5e_borderless_sky_highlight_not_painted_holder_grey`
  - `tests/test_invert.py::test_t5e_block_white_point_caps_inflated_blue_ratio`
- 全量 `pytest` 结果：**`451 passed in 355.70s`**（峰值内存 `3.09 GiB`，退出码 `0`）。

### 2.2 8 帧目标与保护样本实测核验（900 px 预览，独立 `FILMCOLOR_DATA_ROOT`）

| 样本 | 指标 | T5d (`ad88466`) | T5e 实测 |
|---|---|---|---|
| `A342.tif` | `painted_holder_px` / `scene_highlight_px` / `display_sha256` | `17574` / `0` / `8e684c3c...` | **`393` / `17181` / `cb11d943...`** |
| `A502.tif` | `painted_holder_px` / `scene_highlight_px` / `display_sha256` | `5540` / `0` / `fe786448...` | **`439` / `5101` / `078b08f8...`** |
| `A495.tif` | `painted_holder_px` / `scene_highlight_px` / `display_sha256` | `3749` / `0` / `a3255541...` | **`371` / `3378` / `77beced1...`** |
| `A831.tif` | `d_hi` / `central_q50` / `display_sha256` | `[0.3926, 0.9050, 1.8028]` / `[0.4508, 0.5345, 0.3612]` / `63096575...` | **`[0.3926, 0.9050, 1.3122]` / `[0.4384, 0.5249, 0.4269]` / `76d7ea7a...`** (`channel_cap.B = 1.8028`) |
| `A832.tif` | `d_hi` / `central_q50` / `display_sha256` | `[0.2135, 0.6313, 1.3373]` / `[0.5186, 0.5167, 0.3022]` / `f6736a7d...` | **`[0.2135, 0.6313, 0.9154]` / `[0.5054, 0.5158, 0.3828]` / `81dd0468...`** (`channel_cap.B = 1.3373`) |
| `A830.tif` / `A847.tif` / `B096.tif` | `d_hi` / `painted_holder_px` / `display_sha256` | `0d7da775...` / `d2d87047...` / `146e74ce...` | **完全一致** |

---

## 3. 基准集（315 条记录）与设备集（61 条记录）零回归核验

- **315 条基准记录**（`s3_dev_frames.json` 191 帧 × 5 预设 + `bn_rb_frames.json` 124 帧 `standard`）：在独立 `FILMCOLOR_DATA_ROOT`（`/tmp/fcroot_t5e_bench_cand`）下相对 T5d（`ad88466`）除耗时字段（`seconds` / `estimate_seconds`）外**逐位相同**（`benchmark_non_timing_diffs = {"a": [], "b": [], "cross": []}`）。
- **61 条设备记录**（`a1296`、`targets`、`new_targets`、`acros`、`tiff`、`new_tiff`、`special`）：在独立 `FILMCOLOR_DATA_ROOT`（`/tmp/fcroot_t5e_devices_cand`）下相对 T5d（`ad88466`）全部字段（含 `params`、`inversion`、`status`、`flags`、`input_warnings`、`base`、`base_diagnostics`、`holder_share`、`film_box`、`display_sha256`、`preview_image_sha256`）**逐位相同**（`device_diffs = []`）。
