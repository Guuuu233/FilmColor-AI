# FilmColor AI T5f 实验与验证报告（2026-11-08）

## 0. 分支合并与工作区状态

1. **分支合并与基准核验**：
   - 核对 `main` = `ad88466426060d08bd9f0627cd13e3b18b5ff8f8`（T5d）、`t5e` = `2b48500b3e9ba6fbfeb7cf1a55879725c3937191`（T5e）后，经 `git checkout main && git merge --ff-only t5e` 将 `main` 快进至 `2b48500`，并基于该提交创建 `t5f` 分支。
   - 用户未提交的 `src/filmcolor/paths.py` 全程未改动，SHA-256 保持 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`。
   - 所有真图评测均在独立的 `FILMCOLOR_DATA_ROOT` 下运行，且对原始图像调用 `filmcolor.frozen.check_source(path, allow_frozen=False)`。

---

## 1. 本轮两项确定性改动（`src/filmcolor/invert.py`）

### 1.1 改动 A：细碎子块高光白点救援（`block_white_point`）
- **问题根因**：`A623.tif`（Sony ILCE-6700 森林树冠负片）中央区域散布大量直径 `3–8 px` 的亮天空缝隙（占中央区域 `> 2%` 面积），在 `16×16` 块均值中被周围暗树叶平均稀释（`d_hi = [0.3279, 0.7007, 0.8581]`），而下游 `noref.estimation_input` 在 `4×4` 块均值上统计 `st["p99"]`（`4×4` 99.0% 密度达 `[0.3810, 0.8237, 0.9400]`），导致正片 `st["p99"] = 3.6002`（超出白点 `1.85 stops > 1.50 stops`）触发 `white_point_invalid`，回退 `ev = 0`。
- **实现**：在 `block_white_point` 中，当 `block >= 16` 且未触发 `holder_blocks_excluded` 时，在有效 `ok` 块的 `4×4` 子块（`b4 = block // 4`）均值上计算密度 99.0% 分位 `d4_990`；若 `d_hi[1] >= 0.45`、`d4_990[1] - d_hi[1] >= 0.10` 且 `d4_990[1] / d_hi[1] >= 1.15`，采用 `4×4` 子块的 `99.5%` 分位更新 `d_hi`，并记录于 `info["fine_highlight_rescue"]`。

### 1.2 改动 B：无遮挡帧 `zero_escape` 预检门槛收紧（`invert_negative`）
- **问题根因**：`A621.tif` 与 `A623.tif` 的树冠亮天空缝隙处 R、G 略超白点（`s_mid > 1.15`），而 8-bit sRGB 蓝通道触底 `0..1/255`，归一化蓝密度 `s_B ≈ 1.80..3.61 < 5.0`，使 `invert_negative` 中 `Dm` 的 `zero_escape` 预检未达 `0.1%` 占比，爆出电光紫蓝色块。
- **实现**：在 `invert_negative` 的 `Dm` 全局 `zero_escape` 预检中，设置 `max_thr = np.float32(5.0 if (white_info or {}).get("holder_blocks_excluded") else 2.0)`，对无边缘遮挡块的帧将 `s_max` 逃逸门槛从 `5.0` 收紧至 `2.0`，保留含边缘遮挡块的 `A232.NEF`、`B096.tif` 等样本为 `5.0`。

---

## 2. 单元测试与目标样本验证

### 2.1 全量单元测试
- 新增合成单元测试：
  - `tests/test_invert.py::test_t5f_fine_highlight_rescue_recovers_subblock_canopy_gaps`
  - `tests/test_invert.py::test_t5f_zero_escape_clamps_moderate_8bit_floor_spikes`
- 全量 `pytest` 结果：**`453 passed in 342.37s`**（峰值内存 `3.40 GiB`，退出码 `0`）。

### 2.2 8 帧目标与保护样本实测核验（900 px 预览，独立 `FILMCOLOR_DATA_ROOT`）

与研究端原型 `t5f_proto_checks.json` 逐字段比对 **100% 逐位一致**：

| 样本 | 指标 | T5e (`2b48500`) | T5f 实测 |
|---|---|---|---|
| `A623.tif` | `status` / `d_hi` | `manual_review_required` / `[0.3279, 0.7007, 0.8581]` | **`ok` / `[0.4342, 0.9093, 1.1256]`** (`fine_highlight_rescue` 触发) |
| `A623.tif` | `stats_p99` / `spike_clamp_px` / `core_q50` / `display_sha256` | `3.6002` / `123167` / `[0.2580, 0.3321, 0.2186]` / `1780bb14...` | **`1.0193` / `205313` / `[0.3075, 0.3724, 0.2627]` / `fe0acefa...`** |
| `A621.tif` | `stats_p99` / `spike_clamp_px` / `core_q50` / `display_sha256` | `1.0710` / `41055` / `[0.2689, 0.3523, 0.2979]` / `efac8207...` | **`0.9298` / `177362` / `[0.2835, 0.3714, 0.3126]` / `72ede5ad...`** |
| `A622.tif` / `A232.NEF` / `A1007.CR2` | `display_sha256` | `49b9e038...` / `65a69461...` / `6b612172...` | **完全一致** |
| `A305.tif` / `B096.tif` / `B034.tiff` | `display_sha256` | `d1ea0f29...` / `146e74ce...` / `d7ed1a13...` | **完全一致** |

---

## 3. 基准集（315 条记录）与设备集（61 条记录）核验

- **315 条基准记录**（`s3_dev_frames.json` 191 帧 × 5 预设 + `bn_rb_frames.json` 124 帧 `standard`）：在独立 `FILMCOLOR_DATA_ROOT`（`/tmp/fcroot_t5f_bench_cand`）下相对 T5e（`2b48500`）除耗时字段（`seconds` / `estimate_seconds`）外**全部 315/315 逐位相同**（`benchmark_non_timing_diffs = {"a": [], "b": [], "cross": []}`）。
- **61 条设备记录**（`/tmp/fcroot_t5f_devices_cand`）：
  - **60 / 61 条记录**（含 `a1296`、`targets`、`new_targets`、`special`、`acros`、`tiff` 全部样本，以及 `new_tiff` 中除 `A229.TIF` 外的其余 6 帧）全部字段（含 `params`、`inversion`、`status`、`flags`、`input_warnings`、`base`、`base_diagnostics`、`holder_share`、`film_box`、`display_sha256`、`preview_image_sha256`）相对 T5e **逐位相同**。
  - **唯一差异样本 `new_tiff / A229.TIF`**（蓝通道丢失的损坏扫描输入，已触发 `input_warnings = [{"code": "input_channel_empty", "channel": "B"}]` 与 `channel_cap = {"B": 6.0}`）：因全幅蓝通道死零使 `s_B = 6.0 / 3.0274 = 1.9819 > 1.15`（T5e 下仅靠单通道/分块局部逃逸钳制了 `20,947,080` 像素，全局 `Dm` 在 `s_max > 5.0` 下占比为 `0.0%`），在 `max_thr = 2.0` 下中央区域 `0.1588%`（`> 0.1%`）红通道触底像素（`s_R = 6.0 / 2.9514 = 2.0329 > 2.0` 且 `s_G <= 1.6`）激活全局 `zero_escape = True`，将其余局部占比 `< 1%` 分块中的残余零截断蓝通道尖峰一并钳制（`spike_clamp.pixels` 从 `20,947,080` 增至 `20,984,103`，`upper_band_max[2]` 从 `165.68` 降至 `2.62`，`flags` 消除伪高光压暗 `ev_lowered_for_highlights:0.6`，`ev` 从 `-0.0328` 变为 `+0.2421`，`display_sha256` 从 `33d66f31...` 变为 `96770189...`，`d_hi` 与 `input_warnings` 保持不变）。经确认保留 `t5f_proto.diff` 原样并如实记录该损坏通道样本的变化。
