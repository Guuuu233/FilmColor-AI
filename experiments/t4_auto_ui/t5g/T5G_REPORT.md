# FilmColor AI T5g 实验与验证报告（2026-11-08）

## 0. 分支合并与工作区状态

1. **分支合并与基准核验**：
   - 核对 `main` = `2b48500b3e9ba6fbfeb7cf1a55879725c3937191`（T5e）、`t5f` = `195e828e5c6199d9e8d1a1048fc66dbaa1aca4b0`（T5f）后，经 `git checkout main && git merge --ff-only t5f` 将 `main` 快进至 `195e828`，并基于该提交创建 `t5g` 分支。
   - 用户未提交的 `src/filmcolor/paths.py` 全程未改动、未暂存，SHA-256 保持 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`。
   - 所有真图评测均在独立的 `FILMCOLOR_DATA_ROOT` 下运行，且对原始图像调用 `filmcolor.frozen.check_source(path, allow_frozen=False)`。

---

## 1. 本轮三项确定性改动

### 1.1 改动 A：单色/黑白整卷 `natural_roll_base` 密度差混合与青蓝暗部片基松弛（`src/filmcolor/base.py`）
- **单色 `roll_nat` 密度差混合**：在 `natural_roll_base` 中，当 `own_rgb is not None` 时同时读取 `dev = base_deviation(own_rgb, roll_rgb)` 的 `dc = dev["chroma_part"]` 与 `lv = dev["level_diff"]`，令 `eff = max(dc, lv if dc < 0.02 else 0.0)`，使无色差（`dc < 0.02`）但存在密度差（`lv >= 0.04`）的单色/黑白底片向 `d_lo_dq` 平滑混合（受 `d_lo_b1 + max_black_pull_d` `+0.04 D` 上限保护）。
- **C-41 青蓝暗部片基松弛**：在 `estimate_base_blocks` 的 `relaxed_uniformity` 判定中新增 `cyan_shadow = (rgb_arr[0] >= 1.8 * rgb_arr[1] and rgb_arr[2] >= 1.10 * rgb_arr[1] and p98[0] >= 1.01 * rgb_arr[0] and p98[1] >= 1.00 * rgb_arr[1])`，当初始均匀候选块落在画面内青蓝色深暗部（`R >= 1.8 G` 且 `B >= 1.10 G`）时放宽 `max_rel_std` 至 `0.25`，恢复真实橙罩边缘片基（修复 `A1604.dng`、`A1606.dng`、`A1609.dng` 及 `blueneg:19941220D-18-kamakura`）。

### 1.2 改动 B：绿主导背光与白点下的 `blue_white_guard` 门控（`src/filmcolor/invert.py`）
- 在 `blue_white_guard` 中，当传入 4 参数橙色片基守卫（`len(values) == 4 and base is not None`）且 `base[0] <= 0.70 * base[1] and d[0] <= 0.70 * d[1]` 时直接返回原始 `d_hi`（`applied = False`），修复 `A1328.NEF` 因绿主导背光被误推高 `d_hi_B` 导致的黄褐偏色，同时保留 `A239.ORF`（`d_hi_R / d_hi_G = 0.911 > 0.70`）等样本的既有行为。

### 1.3 改动 C：单色（黑白）高底座正片黑位拉回 `bw_lifted_floor`（`src/filmcolor/noref.py`）
- 在 `estimate_params_v3` 的 Step 7b `auto_blacks` 分支之后新增 `elif (p90 <= 1e-4 and span_d_g is not None and span_d_g < 0.80 and base_p["ev"] <= 0.35 and tc["p98_dark_d"] <= 1.32 and p2 > 0.25):`，按 `ex_p2 = float(p2 - 0.22)` 下调 `ev`（`clip(ex_p2 * 1.40, 0.0, 0.45)`）与 `blacks`（`clip(ex_p2 * 1.00, 0.0, 0.28)`）并提升 `contrast`（`clip(ex_p2 * 0.80, 0.0, 0.25)`，上限 `1.65`），记录于 `out["diagnostics"]["bw_lifted_floor"]`。

---

## 2. 单元测试与 12 张目标帧验证

### 2.1 全量单元测试
- 新增合成单元测试：
  - `tests/test_base.py::test_t5g_natural_roll_base_level_diff_and_cyan_shadow_relaxation`
  - `tests/test_invert.py::test_t5g_blue_white_guard_skips_green_dominant_base_and_highlights`
  - `tests/test_noref.py::test_t5g_bw_lifted_floor_pulls_milky_monochrome_shadows`
- 全量 `pytest` 结果：**`456 passed in 355.43s`**（峰值内存 `3.05 GiB`，退出码 `0`）。

### 2.2 12 张 T5g 目标帧实测核验（独立缓存 `/tmp/fcroot_t5g_targets`）

| 目标帧 | 模式 / 所在组 | `base_method` | `d_hi` (`[R, G, B]`) | 关键渲染参数 (`ev / contrast / blacks`) | `core_srgb_q50` | `core_srgb_q05` | 实测 `display_sha256` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `A1328.NEF` | `color` (`A1328..1333`) | `auto` | `[0.294006, 0.525446, 0.431402]` | `ev=-0.1175, T=-0.0625/+0.1032, blk=-0.0346` | `[0.6684, 0.5473, 0.5531]` | `[0.2138, 0.1751, 0.1771]` | `c92271cc65d9292f5fe5abc21f6acf91a45f7f97a7eac43c51c4bb10077f1750` |
| `A1604.dng` | `color` (`A1602..1610`) | `auto` | `[0.453518, 0.431812, 0.056107]` | `ev=-0.6500, T=-0.0818/+0.1240, blk=-0.1800` | `[0.6878, 0.6911, 0.5487]` | `[0.3522, 0.3453, 0.3169]` | `c3413238722a1f53cba2a0ef49634cb2dd5a0734a98b4eda1220ccd195fce6ca` |
| `A1606.dng` | `color` (`A1602..1610`) | `auto` | `[0.486859, 0.508585, 0.129058]` | `ev=-0.0495, T=-0.0609/+0.0045, blk=-0.0514` | `[0.6589, 0.6582, 0.6477]` | `[0.3667, 0.3286, 0.3644]` | `f367faad837ed277b09f1d540a76637cf7003823706f41f88f489f18c4745259` |
| `A1609.dng` | `color` (`A1602..1610`) | `auto` | `[0.987154, 0.835614, 0.235807]` | `ev=+0.3227, T=-0.0372/+0.1156, blk=0.0000` | `[0.5877, 0.6735, 0.6505]` | `[0.2664, 0.2582, 0.2509]` | `7ee1f5d6236c35e1b3eb6fce6650e4e2602987d6fc64f3f95134360ecf16f931` |
| `B052.NEF` | `bw` (`APX 400`) | `roll_nat` | `[0.440215, 0.440215, 0.440215]` | `ev=-0.3990, con=1.4316, blk=-0.2843` | `[0.5297, 0.5297, 0.5297]` | `[0.3552, 0.3552, 0.3552]` | `ea483acfc9aa984f2876f15fe7b3877b0cfb4e77becc080a4ef4bd7bc7d597b3` |
| `B058.NEF` | `bw` (`APX 400`) | `roll_nat` | `[0.449887, 0.449887, 0.449887]` | `ev=-0.2286, con=1.4090, blk=-0.2463` | `[0.5197, 0.5197, 0.5197]` | `[0.3425, 0.3425, 0.3425]` | `31567f09ac5681470d126534bdfba7fd21d1ee5526a2b0e168e1c32741b562cb` |
| `B060.NEF` | `bw` (`APX 400`) | `roll_nat` | `[0.480457, 0.480457, 0.480457]` | `ev=-0.4059, con=1.4458, blk=-0.2618` | `[0.4393, 0.4393, 0.4393]` | `[0.3597, 0.3597, 0.3597]` | `ea4fc022d37b0c999260d7e819f475f7efc19d1c5f380c8ffa04f81c06156903` |
| `B068.NEF` | `bw` (`APX 400`) | `roll_nat` | `[0.534889, 0.534889, 0.534889]` | `ev=+0.1074, con=1.3991, blk=-0.1490` | `[0.5058, 0.5058, 0.5058]` | `[0.3100, 0.3100, 0.3100]` | `50cd45f001da4e837cda6fdcdeb1049a29230d9e4a8ccf5542b91846d49ee04c` |
| `B085.NEF` | `bw` (`APX 400`) | `roll_nat` | `[0.455558, 0.455558, 0.455558]` | `ev=-0.1094, con=1.3530, blk=-0.1707` | `[0.5999, 0.5999, 0.5999]` | `[0.3032, 0.3032, 0.3032]` | `1e850d9e6c3daa99d8746093135e93220f0e1c4909d06b71443fa9d22801c10a` |
| `B003.NEF` | `bw` (`Acros`) | `roll_nat` | `[0.246936, 0.246936, 0.246936]` | `ev=-0.7438, con=1.4679, blk=-0.4099` | `[0.5097, 0.5097, 0.5097]` | `[0.3878, 0.3878, 0.3878]` | `ff518148e3da468575b54253402582948511ef840b7b59c9167f1a1dbbd7184b` |
| `B013.NEF` | `bw` (`Acros`) | `roll_nat` | `[0.431832, 0.431832, 0.431832]` | `ev=-0.4097, con=1.4737, blk=-0.3453` | `[0.5018, 0.5018, 0.5018]` | `[0.3719, 0.3719, 0.3719]` | `4db753335f654c16ef5c84a753d88d4f441cdf20b788c94994402423e67aa795` |
| `B029.NEF` | `bw` (`Acros`) | `roll_nat` | `[0.349061, 0.349061, 0.349061]` | `ev=-0.1352, con=1.4709, blk=-0.2137` | `[0.6653, 0.6653, 0.6653]` | `[0.0200, 0.0200, 0.0200]` | `7d719b1079c679ab29a0e814cf14cf4ef9a3171cd8f15a44700f6584937bc7e1` |

- **说明**：前 11 张目标帧（`A1328.NEF`、`A1604.dng`、`A1606.dng`、`A1609.dng`、`B052.NEF`、`B058.NEF`、`B060.NEF`、`B068.NEF`、`B085.NEF`、`B003.NEF`、`B013.NEF`）的全部字段与 `t5g_proto_checks.json` **100% 逐位一致**。`B029.NEF` 因满足 `t5g_proto.diff` 中 `bw_lifted_floor` 的条件（`p90 = 0.0 <= 1e-4`、`span_d_g = 0.3491 < 0.80`、`ev_before = +0.1640 <= 0.35`、`p98_dark_d = 1.0127 <= 1.32`、`p2 = 0.4337 > 0.25`），实际触发 `bw_lifted_floor`（`ev=-0.1352, con=1.4709, blk=-0.2137`，`display_sha256 = 7d719b1079c679ab29a0e814cf14cf4ef9a3171cd8f15a44700f6584937bc7e1`；`t5g_proto_checks.json` 中记录的 `c09bf03d...` 对应仅应用 `natural_roll_base` 而未触发 `bw_lifted_floor` 的状态），经确认保持 `t5g_proto.diff` 原样并如实记录。

---

## 3. 315 帧基准集与 61 帧设备集验证（独立 `FILMCOLOR_DATA_ROOT`）

### 3.1 315 帧基准集（`/tmp/fcroot_t5g_bench_cand`）
- **`s3_dev`（`candidate_a.json` + `candidate_b.json`，191 帧 `ok` / 185 帧有效色彩评估）**：**191/191 帧与 T5f（`195e828`）除耗时字段外 100% 逐位一致**（`benchmark_non_timing_diffs["a"] = []`, `benchmark_non_timing_diffs["b"] = []`），五大预设 ΔE2000 中位数 / 均值 / P90 保持锁定：
  - `standard`：`8.86 / 9.30 / 13.88`
  - `natural`：`9.18 / 9.87 / 14.55`
  - `portrait`：`9.38 / 10.40 / 15.91`
  - `vivid`：`9.04 / 9.78 / 14.75`
  - `print_classic`：`8.27 / 8.95 / 13.47`
- **`bn_rb` 跨源检查集（`candidate_cross.json`，124 帧）**：**123/124 帧与 T5f 100% 逐位一致**；唯一变化帧 `blueneg:19941220D-18-kamakura` 因 `cyan_shadow` 片基松弛使 `own_rgb` 更贴近整卷片基，`standard` ΔE2000 中位数从 `10.2748` 降至 **`9.8789`**（改善 `-0.3959 ΔE`），124 帧整体 ΔE2000 中位数 / 均值 / P90 为 **`13.49 / 13.97 / 19.94`**（零退步）。

### 3.2 61 帧设备集（`/tmp/fcroot_t5g_devices_cand`）
- **33 / 33 非 Acros 记录**（`a1296` 1 + `targets` 8 + `new_targets` 8 + `tiff` 8 + `new_tiff` 7 + `special` 1）：相对 T5f 全部字段（含 `display_sha256`、`preview_image_sha256`、`params`、`d_hi`）**100% 逐位一致**（`non_acros_device_diffs = []`）。
- **Acros 黑白整卷（`B002.NEF..B029.NEF`，28 帧）**：全部 **28/28** 保持 `status = "ok"`、`flags = []`（仅 `B026.NEF` 保持既有 `["backlit_midtone"]`）、`full_positive_finite = True`；四张目标薄帧 `B003/B011/B016/B017` 的反相跨度分别为 `0.2469 / 0.2419 / 0.2585 / 0.2628 D`（均 `> 0.22 D`），高底座灰雾帧的黑位与反差显著改善。
