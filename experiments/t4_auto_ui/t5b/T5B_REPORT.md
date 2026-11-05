# T5b：相机 RAW 削顶片基剔除、整卷片基保护注释更正与素材极性清单

正式验收 `acceptance.all_passed=true`，13 项门禁全部通过。main 经核对仍为 `acee65d`、t5a 仍为 `1c4c9aa` 后，以 `--ff-only` 快进到 T5a `1c4c9aa10e843b742513dbbb5d84cd312494572f`，再从它创建 t5b。未 push。用户的 `src/filmcolor/paths.py` 本地修改保持原 SHA-256 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`，没有暂存或提交。

候选源码指纹 `0c3e44c044f44326ffb25ec14c48c762252d63e7edfcda53e04c6a94cb9982eb`；隔离 T5a 基线 `2377cf08f109683554309330b0450a03ab3c50c78c5f3cf69059bc62e258a720`，与 T5a 正式候选一致。两者都含同一份用户 paths.py 运行覆盖。noref 未改，`NOREF_V3["rev"]` 仍为 `t4z-2026-11-05`，五个预设快照与 T5a 字节相同。改动 A 改变 `_scan_one` 的输出，所以 `State._disk_key` 前缀改为 `t5b|`，`THUMB_REV` 改为 `t5b`。

## 实现

- **A 相机 RAW 削顶片基剔除（base.py）**：按任务书附录的研究端原型 diff 原样实现，`git apply` 无冲突。新增 `CLIPPED_BASE_MAX = 0.99`。在 bare-backlight 判据之前计算 `clipped_base = near_sat_blocks > 0 and max(rgb) >= 0.99`；成立时与 bare-backlight 一样，只保留 `Y <= 2.5 × p90_Y` 的块重新做共识。诊断里记 `clipped_base_excluded=True`，不记 `bare_backlight_excluded`，并加一条英文 warning。`near_sat_blocks` 只在相机 RAW 上可能大于 0：扫描仪路径传 `exclude_near_saturated=False`，`app._scan_one` 传 `not is_scan`，所以扫描仪不受影响。
  - 已知副作用：`_route_roll_nat` 的回退判据只看 `mode_seed / relaxed_uniformity / bare_backlight_excluded`，不看新键。因此命中新规则的帧不会因为这一项被自动改走 roll_nat。这也包括一种情况：一帧在 T5a 下同时满足 bare-backlight 判据（记 `bare_backlight_excluded`，因而回退 roll_nat），在 T5b 下会改记 `clipped_base_excluded`，不再回退。511 个可读相机 RAW 中没有这样的帧（T5a 带 `bare_backlight_excluded` 的只有 A1051、A1265、A1296，都没有近饱和块）。三帧 A7R4 都是单帧打开，默认片基 method 仍为 `auto`。这是原型和任务书伪代码的行为，本轮没有改动。
  - 规则成立后没有检查重估结果。审查用合成块复现了两种边缘情况：
    - 削顶光（max 通道 0.99–0.9985，近饱和剔除去不掉）占有效块约 12% 以上时，p90 落在削顶光上，`2.5 × p90` 一块也剔不掉，重估得到同一个削顶片基。此时仍会记 `clipped_base_excluded=True` 并加 warning，诊断与结果不符；片基本身与 T5a 相同，不会更差。
    - 剔除后均匀块不足 10 个时，规则静默不生效，没有诊断也没有 warning，与 T5a 相同。
    - 真实素材里两种情况都没有出现：511 个可读相机 RAW 中命中规则的只有三帧 A7R4，重估后片基最大通道为 0.21–0.30。如果将来出现第一种情况，本轮盲扫门禁（`clipped_hits` 必须等于变化帧）会失败。是否加一个“重估后仍 ≥ 0.99 就保留原估计并改记 unresolved”的后验检查，由研究端决定；它不改变本轮任何证据。
  - 任务书正文写“另加一条中文 / 英文 warning”，附录原型只有一条英文。本轮按原型只加了英文；`estimate_base_blocks` 的其他 warning 也都是英文，界面原样显示。
- **B 注释更正（app.py）**：`ROLL_BASE_MAX` 的注释按任务书 §3 改为“整卷片基换算后高于传感器白点的不是片基，出现在误当负片的正片上（反转冲洗的 foma pan100、Eastman 5294 反转片）；基准卷片基最大 1.072”，删去了 “Program AE copy rolls vary the aperture”。阈值 1.25 和代码都没改。
  - 残留：`app.py` 里那条中文提示仍写“（可能光圈或光源有变化）”。`tests/test_app.py` 中 T5a 测试的 docstring 也还写着 Program AE。任务书只要求改注释，这两处本轮没有动。
- **C 素材极性清单（只写数据）**：新增 `data/manifests/material_polarity_20261105.json`，共 9 条，由 `scripts/t5b_material_polarity.py` 生成。
  - 5 个单文件（A850、A851、A852、B094、B040）为 `positive_image`，带 SHA-256。SHA-256 经 `check_source(allow_frozen=False)` 计算，只读取哈希，不解码图像。
  - 4 个目录（Eastman 5294 三卷、foma pan100）为 `reversal_film`，带 NEF 数：21 / 39 / 39 / 38。
  - 每条都有 `evidence` 和 `checked_by`。d_hi_G 取自 T5a 提交的设备记录和 Batch 记录。foma 条目的 NCC 文字沿用任务书的取整值（0.84–0.98，≥ 0.15）；主会话另外人工读了研究端 `foma_polarity_registration.json`（SHA-256 `89813095…`）核对：26 张，0.841–0.977，与第二候选的差距 ≥ 0.157。这次核对没有写进脚本。
  - 素材没有移动。`frozen.MANIFESTS` 只收 `frozen_test_*.json`，新文件不影响冻结守卫。

## 基准与设备

**基准**：308 帧（Dev 185 × 5 个预设 + Check 123 × standard），315 条记录全部重跑。
- 运行设置：`FILMCOLOR_DATA_ROOT=/tmp/fcroot_t5b_candidate`，这是全新的独立缓存根（§0.6）。
- 去掉耗时字段 `seconds` / `estimate_seconds` 后，每条记录都与 T5a 已提交结果逐位相同。六组的 better / worse 均为 0，ΔE 最大差 0.0。
- 309 条非错误记录的 `d_hi`、白点、spike clamp、`base`、`base_diagnostics` 零差异。五张保护帧逐位不变。没有一帧出现 `clipped_base_excluded`。
- 新规则检查的是单帧片基估计 `own_rgb`。它在基准帧上的最大通道最大为 0.877（`nlp:_DSF8948`），低于 0.99。渲染所用片基最大为 0.978（`ext:A135`）。这个值来自 roll_nat，不经过本规则。研究端报告的 0.877 指的是前者。
- Dev 各预设的 median / mean / p90 不变：standard 8.86 / 9.30 / 13.88，natural 9.18 / 9.87 / 14.55，portrait 9.38 / 10.40 / 15.91，vivid 9.04 / 9.78 / 14.75，print_classic 8.27 / 8.95 / 13.47；Check 为 13.52 / 13.98 / 19.95。

**设备**：61 条记录，即 A1296 1 条，targets、new_targets 各 8 条，acros 28 条，tiff 8 条，new_tiff 7 条，special 1 条。基线和候选各用一个新缓存根（`/tmp/fcroot_t5b_devices_<label>`）。
- 隔离 T5a 源码的基线探针与 T5a 提交的 `devices_candidate.json` 逐位相同。比较字段包括 params、inversion、status、flags、report、base、diagnostics、显示哈希、预览图哈希和 core q50；acros 整卷分析也相同。
- 候选与基线 61 条全部逐位相同。61 条显示哈希全部相同；生成了预览图的 26 条（A1296、targets、new_targets、new_tiff、special 和 B029）预览图哈希也相同。
- 有近饱和剔除的设备帧是 A262（1977 块）、A263（2246 块）和 A1061（302 块）。规则检查的单帧片基估计 `own_rgb` 最大通道分别为 0.276、0.914、0.293，都低于 0.99，所以都不触发；这一比较已写进门禁。

## Sony ILCE-7RM4 三帧

| 帧 | 片基 T5a → T5b | T5b R/G, B/G | d_hi_G | core sRGB q50 | EV |
|---|---|---|---|---|---|
| A1298 | [0.456, 0.999, 0.775] → [0.171, 0.212, 0.085] | 0.81, 0.40 | 1.674 → 1.000 | [0.367, 0.393, 0.388] → [0.443, 0.424, 0.338] | 0.555 → 1.227 |
| A1299 | [0.584, 0.999, 0.979] → [0.226, 0.300, 0.114] | 0.75, 0.38 | 1.581 → 1.059 | [0.472, 0.419, 0.438] → [0.490, 0.436, 0.363] | −0.135 → 0.086 |
| A1300 | [0.465, 0.999, 0.792] → [0.176, 0.214, 0.091] | 0.82, 0.42 | 1.539 → 0.871 | [0.438, 0.455, 0.478] → [0.470, 0.468, 0.486] | −0.200 → 0.168 |

- 运行方式：按 app 的方式打开，`State.open(..., "color")`，max_side 1600，基线和候选各用独立缓存根。
- 三帧状态前后都是 ok。片基 G 白点与任务书 §2 的研究端数字一致。表中 T5a 片基的 G 为 0.9993，是渲染路径（全分辨率）的值；扫描路径的 `own_rgb` 是 0.9979–0.9982，两者都 ≥ 0.99。
- 主会话看了 `before_after.jpg`（本地文件，未提交；上排 T5a，下排 T5b）：三帧的灰雾都减轻，反差恢复。
- A1298 左侧片边有一条蓝色斑点带。在 900 px 预览图上，它位于 x 128–151、y 260–471。film_box 左边界在 x 125，所以这条带在 film_box 内侧那条窄窄的片边上，不在景物区域。按 B > R + 60 且 B > 120 统计，这一带的蓝色像素 T5a 有 174 个，T5b 有 521 个（`sony_a7r4.json` 的 `edge_band_blue_px`，由 `t5b_sony_a7r4.py --compare` 在两张带哈希的本地预览图上计算）。T5a 在同一位置涂成黑块，只露出下角一团蓝点；T5b 把这条片边还原成亮的，斑点也就显露出来。斑点的成因（可能是片边划痕或灰尘的反相）本轮没有查。
- 研究端已经指出：A1300 灰卡的偏蓝不是片基造成的，本轮不处理。

## 盲测相机 RAW 片基扫描（`blind_raw_scan.json`）

`scripts/t5b_blind_raw_scan.py` 对每个文件只读一次，读法与 `_scan_one` 相同，然后在同一组块上分别运行 T5a 的 `estimate_base_blocks`（取自隔离树，与 `git show 1c4c9aa` 字节相同）和 T5b 的版本。

- **必查集**：Batch 19–22 彩负目录下的 RAW / DNG，加上 A1298、A1300，共 38 个。其中 Batch 19 的 4 个（A1108、A1130、A1232、A1233）是扫描仪 DNG，相机 RAW 34 个。38 个全部比较完，只有 A1298、A1299、A1300 三帧变化。
- **扩展集**：`01_彩色负片_未反相`、`02_黑白负片_未反相`、`A7R5_RAW` 下的全部 RAW / DNG，共 666 个。其中 547 个做了比较：彩负 455 个（含 31 个扫描仪 DNG）、黑白 86 个（含 5 个扫描仪 DNG）、A7R5 6 个。新规则只可能在其中 511 个相机 RAW 上触发；36 个扫描仪 DNG 走 `exclude_near_saturated=False`，只作不变性对照。另有 119 个是冻结集，只做了 `check_source`，没有解码。解码失败 0 个。
  - 变化的只有上面三帧，没有意外变化。`clipped_base_excluded` 也只命中这三帧。T5a 片基最大通道 ≥ 0.99 的同样只有这三帧。
- **与研究端对照**：研究端 `blind_raw_base_t5a.json` 共 235 个文件，全部在本轮扫描内；其中 219 个带 T5a 片基，与本轮 T5a 片基精确相等（`==`），没有不同的。研究端另有 16 个多平面 DNG 报“only single-plane CFA”；本轮用 `is_scanner_dng` 识别它们，走扫描仪路径读取，两版结果相同。
- 最接近门限的未触发帧：A1233.DNG 0.975、A1043.DNG 0.958（扫描仪 DNG，不适用本规则），A1322.ARW 0.935（没有近饱和块）。有近饱和剔除的 14 帧里，除这三帧外片基最大通道最高的是 A263（0.914）。

## 测试

新增 3 项测试，加在 `tests/test_base.py`；另把 1 项 THUMB_REV 断言改为 t5b。“不触发”和“扫描仪路径”两项按任务书 §5 断言片基与 T5a 逐位相同（与 float32 块值精确相等，在 T5a 树上同样成立）。完整 pytest 实测 **444 passed**，用时 309.01 秒，退出码 0（`tests.log` 末行 `exit=0`，`test_evidence.json` 的 `returncode`），collect-only 计数同为 444。

把这些测试放到隔离的 T5a 源码上运行，结果是 2 项失败、2 项通过：
- 失败：`test_t5b_clipped_base_excluded`。T5a 没有 `CLIPPED_BASE_MAX`，在同一合成块上返回 G 0.9979 的削顶片基。
- 失败：THUMB_REV 断言。
- 通过：“没有近饱和块时不触发”和“扫描仪路径不变”这两项。它们断言的是 T5b 与 T5a 逐位相同的行为，所以在 T5a 上应当通过。

证据见 `baseline_failure_evidence.json`。

HTTP 服务用 `python -m filmcolor.app` 启动，端口 62873，缓存根是独立的 `/tmp/fcroot_t5b_service`。实测结果：
- `/api/open` 返回 ok，`config_rev` 为 `t4z-2026-11-05`。
- `/api/render` 的 PNG 哈希与 T5a 服务检查相同（`b3f4c6c3…`）。
- 缩略图缓存名为 `thumb_color_t5b_<key>.png`（服务缓存根里实际生成的文件，名字由 `THUMB_REV` 决定；没有另写进 JSON）。

检查完服务已关闭。

## 未解决与需确认

- 新规则只看“相机 RAW 有近饱和块、且片基某通道 ≥ 0.99”。如果一张真负片翻拍时曝光过度，片基本身接近满量程，同时画面里又有近饱和块，规则也会把片基改走背光剔除。本轮在 511 个可读的相机 RAW 上没有找到这样的帧（另 36 个扫描仪 DNG 不适用本规则），但没有覆盖冻结集，也没有覆盖未来的新机型。
- 规则成立后不检查重估结果，见“实现”一节的边缘情况。
- 命中新规则的帧不会因为这一项自动改走 roll_nat，见上文“已知副作用”。
- A1300 灰卡偏蓝的来源未查。
- `app.py` 中“整卷片基越界”中文提示里的“可能光圈或光源有变化”，以及 T5a 测试 docstring 里的 Program AE 成因，仍是旧说法。是否一并更正由研究端决定。

## 保护与交付

原片只读。每次读取原片前都先调用 `check_source(path, allow_frozen=False)`，冻结集图像不解码。所有基线 / 候选对比都给每个源码树单独设了 `FILMCOLOR_DATA_ROOT`，共用缓存会让候选读到基线缓存的片基估计。

`t5b_final_audit.py` 实跑通过（输出见 `final_audit.json`），复核了以下内容：
- 分支为 t5b，main 为 `1c4c9aa`，paths.py 哈希不变且未暂存；
- 150 份历史 JSON 与各自提交逐字节相同，T5a 目录的 19 份 JSON 与 `1c4c9aa` 逐字节相同，5 份冻结清单的哈希不变；
- 源码、测试和脚本在验收之后没有改动；
- 提交清单不含 paths.py，覆盖相对 `1c4c9aa` 的全部改动；JSON 中没有邮箱和相机序列号字段。

全程用 Python `-I` 运行，不读 pickle，不训练，不用 Colab，不加依赖，不上传。图片和日志不提交。

交付物在 `experiments/t4_auto_ui/t5b/`：
- TABLES.md、acceptance.json、T5B_REPORT.md
- candidate_a/b/cross.json、devices_baseline/candidate.json
- sony_a7r4*.json、blind_raw_scan.json
- 测试、服务、资源和脚本哈希证据

另有 `data/manifests/material_polarity_20261105.json`。
