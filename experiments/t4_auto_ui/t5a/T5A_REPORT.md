# T5a：片夹涂灰避让过曝窗景、整卷片基越界保护与输入可信度提示

正式验收 `acceptance.all_passed=true`，13 项门禁全部通过。main 经祖先关系核验后由 T4y `4bd014e` 以 `--ff-only` 快进到 T4z `acee65d4428e3f2d71386ee2415410c3b7482b73`，再从它创建 t5a。未 push。用户的 `src/filmcolor/paths.py` 本地修改保持原 SHA-256 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`，没有暂存或提交。

候选源码指纹 `2377cf08f109683554309330b0450a03ab3c50c78c5f3cf69059bc62e258a720`；隔离 T4z 基线 `136a7dbb951ed4c59497adfd2aa789ee0f180246f4e525c6822a78c9253a4756`，与 T4z 正式候选一致。两者都含同一份用户 paths.py 运行覆盖。noref 未改，`NOREF_V3["rev"]` 仍为 `t4z-2026-11-05`，五个预设快照与 T4z 字节相同。`_scan_one` 的输出没有变化，`State._disk_key` 前缀保持 `t4z|`；`THUMB_REV` 改为 `t5a`。

## 实现

- **A 片夹涂灰避让（holder.py / preview.py / app.py）**：新增 `holder.scene_highlight_components`，代码与任务书 §2 一致。只在 `holder.mean() >= 0.15` 时生效：film_box 内深度达到短边 8% 以上、且在所触每条框边上的跨度都不足 60% 的片夹连通块，视为贴边的过曝景物，不涂 `HOLDER_GREY`。`Preview._paint_holder` 改为 `e[holder & ~scene]`，scene 在 Preview 上算一次并缓存；外框涂灰不变。缩略图同样改为 `hm & ~scene_highlight_components(hm)`。`holder` 本身（白点和测光掩膜都用它）不变。导出路径（`pipeline.convert`）本来就不涂片夹灰，无需改动。
- **B 整卷片基越界保护（app.py）**：`ROLL_BASE_MAX = 1.25`。`roll_info` 在算出 `b_roll` 后记录 `roll_implausible = max(b_roll) > 1.25`。`_route_roll_nat` 在帧数和卷警告检查之后，遇到 `roll_implausible` 就返回 False。用户手动选 roll_nat / roll_dq / roll_b1 时照常执行，但片基提示里会加一条：“整卷片基换算到本帧后高于传感器白点（x.xx），本卷的曝光归一化不成立（可能光圈或光源有变化），建议使用单帧片基。”没有加光圈归一化。B0 失败时的自动 roll_nat 回退保留原行为，只附带同一条提示。
- **C 输入可信度提示（pipeline.py / preview.py / app.py / index.html）**：`pipeline.positive` 额外返回中心区（`central_region`，即中心 50%）线性输入的逐通道 p99。`load_preview` 写入 `report["input_center_p99"]` 和 `report["input_warnings"]`，`meta()` 返回 `input_warnings`，界面信息栏显示为“输入提示”。触发规则有两条：`d_hi[1] > 2.40` 给 `input_density_implausible`；某通道 p99 < 0.02 且另一通道 p99 ≥ 0.30 给 `input_channel_empty`。status、params、inversion 都不改。
- 评测脚本 `t4d_source_eval.py` 只增加记录 `input_warnings` 和 `input_center_p99` 两个字段，测量协议不变。它的 SHA-256 因此从 T4z 记录的 `3c74848b…` 变为 `8dd1d9cc…`（见 source_freeze.json / script_sha256.json）。

这些规则都是几何或统计上的启发式判断。片夹规则依靠形状区分窗户和片夹带，不保证所有扫描都分得开。提示只表示输入可能不是可用的负片扫描；没有提示，也不说明颜色正确。

## 基准与设备

308 帧（Dev 185 × 5 个预设 + Check 123 × standard）的 315 条记录全部重跑。除耗时字段 `seconds` / `estimate_seconds` 和两个新字段外，每条记录都与 T4z 已提交结果逐位相同。六组对 T4z 的 better / worse 均为 0，ΔE 最大差 0.0。对 T4o 的 `>.05` 和 `>.25` 退步也都是 0。309 条非错误记录的反演诊断零差异，五张保护帧逐位不变。基准帧上 `input_warnings` 命中 0 帧，所以不需要因基准命中停下。

Dev 各预设 median / mean / p90 不变：standard 8.86 / 9.30 / 13.88，natural 9.18 / 9.87 / 14.55，portrait 9.38 / 10.40 / 15.91，vivid 9.04 / 9.78 / 14.75，print_classic 8.27 / 8.95 / 13.47；Check 13.52 / 13.98 / 19.95。

设备沿用 T4z 设备集：61 条记录（A1296 1、targets 8、new_targets 8、acros 28、tiff 8、new_tiff 7、special 1；acros_roll 是卷分析，不计入）。任务书和研究端写的是 67 条，本仓库的 T4z 设备探针只有 61 条，本轮无法对应多出的 6 条，请研究端核对口径。隔离 T4z 源码的基线探针与 T4z 提交的 `devices_candidate.json` 逐位相同，包括预览图哈希。候选与基线相比，61 条的 params / inversion / status / flags / report（去掉新增键）全部相同。显示渲染哈希只有 B096 变化，core q50 从 0.287 升到 0.377。A1296 全分辨率作用域统计与 T4z 一致。提示只命中 A850（density）和 A229（empty channel）。新门 4（`min(core q50) < 0.03`，或通道 q50 最大与最小之比 > 6）筛出的正是这两帧，而且都带提示。

主会话实际看了 [before_after.jpg](before_after.jpg)（本地文件，未提交）：
- B096 的窗户显示为过曝白，孩子头部周围的黑色硬边消失；
- B029 左侧真正的片夹带仍是灰色；
- A850、A229 的像素不变，只多了提示。

## §3 两卷默认路由

`t5a_roll_routing.py` 在每卷全部 NEF 的符号链接目录上，按 app 的方式运行 `roll_info` 和默认 `open`（film="bw"，与 T4z 盲测一致）。两版的卷分析数值（own / roll base、level_diff、chroma_part）逐位相同，变化的只有路由决定。

| 卷 | 帧 | 卷片基 > 1.25 | 默认 roll_nat 前 → 后 | 改走单帧 |
|---|---:|---:|---|---:|
| foma pan100 | 38 | 24（最大 19.78） | 34 → 10 | 24 |
| Eastman 5294（无后缀） | 21 | 4（最大 2.62） | 17 → 13 | 4 |
| Eastman 5294-1 | 39 | 23 | 0 → 0（卷有警告） | 0 |
| Eastman 5294-2 | 39 | 0 | 0 → 0（卷有警告） | 0 |

改路由帧数与研究端估计的下限 24 / 4 相同。被改的帧恰好是“原本走 roll_nat 且卷片基 > 1.25”的交集，两卷所有帧的状态前后都是 ok。Eastman -1 / -2 两卷原本就因卷警告（达到片基簇的帧只有 2 / 1 帧）不走自动 roll_nat，本轮只做卷分析，没有逐帧打开。foma 卷 24 帧改路由后，G 白点（d_hi_G）从 1.36–2.15 D 变为 0.27–1.87 D；Eastman 4 帧从 2.09–2.22 D 变为 1.47–2.05 D。完整清单见 [roll_routing_changes.json](roll_routing_changes.json) 和 TABLES.md；拼版 `roll_foma_pan100_before_after.jpg` 和 `roll_eastman_5294_before_after.jpg` 只在本地保留。这些卷没有参考图，无法判断哪种路由颜色更准；本轮只能确认，不再使用物理上不可能的卷片基。

## §4 命中清单

- 308 帧基准：0。
- 设备 61 条：A850 `input_density_implausible`、A229 `input_channel_empty`，与预期一致。
- Batch 19–21（41 帧）：A851、A852、B094 命中 `input_density_implausible`，与预期一致，没有多出的命中。B094 原本就是 manual_review。B096 恢复 ok 后 d_hi 为 1.035，不再命中；它的参数和状态与 T4z 相同，只有显示不同。
- §3 两卷（默认路由）：foma 0 帧。**Eastman（无后缀）多出 4 帧**，任务书没有预期：DSC_8049 / 8051 / 8055 / 8061，均为 `input_density_implausible`，d_hi_G 为 2.41–2.48 D。这 4 帧在 T4z 和 T5a 中都走 roll_nat（卷片基 0.52–1.05，未超过 1.25，路由没变），状态 ok，core q50 0.16–0.27。整卷片基比本帧片基低 0.15–0.65 D，所以密度跨度显得偏大。这说明该卷在 roll_nat 下白点偏高，或者这几帧本身密度范围就宽，本轮无法区分。按要求逐条报告，不调阈值。空通道规则在这两卷上没有命中，最小通道 p99 为 0.071。

研究端只在 23 个扫描 TIFF 上测过空通道阈值。本轮在 61 条设备记录（含相机 RAW 和 Acros）、41 帧 Batch、59 帧两卷上统计，只有 A229 命中。BlueNeg 扫描 DNG 只覆盖到基准中那部分，基准上 0 命中。

## 测试

新增 6 项测试，并把 1 项 THUMB_REV 断言更新为 t5a。完整 pytest 实测 **441 passed**，328.32 秒，退出码 0，collect-only 计数同为 441。同样 7 项测试放到隔离的 T4z 源码上全部失败：T4z 缺少相应函数或字段，并且 THUMB_REV 断言不成立。另外在 T4z 代码上实测了同样的合成输入：窗户被涂成灰色，越界帧照样走 roll_nat。证据见 [baseline_failure_evidence.json](baseline_failure_evidence.json)。

HTTP 服务用 preview_start 在 62872 端口启动，实测 `/api/open` 返回 `input_warnings=[]`，`THUMB_REV=t5a`，渲染出 PNG 后服务已关闭。在浏览器里临时注入一条提示，信息栏能显示“输入提示”，随后已恢复页面。

## 未解决与需确认

- Eastman 无后缀卷那 4 帧高密度提示的成因未定。roll_nat 的卷片基可能偏亮，也可能这几帧本身密度范围宽。
- Eastman 5294 的胶片类型（任务书 §9.1）未确认，本轮仍按 `film="bw"` 运行。
- A850–A852 疑似已反相的正片，是否移出“未反相彩负”目录由用户决定。素材按只读处理，本轮没有移动。
- B094 根因未定，仍是 manual_review。

## 保护与交付

原片只读。每次读取原片前都先调用 `check_source(path, allow_frozen=False)`；冻结集图像不解码。`t5a_final_audit.py` 复核了 150 份历史 JSON（与各自提交逐字节相同）、T4z 目录 26 份 JSON（与 `acee65d` 逐字节相同）和 5 份冻结清单的哈希。全程用 Python `-I`，不读 pickle，不训练、不用 Colab、不加依赖、不上传。图片和日志不提交。交付物在 `experiments/t4_auto_ui/t5a/`：TABLES.md、acceptance.json、T5A_REPORT.md、candidate_a/b/cross.json、devices_baseline/candidate.json、rolls_*.json、batches_*.json、roll_routing_changes.json，以及测试、服务、资源和脚本哈希证据。
