# FilmColor AI T5g 实验与验证报告（2026-11-08）

同 `experiments/t4_auto_ui/t5g/T5G_REPORT.md`。

## 核心结论摘要
- **分支状态**：`main` 快进至 `195e828e5c6199d9e8d1a1048fc66dbaa1aca4b0`（T5f），在此基础上切出 `t5g` 分支并落地 `t5g_proto.diff`（`src/filmcolor/{base,invert,noref}.py` 及 `tests/test_{base,invert,noref}.py`）。
- **单元测试**：`456 passed in 355.43s`（退出码 `0`）。
- **315 帧基准集**：`s3_dev` 191/191 帧 100% 逐位一致；`bn_rb` 123/124 帧逐位一致，唯一变化帧 `blueneg:19941220D-18-kamakura` 的 `standard` ΔE2000 中位数从 `10.2748` 降至 `9.8789`（改善 `-0.3959 ΔE`）。
- **61 帧设备集**：全部 33/33 非 Acros 记录 100% 逐位一致；Acros 28/28 帧全部保持 `status = "ok"` 且 `full_positive_finite = True`。
- **12 张 T5g 目标帧**：详见 `experiments/t4_auto_ui/t5g/T5G_REPORT.md` 与 `experiments/t4_auto_ui/t5g/target_checks.json`。
