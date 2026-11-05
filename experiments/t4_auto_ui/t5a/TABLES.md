# T5a 可重算验收表

候选源码 SHA256：`2377cf08f109683554309330b0450a03ab3c50c78c5f3cf69059bc62e258a720`（含用户原有 paths.py 改动）；基线 T4z `136a7dbb951ed4c59497adfd2aa789ee0f180246f4e525c6822a78c9253a4756`。

T5a 不改渲染参数，验收按逐位一致：基准每条记录除耗时字段（seconds / estimate_seconds）与新增 input_warnings / input_center_p99 外完全相同。

| 集合 | 预设 | N | median / mean / p90 | 对T4z 好/差 >.05 | 对T4z 最大 \|ΔΔE\| | 对T4o 好/差 >.05 | 对T4o 好/差 >.25 |
|---|---|---:|---|---|---:|---|---|
| dev | standard | 185 | 8.86 / 9.30 / 13.88 | 0/0 | 0.0 | 21/0 | 21/0 |
| dev | natural | 185 | 9.18 / 9.87 / 14.55 | 0/0 | 0.0 | 28/0 | 26/0 |
| dev | portrait | 185 | 9.38 / 10.40 / 15.91 | 0/0 | 0.0 | 38/0 | 35/0 |
| dev | vivid | 185 | 9.04 / 9.78 / 14.75 | 0/0 | 0.0 | 17/0 | 17/0 |
| dev | print_classic | 185 | 8.27 / 8.95 / 13.47 | 0/0 | 0.0 | 15/0 | 15/0 |
| cross | standard | 123 | 13.52 / 13.98 / 19.95 | 0/0 | 0.0 | 37/0 | 34/0 |

309 条非错误记录反演诊断差异：0；基准 input_warnings 命中：0。

## 设备（T4z 设备集）

61 条记录；除显示外逐位一致且基线重现 T4z 提交记录：61；显示渲染哈希变化：['B096.tif']；失败项：[]。

| 样本 | 状态 | core sRGB q50 | 门 4 需提示 | input_warnings |
|---|---|---|---|---|
| A229.TIF | ok | [0.085, 0.076, 0.939] | True | input_channel_empty |
| A850.tif | ok | [0.007, 0.007, 0.01] | True | input_density_implausible |
| B096.tif | ok | [0.377, 0.377, 0.377] | False | - |

B096 core q50：T4z [0.287, 0.287, 0.287] → T5a [0.377, 0.377, 0.377]；A1296 全分辨率作用域统计与 T4z 一致：True。

## Batch 19–21（41 帧）

命中：{'A851.tif': ['input_density_implausible'], 'A852.tif': ['input_density_implausible'], 'B094.tif': ['input_density_implausible']}；失败项：[]。

## §3 整卷片基越界保护

| 卷 | 帧 | 卷片基 > 1.25 | 默认 roll_nat 前 → 后 | 改走单帧 |
|---|---:|---:|---|---:|
| foma_pan100 | 38 | 24 | 34 → 10 | 24 |
| eastman_5294 | 21 | 4 | 17 → 13 | 4 |
| eastman_5294-1 | 39 | 23 | 0 → 0 | 0 |
| eastman_5294-2 | 39 | 0 | 0 → 0 | 0 |

改走单帧清单：foma ['DSC_7400.NEF', 'DSC_7401.NEF', 'DSC_7402.NEF', 'DSC_7403.NEF', 'DSC_7404.NEF', 'DSC_7405.NEF', 'DSC_7406.NEF', 'DSC_7407.NEF', 'DSC_7410.NEF', 'DSC_7411.NEF', 'DSC_7412.NEF', 'DSC_7414.NEF', 'DSC_7416.NEF', 'DSC_7420.NEF', 'DSC_7421.NEF', 'DSC_7422.NEF', 'DSC_7423.NEF', 'DSC_7426.NEF', 'DSC_7430.NEF', 'DSC_7431.NEF', 'DSC_7432.NEF', 'DSC_7435.NEF', 'DSC_7436.NEF', 'DSC_7437.NEF']；Eastman（无后缀）['DSC_8046.NEF', 'DSC_8048.NEF', 'DSC_8050.NEF', 'DSC_8057.NEF']。

两卷默认路由下 input_warnings 命中：{'eastman_5294/DSC_8049.NEF': ['input_density_implausible'], 'eastman_5294/DSC_8051.NEF': ['input_density_implausible'], 'eastman_5294/DSC_8055.NEF': ['input_density_implausible'], 'eastman_5294/DSC_8061.NEF': ['input_density_implausible']}。

完整 pytest：441 passed，328.32 秒；新增/改动测试在 T4z 源码上 7 项失败。

验收 gates：`{'benchmark_identical_vs_t4z': True, 'benchmark_worse0_vs_t4o': True, 'inversion_diagnostics_identical': True, 'protected_frames_identical': True, 'benchmark_no_input_warning': True, 'devices_identical_display_only_b096_hits_expected_gate4': True, 'batches_hits_expected': True, 'roll_routing_changes_present': True, 'tests_passed_ge_441': True, 'configs_unchanged': True, 'service_ok': True, 'paths_unchanged': True, 'resources_complete': True}`

**all_passed=True**

提示只说明输入可能不是可用的负片扫描，不改变 status、参数或反相；status=ok 与无提示都不等于颜色正确。
