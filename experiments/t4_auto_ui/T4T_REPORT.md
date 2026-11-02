# T4t：扫描仪白点、零截断预检、宽内框与白平衡色区冲突

基线为 T4s `7c5107c`，分支 `t4t`。实现、完整测试、185帧五预设及123帧检查回归、设备和服务验证已完成。**参数与指标相对T4s/T4r逐帧完全一致，相对T4o无>.05或>.25退步；404项测试通过。但三条真实设备条件未通过，因此 all_passed=False。** 最终结果由 `scripts/t4t_report_tables.py` 从量化产物生成，未使用任务书预测替代实测。

## 实现与契约

- `block_white_point` 按任务书增加非绿通道白点 cap：G >= .20 D，R/B > max(2.5 D, 2.2×G) 时设为 1.45×G，`channel_cap` 保留原始值。仅作用于已有 block_mean 分支；小图 pixel_fallback 及直接给定白点保持原契约。该经验规则不构成对所有胶片、扫描仪物理密度范围的证明。
- 宽内框分支：初始 G > 1.15 D 且 R > G+.25 D，密度块四周环满足 mid>1.05、max>1.45 的比例 >=.75 时，沿用边连接连通域、两轮膨胀及最少 100 个剩余块保护。原 G>1.80 的不透明片夹分支保持。
- `invert_negative` 的全局 Dm escape 比例门槛由 >.01 改成 >.001，统一传给各行块；`clamp_spikes` 局部 >.01 契约不改。未增加像素名判断，也未改 `blue_white_guard`。
- cyan confidence floor 增加 bright_px_share >=.10；正 Tint 遇 zone_disagreement>1.25 时乘 .55，替代原 1.20，并记录 `tint_conflict_damped`。这些规则不保证关闭 confidence floor 后剩余 WB confidence 很低。
- 配置版本 `t4t-2026-11-02`，缩略图、扫描缓存前缀 `t4t`；五个 JSON 为运行时 `preset_config()` 的审查快照。

原片只读，所有解码前通过 `check_source(..., allow_frozen=False)`。冻结原片未解码。Python 使用 `-I`，无 pickle、无新增依赖、无训练、无外部源码复制、无 push。`FILMCOLOR_DATA_ROOT` 固定为既有外盘目录。研究方所称 80 张盲测及外部项目逐行比较未被当作本轮执行证据；没有可读取的完整 80 张清单，本轮只独立复验指定帧及既有设备样本。

## 输入与源码来源

T4s 源码从 Git 对象独立提取并逐文件校验，叠加本轮开始时用户既有 paths.py 后运行指纹为 `27ba01f3aa13d29905af25f17284b68bc99da2897a40c957c2a5f4d898b9efb5`，与已提交 T4s 量化产物相符。候选运行指纹 `1e0130e978687017ddb8760373783e5030a35cffc1cc2d069b549b3b710c8099`。指纹按 src/filmcolor 内 .py/.json/.html 的排序相对路径、NUL、内容、NUL 累加 SHA256。

本轮子代理曾误用 checkout 撤掉用户 paths.py 差异；主会话用操作前已保存副本逐字恢复。恢复后 SHA256 为 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`，与开始一致；该文件不暂存、不提交。真实基线始终来自独立提取源码，正式候选冻结及运行均发生在恢复之后。

## 已测真实样本及未通过条件

| 样本 | T4s | T4t 实测 | 判定 |
|---|---|---|---|
| A305.tif | manual_review_required，white_point_invalid | ok，EV=.0860，Tint=.0422 | 指定状态门限通过 |
| A1040.TIF | d_hi_B=6.0 | d_hi_B=.5946196，channel_cap.B=6.0；srgb_q50=[0,0,.16937] | **绿色中位数>.05 未通过** |
| A1041.TIF | d_hi_B=4.402781 | d_hi_B=.6955898；绿色中位数=.199826 | 指定门限通过 |
| A232.NEF / A1069.NEF | d_hi≈[1.68562,1.25605,1.33386]，EV=2.09 | ≈[.877956,.924210,1.025091]，EV=.1446，holder_share≈.69985 | 指定门限通过 |
| A239.ORF | Tint=.4748，cyan_conf_boost=True | Tint=.3729，cyan_conf_boost 不再触发 | **Tint<.22 未通过** |
| A281.RAF | Tint=.2979 | Tint=.1447，tint_conflict_damped=True | <.18 通过，预测 .1365 未精确复现 |
| A1006.CR2（诊断） | d_hi_B=.0861579 | 原白点保持 | 正式四项规范未包含该救援，不宣称修复 |

A239 的 bright_px_share=.03954，关闭 cyan floor 后 confidence 实际为 .5604796（任务书声称 .336 未复现）；由 chroma=[-1.132861,-.504663]、strength=.65 算出 raw Tint=.2982847，乘1.20得 .3579417，再加 base_tint=.015 得最终 .3729。`scripts/t4t_diagnose_targets.py` 能从证据逐项重算，单靠正式规范无法得到预言的 .1845。没有额外压低 confidence 或改 Tint 阈值。

A1040 cap 确实执行，但正片仍存在明显蓝偏和暗部压黑；上带蓝 p99≈2,060,412，不能称所有零截断伪影消除。白点 cap 本身不恢复被裁切通道，也不修复片基或 Dm 以外所有区域。未改曝光/片基算法去追逐验收数字。

### A1296：区分预览门限与统计区域

默认1600px（实际预览shape取整数块均值）上带 rows10–20%、cols20–80%：蓝 p99=1.07499075、max=1.52736545；中心 rows20–80%、cols20–80% p99≈[1.194014,1.227335,1.247261]。因此 **T4t 新预览门限 p99<=1.15、max<2.0、中心<1.35 通过**。T4s 旧 p99<1.0 的失败历史不改。

另做整幅原始正片范围核验：

- 上带区域：任一通道>10 的计数0，占比0；max_rgb≈[4.8276,6.0591,9.0504]。
- 整张图：876,718 / 22,430,520 像素任一通道>10，占比 **.03908594（3.9086%）**。
- 中心区域：419像素任一通道>10，占比 .00005190；仍含极端值。

故任务书“整幅分辨率下>10的极端像素占比为0”若按整图理解，**未通过**。上带为0不替代整图为0；片夹与片外区域贡献不能在没有明示mask时被自动扣除。量化证据为 `t4t/a1296_scope.json`，可用独立脚本重算。

## 完整测试、设备与服务

完整 pytest **404 passed in 414.73s**，collect-only 同为404，peak memory footprint=2.81635 GiB。新增五项测试已在独立T4s基线上实际复现5项失败；相同测试文件在候选完整套件中通过。测试文件及源码SHA记录于test_evidence.json，基线失败证据另存baseline_failure_evidence.json。

真实T4s基线与T4t候选均跑完28帧Acros和8张指定TIFF：全部status=ok、full_positive_finite=True、render_preserves_positive=True。指定疑难样本使用每帧单独文件夹，Acros使用完整28帧目录计算整卷路由，避免把不同片基混成一卷。状态成功只证明处理链可用，不证明颜色正确。

服务使用既有launch配置启动，root仍为data/raw/public/nlp_official、端口62872。真实POST /api/open返回http_config_rev=t4t-2026-11-02，_DSF9239.RAF状态ok、EV=.5492，6个子卷、0片基flag。渲染PNG为3,362,409字节，SHA256=b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff，与T4s该样本一致。浏览器页面可加载，检查时无console error。

## 验证协议与交付

开发清单191条，固定六条无ICC参考排除后185有效，五预设；检查124候选，固定排除NCC<.7的一条后123有效，standard。逐帧完整 params/metrics 与已提交T4s/T4r作精确比较、不设容差；相对已提交T4o按严格>.25及>.05统计。每个历史JSON从相应Git提交读取并核对工作树字节，输入raw/ref SHA逐帧一致。BlueNeg无ICC参考按sRGB仍是评测假设，检查集不能声称未见新卷真值。

从Git提交中的309条非错误历史记录另重算：最大非绿/G白点比1.81124355，最大白点1.73850012 D，最大zone_disagreement=1.18433499；channel_cap及宽内框初始签名均0触发。cyan_boost仅A006/A011/A196，bright_share=.26035/.38133/.51019。这些可从历史诊断核验；Dm escape比例和edge-ring占比未存于历史JSON，不将研究方的相应数字当作本轮重算结果。

可重算入口（均用 `.venv/bin/python -I`，图像任务设置既有 FILMCOLOR_DATA_ROOT）：

- `scripts/t4t_prepare_baseline.py NEW_DIRECTORY`：提取7c5107c源码及冻结守卫，记录paths.py覆盖来源。
- `scripts/t4t_probe_devices.py SRC baseline|candidate`：9个指定/诊断样本、28帧Acros、8TIFF；支持断点时核验源码/脚本/输入SHA。
- `scripts/t4t_run_eval.py`：开发五预设、检查standard，保持旧评测协议。
- `scripts/t4t_a1296_scope.py`：整图/中心/上带范围统计。
- `scripts/t4t_diagnose_targets.py`：重算不达门限的公式及测量。
- `scripts/t4t_visual_compare.py`：从带哈希的派生预览生成前后联系表；JPG不提交。
- `scripts/t4t_export_presets.py --check`：配置快照核验。
- `scripts/t4t_test_evidence.py`：完整测试日志、收集数、源码与测试哈希核验。
- `scripts/t4t_service_check.py`：真实HTTP files/roll/POST open/render与PNG哈希。
- `scripts/t4t_report_tables.py`：重算TABLES、acceptance、device_checks及资源/脚本SHA。
- `scripts/t4t_commit_manifest.py`：精确提交清单，排除paths.py、原件、图片与日志。

完整回归实际产物为开发a96/b95条记录（185有效）和检查124条记录（123有效）：185帧五预设及123帧standard的完整params/metrics相对T4s和T4r最大绝对差均为0.0、差异0条。309条非错误记录（含低NCC排除帧）的d_hi、white_point、spike_clamp及base_method也完全一致。五张保护帧五预设均与T4o一致。

| 集合 / 预设 | N | median / mean / p90 ΔE2000 | 对T4o 好/差>.25 | 对T4o 好/差>.05 |
|---|---:|---|---|---|
| dev standard |185|9.01 / 9.38 / 14.09|5/0|5/0|
| dev natural |185|9.21 / 9.95 / 14.63|9/0|10/0|
| dev portrait |185|9.52 / 10.50 / 16.18|14/0|16/0|
| dev vivid |185|9.04 / 9.86 / 14.84|4/0|4/0|
| dev print_classic |185|8.45 / 9.03 / 13.61|4/0|4/0|
| cross standard |123|13.77 / 14.31 / 20.63|9/0|10/0|

资源峰值按/usr/bin/time -l的peak memory footprint统计：开发a/b约5.33/4.69 GiB，检查8.21 GiB，设备基线/候选3.21/4.29 GiB，完整pytest2.82 GiB。回归曾因调度调整中断并以同源码指纹检查点续跑；产物未混用不同源码。日志不提交，其SHA及原始峰值字节数记录于resource_evidence.json。

正式 `t4t/TABLES.md`、`t4t/acceptance.json` 已生成：除devices_all_passed以外其余八个gate均为True；三条设备失败为A1040绿色中位数、A239 Tint、A1296整图极端像素占比，**all_passed=False**。这次完成的是任务书规定实现和实际验证，不宣称全部目标已解决。
