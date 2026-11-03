# T4u：预拉伸 TIFF 饱和掩模、无中性证据 Tint 封顶与窄蓝白点救援

基线 `567aaf9`（T4t），分支 `t4u`。实现、409项完整测试、185帧开发五预设、123帧有效检查集、真实设备与服务核验均已完成。正式acceptance.json九项gate全部为True，all_passed=True。开发/检查完整params和metrics相对T4t/T4s/T4r最大差均0.0、差异0条；对T4o在>.05及>.25下零退步。研究方的预测和“数学保证”未替代本轮执行证据。

## 改动及信号契约

- `load_scanner` 仅对 TIFF、至少三通道、R触顶比例>.30且G/B各<.05的输入，使用G/B并集作为饱和掩模，记录 `single_channel_prestretched="R"`。它改变的是估计样本的可用范围，不恢复已经截断的R信号。DNG仍保留原any-channel饱和规则；红外不进入RGB信号。
- `estimate_wb_v3` 保留极端色区分歧阻尼的优先级。两区均缺少窄带中性证据、没有cool/cyan增强且原始正Tint>.18时，封顶为.18并记录 `no_narrow_tint_capped`，不再乘1.20。预设偏移随后仍按原流程相加。
- `blue_white_guard` 在既有片基B/G门控之后，对ratio<.55且R白点>1.10×G的输入按.25×weight温和提升B白点，记录 `narrow_blue_rescue`；低R/G极窄蓝跨度仍返回below_valid_ratio。三参数guard或缺base的调用仍按既有可选片基契约处理，没有新增强制base要求。
- 版本 `t4u-2026-11-03`，缩略图/扫描缓存前缀t4u；五个JSON快照与运行时preset_config同步。

原片只读，所有原片解码前执行 `check_source(..., allow_frozen=False)`，不解码冻结图像。Python使用-I，无pickle、无新增依赖、无训练或外部源码复制。用户原有paths.py哈希保持fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913，不暂存、不提交。历史t4o/t4r/t4s/t4t的58份顶层JSON已逐字节比对对应Git提交，其哈希在input_protocol_sha256.json保存。

## 源码和数据来源

独立提取T4t Git对象并逐文件核验，叠加原有paths.py后基线运行指纹为 `1e0130e978687017ddb8760373783e5030a35cffc1cc2d069b549b3b710c8099`。候选冻结运行指纹为 `0c92000ec933c17b559b238d6b54800a7bd0b9aa2eaf3211a1b4f4c30b3995ca`。均按src/filmcolor内.py/.json/.html排序，以相对src路径、NUL、内容、NUL累加SHA256。基线与候选设备探针均实际重新运行，未把旧产物改标签当作本轮结果。

开发191条清单固定排除6条无ICC参考后185有效，五预设；检查124候选固定排除一条NCC<.7后123有效、standard。对T4t/T4s/T4r完整params和metrics按精确相等核验；对T4o以严格>.05和>.25统计。输入raw/ref SHA、成员、协议均核对。检查集并非未见新卷真值；BlueNeg无ICC参考按sRGB仍属评测假设。

历史诊断中309条非错误记录（含低NCC排除帧）有140条ratio_bg<.55，但符合base_bg<.48的新增救援候选为0；开发/检查清单输入后缀均为RAW或DNG，无TIFF。历史记录没有完整保存原始WB Tint，不能仅凭舍入后的params精确复算研究方声称的pre-boost最大值.12825；该论断不写成本轮数学证明，采用完整逐帧回归验证实际零变动。

## 指定样本实测

| 样本 | T4t | T4u候选 | 说明 |
|---|---|---|---|
| A1040.TIF | srgb_q50=[0,0,.16937] | [0,.16095,.24637] | G>.05；红通道截断信息未恢复 |
| A1041.TIF | G中位数=.19983 | .21466 | 同步触发R饱和解耦 |
| A239.ORF | Tint=.3729 | .1950 | no_narrow_tint_capped=True；通过<.22 |
| A1006.CR2 | d_hi_B=.08616；B-G=.31270 | d_hi_B=.22193；B-G=-.03831 | narrow_blue_rescue=True；通过B>.18、B-G<.05 |
| A305.tif | ok | ok | 状态保护继续核验 |
| A232.NEF/A1069.NEF | 宽框已排除 | 宽框仍排除 | 不追加框检测规则 |
| A281.RAF | Tint=.1447 | .1447 | 色区冲突阻尼保留 |

A1040/A1041的真实R饱和比例分别为.84482893/.60385213，G/B均0；新sat掩模与G/B并集逐像素相等。独立量化见scanner_saturation.json。A1040基线片基R<1、新片基R=1后更多估计样本可用，但颜色并非完整恢复。派生图仍可见强蓝偏及颗粒；A1006全图蓝雾减轻，但局部黄蓝差异仍明显。没有色卡真值，srgb_q50和status=ok不证明颜色准确，也不宣称恢复欠曝/截断信息。

### A1296作用域变更

本轮明确采用任务书指定的新作用域门限：full_positive上带rows10–20%、cols20–80%的any-channel>10计数为0；中心rows/cols20–80%的占比<1e-4。候选上带计数0，中心419/8,073,594=.0000518976，均通过指定门限。1600px预览蓝p99=1.07499075、max=1.52736545，中心p99各通道<1.35。

整图any-channel>10仍为876,718/22,430,520=.03908594（3.9086%），保留为诊断；中心也仍有极端值，因此不能把所有整图高值都归因于最外圈片夹，也不声称本轮进一步消除了A1296像素伪影。本轮A1296通过是按用户明确修订的区域和容许比例验收，T4t旧整图门限的失败历史保持不变。

## 测试与服务核验

新增五项测试的最终文件已在独立T4t基线验证：3 failed、2 passed、112 deselected；三个失败分别是TIFF R单通道解耦、无中性证据Tint封顶、窄蓝救援，两个保护测试原本就通过。候选三个完整测试文件117 passed in 40.81s。旧T4t中等分歧Tint测试的合成场景原本没有窄带中性像素，会合理触发新封顶；本轮补入6%分区中性像素以继续验证普通1.20 boost路径，并新增独立测试覆盖无中性封顶分支，没有关掉新规则。完整pytest为409 passed in 515.24s，collect-only同为409，peak memory footprint=2.79296 GiB。测试文件SHA及冻结源码指纹一致，见test_evidence.json。

基线及候选均实际完成28帧Acros与8张代表TIFF：全部status=ok、full_positive_finite=True、render_preserves_positive=True；两组全部params、inversion及status与T4t基线一致。指定8TARGETS+A1296的设备门限独立检查均通过。正式全量回归完成后设备门限集合failures=[]，不使用局部结果预写正式结论。

本地FilmColor服务沿用既有launch配置和root=data/raw/public/nlp_official、port62872，实际POST /api/open返回http_config_rev=t4u-2026-11-03，_DSF9239.RAF状态ok、EV=.5492，6个子卷、0片基flag。合法PNG为3,362,409字节，SHA256=b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff，与T4t相同样本一致。

## 完整回归与最终验收

实际开发a96/b95共191条记录，固定排除6条无ICC参考后185有效；检查124条记录，固定排除一条低NCC后123有效。相对T4t、T4s、T4r逐帧params/metrics均精确相等，max_abs_diff_params=0.0、max_abs_diff_metrics=0.0、diff_count=0。309条非错误记录（含低NCC排除帧）的d_hi、white_point、spike_clamp、base_method相对三代也完全一致。五张保护帧五预设均与T4o相同。

| 集合 / 预设 | N | median / mean / p90 ΔE2000 | 对T4o好/差>.25 | 对T4o好/差>.05 |
|---|---:|---|---|---|
| dev standard |185|9.01 / 9.38 / 14.09|5/0|5/0|
| dev natural |185|9.21 / 9.95 / 14.63|9/0|10/0|
| dev portrait |185|9.52 / 10.50 / 16.18|14/0|16/0|
| dev vivid |185|9.04 / 9.86 / 14.84|4/0|4/0|
| dev print_classic |185|8.45 / 9.03 / 13.61|4/0|4/0|
| check standard |123|13.77 / 14.31 / 20.63|9/0|10/0|

资源peak memory footprint（/usr/bin/time -l）：开发a/b约5.559/5.018 GiB，检查8.288 GiB，设备基线/候选3.176/4.068 GiB，完整pytest2.793 GiB。日志不提交，资源原始字节数和日志SHA见resource_evidence.json。

正式验收九项gate全部通过：all_metrics_params_exact、all_preset_worse0、inversion_diagnostics_exact、protected_frames_exact、devices_all_passed、tests_passed_ge_409、configs_all_match、service_ok、resources_complete。all_passed=True按本轮明确的作用域和数值门限成立，不扩大为颜色准确或所有伪影消失的结论。历史T4t失败记录完整保留。

## 可重算入口

均使用 `.venv/bin/python -I`；图像任务设置既有 `FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data`。

- scripts/t4u_prepare_baseline.py NEW_DIRECTORY：提取567aaf9源码和冻结守卫，并记录paths.py本地覆盖。
- scripts/t4u_probe_devices.py SRC baseline|candidate：A1296、8TARGETS、28Acros、8TIFF；断点校验源码/脚本/输入SHA。
- scripts/t4u_scanner_saturation.py：真实两张TIFF饱和率及新mask一致性。
- scripts/t4u_a1296_scope.py：整图、上带、中心独立统计。
- scripts/t4u_target_summary.py：前后参数、白点与图像统计摘要。
- scripts/t4u_visual_compare.py：从带哈希的派生图生成before_after.jpg；图片不提交。
- scripts/t4u_historical_gate_audit.py：从Git提交诊断重算窄蓝分支候选数与基准输入后缀。
- scripts/t4u_run_eval.py：沿用完整开发五预设及检查standard评测协议。
- scripts/t4u_export_presets.py --check：核对五个预设JSON。
- scripts/t4u_test_evidence.py：完整测试、收集数、测试/源码哈希与峰值资源。
- scripts/t4u_service_check.py：真实HTTP files/roll/POST open/render核验。
- scripts/t4u_report_tables.py：从完整实测生成TABLES、acceptance、device_checks及资源/脚本哈希。
- scripts/t4u_commit_manifest.py：仅在all_passed=true时输出精确提交清单，排除paths.py、原片、图片与日志。
