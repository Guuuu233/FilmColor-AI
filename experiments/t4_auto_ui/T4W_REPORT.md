# T4w：中等曝光暗部阻尼、橙罩片基保护与窄蓝跨度救援

基线为T4v `0d1d7301339e011848cd7a2215ab0eaedffa7e4e`。已核验main原先`98ded3c`为祖先，以`--ff-only`快进到`0d1d730`，再建立`t4w`。没有push。用户`src/filmcolor/paths.py`原有修改SHA256 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`保持，不暂存提交。

完整测试、真实设备及308帧回归均已完成。候选源码SHA256（含用户paths.py）`368eb01a8ba68160612c2f3f29209883a015eb3f6a0c48ace129b17995da3251`。最终重算验收`all_passed=true`，10项总门禁全通过，260项设备检查failures=[]；详见`t4w/TABLES.md`及`t4w/acceptance.json`。309条非错误记录反演诊断零差异，五保护帧对T4v/T4o全预设参数指标精确一致。

## 实现与规格解释

- `noref.py`在原高EV阻尼后添加任务书指定的`elif`：有限跨度证据`span_d_g < 1.15`、`bright_share < .20`、EV至少`.85`、Step4b q25达到`.40 + .35 * max(0, base_shadows)`时，按指定枢轴、渐变和增益下调EV及blacks，记录`overlift_damp.mode="moderate"`。保留原高EV分支优先级。
- 任务书将此称为`[.85,1.50)`，但具体代码没有显式EV上限；本轮遵循其具体`elif`规格，没有额外增加上限。因此旧分支未命中的更高EV输入也可能进入新分支，须以实际诊断而非标题判断作用域。
- `base.py`仅在初始候选不满足`R>G and B<.43G`时允许裸背光高亮剔除，保护深橙罩候选。这是色度启发式，不证明能区分所有胶片与漏光。
- `invert.py`将窄蓝救援R/G门槛由1.10降为.96，保持原B/G<.55、片基B/G护栏和.25倍温和救援。任务书一处标题写.98，但具体代码与两个目标样本要求均为.96，以具体规格为准。
- `app.py`缩略图及磁盘缓存前缀`t4w`；配置rev `t4w-2026-11-03`，五份预设快照同步。

研究方文字分别称28与26改善帧次；逐集合给出的5+4+8+2+1+6为26。本轮不将预测数作为执行结果，正式TABLES/acceptance只记独立实跑。

## 新目标实测

| 样本 | T4v基线 | T4w候选 |
|---|---|---|
| A1004.DNG | manual_review_required，low_information；d_hi_G=.19060096，bare backlight被剔除 | ok、flags=[]；d_hi=[.85335541,1.06495881,1.20057631]，base warnings=[] |
| A1005.CR2 | ok，d_hi_B=.09802755 | ok，narrow_blue_rescue=True，d_hi_B=.20422899 |
| A1007.CR2 | ok，d_hi_B=.22536910 | ok，narrow_blue_rescue=True，d_hi_B=.32222252 |

A1004/A1007数值与研究方预测略有差异，按本轮值保留，没有调整阈值迎合预测。status=ok与有限像素不构成颜色准确证明。最终对照图中A1004仍很淡，A1005/A1007仍有明显黄绿偏色；不宣称视觉完美或完全消除色偏。基准例ext:A124 standard ΔE由7.9962888146降至7.3444875758，19960812D-25由25.8009444998降至21.5210307202；精确值见benchmark_visuals_candidate.json。

## 完整308帧回归结果

| 集合 / 预设 | N | median / mean / p90 | 对T4v好/差 >.25 | 对T4v好/差 >.05 | 对T4o好/差 >.25 | 对T4o好/差 >.05 |
|---|---:|---|---|---|---|---|
| Dev standard |185|9.01 / 9.37 / 14.00|5/0|5/0|10/0|10/0|
| Dev natural |185|9.21 / 9.93 / 14.63|4/0|4/0|13/0|14/0|
| Dev portrait |185|9.52 / 10.47 / 15.96|7/0|8/0|21/0|24/0|
| Dev vivid |185|9.04 / 9.85 / 14.84|2/0|2/0|6/0|6/0|
| Dev print_classic |185|8.45 / 9.02 / 13.61|1/0|1/0|5/0|5/0|
| Check standard |123|13.77 / 14.24 / 20.63|6/0|6/0|15/0|16/0|

相对T4v在>.05门限共26改善帧次，>.25为25帧次；这不是26个独立场景，预设间可重复。六项均有净正改善，对T4v/T4o双阈值退步为0。改善仍是与既有冲印/调色参考的接近程度，不构成场景颜色真值或对新卷的无条件保证。

## 完整设备探针结果

基线及候选真实设备探针均已完成，独立运行258项设备门禁全部通过，failures=[]（正式汇总另追加A1296作用域两项）。全部既有样本d_hi与T4v一致；A1296完整inversion/params/status精确一致，裸背光排除警告仍存在。

旧设备唯一参数变化为A1027.TIF：moderate阻尼将EV从1.3194降到1.0784，新增blacks=-.0241，其余参数精确一致；诊断pull=.2410228062、q25_before=.4402557015。任务书允许个别旧设备命中moderate产生曝光变化，本轮据实际诊断记录，不把该帧宣称为参数零变动。设备status、finite与render-preservation均通过。

## 测试进展

新增五项真实合成测试在候选上独立通过，在隔离T4v源码上为4 failed、1 passed、115 deselected，基线/测试哈希见baseline_failure_evidence.json。首次完整测试暴露旧T4q测试仅关闭overlift_damp_check便断言无阻尼，与T4w独立开关overlift_mod_check冲突；已单独复现，并在旧分支ON/OFF比较中显式关闭新分支。原有数值与上游诊断断言保留，未改生产行为。修订后的完整pytest实际419 passed in 335.38s，0 failed，collect-only同为419；memory footprint为3254275216 bytes（约3.03 GiB）。本轮已核验测试哈希与冻结生产源码一致。

## 验证协议

185有效开发帧五预设与123有效检查帧standard，沿用固定manifest、1000px回归、原NCC门限及排除项。对T4v/T4o严格>.05、>.25统计better/worse；六项均须净改善且零退步。309非错误反演记录d_hi/white_point/spike_clamp/base_method对T4v精确相等；五保护帧对T4v/T4o全参数指标精确相等。

真实设备1600px：A1296、8TARGETS、7张T4v新增TIFF、28Acros、8既有TIFF和三new_targets，基线/候选同脚本、同输入SHA；旧样本d_hi守卫，曝光变化必须由moderate诊断解释并逐项记录。A1296保持参数/白点/状态精确一致，沿用上带及中心作用域门限，整图极端值诊断保留。

原片只读，读取前check_source(..., allow_frozen=False)；冻结图像不解码。Python -I，无pickle、新依赖、训练、Colab、GPL外部源码复制或外部上传。历史96份JSON及5份冻结manifest已核验字节/哈希，输入协议文件记录任务书和研究JSON哈希。fresh_batches_t4v.json为研究来源，不将其62张记录视作本轮完整复验，不声称完美出片。

## 资源与服务

memory footprint：回归a=6079960368 bytes、b=5493494608 bytes、cross=8439683984 bytes；设备基线3593342648 bytes、候选4802072872 bytes。并发负载不同，不能用墙钟时间直接推断速度改进。完整测试3254275216 bytes。

本地服务实际复验HTTP配置rev和缩略图rev为t4w，打开_DSF9239.RAF为ok、EV=.5492，PNG 3362409 bytes，SHA256 `b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff`。此帧输出保持T4v一致。

## 可重算入口

- `scripts/t4w_protocol_evidence.py`：历史提交字节、冻结manifest、任务书/研究JSON来源及main快进记录。
- `scripts/t4w_prepare_baseline.py NEW_DIRECTORY`：提取0d1d730源码并记录paths.py运行时覆盖。
- `scripts/t4w_export_presets.py --check`：五份配置与运行时精确一致。
- `scripts/t4w_probe_devices.py SRC baseline|candidate`：真实设备基线/候选，可断点续跑并核对source/script/input哈希。
- `scripts/t4w_run_eval.py`：完整开发五预设及检查standard，分片日志与memory footprint。
- `scripts/t4w_baseline_test_evidence.py`：隔离T4v源码上复现新增测试失败。
- `scripts/t4w_test_evidence.py`：完整pytest、collect一致、测试与源码哈希。
- `scripts/t4w_a1296_scope.py`：A1296整图/中心/上带独立复算。
- `scripts/t4w_benchmark_visuals.py SRC baseline|candidate`：两张代表基准帧使用同版评测参数渲染。
- `scripts/t4w_visual_compare.py`：三新目标与两代表基准帧拼图；图片不提交。
- `scripts/t4w_service_check.py`：真实HTTP打开、版本及PNG验证。
- `scripts/t4w_report_tables.py`：正式TABLES/acceptance/device_checks/资源及脚本哈希。
- `scripts/t4w_commit_manifest.py`及`t4w_final_audit.py`：仅all_passed=True后精准提交清单与最终审计。
