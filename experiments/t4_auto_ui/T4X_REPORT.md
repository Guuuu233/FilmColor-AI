# T4x：有界窄蓝救援、平灰负片恢复与准饱和片基保护

基线T4w `b3a751038958600a99f0fb9345852830b2de2d58`。已核验main `0d1d730`为T4w祖先，本地--ff-only快进至b3a7510后创建t4x。没有push。用户paths.py原修改SHA256 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`保留，不暂存提交。

完整验证结束，正式重算验收all_passed=true，10项总门禁全部通过。设备226项检查加2项scope门禁通过，额外5条诊断差异原样保留。冻结候选源码SHA256为`967dbd1fbf4ab5762df118bf132ecd8fc5481b5cdaf402f5300ba6a09e53ece2`（含用户paths.py运行覆盖），隔离T4w源码为`368eb01a8ba68160612c2f3f29209883a015eb3f6a0c48ace129b17995da3251`。研究方数字仅作为待独立核验的预测；不调整阈值、跳过失败或使用文件名特判迎合预测。

## 实现及范围

- `invert.blue_white_guard`保留片基B/G和低R/G保护，对ratio<.55内部仅R/G>.96且ratio<.28时救援，按固定85%向.28G收拢；ratio>=.28且<.55保持原值。该分支按任务书不再使用guard.weight/target控制拉升量。
- `base.estimate_base_blocks`在crop约束后排除有效块中`.9985 < max_channel < 1.0`的准饱和块，前提是剩余至少100块；仅实际排除时新增`near_saturated_blocks_excluded`诊断。共享函数没有Sony型号限定，适用于所有调用路径，因此这是亮度启发式；设备归因与无误伤需分别验证。精确1.0保留是本轮指定兼容契约，不能据此认定真实1.0白块均未饱和。
- `noref.estimate_params_v3`在Step4b记录q25/q90后按任务书顺序加入thin_flat与ultra_flat分支，保留后续backlit/highlight/contrast流程；thin_flat优先于ultra_flat。两分支未额外增加显式w_bl==0判断，而是遵循任务书q25条件；应通过实际诊断确认效果。
- thin_flat对span<.60、原thin未应用、EV<=0、q25>=.62和IQR<=.145的输入降低EV/blacks；ultra_flat要求range_stops>=1.20、EV<=0、q25>=.72、IQR<=.068，降低EV/blacks并提高contrast。
- moderate EV配置由(.85,.70,.45)变为(.70,.58,.55)，原高EV分支优先顺序和其它参数保持。此分支仍没有显式EV上限。
- app缓存前缀/THUMB_REV为t4x，rev按任务书保留`t4x-2026-11-03`，五份配置快照同步。

这些均为参数化启发式；没有从高密度或截断数据中恢复唯一真实场景的保证。status=ok及有限像素不是色彩准确性证明。

## 已测指定目标

- A1005：ok，有界ratio由.15390684至.26108603，d_hi_B=.16629296。
- A1006：ok，有界ratio由.11030421至.25454563，d_hi_B=.19882384。
- A1007：ok，guard.applied=False，d_hi_B=.22536910保持原始值。
- A1004：ok，ultra_flat_bright.applied=True，EV=-.7505、contrast=1.7254，pull=.50049268、blacks降低.27527097。
- A1033：ok，ultra_flat_bright.applied=True，EV=-.4366、contrast=1.4586，pull=.18657761、blacks降低.10261769。
- A262：ok，片基[.13936065,.27678341,.12976272]，近饱和块排除1977个。
- A263：ok，片基[.54563212,.91461051,.42517740]，近饱和块排除2246个。

Sony片基与研究预测有小差异，记录本轮原值，不为贴合预测修改参数。服务HTTP配置已独立核验为t4x-2026-11-03，PNG打开渲染通过。

## 已发现跨设备副作用

共享near_sat规则也命中旧扫描TIFF A1041，排除84块；片基由[.98579627,.52690488,.43040568]变为[.98575056,.52683282,.42946503]，d_hi由[.30475023,.44258389,.67992812]变为[.30473009,.44252449,.67897791]。EV从-.0653到-.0655、temp从-.0335到-.0340、blacks从-.1174到-.1175。它不是Sony RAW，证实设备无关的共享规则会影响其它输入；具体验收归属需区分任务书明确的309基准诊断守卫与旧设备状态要求，不能将新加的执行方守卫冒充原任务书要求。保留全部差异，不擅自调整生产阈值。

## 完整设备探针核验

基线与候选设备已全部实跑结束：226项任务及继承保护门禁通过，failures=[]，另保留5条额外诊断差异。A1296完整参数/反演/状态精确一致，独立scope结果保持。

除A1041外，A1027.TIF也因共享near_sat规则排除24块：片基由[.71935606,.34094760,.36549935]变为[.71921873,.34088656,.36545357]，EV从1.0784变为1.0788，d_hi有微变。这两张非Sony TIFF说明不能宣称该规则仅修复Sony或对所有旧设备像素零变动。

A1033的额外参数检查原只允许EV/blacks/contrast变化，实际highlights由-.033变为省略（默认0）；其ultra_flat降低EV后，后续Highlight Hold不再需要同样高光压缩。实际诊断保留为检查不匹配，不将其静默改成精确一致。B010.NEF命中thin_flat；A1006/A1005/A1007因有界白点策略改变参数；A262/A263因片基恢复改变参数，详见最终device_checks.param_changes。

## 完整测试结果

主会话专项126 passed；新增六项在隔离T4w源码上为5 failed、1 passed、120 deselected。最终完整pytest实际425 passed in 429.06s，0 failed，collect-only同为425；memory footprint 3178613760 bytes（约2.96 GiB）。测试文件哈希与冻结候选源码指纹已逐项核验一致。

## 完整308帧回归结果

| 集合 / 预设 | N | median / mean / p90 | 对T4w好/差 >.25 | 对T4w好/差 >.05 | 对T4o好/差 >.25 | 对T4o好/差 >.05 |
|---|---:|---|---|---|---|---|
| Dev standard |185|9.01 / 9.36 / 14.00|2/0|6/0|12/0|12/0|
| Dev natural |185|9.21 / 9.93 / 14.63|2/0|6/0|15/0|16/0|
| Dev portrait |185|9.52 / 10.46 / 15.96|4/0|8/0|24/0|28/0|
| Dev vivid |185|9.04 / 9.85 / 14.84|1/0|2/0|7/0|7/0|
| Dev print_classic |185|8.45 / 9.02 / 13.61|1/0|2/0|6/0|6/0|
| Check standard |123|13.77 / 14.16 / 20.63|7/0|9/0|21/0|22/0|

六项均净正改善，对T4w/T4o的>.05、>.25退步均为0；相对T4w合计33帧次改善>.05，17帧次改善>.25（预设间可重复，不是独立场景数）。309条非错误记录d_hi/white_point/spike_clamp/base_method对T4w差异0条。全量采用固定排除6条无ICC及1条低NCC的既有协议，没有为达标更换成员或阈值。

## 验证条件

完整pytest至少425 passed；新增至少6项合成测试并更新两项旧窄蓝测试，使用同一最终测试文件在隔离T4w源码上核验失败行为。完整开发185帧五预设及检查123帧standard实跑，相对T4w/T4o严格>.05、>.25双门限零退步，六项均净改善；309非错误记录的d_hi/white_point/spike_clamp/base_method精确一致，五保护帧对w/o全参数指标精确一致。

真实设备基线/候选同脚本、同输入SHA，保留A1296、28Acros、8TIFF、8TARGETS、7张T4vTIFF及扩展六new_targets。A1005/A1006有界ratio门禁；A1007不再触发；A1004/A1033 ultra恢复；A262/A263片基最大值<.95。允许指定目标白点/片基改变，旧设备其它变化完整记录为额外诊断，不能冒充全部参数零变动。任务书第4.2.1只要求旧设备组status==ok，并未将旧设备d_hi/params精确相等列为本轮硬门禁；309基准反演及五保护帧精确守卫仍为硬门禁。执行方最初沿用T4w额外旧设备相等守卫，复核原任务书后已将其归为diagnostic_checks/diagnostic_differences，未改任何任务书阈值或生产代码。

输入原片只读，读取前check_source(..., allow_frozen=False)，冻结图像不解码；Python -I，不加载pickle、不训练、不使用Colab、不引入外部GPL源码或新依赖、不外部上传。历史JSON字节与冻结manifest哈希需在最终审计保持。fresh_batches_t4w.json的50条记录为研究来源，除明确独立重跑样本外不将其视作本轮50张全量复验。

## 视觉边界与资源

已检查最终9行对照图：Sony两样本灰紫底与齿孔边有所改善；A1005/A1007仍明显偏黄，A1004/A1033仍偏淡。数值门禁通过不能支持“彻底消除黄绿”“恢复真实场景”或任意新样本零误伤结论。没有色卡测量配准与光照真值，本轮不报告色卡准确度。

回归memory footprint：a=5741761600 bytes、b=6041670912 bytes、cross=8263490280 bytes；设备基线3380399392 bytes、候选4101099384 bytes；测试3178613760 bytes。并发负载不同，不据墙钟作速度提升结论。服务已复验t4x配置及缩略图版本，_DSF9239.RAF打开和PNG输出通过。

## 已生成证据

- t4x/source_freeze.json及baseline_source.json：运行源码与隔离T4w基线。
- test_evidence.json及baseline_failure_evidence.json：完整测试及新增测试失败对照。
- candidate_a/b/cross.json、TABLES.md、acceptance.json：完整回归及正式门禁。
- devices_baseline/candidate.json、device_checks.json、a1296_scope.json：真实设备与作用域。
- before_after.jpg及benchmark_visuals_*.json：六目标和三代表改善基准帧；图片不提交。
- service_check.json：实际HTTP版本、打开及PNG输出。
- input_protocol_sha256.json、script_sha256.json、resource_evidence.json：可重算协议与资源证据。

仅all_passed=true后按t4x_commit_manifest.py精准本地提交；不提交paths.py、原片、日志、派生图片，不push。
