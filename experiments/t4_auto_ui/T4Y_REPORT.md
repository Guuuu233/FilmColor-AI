# T4y：相机矩阵边界、亮场短跨度阻尼与扫描仪片基隔离

完整验证结束，正式重算 acceptance.all_passed=true，10项总门禁全部通过。main 已从 T4w `b3a751038958600a99f0fb9345852830b2de2d58` 经祖先关系核验后 --ff-only 快进至 T4x `f6340aa5fe053332a1d383702b3ad218f0c81624`，随后创建 t4y；未 push。用户 paths.py 保留原 SHA256 `fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913`，不暂存提交。

候选源码冻结 SHA256 `db2d825798bc3edf7fe99e128d57ae56df27c466db876b633b067cca91fbc629`，隔离 T4x 基线 SHA256 `967dbd1fbf4ab5762df118bf132ecd8fc5481b5cdaf402f5300ba6a09e53ece2`，均含相同用户 paths.py 运行覆盖。研究预测只作为待检验线索，不据预测调整门槛或更换评测成员。

## 实现与验证范围

- render 的主对角边界改为 (.5,2.4)，非对角边界改为 (-1.6,.8)，条件数上限仍为10。主会话用任务书两份 LibRaw 矩阵独立复现旧版拒绝：Canon 被非对角边界拒绝，Sony ILCE-7 被主对角边界拒绝；新版100%矩阵均通过，条件数分别约2.70、2.60，40%矩阵首元分别1.34491、1.40494。这证明所测矩阵的回退原因，不将研究方64款机身扫描冒充本轮64款全量复验。
- estimate_base_blocks 新增 exclude_near_saturated=True；scanner.estimate_base_scanner 默认关闭，app._scan_one 显式传入 not is_scan。相机路径保留准饱和启发式，扫描仪 TIFF/LinearRaw DNG 路径默认隔离。更新 t4y 缓存键与缩略图版本。
- 在原高EV、moderate阻尼之后增加 bright_short_span 分支：span<.76、bright_share>=.20、EV>=.45、ev_anchor_raw<=1.40、med_tex_before>=.46，并限制 q25 在 [.475,.68]+预设阴影补偿内。pull 按任务书配置计算，最大.42且不超过当前正EV，同时降低blacks；后续contrast/shadow/round流程保留。
- noref revision 为 t4y-2026-11-04；五份配置快照已导出并与运行配置逐项比对。

矩阵应用在逐通道反演后的正像上，属于现有渲染管线的近似，不是严格的光谱或传感器前向逆模型。亮场阻尼和准饱和判定仍是参数化启发式。status=ok、矩阵非单位阵和有限输出均不构成色彩真值证据。A1005..A1007 的镜头反射热点解释来自研究方，本轮没有独立验证光学成因。

## 测试与保护

新增五项合成测试，覆盖 Canon/Sony 矩阵接受及病态矩阵拒绝、扫描仪与相机片基差异、亮场阻尼触发、雪景/宽跨度/深暗锚点豁免，以及 app 的扫描分流。主会话独立运行新增五项为5 passed；同一测试在隔离T4x上为4 failed、1 passed，失败对应四项修复入口。完整测试实际430 passed in400.41s，退出码0；peak memory footprint3123645416 bytes。test_evidence脚本核验collect-only同为430，主会话逐项核验测试文件哈希与冻结源码一致。

A1296 全分辨率 full_positive 的全部区域统计已与T4x精确相等：中心任一通道>10计数419，占比0.000051897581176；上带计数0。整幅仍有3.908594%像素>10，保留该诊断，不将局部保护误称为整幅无极端值。

实际HTTP服务已复验 config_rev=t4y-2026-11-04、THUMB_REV=t4y；_DSF9239.RAF打开ok，PNG通过解码验证。

## 完整回归及设备结果

candidate_a/b/cross已完整实跑结束，记录数分别96/95/124；沿用既有6条无ICC和1条低NCC排除协议，得到185开发帧五预设与123检查帧standard。主会话独立重算如下；正式汇总确认五保护帧对T4x/T4o的五预设参数与指标全部精确一致，all_passed=true。

| 集合 / 预设 | N | median / mean / p90 | 对T4x 好/差 >.25 | 对T4x 好/差 >.05 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 |
|---|---:|---|---|---|---|---|
| Dev standard |185|9.01 / 9.33 / 13.94|6/0|6/0|18/0|18/0|
| Dev natural |185|9.18 / 9.90 / 14.55|6/0|7/0|21/0|23/0|
| Dev portrait |185|9.45 / 10.43 / 15.96|6/0|6/0|30/0|34/0|
| Dev vivid |185|9.04 / 9.81 / 14.80|6/0|6/0|13/0|13/0|
| Dev print_classic |185|8.38 / 9.00 / 13.47|5/0|5/0|11/0|11/0|
| Check standard |123|13.52 / 14.09 / 20.63|6/0|6/0|27/0|28/0|

相对T4x合计36帧次改善>.05、35帧次改善>.25；六项均净正改善，对T4x/T4o双阈值退步均0。309条非错误记录的d_hi/white_point/spike_clamp/base_method对T4x差异0。帧次包含预设间重复，不能解释为36个独立场景。

两版设备探针各60条全部结束；主会话独立重算263项设备门禁全部通过，failures=[]，正式汇总加入5项A1296 scope门禁后共268项全部通过。A1041/A1027片基与d_hi精确恢复T4w，A1041完整params亦相同；A262/A263分别排除1977/2246块且maxbase<.95，八指定矩阵目标首元均>1.25。共10条设备参数变化完整记录；矩阵修复后的参数变化不套用旧版参数零变化门禁。测试证据的实际source_sha256与冻结候选db2d8257…一致；代理文字报告曾误报其他指纹，以主会话读取JSON及重算结果为准。

## 输入保护与交付

本轮协议证据已独立核验132份历史JSON与对应git提交字节一致、5份冻结manifest哈希不变。原片只读，读取前check_source(...,allow_frozen=False)，冻结图像不解码；Python -I、不加载pickle、不训练、不使用Colab、不引入新依赖、不外部上传。

fresh_batches_t4x.json含12+16+10=38条记录，已存证JSON哈希与计数；本轮独立复验指定目标，未将研究方38/38结论当作本轮38张全量复验。

已交付T4Y_REPORT.md、t4y/TABLES.md、acceptance.json、设备/测试/协议/服务/资源证据及14个可重算脚本。五行before_after.jpg由A1061、A237与三个指定基准代表帧组成，主会话已读取最终图像：相机样本草木与苔藓色彩分离更明显，亮场三个样本整体亮度回落；BlueNeg样本仍可见偏粉/偏紫和原有耀斑，不能称为色彩真值恢复。两张横置样本保留原解码方向，拼图未另作人工旋转。图片与日志仅本地保留。

三代表standard指标实测：blueneg:199807B-20-seattle 10.603968→7.733378；blueneg:19960811B-25-arizona-london-bridge 15.831706→13.271141；ext:A205 8.168879→6.755158。代理文字曾给出不同baseline数字，已由主会话以两版benchmark JSON与正式评测记录精确比对，保留上述实际值。

资源peak memory footprint（bytes）：candidate_a6088545608、candidate_b5438853992、candidate_cross8023349752、devices_baseline3786050944、devices_candidate5667460088、tests3123645416。并发负载不同，不从这些耗时推断速度提升。

正式all_passed=true后按t4y_commit_manifest.py精准本地提交，排除paths.py、原片、日志与派生图片，不push。
