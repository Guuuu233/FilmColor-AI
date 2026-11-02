# T4s：16-bit 扫描仪 TIFF、全局零截断预检与黑白整卷路由

基线 fbd48a0（T4r），分支 t4s。实施、完整测试、185+123 帧回归、设备实测和服务核验均已完成。**308 帧回归相对 T4r 逐帧参数与指标完全一致，相对 T4o 五预设无 >.05 退步；TIFF 与 Acros 目标验收通过。但任务书的 A1296 上带蓝通道 p99<1.0 在默认 1600px 预览口径下未通过（1.07499），因此不能称全部验收条件通过。** acceptance.json 的 all_passed 为 False，原因仅此一项。

## 变更与信号契约

- 扫描仪入口扩展至 uint16 的 TIFF，支持 1/3/4 通道；DNG 仍要求 LinearRaw。优先 tifffile 解码，缺 imagecodecs 时使用本项目的纯 Python/NumPy LZW、PackBits 条带解码与 Predictor 2 还原，不新增依赖。
- 通过 Software 或 FilmColor ICC 描述拦截可识别的本软件导出正片。这不是任意第三方正片/负片语义识别。普通 TIFF 直接按黑白电平归一化并假设信号可用于现有反演；未应用 ICC/TRC，不声称其为已校准的物理线性信号。
- 反演在最终白点确定后使用 Dm（测光/白点 mask 内的像素密度）预检 escape 比例。超过 1% 时将 True 传给所有行块；False 时保留局部自行触发的历史契约。因而只保证全局阳性时各块统一开启，不能宣称所有输入都与 CHUNK_ROWS 无关。
- film=bw 时新增 >=.06 D 的健康整卷路由条件；彩色的密度差阈值仍为 None。最少 3 帧、卷片基无警告等保护保持。
- 配置版本 t4s-2026-11-02、缩略图/扫描缓存前缀 t4s。根目录 configs 的五个 JSON 是 preset_config() 的可审查快照；运行时仍由现有 Python 配置提供。scripts/t4s_export_presets.py --check 可重算一致性。
- 实际 HTTP 打开接口为 POST /api/open；本轮在返回 auto 对象中暴露真实 config_rev，不增加 GET 打开接口。

## 数据与验收口径

原件只读，所有图像读取前执行 check_source(..., allow_frozen=False)；冻结图像不解码。使用 python -I，不读取 pickle。保留用户既有 paths.py 差异，不提交原件、图片、日志或临时文件，不 push。

开发清单共 191 条，固定排除 A016/A018/A025/A026/A027/A028 六个无 ICC 参考，185 有效，保持 a96/b95 分片。检查清单 124 候选，NCC>=.7 的 123 有效，固定排除 blueneg:199807A-21-orlando-disney。每帧先取 median ΔE2000，再跨帧计算 median/mean/p90；相对 T4r 按完整 params/metrics 精确比较（不设容差），相对 T4o 按严格 >.25 和 >.05 统计好差。BlueNeg 无 ICC 参考按 sRGB 是评测假设；检查集来自已有开发卷，不作为未见新卷颜色真值。

任务书称 B002..B029 为 Acros 24 帧，磁盘与既有原片索引均显示 28 张连续 NEF，且全部通过冻结守卫。没有提供 24 帧的明确子集，因此本轮按实际完整 28 张验证，并单列指定 B003/B011/B016/B017，不自行挑除四张。任务书所称 APX 32 帧范围也实际为 B050..B085 共 36 张；APX 不在本轮正式硬验收清单中，不凭研究方声称写成已全卷实测。95 张新盲测未有逐张可核验清单，不声称复验全部 95 张。

## 可重算入口

均使用 `.venv/bin/python -I`；图像任务设置 `FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data`。

- scripts/t4s_prepare_baseline.py NEW_DIR：从 fbd48a0 提取源码与冻结守卫清单，不复制图像。纯提交源码指纹为 16862d53f8873cdf82d9659674da0f372114c8ab6ad7a284fca2dd1a0c1158a7，已由 git 对象独立重算一致。
- scripts/t4s_probe_devices.py SRC baseline|candidate：A1296、实际 28 帧 Acros；候选另含 8 张指定 TIFF。记录输入/源码 SHA、状态、路由、跨度、分位数、finite 与渲染不修改正片等证据。
- scripts/t4s_run_eval.py：完整开发五预设与检查 standard，沿用既有对齐、成员和度量协议。
- scripts/t4s_test_evidence.py：提取完整 pytest 日志，验证 collect 数量、测试及源码 SHA。
- scripts/t4s_service_check.py：核验已运行服务的 files/roll/POST open/render，记录真实 HTTP config_rev 与 PNG 哈希。
- scripts/t4s_a1296_sensitivity.py [MAX_SIDE [SRC LABEL]]：A1296 在指定预览尺寸、指定源码树（如 T4r 基线）下的区域分位数、极值与极端像素占比。
- scripts/t4s_codec_crosscheck.py、scripts/t4s_lzw_strip_check.py：用本机 Pillow 自带 libtiff 做 LZW 独立交叉核验，只作验证，不进入运行时。
- scripts/t4s_export_presets.py：导出/核验五个预设快照。
- scripts/t4s_report_tables.py：重算 TABLES、acceptance、device_checks、资源与脚本哈希，保留失败门槛和精确差异。
- scripts/t4s_commit_manifest.py：列出精确提交范围，不自动 stage，不含用户 paths.py、图片、日志、原件或被隔离的中断目录。

## 实际结果

候选源码指纹 27ba01f3aa13d29905af25f17284b68bc99da2897a40c957c2a5f4d898b9efb5，包含用户既有未提交的 paths.py，因此不等于纯提交指纹。

### 测试

完整 pytest **399 passed in 602.18s**，collect-only 同为 399，peak memory footprint=2.98 GiB。新增测试覆盖 TIFF mono/RGB/RGBI、DNG 严格契约、LZW/PackBits/Predictor2、大小端与 planar、非法/截断流与冻结守卫、本软件正片拒绝、HTTP 全链路、全局阳性跨块传播、局部兜底与黑白路由健康门槛。

### 308 帧回归

| 集合 / 预设 | N | T4s 中位/均值/p90 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 | 对T4r |
|---|---:|---|---:|---:|---|
| 开发 standard | 185 | 9.01 / 9.38 / 14.09 | 5/0 | 5/0 | 逐帧完全一致 |
| 开发 natural | 185 | 9.21 / 9.95 / 14.63 | 9/0 | 10/0 | 逐帧完全一致 |
| 开发 portrait | 185 | 9.52 / 10.50 / 16.18 | 14/0 | 16/0 | 逐帧完全一致 |
| 开发 vivid | 185 | 9.04 / 9.86 / 14.84 | 4/0 | 4/0 | 逐帧完全一致 |
| 开发 print_classic | 185 | 8.45 / 9.03 / 13.61 | 4/0 | 4/0 | 逐帧完全一致 |
| 检查 standard | 123 | 13.77 / 14.31 / 20.63 | 9/0 | 10/0 | 逐帧完全一致 |

相对 T4r 的 params 与 metrics 最大绝对差均为 0.0，差异记录 0 条，由脚本以不设容差的精确比较得出。print_classic 相对 T4o 有一帧最大 +0.0093，未超过 .05，不构成退步。五张保护帧 A008/A012/A013/DSC_3833/_DSF8948 参数与完整指标与 T4o 保持。全局预检同样经过这些帧（含 151 条 BlueNeg 扫描仪 DNG 记录）：309 条非错误记录（308 有效帧加检查集中被 NCC 排除的 1 条）逐条比较，spike_clamp 的钳制像素数与占比、d_hi、白点信息和片基方法与 T4r 完全一致，差异 0 条，说明全局预检在这些帧上没有改变任何钳制结果。这是已测素材上的实测，不是对未知素材的安全性证明。

### A1296（分块全局预检）

同一区域（上带 rows 10–20%、cols 20–80%）在不同预览尺寸下的实测：

| 口径 | 基线上带蓝 p99 | 候选上带蓝 p99 | 基线上带蓝最大值 | 候选上带蓝最大值 | 基线 / 候选任一通道>10 占比 |
|---|---:|---:|---:|---:|---|
| 1000px 预览 | 9,407,591 | .964962 | 2.82e7 | 1.163 | 7.98% / 0 |
| 1600px 预览（默认设备口径） | 10,583,540 | **1.074991** | — | — | — |
| 整幅 3870×5796 | 1.5626 | 1.3762 | 1.69e8 | 9.05 | 0.34% / 0 |

- 全局预检直接消除了第 1、2 行块上的稀疏蓝爆点：整幅分辨率下最大值从 1.69e8 降到 9.05，任一通道>10 的像素占比从 0.34% 降到 0。预览里的百万量级条带也随之消失。
- 但 **p99<1.0 的硬验收在默认 1600px 口径下失败**（1.074991），1000px 通过（.964962），整幅为 1.3762。p99 对预览块均值与尺寸敏感：基线在整幅下 p99 只有 1.56，说明极端值是稀疏像素，经块均值被放大后才体现在 p99。因此该判据既不能证明整幅修复，也不能替代默认口径的失败；失败如实保留，不追加阈值。
- 中心区域（20–80%）候选与基线逐通道完全相同（本改动未触及）：1600px 为 [1.194014,1.227335,1.247261]（<1.35 通过），1000px 为 [1.131198,1.157580,1.148001]，整幅为 [1.5286,1.7174,1.9216]（整幅口径超过 1.35）。这张图是大片白雪，p99 超过白点不一定是伪影。
- 任务书预测的具体分位数（上带蓝 .8282、中心 [1.0658,1.0978,1.0691]）未复现。衍生图对照中上部蓝色噪点水平带明显减少，但全图仍有噪点、片外高值与偏色，不宣称恢复欠曝信息或完全消除伪影。status=ok，full_positive_finite=True，渲染不修改正片。

### 黑白整卷（Acros）

真实卷 28 帧，n_base=28、卷片基无 warnings。基线 24/28 ok，B003/B011/B016/B017 四帧均 manual_review_required（span .1889/.1841/.2047/.2151）。候选 28/28 ok，四帧均自动 roll_nat、flags=[]，span 分别 .2869/.2455/.2985/.2977 D（均 ≥.24）。全部 28 帧 full_positive_finite=True、render_preserves_positive=True。整卷另有 11 帧（B009/B010/B012/B013/B019/B020/B021/B022/B025/B026/B029）也被路由到 roll_nat，其中 B026 带 backlit_midtone 标志；这些作诊断记录，不属于任务指定验收，也未单独评估其颜色是否更好。status=ok 只表示自动估计状态，不等于颜色正确。

### 扫描仪 TIFF（8 张指定代表样本）

A1310、A304、A1301、A255、A1027、A1028、B032、B034 经 State.open 全部 status=ok、full_positive_finite=True、渲染不修改正片；G 密度跨度分别为 .505/.910/1.010/.446/.800/.908/1.441/.258。B034 为 4 通道黑白，红外以 interleaved_4ch 仅记录。该验证只证明读取与自动估计链路可用，不证明颜色正确，也不证明 TIFF 信号为物理线性（未应用 ICC/TRC）。

真实样本暴露并修复了两个解码缺陷：

1. A255（LZW+Predictor2）首次冒烟数分钟未完成：读取器没有丢弃已消费位，Python 整数随条带长度增长。修正为取码后掩蔽已消费位。最终源码上 A255 的 2000×1330 RGB 实际解码 6.48 秒。
2. A1027（VueScan 4 通道 LZW，7504 个单行条带）报 corrupt LZW code 514：部分条带在像素写满后不写 EOI，解码器仍继续读，把填充位当成非法码。修正为给定条带预期字节数且已满额即结束；未满额、超额、无预期长度却缺 EOI 仍报错。A1027 的 11 个抽样条带（含出错的第 405 条带）经解码与本机 libtiff 逐字节一致（lzw_strip_check.json）。

每次修复后均隔离旧指纹下的候选运行与检查点（experiments/t4_auto_ui/t4s/interrupted_before_*，不提交），并在新冻结源码上重跑全部正式验证。另用 Pillow libtiff 4.7.1 独立编码 512×1024 随机满字典与渐变 uint16，本项目解码逐像素相等（codec_crosscheck.json）；不由此推广为所有 LZW 文件已兼容。

### 服务

服务按最终源码启动。POST /api/open 实测返回 auto.config_rev=t4s-2026-11-02；_DSF9239 status=ok、EV=.5492，6 个子卷、0 片基 flag；渲染为 3,362,409 字节合法 PNG。版本字段 http_config_rev 来自真实 HTTP 响应，source_thumb_rev=t4s、source_config_rev=t4s-2026-11-02 来自加载的源码。

峰值内存为 /usr/bin/time -l 的 peak memory footprint：开发 a/b 5.33/4.97 GiB，检查 7.89 GiB，设备候选 3.40 GiB，完整 pytest 2.98 GiB。原始字节数和日志 SHA 见 resource_evidence.json。

## 交付结论

实现、测试、308 帧锁定、Acros 四个目标帧、8 张 TIFF 和服务均已验证；A1296 的 p99<1.0 默认口径未通过，已如实记录，并以极值与占比的直接统计证明稀疏蓝爆点已消除。未验证的边界：TIFF 的物理线性与颜色正确性、任意第三方正片的识别、全局预检对 Dm 以外区域的保证（仅在全局阳性时统一开启）、APX 与 95 张新盲测的逐张复验。
