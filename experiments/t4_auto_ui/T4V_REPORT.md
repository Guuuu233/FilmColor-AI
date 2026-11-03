# T4v：扫描仪 TIFF 位深、传递曲线与预拉伸重构

基线98ded3c（T4u）。本地main已由e8a8c3d以--ff-only快进至98ded3c，再创建t4v分支；没有push。用户原有paths.py哈希fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913保持不变，不暂存、不提交。候选源码SHA256为`0b711db2f9f61cc53f50aaeb9a3b6503b436c9916014e6a899e50affb9819071`；九项正式门禁全部通过（`all_passed=true`，234项设备子门禁`failures=[]`）。

## 实现与契约

- TIFF选择分支接受统一uint8/uint16，继续拒绝float、有符号、非标位深及可识别的FilmColor导出正片；DNG页面选择契约不改。
- 缺imagecodecs时，条带回退支持LZW、PackBits、Deflate8/32946，uint8/uint16的Predictor2按对应无符号位宽累加。Deflate解码错误和长度不符报ValueError；未扩大到tiled等布局。
- 独立NumPy sRGB EOTF避免scanner依赖render。8-bit默认sRGB或使用嵌入单曲线；16-bit仅对mntr/spac采用任务书指定的TRC策略，scnr和无ICC保持既有像素。解析curv gamma/LUT及任务书指定para处理，记录tiff_eotf；不执行完整ICC色彩转换或矩阵变换。
- ICC读取验证tag条目和payload长度，避免越界/错误跨tag读取。当前单曲线/para策略不等于通用ICC解释器；尤其任务书将para function3视为sRGB、其他正gamma只用幂次，是本轮限定的简化策略，不证明任意ICC都准确。
- 对16-bit TIFF、R单通道大面积触顶、无工作空间TRC、B p01<.03且R p01>.40时，执行任务书指定的R尾外推、共享pedestal搜索及gamma2.2重构。未命中时不改像素。此为基于跨通道关系的估计，不是从截断原件恢复唯一真实值。
- pipeline input.kind区分scanner_tiff8/scanner_tiff16，并如实说明实际TRC或重构步骤；DNG描述保持。版本t4v-2026-11-03，缩略图/扫描缓存前缀t4v，五份JSON快照同步。

原件只读，所有外部原片在读取前经过check_source(..., allow_frozen=False)，冻结图像未解码。Python-I，无pickle、无训练、无新依赖、无外部GPL源码复制。历史t4o/t4r/t4s/t4t/t4u共78份顶层JSON与对应提交字节一致，冻结清单哈希、主干快进记录和任务书哈希在input_protocol_sha256.json。

## 已测目标

| 样本 | T4u基线 | T4v候选实测 |
|---|---|---|
| A1040.TIF | RGB中位数[0,.16095,.24637] | [.14534,.18671,.23498]，ok、flags=[]；R>.10、G>.10、B-G<.08 |
| A1041.TIF | [0,.21466,.39837] | [.20340,.24639,.41699]，ok、flags=[]；R>.15、G>.15 |
| A1228.tif | manual_review_required，low_information，G白点=.20502 | ok、flags=[]，应用gamma:1.8008，G白点=.37166 |
| A229/A655/A849/A850/A1093/B036 | 均无法打开 | 六张均ok，finite及render-preservation按探针保存 |

A1040的重构r_base_est=1.03、slope_rg=.44；A1041分别1.0939/.6409；两者pedestal_p0均=.55，命中任务书搜索上界。不能把这个搜索结果认作测得的真实扫描黑电平，也不能据此证明扫描编码确为gamma2.2。重构改变图像层次与色彩，存在模型假设；只按指定样本量化门限评估。

独立审计18张指定TIFF的头部：A1301/A1028/B032为scnr并保持linear；A1228为mntr、curv存储的gamma=1.80078125；A229为mntr LUT4096；A655/A849/A850按任务书para策略为sRGB；A1093/B036无ICC按8-bit默认sRGB。没有复验研究方所有22张ICC或97张新盲测，不宣称完整覆盖。

图像质量边界：A229仍明显偏蓝，A850整幅中位数约[.00734,.00663,.01022]，虽status=ok不代表视觉正确；没有色卡或独立颜色真值。8-bit量化损失不可恢复。scnr设备类别本身不证明像素线性，本轮忽略scnr曲线是保持既有信号契约的政策，不是普遍物理定律。A1296沿用T4u上带和中心作用域门限，整图极端值仍保留诊断。

## 308帧回归与资源结果

开发集185帧（191条manifest固定排除6条无ICC参考）五套预设（standard/natural/portrait/vivid/print_classic）与检查集123帧（124条非s3_5固定排除1条低NCC `blueneg:199807A-21-orlando-disney`）standard完整实跑完成：
- 对已提交T4u（`98ded3c`）、T4t（`567aaf9`）、T4s（`7c5107c`）、T4r（`fbd48a0`）：`max_abs_diff_params = 0.0`、`max_abs_diff_metrics = 0.0`、`diff_count = 0`；全部309条非错误记录的`d_hi`、`white_point`、`spike_clamp`、`base_method`反演诊断差异为0条。
- 对已提交T4o（`4cd76b7`）：全部五套预设及检查集保持`worse_025 = 0`、`worse_005 = 0`（dev standard 9.01/9.38/14.09，好/差 `>.25` 5/0、`>.05` 5/0；cross standard 13.77/14.31/20.63，好/差 `>.25` 9/0、`>.05` 10/0）。
- 五帧保护样本（`ext:A008`、`ext:A012`、`ext:A013`、`shop:DSC_3833`、`nlp:_DSF8948`）五预设参数与指标精确一致。
- 回归峰值memory footprint：`candidate_a` 5511140176 bytes（5.13 GiB）、`candidate_b` 5858170088 bytes（5.46 GiB）、`candidate_cross` 8371919712 bytes（7.80 GiB）。
- 本地HTTP服务（`http://127.0.0.1:62872`）：`http_config_rev == "t4v-2026-11-03"`、`source_thumb_rev == "t4v"`，打开`_DSF9239.RAF`（`status="ok"`，`ev=0.5492`）并渲染PNG（3362409 bytes，SHA256 `b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff`）验证通过。

## 设备回归结果

设备基线与候选均已完整结束：28张Acros、8张既有TIFF、8个指定目标、A1296及7张新增TIFF。独立调用device_checks得到230项全通过、failures=[]；A1296作用域和原始编码饱和掩模还会在正式汇总中追加核验。既有28张Acros/8张TIFF的完整inversion、params、status直接比较完全一致，指定保护目标的d_hi/params/status也一致。

A1296整图任一通道>10为876718/22430520=.03908594；中心为419/8073594=5.189758e-5；上带为0/1345599。仍有中心极端值，不宣称全图没有问题；本轮沿用任务书指定作用域验收。

设备实跑峰值memory footprint：基线3815919216 bytes，候选4079013896 bytes；墙钟998.69与1491.79秒。两者有并发负载差异，不据此作速度比较。设备JSON中的source/script/input SHA均需通过最终汇总核验。

## 测试契约补充

本轮新增五项合成测试，并将旧8-bit拒绝断言改为接受和EOTF数值校验；保留12-bit、4-bit、32-bit和FilmColor正片拒绝保护。新增测试覆盖缺imagecodecs时的四种压缩标识、Predictor2、ICC gamma/LUT与异常tag、scnr不变、预拉伸触发/跳过及R触顶过渡的单调连续性。ICC LUT期望值使用实际存储的uint16量化曲线，不放宽容差。异常ICC的unknown_type fixture明确写入xxxx签名并断言，避免测试名与数据不符。

完整测试首次暴露旧HTTP导出测试仍匹配固定句子“ICC / TRC not applied”；打开、渲染和导出均已成功，失败仅在描述文本断言。tests/test_app.py同步核验线性路径的实际说明“signal assumed linear; no TRC applied”及“no ICC colour matrix transform”，没有为通过测试更改生产行为。最终完整测试按冻结后的测试文件重新运行，实际414 passed in 413.15s，0 failed，collect-only同为414；峰值memory footprint为3254684768 bytes（约3.03 GiB）。基线通过Python -I重跑新增五项测试，实际4 failed、1 passed、24 deselected；测试和隔离源码哈希已记录。35项scanner/import-check/HTTP导出专项测试也通过。

## 验证协议和入口

开发清单191条固定排除6条缺ICC后185有效，五预设；检查124候选固定排除一条低NCC后123有效standard。完整params/metrics及反演诊断对T4u/T4t/T4s/T4r精确比较，不设容差；相对T4o使用严格>.05、>.25。输入raw/ref SHA、成员和配置协议均校验。检查集并非未见新卷颜色真值，BlueNeg无ICC参考按sRGB仍属假设。

所有脚本使用.venv/bin/python -I；图像任务设置FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data。

- scripts/t4v_prepare_baseline.py NEW_DIRECTORY：提取T4u源码和冻结守卫，保存paths.py覆盖来源。
- scripts/t4v_probe_devices.py SRC baseline|candidate：重跑A1296、8TARGETS、7新增TIFF、28Acros、8既有TIFF。只保存选定scanner字段，无序列号。
- scripts/t4v_icc_audit.py：独立记录指定TIFF位深、压缩、ICC类别/哈希和曲线策略。
- scripts/t4v_scanner_saturation.py：从编码原始样本测饱和率，核对重构前sat掩模，避免从重构后的像素反推饱和。
- scripts/t4v_a1296_scope.py：整图/中心/上带范围统计。
- scripts/t4v_visual_compare.py：新增7TIFF及两张重构样本的前后图；基线打不开的格子标明unavailable，JPG不提交。
- scripts/t4v_run_eval.py：完整开发五预设、检查standard实跑。
- scripts/t4v_baseline_test_evidence.py：Python -I 显式载入隔离T4u源码，重跑本轮五项合成测试，保存失败数、实际模块路径及测试/基线源码哈希。
- scripts/t4v_test_evidence.py：全量pytest与收集数/测试和源码哈希/峰值证据。
- scripts/t4v_service_check.py：实际HTTP版本、打开和PNG渲染。
- scripts/t4v_export_presets.py --check：运行时与五个快照完全一致。
- scripts/t4v_report_tables.py：生成TABLES、acceptance、device_checks、资源与脚本哈希。
- scripts/t4v_commit_manifest.py：仅all_passed=True时输出精准清单，排除原件/图片/日志/paths.py。
- scripts/t4v_final_audit.py：核对分支、main里程碑、历史JSON提交字节、冻结清单、用户paths.py哈希、最终diff/暂存清单与JSON敏感字段。
