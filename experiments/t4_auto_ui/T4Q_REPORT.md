# T4q：T4p 固定半径收口与高曝暗部阻尼（2026-11-01）

实施、设备核验、完整测试与185+123有效帧回归均已完成。相对T4o，standard开发5好/0差、检查9好/0差（>.25），>.05分别5/0、10/0；所有standard帧ΔE均未增加，T4p既有退步全部消除。**standard好差计数达成任务书目标，汇总预测值未完全复现；natural/portrait仍存在退步，五预设全局零退步条件未通过。** 不将任务书预测指标视为本轮证据。基线为T4o提交4cd76b7加此前未提交的T4p工作区。当前分支t4q；用户既有src/filmcolor/paths.py改动保持原样，不纳入提交。

## 实现与边界

- 模块A：上下半幅quads恢复配置中的固定最大半径（默认.35）；亮部/中灰区本身的自适应半径保持。保留T4p单通道LinearRaw、IR排除、Step5降曝最大.60、救援前序分位数、bright_px_share>=.45分支。缓存扫描键保留t4p（输入/片基算法本轮未改），THUMB_REV与自动参数版本更新t4q。
- 模块B：在Step5高光保护及救援结束之后、Step6反差调整之前，以Step4b已经测得的q25和当前EV判断。EV>=1.50且q25>=.35时，pull=clip(3.2*(q25-.33)*clip((EV-1.30)/.50,0,1),0,.65)，下调EV并按min(.06,.10*pull)微压blacks，两者仍受现有参数下限约束。新增overlift_damp诊断。采用任务书明确允许的后置位置，避免改变Step4b原有逆光权重及Step5救援判据。
- 任务书配置清单的overlift_ev_Pivot与代码小写不一致；实现统一为overlift_ev_pivot。
- 这是本项目的经验阻尼规则，不是复刻NegPy完整Boyack & Juenger锚点或1.9D相纸模型。本轮未读取或复制GPL实现源码，未加入密度域解混、TIFF支持、肤色保护或其它对比报告建议。
- 物理说明修正：sRGB中性灰编码值.35对应L*约37.93，不是L*>65；编码亮度分位数本身也不等同于CIELAB分位数。固定阈值的有效性以实测为准。宽光谱传感器的log积分响应也不宜泛称严格线性的密度串扰矩阵；本轮不据此改动反演。
- 按任务书精确公式，阻尼没有另加“p98未触顶”或场景分类条件。EV/q25门槛命中只说明这组统计成立，不能据此证明原场景是低调、算法必然过曝或失去所有黑位；验证结论限于本轮配对参考的改善/退步。
- 高光.60上限仍可能保留超过.2%的高光裁切；保留T4o救援后预算校验/回退。A1036的本地输入与任务书场景描述差异已记录于T4P_REPORT.md，本轮不为追逐Temp目标调整阈值。

## 验证协议

开发集191记录、固定六个缺ICC参考排除后185有效，沿用a96/b95拆分；检查124候选、固定NCC>=.7后123有效。候选重新full-resolution解码/反演/估计，1000px预览对齐，逐帧核对输入与参考SHA、有效成员、协议及源码指纹。参考只用于离线评测，估计器不读参考。BlueNeg无ICC参考按sRGB是假设，检查帧来自已有开发卷，不作为未见新卷或颜色真值。

开发五种预设完整运行；standard另有no_overlift消融，检查集运行standard与no_overlift。no_overlift为模块A独立结果，用于区分固定半径收口与新增阻尼。设备18个独立文件跑color，B086/B087/B091另跑bw，共21次。设备对照沿用T4p当轮从4cd76b7提取源码的新鲜基线，逐原件SHA核对。未提供逐张清单的59张盲测不宣称全部复验。

## 复现入口

全部命令使用.venv/bin/python -I，图像任务设置FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data。

- scripts/t4q_run_eval.py：按T4o的a96/b95划分跑开发五预设及no_overlift，随后跑124检查候选的standard/no_overlift。checkpoint不匹配源码/变体/清单时拒绝复用。
- scripts/t4q_run_devices.py src candidate：18个color与3个bw App路径探测；原件路径清单与T4p相同。
- scripts/t4q_test_evidence.py：从完整pytest日志提取数量/耗时/内存和文件哈希，并核对最终collect数量。
- scripts/t4q_report_tables.py：核对191/185与124/123成员、输入/参考/源码哈希、协议、模块B on/off前序诊断、保护帧、设备状态与有限性，重建TABLES.md、acceptance.json、resource_evidence.json及脚本哈希。
- scripts/t4q_device_sheet.py：从T4p的T4o基线衍生预览与T4q候选衍生预览重建本地对照图；不读原件。
- scripts/t4q_service_check.py：验证已经运行的FilmColor服务files/roll/open/render并记录最小证据。
- scripts/t4q_commit_manifest.py：列出本轮T4p/T4q文本证据、脚本、实现和测试的精确提交范围；不stage，不含paths.py、原件、图片、日志或.tmp.driveupload。

T4p实现未单独提交，其脚本与历史失败证据会随本轮正式提交保留；T4P_REPORT.md描述的是T4p当轮候选状态，T4Q_REPORT.md描述当前收口结果。

## 当前已验证结果

开发185有效帧已经完成：standard相对T4o **5好/0差**，>.25与>.05均相同；median/mean/p90为 **9.007481 / 9.382650 / 14.089629**（显示9.01/9.38/14.09），并未复现任务书所述8.95/9.38/13.97，不能声称中位数首次低于9、p90低于14。排序后第93帧为未变化的_DSF9238（ΔE=9.007481）；p90由A227（14.025932）与A127（14.132093）插值得到，不以任务书四舍五入预测替代实际统计。

开发目标199807A-09 ΔE=9.078063、EV=1.90、mean_dL=-2.937310。新增阻尼命中A105（14.973155→13.810336）、A211（11.934733→9.376038）、A212（15.420377→10.948056）、A218（10.968294→7.841977）。

五预设相对T4o：standard 5/0、natural 9/2、portrait 15/1、vivid 4/0、print_classic 4/0（>.25）；>.05分别5/0、10/3、17/3、4/0、4/0。natural退步：A033 +1.457500、A038 +.442415、A123 +.153779；portrait退步：A033 +1.246058、A123 +.171874、A038 +.140025。均触发overlift_damp，本轮保留指定默认机制，不为了使五预设零退步另加门控或阈值。零退步结论只限实际满足条件的预设。

开发消融中，模块A独立仅改变199807A-09（ΔE -1.193101）；模块B独立仅命中上述4帧且全部改善；全部185帧的backlit/highlight_hold诊断在阻尼开关下完全一致。

开发standard的Temp/Tint相对T4o全部保持；NLP与店扫域指标不变，BlueNeg开发域1好/0差，自有素材4好/0差。A008/A012/A013/DSC_3833/_DSF8948五张保护帧与19970620C-10、A103两张已有救援帧的参数及完整指标字典均与T4o相同。

检查124候选、123有效帧全部完成。相对T4o，standard **9好/0差（>.25）、10好/0差（>.05）**；median/mean/p90实测 **13.772807 / 14.314420 / 20.634657**，显示13.77/14.31/20.63，并非任务书的13.68/14.29/20.63。模块A仅改变19/33两帧，2好/0差；模块B命中9帧，7好/0差（>.25）、8好/0差（>.05）。13次阻尼命中（开发4+检查9）的ΔE均严格减少，但任务书中近零收益的19960811B-36实测仅改善.000057759（5.955942589→5.955884831），不把13次命中都称为显著改善。

| 集合 / standard | N | T4o 中位/均值/p90 | T4q 中位/均值/p90 | 好/差 >.25 | 好/差 >.05 |
|---|---:|---:|---:|---:|---:|
| 开发 | 185 | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 5/0 | 5/0 |
| 检查 | 123 | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 9/0 | 10/0 |

检查救援199807A-19 ΔE 15.246108→13.909698，199807A-33 20.862067→18.681808。检查阻尼较大收益包括199807B-01 15.869348→12.199066、19960812D-27 15.775887→13.520462。全部308有效standard帧的Temp/Tint相对T4o保持，cool_cast开发13/检查9；no_overlift与standard的前序backlit/highlight_hold诊断全部相同。T4p历史>.05退步在当前标准结果中均已消除。

最终输入/参考SHA、有效成员、协议、源码指纹、保护帧、设备及测试哈希均经scripts/t4q_report_tables.py核验，生成[t4q/TABLES.md](t4q/TABLES.md)与机器可读[t4q/acceptance.json](t4q/acceptance.json)。验收JSON明确primary_variant=standard并包含五种开发预设的退步名单，不以主预设计数掩盖其它预设问题。开发199807A-09最终clip=.00278（.278%），为唯一超过.002预算的有效standard开发帧；检查集无超预算帧。

完整pytest **377 passed in 339.71s**，最终collect-only为377项，peak memory footprint为2.77 GiB。test_evidence.json保存完整日志哈希、测试文件哈希与源码指纹，源码与设备及回归检查点一致。noref专项53项通过（python -I -m pytest），新增覆盖阻尼公式、EV/q25不触发条件、等号边界、pull上限以及前序backlit/highlight诊断不变量；固定半幅半径测试用全部色度范数>.35的合成图确认即使zone自适应扩半径，半幅spread仍缺失，并补充低色度候选充足时正常产出spread。scanner的4项T4p新增测试完整保留。全部新增及既有测试均包含在本轮完整运行中。

18次color与3次bw均status=ok、full_positive_finite=True、render_preserves_positive=True；B086/B087均记录monochrome_dng=True、infrared_layer=None。A277 Temp恢复.0707，A1170保持.2508，B091 bw EV保持.2349、clip_share=.00456。A1036仍Temp=.5，本轮未解决。该设备清单均无overlift_damp命中。

服务已用preview配置重新启动（旧T4p服务已被App停止，启动前端口无监听，无需终止用户其它进程）。HTTP验证6个子卷、0个片基flag，_DSF9239 status=ok、EV=.5492，渲染返回3,362,409字节合法PNG。THUMB_REV=t4q、配置版本t4q-2026-11-01，证据t4q/service_check.json。地址http://127.0.0.1:62872/，root保持data/raw/public/nlp_official，FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data。

资源证据边界：开发a的peak memory footprint为5.75 GiB（6,177,838,384 bytes），检查集6.37 GiB，设备color/bw为3.43/1.08 GiB，完整测试2.77 GiB；原始字节数和日志SHA见resource_evidence.json。开发b的JSON完整95条且父进程成功进入并完成检查集，但其日志文件为空，本轮缺少b进程内存统计，未用a或其它运行数字替代。

候选运行源码SHA256：8abc02b946fcd9408cf2f3eaff137837781a4841387facb6f874259ceadf5f4c；含用户既有paths.py工作区差异，纯提交指纹不能冒充该运行指纹。

## 交付结论

本轮按任务书完成模块A与B、完整测试、五预设开发集、检查集消融及设备验证，并以正式本地T4q提交交付。相对已提交T4o基线，standard合计14好/0差（>.25），15好/0差（>.05）；仅这些测试成员与预设支持该结论。开发/检查集汇总预测值、五预设无单帧>.05退步这两项目标未全部达到，不能称任务书所有验收条件均通过。A1036未解决、稀疏高光预算超限和资源证据缺口均如上保留。

原件只读；全部图像读取先check_source(...allow_frozen=False)，冻结原件不解码，不载入pickle。T4p历史失败报告与完整文本证据随本轮提交保留，不改写为已通过；图片、日志、.tmp.driveupload和用户paths.py差异不纳入提交。无push或对外发布。服务已更新并在最终交付前再次通过HTTP打开/渲染验证。
