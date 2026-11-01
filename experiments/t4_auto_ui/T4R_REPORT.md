# T4r 五预设阻尼、裸背光片基与晴空/树荫收口（2026-11-01任务书）

实施、完整测试、20+3设备探测与185+123有效帧全量回归均已完成。相对T4o五预设均无>.05退步，standard检查集保持9好/0差（>.25）、10好/0差（>.05）；但portrait的A094相对T4q退步+.314241，**未通过双基线五预设零退步的完整验收条件**。基线4d31a38（T4q），分支t4r。用户既有src/filmcolor/paths.py改动保持原样，不纳入提交。任务书目标实图精确数值未全部复现，以下保留实测与边界。

## 实施范围

- 阻尼豁免：span_d_g>=1.15且tone bright_share>=.05时跳过overlift_damp；span未知继续原行为。按五预设分别相对T4o及T4q核验，不能把恢复旧误伤等同于相对所有基线都无收益损失。
- 所有新增条件均是统计启发式；裸背光强阶跃不等同于已识别物理光路，WB分区分歧也不是场景真值。未按目标文件名添加特殊分支。
- 片基：初次候选之后、均匀度放宽之前检测极端裸背光阶跃；p98Y>5*p90Y且超过88%有效块<=.25*候选阈值，至少10个可用均匀film块时排除Y>2.5*p90Y重估，保留诊断与警告。健康多帧子卷将bare_backlight_excluded与已有回退诊断同等路由至roll_nat。
- 尖峰：按任务书最小/中/最大归一化密度条件检测批量零截断逃逸，并用最小通道建立额外钳位。**作用域边界**：任务书示例函数按传入数组自身计算>1%；现有invert_negative以CHUNK_ROWS=256逐块调用invert，因此实际门控是每个行块，不是整图统一预判。全图<=1%仍可能因局部集中而触发；全图>1%的稀疏行块也可能不触发。preview.render_crop直接切缓存full_positive，无局部重反演。保持最小实现并核验实测，不能据全图占比声称308帧逐像素保持或任意分块一致。
- WB：缺少top_bottom_spread时亮占比门槛.12，否则.45；cyan_conf_boost要求zone_disagreement<=.60。半幅固定半径保持。
- 自动配置版本t4r-2026-11-01，THUMB_REV=t4r，扫描缓存键t4r使片基旧缓存失效；旧缓存不删除。

## 验证口径与约束

191开发记录固定排除六个缺ICC参考，185有效，沿用a96/b95，五种预设完整重跑；124检查候选按NCC>=.7沿用123有效。与T4o/T4q分别核对raw/ref SHA、有效成员、协议及源码指纹。BlueNeg无ICC参考按sRGB是假设，检查集来自已有开发卷，不作为未见新卷颜色真值。估计器不读取配对参考。

设备为原有18个文件加A1296.CR2、A1102.ARW，共20个color探测，B086/B087/B091另3次bw。基线从真实4d31a38源码重新提取并实跑，候选用当前源码，不将转发诊断当执行证据。64张盲测的逐张本地清单未提供，不声称复验全部64张；资料清单还含已有B087与A1010/_DSF8948同源说明，不将整批宣称完全独立未见。

全部原件读取先check_source(...allow_frozen=False)，只读原件，冻结集不解码；使用python -I，不载入pickle。未读取或复制GPL实现源码，不扩展实现全局正片16上限。图片、日志、临时目录和用户paths.py不提交，不push、不对外发布。

## 复现入口

全部使用.venv/bin/python -I；图像任务设置FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data。

- scripts/t4r_prepare_baseline.py NEW_DIR：从真实4d31a38提取源码及冻结守卫清单；不读原件。设备基线SHA=d9a29d38e01b6415808f795acc47d62184a68c3bb43d8d7205bed4079c7e9f23，经git对象重算一致。
- scripts/t4r_run_devices.py SRC baseline|candidate：20次color和3次bw；源与原件哈希不匹配时拒绝沿用checkpoint。
- scripts/t4r_run_eval.py：五预设及各预设old_overlift消融，standard另加old_wb_gate；检查集standard与这两项消融。片基/尖峰变化通过全链路基线对照，不称独立像素消融。
- scripts/t4r_test_evidence.py：提取完整pytest日志、数量、内存及测试文件哈希，并验证collect数量。
- scripts/t4r_report_tables.py：生成TABLES.md与acceptance.json，逐预设分别对T4o/T4q统计好差与退步名单；重算资源摘要及脚本SHA。失败设备状态保留，不因status断言而隐藏。
- scripts/t4r_device_sheet.py：本轮基线/候选衍生预览对照；图片只本地保留。
- scripts/t4r_service_check.py：验证已运行服务的files/roll/open/render，保留源版本字段与PNG哈希。
- scripts/t4r_commit_manifest.py：列出精确提交范围，不自动stage，不含用户paths.py、图片、日志、原件或.tmp.driveupload。

## 已确认的验收差异

完整回归确认portrait的A094为相对T4q唯一>.05退步（+.314241）：T4o ΔE=8.317867，T4q阻尼后=8.003626，T4r宽亮豁免后回到8.317867；old_overlift_portrait消融恢复8.003626。故这项改动相对T4o无退步，但相对T4q撤销了真实收益，任务书“同时对两个基线五预设零退步”不能成立。保留指定公式与退步证据，不追加样本特判。

## 全量回归与消融

以下每帧先取median ΔE2000，再跨帧统计中位/均值/p90。所有输入/参考SHA、源码指纹、有效成员、固定排除及评测协议经scripts/t4r_report_tables.py核验；逐帧与机器结果见[t4r/TABLES.md](t4r/TABLES.md)、[t4r/acceptance.json](t4r/acceptance.json)。好差为严格超过门槛。

| 集合 / 预设 | N | T4r 中位/均值/p90 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 | 对T4q 好/差 >.25 | 对T4q 好/差 >.05 |
|---|---:|---:|---:|---:|---:|---:|
| 开发 standard | 185 | 9.01 / 9.38 / 14.09 | 5/0 | 5/0 | 0/0 | 0/0 |
| 开发 natural | 185 | 9.21 / 9.95 / 14.63 | 9/0 | 10/0 | 2/0 | 3/0 |
| 开发 portrait | 185 | 9.52 / 10.50 / 16.18 | 14/0 | 16/0 | 1/1 | 3/1 |
| 开发 vivid | 185 | 9.04 / 9.86 / 14.84 | 4/0 | 4/0 | 0/0 | 0/0 |
| 开发 print_classic | 185 | 8.45 / 9.03 / 13.61 | 4/0 | 4/0 | 0/0 | 0/0 |
| 检查 standard | 123 | 13.77 / 14.31 / 20.63 | 9/0 | 10/0 | 0/0 | 0/0 |

- 宽亮豁免消融：standard/vivid/print_classic的自动参数0变化；natural只改变A033/A038/A123，2好/0差（>.25）、3好/0差（>.05）；portrait改变A033/A038/A094/A123，1好/1差（>.25）、3好/1差（>.05）。natural/portrait原先相对T4o的三帧误伤均恢复，但portrait同时撤销A094既有收益。
- WB新门槛消融：185开发及123检查standard均0参数变化；不将这项结果推广为未知素材安全性证明。
- 裸背光片基过滤：308有效基准帧0触发，目标A1296触发。尖峰报告在开发19970619B-09-notre-dame发生变化，钳位像素33359→111166（share .000488416→.001627605），其d_hi、standard ΔE=15.922052均保持；其余基准帧尖峰报告不变。因此“308帧逐像素不变”的预测不成立，与按256行块触发的已知边界相符。
- 五张保护帧A008/A012/A013/DSC_3833/_DSF8948的standard参数及完整指标字典与T4o保持。
- 峰值内存为/usr/bin/time -l的peak memory footprint：开发a/b 5.95/5.10 GiB，检查7.86 GiB，设备候选color/bw 3.22/1.08 GiB，完整pytest2.97 GiB。resource_evidence.json保存原始字节数与日志SHA。

## 当前证据

新增及调整测试包括裸背光阶跃、样本不足/正常场景不触发、恰好1%不触发与超过1%触发、holder最小通道>1.6保护、局部行块集中反例；阻尼宽亮双条件/单条件/未知跨度、no-TB亮阈值分支、cyan最大分歧门槛及健康多帧路由。专项base/invert 48项、noref/App 76项实际通过；完整pytest **385 passed in 389.71s**，collect-only为385项，peak memory footprint=2.97 GiB。test_evidence.json保存日志SHA、测试文件SHA及候选源码指纹；全部新测试纳入本轮完整运行。

T4q设备基线已复现A1296为manual_review_required、正片有限，警告white_point_invalid:p99 12.46stops；A1036与A1102的Temp均=.5。

T4r候选20个color+3个bw均status=ok、full_positive_finite=True、render_preserves_positive=True。候选源码指纹276fa74b10912b20a38f47ee8edd7db2cbe59412d02ae90b80ff05c14afca1a1；包含用户既有paths.py工作区差异，不等于纯提交指纹。

- A1296：bare_backlight_excluded=True，本帧估计片基[.01419604,.00909135,.00271867]，实际渲染片基[.01414511,.00906386,.00268559]；status恢复ok，EV=-.25、Temp=.0458、Tint=.0177，white_point_invalid消失，但仍有white_point_unrepresentative警告。预览画芯positive的各通道p99=[1.18128,1.21504,1.24661]；极少像素最大值仍为[5.1029,58074960,31750616]，图上仍可见稀疏彩噪/边缘亮点。因此不能声称纯蓝/高值像素已全部消除或与任务书p99=1.176口径一致。本轮只证明主体分布和自动状态改善，不证明欠曝原始信息恢复。
- A1036：split_sky=True、cool_cast=False、gw_cool_undamped=False，Temp .5→.3448、Tint→.0169；未复现任务书Temp≈.2512。
- A1102：zone_disagreement=.622423，cyan_conf_boost不再触发，Temp .5→.2357、Tint→.1515；未复现任务书Temp≈.2808/Tint≈.1626。

本地前后对照由scripts/t4r_device_sheet.py生成，仅衍生预览，未作为配对颜色真值。A1296的“p99”按预览画芯逐通道统计，和任务书的单值口径不同；峰值必须与主体分位数一并看，不能以finite=True替代无伪影验证。T4r服务已重新启动，HTTP实测6个子卷、0个片基flag，_DSF9239 status=ok、EV=.5492，返回3,362,409字节合法PNG，source_config_rev=t4r-2026-11-01、source_thumb_rev=t4r。版本字段记录于service_check.json，现有API本身不暴露同名source_*字段；没有为验收虚构HTTP字段。

## 交付结论

按任务书精确公式完成三组改动、完整测试、双基线五预设回归、消融、设备与服务核验。以本地T4r提交交付实现与全部文本证据；相对T4o五预设零>.05退步目标成立，相对T4q的portrait零退步目标未成立。A1296仍有彩噪/高值残留，A1036/A1102只复现门控和色温收敛，未复现目标精确值。保留这些失败和作用域边界，不把status=ok或测试通过等同于全部验收通过。

原件未修改、冻结图像未解码；用户paths.py、图片、日志及.tmp.driveupload保持在提交之外，无push或对外发布。服务在最终交付前再次通过HTTP打开与渲染核验。
