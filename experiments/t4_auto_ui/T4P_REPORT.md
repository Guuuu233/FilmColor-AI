# T4p 单通道扫描仪与高光/晴空收口核验（2026-10-31任务单）

四项实现、完整测试、185+123帧回归、18文件设备验证与本地服务更新均已完成。**未通过任务单零退步验收**：相对本地基线 `4cd76b7`，标准预设开发集3好/2差，检查集2好/6差（单帧ΔE变化严格>.25）；>.05分别为6好/6差、7好/12差。两张检查集救援目标已达成，A1036暖偏目标未达成。工作分支 `t4p`，用户既有paths.py保持不动；本轮变更保留在工作区，未创建新提交或push。不将任务单“无副作用”或示例指标视作实测结论。

## 实施范围

1. 单通道LinearRaw DNG：优先3/4通道彩色主图，无该主图才选择非缩略图1通道；二维或H×W×1复制为等值RGB，记录monochrome_dng，IR列表排除主图本身。
2. Step5高光降EV上限.60：记录highlight_hold.ev_lowered；按指定循环，未以新增p98条件另行门控。触顶后允许保留高于clip_budget的极少高光，不宣称所有输出都满足旧.2%预算。
3. 逆光救援按任务单公式保留Step4b证据：未降EV时用refine.med_tex_after与旧q25，q90=max(旧,当前)；有EV降幅时中位数取两者最小、q25取当前。保留T4o救援后裁切校验和回退。
4. 空间WB：半幅筛选半径取各区radius_effective最大值；区域分歧>.22且(空间分歧>.22或bright_px_share>=.45)为split_sky。采用任务单代码公式，没有加入文字段落中额外的灰世界蓝轴/亮区share门槛。

## 验证口径

185有效开发帧（191原记录、固定六项ICC参考排除、a96/b95），124检查候选经NCC>=.7筛选，历史123有效。full-resolution估计与1000px预览对齐。基线用已提交T4o证据，T4p候选重新运行并逐帧核对raw/ref SHA、有效成员和源码指纹。BlueNeg检查集是已有卷其他画格，无ICC参考按sRGB假设，不作颜色真值或独立新卷声明。

设备实际清单为18个独立文件（既有14+新增A1036/B086/B087/B091）；18个color探测与B086/B087/B091的3个bw探测。任务单“五个目标及既有14”中A1170重复，只计一个文件。所有原件先check_source(...allow_frozen=False)，只写symlink和衍生证据，未提供清单的59张盲测不声称已复验。

## 全量回归结论

以下为每帧median ΔE2000，再跨帧取中位/均值/p90。原件与参考SHA、固定排除项、有效成员、评测协议及候选源码指纹均已由scripts/t4p_report_tables.py核对，详情见[t4p/TABLES.md](t4p/TABLES.md)。

| 集合 | N | T4o | T4p | 好/差 >.25 | 好/差 >.05 |
|---|---:|---:|---:|---:|---:|
| NLP | 11 | 8.44 / 8.72 / 10.25 | 8.57 / 8.73 / 10.25 | 0/0 | 0/1 |
| 店扫 | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0/0 | 1/1 |
| BlueNeg开发 | 27 | 13.62 / 13.49 / 16.08 | 13.58 / 13.45 / 16.08 | 1/0 | 1/0 |
| 自有素材 | 126 | 8.75 / 8.91 / 13.26 | 8.71 / 8.91 / 13.26 | 2/2 | 4/4 |
| **开发集全部** | **185** | **9.03 / 9.45 / 14.14** | **9.01 / 9.45 / 14.15** | **3/2** | **6/6** |
| **BlueNeg检查集** | **123** | **13.84 / 14.43 / 20.73** | **13.85 / 14.43 / 20.63** | **2/6** | **7/12** |

- 开发目标199807A-09：EV 1.10 → 1.90，ΔE 10.271164 → 9.162867（-1.108297），mean_dL -9.323 → -2.953。实际ΔE不是任务单预测的约9.08；WB也随自适应半幅半径改变。最终clip_share=.00274（.274%），是开发集唯一超过.002的标准帧。
- 检查目标199807A-19：rescue_c1、weight=.4，ΔE 15.246108 → 13.909698（-1.336410），mean_dL -17.437 → -14.120；最终clip=.00070。
- 检查目标199807A-33：rescue_c2、weight=.85，ΔE 20.862067 → 18.676226（-2.185841），mean_dL -12.984 → -5.440；最终clip=0。其WB参数也有小幅变化。
- 既有开发救援19970620C-10和A103参数完全保持，ΔE分别为14.441746、9.095242。五张保护帧A008/A012/A013/DSC_3833/_DSF8948参数与指标保持，ΔE分别8.720626/4.460079/6.661887/4.664065/8.343404。
- 两张开发退步>.25：A149 6.733339 → 7.425905（+.692566），A150 6.767307 → 7.242958（+.475651）。六张检查退步>.25：19941222E-10 +.294157、19960808B-08 +.317291、19941222E-02 +.354523、19960812D-35 +.414733、19960812D-11 +.507488、199807A-22 +1.712863。
- 主要副作用来自半幅自适应半径：A149/A150原先缺失半幅差异，候选分别为.314999/.331011，改变空间冲突置信度；199807A-22从缺失spread变为.251758，zone_disagreement=.402158，使split_sky=True、cool_cast从True变False，Temp .5 → .1615，ΔE 14.233180 → 15.946044。其bright_px_share仅.019859；任务单“暗调真实冷偏帧tb_spread<=.05”未复现。_DSF8951的spread .018367 → .116718还影响灰世界不阻尼分支，ΔE +.126505。不能将这些变化归因于新增bright-share分支。
- WB的Temp/Tint改变开发15帧、检查21帧；cool_cast开发13帧、检查8帧（T4o为13/9）。old_sky_gate消融在两集合均0参数变化，说明本次新增亮占比分支未影响这些基准帧，不能证明自适应半径无副作用。
- old_highlight_drop消融仅开发199807A-09参数变化，1好/0差；检查集0变化。救援时机恢复了两张检查目标，完整组合仍有上述WB退步。
- 检查集所有有效标准帧clip<=.002；开发Step5降EV的为199807A-09（.6）与A035（.1）。相对T4n，开发7好/2差、检查3好/6差（>.25），也未复现任务单5/0与3/0。

五种开发预设均已完整评测，结果详表列出；natural、portrait均1好/2差，vivid 1好/4差，print_classic 2好/3差（>.25）。不将standard的局部收益推广为所有预设收益。

峰值内存（/usr/bin/time -l的peak memory footprint）：开发a/b为5.36/6.08 GiB，检查集7.49 GiB；设备候选color/bw为2.76/1.05 GiB，完整测试2.99 GiB。resource_evidence.json保存原始字节数与日志哈希。

## 设备实测

18次color与3次bw均为status=ok、full_positive_finite=True，且render_preserves_positive=True。B086/B087的两种模式都记录monochrome_dng=True、infrared_layer=None；基线两者均抛出IndexError，候选完成加载/反演/渲染。

- B091.RW2 / bw：EV -1.7651 → +.2349；Step5降幅由2.6限制为.6。显示画芯RGB各通道中位数 .204544 → .349502，最终clip_share=.00456，即.456%，高于旧.2%预算。这里用画芯显示RGB统计，不能与任务单未定义区域的srgb_q50=.156混用。color模式EV=.9787、clip=.00196，参数与基线相同。
- A1170.NEF：Temp .5000 → .2508，EV 1.2328 → 1.2445。WB有效样本bright_px_share=.537315（tone_pixels另一个样本口径为.52414）、zone_disagreement=.293472、top_bottom_spread=.124691，split_sky=True、cool_cast=False。实际Temp接近但不等于任务单的.246。
- A1036.RW2：**未达到暖偏修复目标**，Temp仍=.5000。实际WB bright_px_share=.16748、zone_disagreement=.267796、top_bottom_spread=.047315，因此指定公式不触发split_sky。任务单所述bright_share≈.519未复现。本地对照预览是蓝天桅杆画面，与任务单“雪地树林”的描述不同；基线和候选原件SHA相同，不能据此认定研究方用了哪个文件，也不按目标数值改阈值。
- 既有救援仍触发，但并非逐参数不变：A240/A328 rescue_weight .596183 → .587163；A329因q90取前后最大值，weight .4 → .85，midtone .262 → .298、shadows .064 → .136，最终clip仍=.0019。A099/A1094最终clip分别=.00134/.00170。
- 自适应半幅半径还影响既有空间冲突置信度：A277从缺失top_bottom_spread变为.731461，触发conflict；Temp .0707 → .0530、Tint .0624 → .0505。不能称彩色设备自动参数完全零变化，也没有配对参考可量化其颜色收益。

本地对照由scripts/t4p_device_sheet.py生成targets_before_after.jpg，仅供查看，不作为准确还原的颜色真值。

## 测试、全量回归与服务

完整pytest **376 passed in 394.26s**，最终collect-only为376项，峰值内存2.99 GiB。test_evidence.json记录日志哈希、源码指纹与全部测试文件哈希；scripts/t4p_test_evidence.py可由日志重建，不把日志纳入提交。noref新增3项后52项通过，scanner新增4项后16项通过。185+123有效帧回归已完成，固定六项ICC排除及一项检查集NCC排除保持。测试覆盖Step5默认.60、小数.25与配置EV下限，真实高光压制参数下q90跌破.45后保留Step4b证据，以及亮区占比与自适应半幅半径；旧T4o空间双条件测试显式关闭新增亮区分支，继续独立验证原契约。全量结果见最终TABLES.md；任务单预测的1好/0差与2好/0差不成立。

候选运行源码SHA256为45c7818133e3fb33b6f5aca24fcc3814783e3a905b1f591cc699cdb35f3efbd6，包含用户既有paths.py差异；不能将运行指纹说成纯提交指纹。开发集输出五种预设，检查集输出standard；old_highlight_drop将上限恢复3.0，old_sky_gate仅关闭新增亮像素占比分支。后者保留自适应空间半径，不能用其零参数变化证明整个WB升级无影响；救援时机与空间半径按T4o/T4p组合回归评估，没有宣称四项各自独立的全量消融。

已重新启动FilmColor服务（此前preview服务已被App停止，端口无监听，不涉及终止用户未知进程）。地址http://127.0.0.1:62872/，root保持data/raw/public/nlp_official，使用FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data。HTTP实测6个子卷、0个片基flagged，_DSF9239 status=ok、EV=.5492，返回3,362,409字节合法PNG；source_thumb_rev=t4p、source_config_rev=t4p-2026-10-31，highlight_hold.max_drop=.6。证据见t4p/service_check.json，可用scripts/t4p_service_check.py重算。

## 复现入口

全部脚本使用.venv/bin/python -I。图像任务设置FILMCOLOR_DATA_ROOT=/Volumes/waizhiyingpan/filmcolor_data。

- scripts/t4p_prepare_baseline.py NEW_DIR：从4cd76b7提取源码与清单，不读取原件。设备基线源码指纹8275d812508e544d4812509097d03a8d39ce5508084dda1f58505b6bfcd4ee50，经git对象重新计算与baseline记录一致。
- scripts/t4p_run_devices.py SRC baseline|candidate：18个文件color及3个bw，经App路径完成自动处理；源或原件哈希变动时拒绝沿用旧checkpoint。
- scripts/t4p_run_eval.py：开发两进程a96/b95，之后检查集单进程，输出同正片的两项消融及五种开发预设。
- scripts/t4p_test_evidence.py：完整pytest后提取结果、内存、日志哈希与测试文件哈希，并复核最终collect数量。
- scripts/t4p_report_tables.py：重算指标、变化名单、逐预设结果、保护帧和设备表；校验有效成员、输入与源码哈希、评测协议、全部设备状态和有限性。日志中的内存摘要另存resource_evidence.json，日志不提交。
- scripts/t4p_device_sheet.py：由本轮衍生预览生成本地对照图。
- scripts/t4p_service_check.py：对已运行的本地HTTP服务核验files/roll/open/render，不自行启动服务，不扩大服务root。

## 已知机制边界

- Step5按任务单累计请求步长，highlight_hold.ev_lowered在最后一步碰到EV下限且实际剩余不足.1时，可能比实际降幅多记不足.1 EV；实际EV仍受下限和最大降幅约束，救援只依赖其是否大于0。当前不将这个诊断字段泛称为任意配置下的精确EV差。
- 保留T4o救援后clip_budget保护。若Step5达到.60上限、肩部已到.85且仍超预算，后续救援即使命中c1/c2，也可能因裁切校验回退、rescue_weight归零；不为了救援重新无限降EV。

本轮实施与核验已完成，零退步验收未通过。保留任务单指定阈值及全部失败证据，没有按单帧追加调参。实现、测试、复现脚本和报告保留在t4p工作区，HEAD仍为基线4cd76b7，未创建新提交；未stage用户既有paths.py或无关文件。原件只读，冻结图像不解码，不读pickle；图片和日志仅本地保留，没有push或外部发布。
