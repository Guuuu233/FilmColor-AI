# 数据来源总表

| source_id | 来源 | 获取日期 | 内容 | 许可 / 可用范围 | 备注 |
|---|---|---|---|---|---|
| ultramax400_r01 | 用户自有：用户本人拍摄的一卷 Kodak UltraMax 400（全能 400）135 彩负，冲扫店用 Nikon Z 7_2 + NIKKOR Z MC 105mm f/2.8 VR S 翻拍（2024-03-30，62 秒内拍完 37 格）；复制自 `~/Downloads/原件`，复制后逐文件 sha256 与原件一致 | 2026-10-14 | 37 张 NEF（Bayer RGGB，14 bit），翻拍时只拍画面、无片边；光圈优先 f/16、ISO 100、矩阵测光 +1 EV，快门随画面在 1/25～1/60 之间变化；另有 `DSC_3820.JPG`（sha256 `dfed0c89…abb2`），拍的是一段贴有“0625”标签的胶片，只有 JPEG，不列入 RAW 清单 | **training-ok**：用户拥有照片的完整版权 | 参考图见 shop_r01 |
| shop_r01 | 冲扫店为 ultramax400_r01 逐张去色罩、校色的正片；复制自 `~/Downloads/逐张去色罩校色`，复制后逐文件 sha256 与原件一致 | 2026-10-14 | 23 张 sRGB JPEG，7956×5300，Adobe Photoshop 22.4 (Windows) 处理；按画面内容与 NEF 配对，23 张全部唯一配对，且与文件顺序一致（`shop_r01_reference.json`） | **training-ok**：用户拥有完整版权，店家只做加工、无版权主张，交易协议中已注明（用户 2026-10-14 告知） | 人工修图师的审美结果，不是色彩真值；8 bit JPEG；只用于评估与参考图拟合，不参与无参考优化器求解 |
| nlp_official | https://negativelabpro.s3-accelerate.amazonaws.com/sample-raw-scans.zip（zip sha256 `08e98251…3ca4`，Last-Modified 2020-09-30） | 2026-10-13 | 12 张 Fujifilm X-T2 + XF80mm Macro 翻拍的彩色负片 RAF（X-Trans），从画幅外观推断为 120 中画幅，均含片基边缘 | **dev-test-only**：压缩包内无任何许可或说明文件，未见明确授权；不得用于商业训练或随软件分发 | 无 NLP 输出参考，需用户自行用 NLP 默认设置转换 |
| fabio_sirna_s5m2 | https://fabiosirna.com/notes/2026/a-data-driven-approach-to-film-inversion/ 附带的压缩包（zip sha256 `b3955c4e…1f27`） | 2026-10-15 | 1 张 Panasonic S5II 高分辨率合成模式 RW2（12000×8000 有效区，Bayer RGGB，黑电平 512、白电平 16319，crop 边距 4/4，sha256 `ea0ee6e6…80e1`）+ darktable xmp + README；文章称 Kodak Portra 400、Pentax 67II 拍摄，Sigma 105mm 微距 + Kaiser 5000K 背光翻拍（未独立核实） | **dev-test-only**：README 声明 CC BY-NC 4.0，禁止商业用途；分享结果需署名并链接原文 | 翻拍曝光很低（最大值约满量程 13%）；2026-10-15 用冻结的 noref_v2 转换并目视检查（`experiments/s3_intake/camera/`，无参考图），未参与任何调参或研发决策 |
| blueneg_sample3 | https://huggingface.co/datasets/ttgroup/blueneg-release（ICCV 2025 BlueNeg），按 meta.json 从 blue-intact 子集抽 3 张：`199807A-06-hk`（Fuji 400，白天室外）、`199910-01-korea`（Kodak Gold 400，夜景）、`19960808B-03-new-orleans-siggraph96`（Fuji NPH400，室内） | 2026-10-15 | 每张：负片 DNG（约 550 MB）、冲印 16 bit TIFF（约 3000 px 宽、无 ICC）、8 bit 预览 PNG；6 个大文件 sha256 与 Hugging Face LFS 记录一致；另存 LICENSE、README、meta.json（未下载、不加载 transformations.pkl） | **training-ok（需署名）**：LICENSE 允许学术与商业使用，条件是任何发表、复制、再分发或衍生中注明 “Copyrighted by Tien-Tsin Wong” | DNG 由 Plustek OpticFilm 8200i 胶片扫描仪经 SilverFast 8.8 生成，**不是相机 RAW**。2026-10-15 读标签核实：主图在 SubIFD，3 通道 16 bit `LinearRaw`（无 CFA），无 BlackLevel / WhiteLevel 标签（按 DNG 默认 0 / 65535；实测最大值约满量程一半）；另有同尺寸单通道层（NewSubfileType 4），XMP 中 SilverFast 设置 `isWithIR=TRUE`，判为红外层，不进颜色处理；ColorMatrix1 即标准 XYZ→线性 sRGB 矩阵、AsShotWhiteXY 为 D65，属名义标注而非校准；XMP 记录负片模式、iSRD 除尘、MultiExposure 等任务设置，像素未反转，iSRD 是否作用于 RGB 无法确认。由 `filmcolor.scanner` 读入。冲印 TIFF 无 ICC、由 tifffile 重存，色彩空间未知（如需比较只能假设 sRGB 编码，须标为假设）；1980～2000 年代老胶片、冲印可能褪色。仅 `199807A-06-hk` 做过一次冒烟转换与目视检查，未参与任何调参 |
| nlp_forum_9652 | NLP 论坛帖 https://forums.negativelabpro.com/t/9652 （kur1j，2025-08-11）第 6 楼分享的 Google Drive 文件 | 2026-10-15 | `DSC03060.ARW`（Sony A7R II + FE 90mm Macro + VALOI 360 背光翻拍，sha256 `e0dc3d28…07d3`）；`000352720003.jpg`（同一画格的 Noritsu 店家扫描，1920 px 级 JPEG，sha256 `380f88c9…a96b`）；帖中称 NLP 用 B+W 设置转换，即**黑白负片** | **dev-test-only**：公开分享不等于授权，帖中无许可声明；不得用于商业训练 | 无片边；用冻结的 noref_v2 转换后明显偏蓝紫（固定观感套到了黑白片上），见 s3_intake；未参与调参 |
| nlp_forum_2023 | NLP 论坛帖 https://forums.negativelabpro.com/t/2023 （andriy，2020-06-27）第 1、9 楼分享的 Google Drive 文件 | 2026-10-15 | `DSC_5889.NEF`（Nikon D750 翻拍，sha256 `8c696fdf…eba0`，含片夹与相邻画格）；`85840009.JPG`（Noritsu 店家扫描，sha256 `21553600…9c97`）；`DSC_5889_LR75_NLP212_1255px.jpg`（发帖人的 NLP 2.1.2 转换，1255 px，sha256 `9de59ec6…792e`）；帖子题目所述胶片为 Portra 400（未独立核实） | **dev-test-only**：同上 | 用冻结的 noref_v2 转换并目视检查，画面正常，比 Noritsu 偏冷偏淡，无 NLP 版那种青色偏差；未做数值对照、未参与调参 |
| （未获取）nlp_forum_7898 | NLP 论坛帖 https://forums.negativelabpro.com/t/7898 （Portra 400 偏绿案例）第 4 楼的 Drive 文件夹 | 2026-10-15 尝试 | — | — | 文件夹返回 404（已删除或改为私有），未下载 |

## 数据用途划分（2026-10-15 定）

| 数据 | 用途 | 说明 |
|---|---|---|
| nlp_official（12 张）+ NLP 参考 | 算法开发与调参 | 已参与参数设计、λ / 高光权重选择 |
| shop_r01 的 23 对 | 第二参考域验证、失败分析与后续开发 | 已参与超参数选择与失败分析（S2-6b / S2-6c），**不再是独立测试集**；按奇偶等方式再拆分也不能恢复独立性（同卷、同设备、已参与决策） |
| ultramax400_r01 中无参考图的 14 帧 | 无参考算法的运行与鲁棒性测试 | 无参考图，只看能否运行、诊断是否合理 |
| 今后新增、未参与任何研发决策的独立胶卷（最好不同翻拍设备） | **正式保留测试集** | 在使用前不得用于架构选择、调参或失败分析 |

- 训练神经网络需要验证集时，从现有数据中**按胶卷或场景分组**划分，并标为“开发验证集”，不得称为保留测试集。
- 异常帧（ultramax400_r01 的 DSC_3822 卷首、DSC_3858 大半未曝光）在普通彩色照片的汇总指标中单列，但保留在完整的鲁棒性测试中。
