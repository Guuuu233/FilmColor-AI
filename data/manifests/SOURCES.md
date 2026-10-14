# 数据来源总表

| source_id | 来源 | 获取日期 | 内容 | 许可 / 可用范围 | 备注 |
|---|---|---|---|---|---|
| ultramax400_r01 | 用户自有：用户本人拍摄的一卷 Kodak UltraMax 400（全能 400）135 彩负，冲扫店用 Nikon Z 7_2 + NIKKOR Z MC 105mm f/2.8 VR S 翻拍（2024-03-30，62 秒内拍完 37 格）；复制自 `~/Downloads/原件`，复制后逐文件 sha256 与原件一致 | 2026-10-14 | 37 张 NEF（Bayer RGGB，14 bit），翻拍时只拍画面、无片边；光圈优先 f/16、ISO 100、矩阵测光 +1 EV，快门随画面在 1/25～1/60 之间变化；另有 `DSC_3820.JPG`（sha256 `dfed0c89…abb2`），拍的是一段贴有“0625”标签的胶片，只有 JPEG，不列入 RAW 清单 | **training-ok**：用户拥有照片的完整版权 | 参考图见 shop_r01 |
| shop_r01 | 冲扫店为 ultramax400_r01 逐张去色罩、校色的正片；复制自 `~/Downloads/逐张去色罩校色`，复制后逐文件 sha256 与原件一致 | 2026-10-14 | 23 张 sRGB JPEG，7956×5300，Adobe Photoshop 22.4 (Windows) 处理；按画面内容与 NEF 配对，23 张全部唯一配对，且与文件顺序一致（`shop_r01_reference.json`） | **training-ok**：用户拥有完整版权，店家只做加工、无版权主张，交易协议中已注明（用户 2026-10-14 告知） | 人工修图师的审美结果，不是色彩真值；8 bit JPEG；只用于评估与参考图拟合，不参与无参考优化器求解 |
| nlp_official | https://negativelabpro.s3-accelerate.amazonaws.com/sample-raw-scans.zip（zip sha256 `08e98251…3ca4`，Last-Modified 2020-09-30） | 2026-10-13 | 12 张 Fujifilm X-T2 + XF80mm Macro 翻拍的彩色负片 RAF（X-Trans），从画幅外观推断为 120 中画幅，均含片基边缘 | **dev-test-only**：压缩包内无任何许可或说明文件，未见明确授权；不得用于商业训练或随软件分发 | 无 NLP 输出参考，需用户自行用 NLP 默认设置转换 |
