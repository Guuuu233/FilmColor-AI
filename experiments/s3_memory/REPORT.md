# S3-2 内存：定位与优化

日期 2026-10-15。改前代码 `9603fe8`，改后见提交记录。内存口径：`/usr/bin/time -l` 直接测 Python 进程（`.venv/bin/python`，不经 `uv` 外壳）的 peak memory footprint，单位 GiB（1 GiB = 2^30 字节）。上限：单进程 ≤ 6 GiB（本机 18 GB）。

## 结论

峰值来自两个共用的逐像素阶段：参数化渲染（`render`）和密度反转（`invert_negative`）。解码、反马赛克和 TIFF 写出都不是瓶颈。在这两个阶段内部改为按行分块（每块 256 行）之后，四个回归用例的峰值都降到 3.84 GiB 以下。**TIFF 文件逐字节相同**，sidecar 除 `software` 字段（代码版本）外完全相同。反马赛克没有改动。

| 用例 | 像素 | 改前峰值 | 改后峰值 | 改前耗时 | 改后耗时 |
|---|---|---|---|---|---|
| BlueNeg `199807A-06-hk`，`--auto` | 6800 万（扫描仪 DNG） | **8.89** | **3.84** | 23.3 s | 16.5 s |
| Nikon Z7 II `DSC_3826`，`--auto` | 4570 万（Bayer） | **6.58** | **2.44** | 13.4 s | 12.4 s |
| Nikon Z7 II `DSC_3826`，默认参数 | 同上 | **6.05** | **2.70** | 9.9 s | 10.0 s |
| Fuji X-T2 `_DSF9235`，全部渲染分支 | 2430 万（X-Trans） | 3.60 | 1.90 | 13.9 s | 13.9 s |

数据：`before/memory.tsv`、`after/memory.tsv`；全部渲染分支的参数见 `all_branches_render.json`（非单位色彩矩阵，饱和度、反差、中间调、高光压缩、曝光、白平衡均取非默认值）。

## 定位（`scripts/diag_memory.py`）

用 tracemalloc 包住管线的各阶段函数，记录每个阶段 numpy 分配的峰值。tracemalloc 看不到 LibRaw 的 C++ 内存，但扫描仪路径没有 LibRaw，所以 BlueNeg 的数字是完整的。

| 阶段（BlueNeg） | 改前：进入时已占用 → 峰值 | 改后 |
|---|---|---|
| `invert_negative` | 0.89 → 5.68 | 0.90 → 2.44 |
| `render_image` | 2.42 → **8.23** | 0.90 → 1.96 |
| `write_tiff16` | 0.77 → 2.30 | 0.77 → 2.30 |
| 其他（读入、片基、估计、画质检查） | ≤ 2.9 | ≤ 1.4 |

原因：

1. `render_full` 每一步都新建一张全分辨率数组，`x`、`pre_clip`、`v`、`e` 同时存在；`srgb_oetf` 和反差那一步的 `np.where` 还要算出两个分支的完整数组。
2. `invert_negative` 一次算出全图密度 `D`，再算 `P`，而且额外算了一张管线不用的 `display`（0.78 GiB），这张图一直留到渲染结束。
3. 管线把 `positive × exposure` 算了两次。

float64：没有逐个运算核查。各阶段的内存量和 float32 全图（BlueNeg 0.78 GiB）的整数倍对得上，输出数组也都是 float32，所以推断不存在全分辨率的 float64 中间数组；这一点未直接验证。

## 改动

- `render.render`：图像高于 `chunk_rows` 时按行分块，调用同一个逐像素函数 `_render_core`，写入一张预先分配的输出数组；诊断比例按计数累加，和原来 `np.mean` 的结果相同。`render_full`（拟合器和 noref 在小图上使用）保持整图计算，接口不变。
- `invert.invert_negative`：全局统计（白点分位数、范围报告）只依赖画面区域内的像素，**先**在这批像素上单独算密度（数值和顺序都与 `density(rgb)[mask]` 相同），确定 `d_hi`；**然后**按行分块生成全图 `P`，并累加“全图高于白点的比例”。`display` 改为访问时才计算的属性。
- `pipeline.convert`：`positive × exposure` 只算一次；曝光为 1 时直接复用原数组（乘 1.0 逐位不变）。

## 验证

- 现有测试全部通过，其中包括锁定 Sprint 1 输出哈希的真实文件测试，它会走新的分块路径。
- 新增 `tests/test_chunked.py`（19 项）：块行数取 1、3、4、40、100，带 mask 和不带 mask，渲染全部分支打开，要求和整图计算逐位相同，诊断量相同，NaN 照样报错。
- 四个真实大图用例：改前改后 TIFF **逐字节相同**；ICC 和编码因此也相同；解码后 16 位像素逐值相同；sidecar 除 `software` 外 0 处差异。

## 未做 / 局限

- 没有用 memmap，也没有流式写 TIFF。写出阶段新增的 1.53 GiB 现在不是瓶颈，所以没改，文件字节也就不会因此变化。
- tracemalloc 的阶段数字不包含 LibRaw 的内存，相机路径的解码和反马赛克阶段只能看整个进程的峰值。
- 按现在的峰值，本机同时跑多个转换时，BlueNeg 每个进程约 3.9 GiB，最多并行 3 个才能留出余量；这个上限是估算，没有实测。
