# 数据目录与元数据规范（草案 v0.1）

## 目录

```text
data/raw/<source_kind>/<source_id>/      原始文件，只读，不修改
  source_kind: public | own | nlp_output | lab_scan
data/manifests/<source_id>.csv           每个来源一份清单
data/manifests/SOURCES.md                来源、许可与可用范围总表
```

## 清单字段（每个文件一行）

| 字段 | 说明 |
|---|---|
| file | 相对 `data/raw/` 的路径 |
| sha256 | 文件校验和 |
| source_id | 来源标识 |
| format | ARW / NEF / CR2 / DNG / TIFF … |
| camera_make, camera_model | 来自 EXIF |
| width, height, bit_depth | 解码后实测 |
| film_stock | 胶片型号；未知填 `unknown` |
| roll_id | 同一卷的分组键（划分数据集用） |
| has_film_border | 是否有可用于片基估计的未曝光边缘：yes / no / partial |
| reference_files | 对应的参考正片（NLP 输出、冲扫店扫描等），分号分隔 |
| license | 许可 / 可用范围：dev-test-only / training-ok / unknown |
| notes | 其他 |

## 划分规则

- 同一 `roll_id` 不跨训练/验证/测试集。
- `license != training-ok` 的数据不得用于商业模型训练。
