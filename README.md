# FilmColor AI

彩色负片转换与学习型校色研究项目（工作名称）。当前阶段：**Sprint 1 — 确定性负片反转引擎**。

项目依据：[docs/PLAN_v0.2.md](docs/PLAN_v0.2.md)（所有编码 Agent 开工前必读）。

## 目录

```text
docs/               方案、数据规范、算法说明、审计文档（docs/audits/）
data/raw/public/    公开负片样本（不入 git）
data/manifests/     样本清单、校验和、来源与许可记录（入 git）
experiments/        实验记录与对照报告（每个实验一个子目录）
src/filmcolor/      引擎源码
tests/              自动化测试
```

## 规则摘要

- 不复制 GPL 项目（NegPy、spektrafilm）的实现；只读其算法说明（见方案第六节）。
- 每个工作单元需附测试、运行记录和输出样本才算验收。
- 原始数据不入 git；清单和校验和入 git。
