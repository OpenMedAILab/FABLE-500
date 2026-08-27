# FABLE-500

FABLE-500（Fundus And B-scan Linked Evaluation）是一个同日眼底超广角图像与眼科 B 超图像配对的多模态眼科数据集，用于评价视觉语言模型和 workflow-based AI 系统在患者级诊断分类任务中的表现。

本仓库只包含论文中使用的公开代码，包括 metadata 校验、benchmark 输入导出、API 模型调用、指标分析和图表生成。脱敏后的数据集压缩包单独发布。

## 数据集

数据集 Zenodo 链接：

https://zenodo.org/records/22119619

发布压缩包：`FABLE-500_v1.0_20260825.zip`

下载数据后，建议解压到：

```text
data/FABLE-500/
```

该目录应包含：

```text
FABLE-500_metadata.xlsx
FABLE-500_metadata.jsonl
FABLE-500_original_chinese_fields.xlsx
FABLE-500_original_chinese_fields.jsonl
checksums_sha256.csv
docs/
images/fundus/
images/bscan/
```

数据集本身用于科研用途，采用 CC BY-NC 4.0 许可。正式引用和数据许可请以 Zenodo 记录为准。

## 研究设计

FABLE-500 是一个面向 benchmark 的同日、患者级、类别均衡的多模态眼科数据资源。数据集从 8,740 条严格筛选后的同日多模态病例记录中构建，这些记录来自 6,262 名唯一患者。纳入病例需要满足：

- 同日超广角眼底图像和眼科 B 超图像；
- 非空的结构化 B 超所见和 B 超提示字段；
- 一个经过整理的患者级诊断标签；
- 有效的公开图片相对路径；
- 每例对应一个 B 超检查来源。

术后状态、诊断不确定、眼别文本-图像不一致、多源诊断标签、同日多个 B 超检查来源等病例被排除或用同病种合格病例替换。最终公开数据集包含 500 个唯一患者级病例，5 个诊断类别各 100 例，并固定划分为 train/validation/test。

## 数据集主要特征

| 特征 | 数值 |
|---|---:|
| 病例数 | 500 |
| 唯一公开 patient ID | 500 |
| 眼底图像 | 1,059 |
| B 超图像 | 1,436 |
| 总图像数 | 2,495 |
| 诊断类别 | 5 |
| 每类病例数 | 100 |
| 数据划分 | 300 train / 100 validation / 100 test |

诊断类别包括：

- 白内障
- 玻璃体积血
- 高度近视
- 屈光不正
- 视网膜脱离

每个病例包含眼底和 B 超图片相对路径、英文 B 超所见和提示、患者级参考诊断标签、公开 ID、split、年龄、性别和图像数量。原始中文诊断、性别、B 超所见和 B 超提示字段以 companion workbook 的形式提供，并通过公开 ID 关联。

## 参考 benchmark

本仓库复现论文中的 API-based reference benchmark。benchmark 在固定 100-case test split 上进行五分类诊断评价，主要用于输入消融、模型家族敏感性分析和 workflow-based evaluation，而不是临床部署性能声明。

GPT-5.6 Sol 主要参考结果：

| 输入设置 | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| Report-text only | 0.39 (0.30-0.49) | 0.31 (0.24-0.37) |
| Fundus only | 0.38 (0.29-0.48) | 0.37 (0.27-0.46) |
| B-scan only | 0.25 (0.17-0.34) | 0.21 (0.14-0.28) |
| Fundus+B-scan image only | 0.35 (0.26-0.44) | 0.32 (0.22-0.41) |
| Full case multimodal | 0.40 (0.30-0.50) | 0.31 (0.24-0.37) |
| Four-stage workflow-based comparator | 0.40 (0.31-0.50) | 0.35 (0.26-0.43) |

Full-case model-family sensitivity：

| 模型 | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| GPT-5.6 Sol | 0.40 (0.30-0.50) | 0.31 (0.24-0.37) |
| Gemini 3.7 Flash | 0.52 (0.42-0.62) | 0.52 (0.42-0.60) |
| Claude Sonnet 5 | 0.33 (0.24-0.43) | 0.26 (0.18-0.33) |
| Qwen3.7 Plus | 0.43 (0.34-0.53) | 0.35 (0.28-0.40) |

## 安装

建议使用 Python 3.10 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

根据模板创建本地 `.env`：

```bash
cp .env.example .env
```

只填写需要使用的 API key。不要把 `.env` 上传到 GitHub。

## 运行 benchmark

导出 benchmark 输入：

```bash
python scripts/export_benchmark_inputs.py \
  --release-root data/FABLE-500 \
  --output-dir benchmark/inputs

python scripts/export_masked_report_benchmark_inputs.py
```

检查 API 环境：

```bash
python scripts/check_api_environment.py \
  --providers openrouter_gpt56_sol,openrouter_gemini37_flash,openrouter_claude_sonnet5,bailian_qwen_vl
```

运行完整实验：

```bash
bash scripts/run_fable500_all_required_experiments.sh
```

单例 smoke test：

```bash
python scripts/run_api_benchmark.py \
  --input benchmark/inputs/test100_cases.jsonl \
  --output benchmark/outputs/smoke_full_case.jsonl \
  --systems full_case_multimodal \
  --provider-chain openrouter_gpt56_sol \
  --limit 1
```

## 公开范围

本仓库包含：

- benchmark prompts 和 API 调用脚本；
- 指标分析脚本；
- 图表生成脚本；
- dataset card 和 metadata dictionary。

本仓库不包含：

- 临床图像或完整 metadata 压缩包；
- 私有源路径映射；
- 医生审核材料；
- 原始 PDF 报告；
- API key 或 `.env`；
- 论文 Word/PDF 文件；
- 缓存的模型输出。
