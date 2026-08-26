# FABLE-500

FABLE-500（Fundus And B-scan Linked Evaluation）是一个同日眼底超广角图像与眼科 B 超图像配对的多模态眼科数据集，用于评价视觉语言模型在患者级诊断分类任务中的表现。

本仓库只包含论文中使用的公开代码，包括 metadata 校验、benchmark 输入导出、API 模型调用、指标分析和图表生成。脱敏后的数据集压缩包单独发布。

## 数据集

数据集 Zenodo 链接待替换。

当前论文中的临时占位链接：

```text
https://zenodo.org/records/FABLE-500-V1-PLACEHOLDER
```

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

数据集本身用于科研用途，采用 CC BY-NC 4.0 许可。正式引用和数据许可请以后续 Zenodo 记录为准。

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

