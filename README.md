# FABLE-500

FABLE-500 (Fundus And B-scan Linked Evaluation) is a same-day multimodal
ophthalmic dataset and benchmark for evaluating vision-language models on
patient-level diagnosis classification.

This repository contains the public code used for metadata validation,
benchmark input export, API-based model evaluation, metric analysis, and figure
generation. The de-identified dataset archive is distributed separately.

[中文说明](README_CN.md)

## Dataset

Dataset archive: Zenodo link pending.

Temporary manuscript placeholder:

```text
https://zenodo.org/records/FABLE-500-V1-PLACEHOLDER
```

After downloading the dataset archive, extract it as:

```text
data/FABLE-500/
```

The expected release directory contains:

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

The dataset itself is released for research use under CC BY-NC 4.0. See the
dataset archive for the official data license and citation information.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file from the template:

```bash
cp .env.example .env
```

Set only the providers you plan to use. Do not commit `.env`.

## Export Benchmark Inputs

```bash
python scripts/export_benchmark_inputs.py \
  --release-root data/FABLE-500 \
  --output-dir benchmark/inputs

python scripts/export_masked_report_benchmark_inputs.py
```

This creates the fixed `smoke10_cases.jsonl`, `test100_cases.jsonl`, and masked
report-text sensitivity inputs from the public metadata.

## Run API-Based Benchmarks

Check provider configuration:

```bash
python scripts/check_api_environment.py \
  --providers openrouter_gpt56_sol,openrouter_gemini37_flash,openrouter_claude_sonnet5,bailian_qwen_vl
```

Run the full experiment suite:

```bash
bash scripts/run_fable500_all_required_experiments.sh
```

For a one-case smoke test:

```bash
python scripts/run_api_benchmark.py \
  --input benchmark/inputs/test100_cases.jsonl \
  --output benchmark/outputs/smoke_full_case.jsonl \
  --systems full_case_multimodal \
  --provider-chain openrouter_gpt56_sol \
  --limit 1
```

## Benchmark Settings

Single-call input settings:

- `report_text_only`
- `fundus_only`
- `bscan_only`
- `fundus_bscan_image_only`
- `full_case_multimodal`

Workflow-based comparator:

- `api_agentic_full_case_workflow`

The workflow comparator uses the same released inputs but decomposes the case
into fundus-image evidence extraction, B-scan-image evidence extraction,
B-scan-report evidence extraction, and cross-modal adjudication. It does not use
retrieval, web search, external guideline text, or fine-tuned specialist tools.

## Analysis and Figures

After prediction files are generated:

```bash
python scripts/analyze_api_reference_benchmark.py
python scripts/analyze_multimodel_fullcase.py \
  --predictions benchmark/outputs/test100_fullcase_openrouter_gpt56sol.jsonl \
    benchmark/outputs/test100_fullcase_openrouter_gemini37flash.jsonl \
    benchmark/outputs/test100_fullcase_openrouter_claude_sonnet5.jsonl \
    benchmark/outputs/test100_fullcase_bailian_qwen37plus.jsonl
python scripts/generate_figure3_api_reference_benchmark.py
python scripts/generate_supplementary_confusion_heatmap.py
```

Generated results are written under `benchmark/analysis/` and `figures/`.

## Repository Scope

Included:

- benchmark prompts and API runners;
- metric analysis scripts;
- figure-generation scripts;
- dataset card and metadata dictionary.

Not included:

- clinical images or full metadata archive;
- private source mappings;
- physician review packages;
- raw PDF reports;
- API keys or local `.env` files;
- manuscript Word/PDF files;
- cached model outputs.
