# FABLE-500

FABLE-500 (Fundus And B-scan Linked Evaluation) is a same-day multimodal
ophthalmic dataset and benchmark for evaluating vision-language and
workflow-based AI systems on patient-level diagnosis classification.

This repository contains the public code used for metadata validation,
benchmark input export, API-based model evaluation, metric analysis, and figure
generation. The de-identified dataset archive is distributed separately.

[中文说明](README_CN.md)

## Dataset

Dataset archive: https://zenodo.org/records/22119619

Release archive: `FABLE-500_v1.0_20260825.zip`

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

## Study Design

FABLE-500 was constructed as a benchmark-oriented, same-day, patient-level,
class-balanced ophthalmic data resource. The release was curated from a strict
eligible pool of 8,740 same-day multimodal case records from 6,262 unique
patients. Eligible cases required:

- same-day ultra-widefield fundus and ophthalmic B-scan images;
- nonempty structured B-scan finding and impression fields;
- one curated patient-level diagnosis label;
- valid public image paths and one B-scan examination source.

Records with postoperative status, uncertain impressions, laterality
text-image mismatch, multiple source diagnosis labels, or multiple same-day
B-scan examination records were excluded or replaced. The final release contains
500 unique patient-level cases, sampled as 100 cases for each of five diagnosis
categories and assigned to fixed patient-level train/validation/test splits.

## Dataset Characteristics

| Characteristic | Value |
|---|---:|
| Cases | 500 |
| Unique public patient IDs | 500 |
| Fundus images | 1,059 |
| B-scan images | 1,436 |
| Total images | 2,495 |
| Diagnosis categories | 5 |
| Cases per category | 100 |
| Split | 300 train / 100 validation / 100 test |

Diagnosis categories:

- Cataract
- Vitreous hemorrhage
- High myopia
- Refractive error
- Retinal detachment

Each released case includes relative paths to all released fundus and B-scan
images, English B-scan finding and impression fields, a patient-level reference
diagnosis label, public identifiers, split assignment, age, sex, and image
counts. Original Chinese diagnosis, sex, B-scan finding, and B-scan impression
fields are provided in a companion workbook keyed by public identifiers.

## Reference Benchmark

The repository reproduces the API-based reference benchmark reported in the
manuscript. The benchmark evaluates fixed diagnosis classification on the
100-case test split. It is intended as a reproducible reference for input
ablation, model-family sensitivity, and workflow-based evaluation rather than
as a clinical deployment benchmark.

Primary GPT-5.6 Sol reference benchmark:

| Input setting | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| Report-text only | 0.39 (0.30-0.49) | 0.31 (0.24-0.37) |
| Fundus only | 0.38 (0.29-0.48) | 0.37 (0.27-0.46) |
| B-scan only | 0.25 (0.17-0.34) | 0.21 (0.14-0.28) |
| Fundus+B-scan image only | 0.35 (0.26-0.44) | 0.32 (0.22-0.41) |
| Full case multimodal | 0.40 (0.30-0.50) | 0.31 (0.24-0.37) |
| Four-stage workflow-based comparator | 0.40 (0.31-0.50) | 0.35 (0.26-0.43) |

Full-case model-family sensitivity:

| Model | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| GPT-5.6 Sol | 0.40 (0.30-0.50) | 0.31 (0.24-0.37) |
| Gemini 3.7 Flash | 0.52 (0.42-0.62) | 0.52 (0.42-0.60) |
| Claude Sonnet 5 | 0.33 (0.24-0.43) | 0.26 (0.18-0.33) |
| Qwen3.7 Plus | 0.43 (0.34-0.53) | 0.35 (0.28-0.40) |

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
