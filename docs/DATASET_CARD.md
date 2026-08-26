# Dataset Card: FABLE-500

## Dataset Description

FABLE-500 is a class-balanced, same-day multimodal ophthalmic dataset for
benchmarking vision-language and multimodal AI systems. Each row corresponds
to one unique public patient ID and includes same-day fundus photography,
B-scan ultrasonography, structured B-scan finding/impression text, and one
patient-level reference diagnosis category derived from clinical records. The
default metadata is English for international reuse and benchmark
reproducibility. The diagnosis label is not an eye-level lesion annotation.

## Composition

- Total cases: 500
- Unique public patient IDs: 500
- Fundus images: 1059
- B-scan images: 1436
- Diagnostic categories: 5
- Cases per category: 100
- Split: 300 train, 100 validation, 100 test

## Data Fields

The default metadata file contains English fields for experiments: public
case/patient identifiers, split, reference diagnosis label, age, sex, same-day
matching indicator, image counts, relative image paths, and English B-scan
finding/impression text. The original Chinese clinical text is provided only
in the optional companion file `FABLE-500_original_chinese_fields.xlsx`.

## Intended Uses

- Fundus-only diagnosis classification
- B-scan-only diagnosis classification
- Same-day fundus+B-scan multimodal diagnosis
- B-scan finding/impression generation
- Report-text-based diagnosis classification
- API-based or local multimodal model benchmarking

## Out-of-Scope Uses

- Direct clinical diagnosis or treatment decision-making
- Estimating real-world disease prevalence
- Re-identification or linkage to external patient records
- Deployment of unvalidated clinical AI systems

## Known Limitations

FABLE-500 is retrospective, single-center, and class-balanced by design. It
contains five common diagnostic categories and does not represent the full
spectrum of ophthalmic disease. The public package includes structured B-scan
finding and impression fields rather than raw B-scan PDF reports.
