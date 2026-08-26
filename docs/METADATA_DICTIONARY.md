# Metadata Dictionary

| Field | Description |
|---|---|
| `case_id` | Public case identifier. |
| `patient_id` | Public patient identifier; one public patient ID appears once in the dataset. |
| `split` | Predefined split: `train`, `validation`, or `test`. |
| `reference_diagnosis` | Patient-level reference diagnosis label in English, derived from clinical records. |
| `age_years` | Age in years at examination; ages 90 years or older are top-coded as 90. |
| `sex` | Sex field in English. |
| `same_day_fundus_bscan_pair` | Whether the fundus and B-scan data were matched as same-day multimodal data. |
| `n_fundus_images` | Number of released fundus images for the case. |
| `n_bscan_images` | Number of released B-scan ultrasound images for the case. |
| `fundus_image_paths` | Semicolon-separated relative paths to released fundus images. |
| `bscan_image_paths` | Semicolon-separated relative paths to released B-scan images. |
| `bscan_finding` | Structured B-scan finding text in English. |
| `bscan_impression` | Structured B-scan impression text in English. |

The optional file `FABLE-500_original_chinese_fields.xlsx` contains
`reference_diagnosis_zh`, `sex_zh`, `bscan_finding_zh`, and
`bscan_impression_zh`, keyed by `case_id` and `patient_id`.
