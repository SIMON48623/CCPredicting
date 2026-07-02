# CCPredicting

CCPredicting is a research codebase for cervical high-grade lesion risk prediction from routine structured clinical variables. The repository implements a Tabular Transformer workflow with Masked Feature Modeling (MFM) pretraining, probability calibration, Integrated Gradients explanation, and Streamlit deployment.

This repository is intended for reproducible code review and demonstration. It is **not** a medical device and must not be used as a standalone clinical decision system.

## What this repository contains

- A cleaned inference package under `src/ccpredicting/`.
- A Streamlit demo under `app/streamlit_app.py`.
- Exported model artifacts under `final_model/`.
- Legacy experiment scripts under `scripts/legacy/` for transparency.
- Curated generated figures/tables under `reports/`.
- Schema-only and synthetic input examples under `data/`.

## Repository structure

```text
CCPredicting/
├── app/                         # Maintained Streamlit app
├── data/                        # Schema and synthetic sample only; no patient data
├── final_model/                 # Exported model artifacts and compatibility API
├── reports/                     # Curated figures/tables generated during analysis
├── scripts/legacy/              # Original numbered experiment scripts
├── src/ccpredicting/            # Maintained Python package
│   ├── inference/               # Predictor and Integrated Gradients API
│   ├── models/                  # Tab-MFM model definition
│   └── schema.py                # Input schema, aliases, and variable dictionary
└── tests/                       # Lightweight inference and schema tests
```

## Model summary

The deployed model is an MFM-pretrained Tabular Transformer. It uses structured variables covering demographics, reproductive history, HPV status, cytology, colposcopic impression, transformation-zone type, iodine test result, atypical vessels, and a clinician-recorded image/colposcopy-derived assessment variable.

The exported artifact includes:

- `preprocess.pkl`: fitted preprocessing metadata;
- `transformer_state.pt`: model state dict;
- `calibrator.pkl`: post-hoc probability calibrator;
- `metadata.json`: model metadata and decision thresholds;
- `arch.json`: architecture parameters.

The internal model feature name `pathology_fig` has been retained inside the artifact for compatibility, but the public API and UI use the clearer alias `clinician_image_assessment`. This variable is **not** the histopathological outcome label.

## Privacy note

The original clinical dataset is not included. Only a schema file and a synthetic example row are provided. Raw clinical data, local Excel files, patient-level predictions, and temporary outputs should not be committed.

## Quick start

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Run a minimal inference check:

```bash
PYTHONPATH=src python - <<'PY'
from ccpredicting import predict_one

record = {
    "age": 45,
    "menopausal_status": 0,
    "gravidity": 2,
    "parity": 1,
    "child_alive": 1,
    "HPV_overall": 1,
    "HPV16": 0,
    "HPV18": 0,
    "HPV_other_hr": 0,
    "cytology_grade": 3,
    "colpo_impression": 2,
    "TZ_type": 2,
    "iodine_negative": 0,
    "atypical_vessels": 0,
    "clinician_image_assessment": 0,
}
print(predict_one(record, mode="balanced"))
PY
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## Decision modes

The exported metadata originally used the names `screen`, `triage`, and `youden`. The maintained API exposes clearer names while preserving backward compatibility:

| Public mode | Legacy metadata key | Intended use |
|---|---|---|
| `high_sensitivity` | `screen` | Lower threshold intended to reduce false negatives |
| `balanced` | `triage` | Balanced sensitivity/specificity trade-off |
| `youden` | `youden` | Youden-index threshold |

Legacy calls such as `predict_one(record, mode="triage")` still work through compatibility mapping.

## Legacy scripts

The original numbered scripts have been moved to `scripts/legacy/`. They are retained to document the research trajectory, but the maintained inference and deployment code is under `src/ccpredicting/` and `app/`.

Some legacy scripts require the private clinical dataset and are not expected to run without approved local data. Local absolute paths have been replaced by environment/configuration placeholders where practical.

## Manuscript / project status

This repository supports an under-review tabular deep-learning project on cervical high-grade lesion risk prediction and web deployment. The codebase is provided as a research prototype and implementation record.
