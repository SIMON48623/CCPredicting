# Exported model artifact

This directory contains the exported Tab-MFM model artifact used by the maintained inference API.

## Files

- `preprocess.pkl`: fitted preprocessing metadata;
- `calibrator.pkl`: fitted post-hoc probability calibrator;
- `metadata.json`: model metadata and decision thresholds;
- `arch.json`: architecture parameters;
- `transformer_state.pt`: PyTorch state dict;
- `transformer.pt`: archived full model object from the original workflow;
- `predict_api.py`: backward-compatible wrapper importing from `src/ccpredicting`;
- `predict_api_legacy.py`: original single-file inference implementation retained for traceability.

## Maintained inference API

Preferred usage from the repository root:

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

Backward-compatible usage also remains available:

```python
from final_model.predict_api import predict_one
```

## Variable naming note

The exported artifact uses the internal feature name `pathology_fig` because that was the name used during original training/export. The public API maps `clinician_image_assessment` to `pathology_fig` for clarity. This feature is a clinician-recorded image/colposcopy-derived assessment and is **not** the histopathology outcome label.
