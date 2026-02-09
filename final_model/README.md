# Cervix Model A — Final Deployment Package (Tab-MFM Transformer)

This folder contains a portable inference package for the **Tabular Transformer (MFM-pretrained)** model.

## Files
- `predict_api.py`  
  Main inference API. Provides `predict_one(record, mode="triage")`.
- `transformer_state.pt`  
  Model weights (state_dict).
- `arch.json`  
  Model architecture configuration.
- `preprocess.pkl`  
  Feature preprocessing artifacts (imputation/scaling/mapping).
- `calibrator.pkl`  
  Platt (sigmoid) probability calibrator.
- `metadata.json`  
  Model metadata + thresholds.

## Supported input schema
`predict_one()` expects a **dict** containing (recommended to provide all):

### Numeric / ordinal
- `age`
- `gravidity`
- `parity`
- `cytology_grade`
- `colpo_impression`
- `TZ_type`
- `pathology_fig`  *(clinician image/colposcopic score; NOT gold-standard pathology)*

### Binary / categorical (0/1)
- `menopausal_status`
- `HPV_overall`
- `HPV16`
- `HPV18`
- `HPV_other_hr`
- `iodine_negative`
- `atypical_vessels`
- `child_alive`

Missing fields are allowed (will be treated as missing), but providing all fields is recommended.

## Output
`predict_one()` returns a dict:
- `prob_raw`: uncalibrated probability (sigmoid(logit))
- `prob`: calibrated probability (Platt sigmoid)
- `decision_mode`: threshold mode used
- `threshold`: threshold value
- `label`: `positive` / `negative`
- `meta`: model metadata

## Decision modes (thresholds)
Defined in `metadata.json`:
- `screen` : high sensitivity screening threshold
- `triage` : balanced triage threshold (recommended default)
- `youden` : threshold around max Youden index

## Quick start
From project root (`cervix_modelA/`):

### Single example
```bash
python -c "from final_model.predict_api import predict_one; print(predict_one({'age':45,'menopausal_status':0,'gravidity':2,'parity':1,'HPV_overall':1,'HPV16':0,'HPV18':0,'HPV_other_hr':1,'cytology_grade':3,'colpo_impression':2,'TZ_type':2,'iodine_negative':1,'atypical_vessels':0,'child_alive':1,'pathology_fig':2}, mode='triage'))"
