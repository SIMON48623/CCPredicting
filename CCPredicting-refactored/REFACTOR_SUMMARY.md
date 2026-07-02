# Refactor summary

## Scope

This refactor reorganized the repository from a root-level collection of research scripts into a public-facing research codebase with a maintained inference package, Streamlit app, privacy-safe data examples, and retained legacy scripts.

## Major changes

1. Added `src/ccpredicting/` as the maintained Python package.
   - `models/tab_mfm.py`: model architecture compatible with exported weights.
   - `inference/predictor.py`: model loading, prediction, probability calibration, decision-mode aliases, and Integrated Gradients.
   - `schema.py`: feature dictionary, public alias mapping, and default synthetic record.

2. Rebuilt the Streamlit app.
   - New maintained app: `app/streamlit_app.py`.
   - Added `webapp/streamlit_app.py` compatibility wrapper.
   - Fixed the previous syntax issue in the original Streamlit file.
   - Replaced confusing UI field `pathology_fig` with `clinician_image_assessment` while preserving model compatibility.

3. Preserved exported model artifacts.
   - Kept `final_model/` as the active artifact directory.
   - Replaced `final_model/predict_api.py` with a compatibility wrapper.
   - Retained the original implementation as `final_model/predict_api_legacy.py`.

4. Organized legacy research scripts and outputs.
   - Moved numbered scripts into `scripts/legacy/`.
   - Moved generated plots/tables into `reports/`.
   - Moved older baseline artifacts into `artifacts/legacy_models/`.

5. Improved public repository hygiene.
   - Added root `README.md`.
   - Added `data/README.md`, `data/sample_schema.csv`, and `data/sample_input.csv`.
   - Cleaned `.gitignore` for raw clinical data, local Excel files, caches, and temporary outputs.
   - Added `requirements.txt`, `pyproject.toml`, and basic tests.

## Compatibility checks

- Original and refactored predictors returned identical calibrated probability on a sample record: `0.7880170218129844`.
- `PYTHONPATH=src pytest -q` passed: 4 tests passed.
- Known warnings are from scikit-learn version mismatch when loading the existing calibrator and PyTorch Transformer nested-tensor behavior; they do not indicate a refactor failure.

## Recommended GitHub upload workflow

1. Upload this refactored directory to a new branch, not directly to `main`.
2. Create a pull request and inspect changed files.
3. After merge, set the repository About description to:

   `Tabular deep learning and Streamlit deployment for cervical high-grade lesion risk prediction.`

4. Add topics such as `tabular-deep-learning`, `streamlit`, `medical-ai`, `risk-prediction`, `pytorch`.
