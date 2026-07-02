# Data availability and privacy

The original clinical dataset is not included in this repository because it contains sensitive patient-level information. The repository provides only:

- `sample_schema.csv`: variable dictionary and allowed values;
- `sample_input.csv`: one synthetic example row for checking the inference API and Streamlit app.

To reproduce training on private data, place the approved de-identified dataset outside the repository or under `data/raw/`, which is ignored by Git. Do not commit raw Excel, CSV, or patient-level prediction files.
