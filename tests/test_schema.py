from pathlib import Path
import csv


def test_sample_schema_exists_and_has_no_patient_label():
    path = Path(__file__).resolve().parents[1] / "data" / "sample_schema.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    variables = {row["variable"] for row in rows}
    assert "clinician_image_assessment" in variables
    assert "pathology_group" not in variables
