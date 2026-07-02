"""Input schema and user-facing variable dictionary.

The exported model was trained with the internal feature name ``pathology_fig``.
For public documentation and the Streamlit UI we expose the safer alias
``clinician_image_assessment`` and map it back during inference. This variable
is a clinician-recorded image/colposcopy-derived assessment, not the
histopathological outcome label.
"""

from __future__ import annotations

from typing import Any

INTERNAL_CLINICIAN_IMAGE_FIELD = "pathology_fig"
PUBLIC_CLINICIAN_IMAGE_FIELD = "clinician_image_assessment"

FEATURE_COLUMNS = [
    "age",
    "menopausal_status",
    "gravidity",
    "parity",
    "HPV_overall",
    "HPV16",
    "HPV18",
    "HPV_other_hr",
    "cytology_grade",
    "colpo_impression",
    "TZ_type",
    "iodine_negative",
    "atypical_vessels",
    "child_alive",
    INTERNAL_CLINICIAN_IMAGE_FIELD,
]

NUMERIC_COLUMNS = [
    "age",
    "gravidity",
    "parity",
    "cytology_grade",
    "colpo_impression",
    "TZ_type",
    INTERNAL_CLINICIAN_IMAGE_FIELD,
]

CATEGORICAL_COLUMNS = [
    "menopausal_status",
    "HPV_overall",
    "HPV16",
    "HPV18",
    "HPV_other_hr",
    "iodine_negative",
    "atypical_vessels",
    "child_alive",
]

VARIABLE_DICTIONARY = [
    {"variable": "age", "type": "integer years", "allowed": "10-100", "description": "Age at index visit/exam."},
    {"variable": "menopausal_status", "type": "binary", "allowed": "0/1", "description": "0=pre-menopause, 1=post-menopause."},
    {"variable": "gravidity", "type": "integer", "allowed": "0-30", "description": "Number of pregnancies."},
    {"variable": "parity", "type": "integer", "allowed": "0-20", "description": "Number of deliveries."},
    {"variable": "child_alive", "type": "binary", "allowed": "0/1", "description": "1=at least one living child; 0=none."},
    {"variable": "HPV_overall", "type": "binary", "allowed": "0/1", "description": "Overall high-risk HPV positivity."},
    {"variable": "HPV16", "type": "binary", "allowed": "0/1", "description": "HPV16 status."},
    {"variable": "HPV18", "type": "binary", "allowed": "0/1", "description": "HPV18 status."},
    {"variable": "HPV_other_hr", "type": "binary", "allowed": "0/1", "description": "Other high-risk HPV types."},
    {"variable": "cytology_grade", "type": "integer", "allowed": "0-5", "description": "0=NILM, 1=ASC-US, 2=ASC-H, 3=LSIL, 4=HSIL, 5=AGC."},
    {"variable": "colpo_impression", "type": "integer", "allowed": "0-4", "description": "0=normal, 1=mild, 2=moderate, 3=severe, 4=highly suspicious for CIN3+."},
    {"variable": "TZ_type", "type": "integer", "allowed": "1/2/3", "description": "Transformation-zone type: 1=visible, 2=partly visible, 3=not visible."},
    {"variable": "iodine_negative", "type": "binary", "allowed": "0/1", "description": "Iodine test negativity."},
    {"variable": "atypical_vessels", "type": "binary", "allowed": "0/1", "description": "Atypical vessels present."},
    {"variable": PUBLIC_CLINICIAN_IMAGE_FIELD, "type": "binary", "allowed": "0/1", "description": "Clinician-recorded image/colposcopy-derived assessment; not the histopathology outcome label."},
]

DEFAULT_RECORD = {
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
    PUBLIC_CLINICIAN_IMAGE_FIELD: 0,
}

MODE_ALIASES = {
    "high_sensitivity": "screen",  # legacy metadata threshold: 0.12
    "screen": "screen",
    "balanced": "triage",          # legacy metadata threshold: 0.28
    "triage": "triage",
    "youden": "youden",
}

MODE_DISPLAY = {
    "screen": "high_sensitivity",
    "triage": "balanced",
    "youden": "youden",
}


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map public aliases to internal model feature names without changing semantics."""
    out = dict(record)
    if PUBLIC_CLINICIAN_IMAGE_FIELD in out and INTERNAL_CLINICIAN_IMAGE_FIELD not in out:
        out[INTERNAL_CLINICIAN_IMAGE_FIELD] = out[PUBLIC_CLINICIAN_IMAGE_FIELD]
    return out


def normalize_mode(mode: str | None) -> str:
    """Return the legacy threshold key stored in metadata.json."""
    return MODE_ALIASES.get(str(mode or "balanced").strip().lower(), "triage")


def display_mode(mode: str) -> str:
    return MODE_DISPLAY.get(mode, mode)


def validate_probability(value: float) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"Predicted probability is outside [0, 1]: {value}")
    return float(value)
