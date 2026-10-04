# 8-class OIA-ODIR disease taxonomy used throughout the project

CLASS_NAMES = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other",
]

CLASS_CODES = {n: c for n, c in zip(CLASS_NAMES, ["N", "D", "G", "C", "A", "H", "M", "O"])}

CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

IDX_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

# Rough class prevalence in the ODIR dataset (used for reference / display)
CLASS_METADATA = {
    "Normal":       {"prevalence_pct": 28.9, "icd": "Z01.00"},
    "Diabetes":     {"prevalence_pct": 28.8, "icd": "E11.319 / H36.0"},
    "Glaucoma":     {"prevalence_pct":  6.4, "icd": "H40.11X0"},
    "Cataract":     {"prevalence_pct":  5.8, "icd": "H26.9"},
    "AMD":          {"prevalence_pct":  5.8, "icd": "H35.30"},
    "Hypertension": {"prevalence_pct":  4.6, "icd": "H35.03"},
    "Myopia":       {"prevalence_pct":  5.8, "icd": "H52.1"},
    "Other":        {"prevalence_pct": 13.9, "icd": "H47.299"},
}
