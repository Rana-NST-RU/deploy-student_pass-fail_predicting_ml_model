from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    "parental_education_level",
    "daily_study_hours",
    "attendance_rate",
    "sleep_hours",
    "stress_level",
    "motivation_score",
    "math_score",
    "reading_score",
    "writing_score",
]
