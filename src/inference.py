import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.config import MODELS_DIR, FEATURES


class StudentPredictor:
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        self.models_dir = models_dir
        self.feature_cols = FEATURES
        self.model_files = {
            "model":  os.path.join(self.models_dir, "model.pkl"),
            "scaler": os.path.join(self.models_dir, "scaler.pkl"),
            "kmeans": os.path.join(self.models_dir, "kmeans.pkl"),
            "meta":   os.path.join(self.models_dir, "meta.pkl"),
        }
        self.bundle = self._load_pretrained_model()

    def _load_pretrained_model(self) -> Optional[Dict[str, Any]]:
        if not all(os.path.exists(v) for v in self.model_files.values()):
            return None
        bundle = {
            "model":  joblib.load(self.model_files["model"]),
            "scaler": joblib.load(self.model_files["scaler"]),
            "kmeans": joblib.load(self.model_files["kmeans"]),
            "meta":   joblib.load(self.model_files["meta"]),
        }
        scaler = bundle["scaler"]
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(self.feature_cols):
            raise RuntimeError(
                f"Loaded scaler expects {scaler.n_features_in_} features but config defines "
                f"{len(self.feature_cols)}. Re-run train_model.py to retrain."
            )
        return bundle

    def predict_bundle(
        self, X_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.bundle:
            raise RuntimeError("Models not loaded. Run train_model.py first.")
        Xc = X_df[self.feature_cols].copy()
        for col in self.feature_cols:
            Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(0.0)
        scaler = self.bundle["scaler"]
        if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
            for i, col in enumerate(self.feature_cols):
                Xc[col] = Xc[col].clip(scaler.data_min_[i], scaler.data_max_[i])
        X_s = scaler.transform(Xc)
        probas = self.bundle["model"].predict_proba(X_s)[:, 1]
        preds = (probas >= 0.5).astype(int)
        clusters = self.bundle["kmeans"].predict(X_s)
        return probas, preds, clusters

    def get_student_recommendations(self, row: pd.Series) -> List[str]:
        recs = []
        if float(row.get("daily_study_hours", 5)) < 2:
            recs.append("📚 Increase daily study to at least 2–3 hours.")
        if float(row.get("attendance_rate", 1.0)) < 0.75:
            recs.append("🏫 Aim for 75%+ attendance to reduce knowledge gaps.")
        if float(row.get("math_score", 100)) < 50:
            recs.append("🔢 Focus on math — practice problem sets daily.")
        if float(row.get("reading_score", 100)) < 50:
            recs.append("📖 Improve reading — summarize one article daily.")
        if float(row.get("writing_score", 100)) < 50:
            recs.append("✍️ Work on writing structure and grammar daily.")
        if float(row.get("stress_level", 5)) > 7:
            recs.append("🧘 High stress detected — try the Pomodoro technique.")
        if float(row.get("sleep_hours", 8)) < 6:
            recs.append("😴 Aim for 7–8 hours of sleep for better retention.")
        if float(row.get("motivation_score", 50)) < 30:
            recs.append("🎯 Set small SMART goals to rebuild motivation daily.")
        if not recs:
            recs.append("🌟 Great habits! Keep consistency and add spaced revision.")
        return recs

    def get_meta(self) -> Dict[str, Any]:
        return self.bundle["meta"] if self.bundle else {}

    def is_ready(self) -> bool:
        return self.bundle is not None
