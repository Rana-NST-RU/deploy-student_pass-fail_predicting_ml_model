import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Cloud, API keys come from st.secrets instead of .env
for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    if key not in os.environ and key in st.secrets:
        os.environ[key] = st.secrets[key]

st.set_page_config(page_title="Student Study Coach", layout="wide", page_icon="🎓")

# ── Auto-train if models are missing (e.g. first deploy on Streamlit Cloud) ──
from src.config import MODELS_DIR

required = ["model.pkl", "scaler.pkl", "kmeans.pkl", "meta.pkl"]
models_missing = not all((MODELS_DIR / f).exists() for f in required)

if models_missing:
    with st.spinner("Setting up models for the first time — this takes about 30 seconds..."):
        from src.data_processor import DataProcessor
        from src.model_trainer import ModelTrainer

        processor = DataProcessor()
        trainer = ModelTrainer()

        df = processor.load_data()
        df = processor.fill_nulls(df)
        df_clean = processor.remove_outliers(df)
        X_scaled, y, df_clean = processor.preprocess(df_clean)
        X_train_sm, X_test, y_train_sm, y_test = processor.split_and_balance(X_scaled, y)
        trainer.train_and_evaluate(X_train_sm, y_train_sm, X_test, y_test)
        trainer.train_kmeans(df_clean, processor.feature_cols, processor.scaler)
        trainer.save_artifacts(processor.scaler, processor.feature_names, processor.feature_cols)

    st.success("Models ready!")
    st.rerun()

dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="🎓", default=True)
chat = st.Page("pages/chat_interface.py", title="Study Coach", icon="🤖")

pg = st.navigation([dashboard, chat])
pg.run()
