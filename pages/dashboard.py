import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.inference import StudentPredictor
from src.ui_components import UIBuilder


@st.cache_resource
def load_predictor():
    return StudentPredictor()


predictor = load_predictor()
ui = UIBuilder()
ui.load_css()

if not predictor.is_ready():
    st.error("⚠️ Models not found. Run `python train_model.py` first.")
    st.stop()

st.title("🎓 Student Performance Predictor")
st.markdown("Enter your academic details below to get a Pass/Fail prediction.")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        parental_education = st.slider("Parental Education Level", 1, 6, 3,
            help="1=No formal education, 6=Postgraduate")
        daily_study = st.slider("Daily Study Hours", 0.0, 12.0, 3.0, step=0.5)
        attendance = st.slider("Attendance Rate", 0.0, 1.0, 0.85, step=0.01,
            help="0.0 = 0%, 1.0 = 100%")

    with col2:
        sleep_hours = st.slider("Sleep Hours per Night", 3.0, 12.0, 7.0, step=0.5)
        stress_level = st.slider("Stress Level", 1, 10, 5,
            help="1 = very low stress, 10 = extremely high stress")
        motivation = st.slider("Motivation Score", 0, 100, 60)

    with col3:
        math_score = st.slider("Math Score", 0, 100, 65)
        reading_score = st.slider("Reading Score", 0, 100, 65)
        writing_score = st.slider("Writing Score", 0, 100, 65)

    submitted = st.form_submit_button("🔍 Predict My Outcome", type="primary")

if submitted:
    input_data = {
        "parental_education_level": parental_education,
        "daily_study_hours": daily_study,
        "attendance_rate": attendance,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "motivation_score": motivation,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }
    df = pd.DataFrame([input_data])
    prob, pred, cluster = predictor.predict_bundle(df)

    st.session_state.student_data = {
        **input_data,
        "prediction": "Pass" if pred[0] == 1 else "Fail",
        "probability": float(prob[0]),
        "cluster": int(cluster[0]),
    }
    st.session_state.prediction_done = True

    st.markdown("---")
    ui.render_prediction_card(float(prob[0]), int(pred[0]), int(cluster[0]))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Score Breakdown")
        fig, ax = plt.subplots(figsize=(6, 3))
        scores = [math_score, reading_score, writing_score]
        labels = ["Math", "Reading", "Writing"]
        colors = ["#6366f1" if s >= 50 else "#ef4444" for s in scores]
        ui.mpl_bar(ax, labels, scores, "Subject Scores", colors)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📋 Actionable Tips")
        recs = predictor.get_student_recommendations(pd.Series(input_data))
        for rec in recs:
            st.markdown(f"- {rec}")

    st.markdown("---")
    st.markdown("### Ready to improve? Chat with your AI Study Coach below 👇")
    if st.button("💬 Open Study Coach Chat", type="primary"):
        st.switch_page("pages/chat_interface.py")
