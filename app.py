import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

st.title("📊 Intelligent Learning Analytics Dashboard")

# ===============================
# 🎛️ SIDEBAR MODE SELECTION
# ===============================
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Batch Dataset Analysis", "Individual Student Prediction"]
)

# ===============================
# 📊 MODE 1 — YOUR EXISTING APP
# ===============================
if mode == "Batch Dataset Analysis":

    uploaded_file = st.file_uploader("Upload Student Dataset CSV", type="csv")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.write(df.head())

        # 🔹 Convert grade → pass/fail
        df["pass"] = df["grade"].apply(
            lambda g: 1 if g in ["A", "B", "C"] else 0
        )

        # 🔹 Feature selection
        X = df[[
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation"
        ]]
        y = df["pass"]

        # 🔹 Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 🔹 Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 🔹 Logistic Regression
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # 🔹 Predictions
        df["prediction"] = model.predict(scaler.transform(X))

        st.subheader("Pass/Fail Predictions")
        st.write(df[["student_id", "prediction"]])

        # 🔹 K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["cluster"] = kmeans.fit_predict(scaler.transform(X))

        st.subheader("Cluster Groups")
        st.write(df[["student_id", "cluster"]])

        # =======================
        # 📊 VISUALIZATIONS
        # =======================

        st.subheader("📈 Score Distribution")

        fig1, ax1 = plt.subplots()
        sns.histplot(df["total_score"], bins=20, kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("📊 Study Hours vs Score")

        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            x="weekly_self_study_hours",
            y="total_score",
            hue="cluster",
            data=df,
            ax=ax2
        )
        st.pyplot(fig2)

        st.subheader("📊 Attendance vs Score")

        fig3, ax3 = plt.subplots()
        sns.scatterplot(
            x="attendance_percentage",
            y="total_score",
            hue="cluster",
            data=df,
            ax=ax3
        )
        st.pyplot(fig3)

        # =======================
        # 📌 RECOMMENDATIONS
        # =======================

        st.subheader("📌 Study Recommendations")

        for _, row in df.iterrows():
            if row["prediction"] == 0:
                st.write(
                    f"Student {row['student_id']}: Increase study hours and class participation."
                )
# =====================================
# 🎯 MODE 2 — INDIVIDUAL PREDICTION
# =====================================
elif mode == "Individual Student Prediction":

    st.subheader("🎓 Enter Student Details")

    # 🔹 Input fields (Removed total_score)
    study_hours = st.slider("Weekly Self Study Hours", 0, 40, 10)
    attendance = st.slider("Attendance Percentage", 50, 100, 80)
    participation = st.slider("Class Participation (0–10)", 0, 10, 5)

    # Dummy training data (without total_score)
    data = {
        "weekly_self_study_hours": [5, 10, 15, 20, 25, 30],
        "attendance_percentage": [60, 70, 75, 85, 90, 95],
        "class_participation": [2, 4, 5, 7, 8, 9],
        "pass": [0, 0, 1, 1, 1, 1]
    }

    df_train = pd.DataFrame(data)

    X = df_train[[
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation"
    ]]
    y = df_train["pass"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    # 🔹 Predict button
    if st.button("Predict Result"):

        input_df = pd.DataFrame([{
            "weekly_self_study_hours": study_hours,
            "attendance_percentage": attendance,
            "class_participation": participation
        }])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("PASS ✅")
            st.info("Great job! Keep maintaining your performance.")
        else:
            st.error("FAIL ❌")
            st.subheader("📌 Improvement Suggestions")

            # 🔹 Personalized Recommendations
            if study_hours < 15:
                st.write("• Increase weekly self-study hours to at least 15–20 hours.")
            
            if attendance < 75:
                st.write("• Improve attendance to above 80% for better understanding.")
            
            if participation < 5:
                st.write("• Participate more actively in class discussions.")
            
            if study_hours >= 15 and attendance >= 75 and participation >= 5:
                st.write("• Focus on consistency and structured revision strategy.")
