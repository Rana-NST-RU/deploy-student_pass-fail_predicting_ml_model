import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agents.study_coach_agent import StudyCoachAgent
from src.ui_components import UIBuilder


@st.cache_resource
def load_agent():
    return StudyCoachAgent()


agent = load_agent()
ui = UIBuilder()
ui.load_css()

st.title("🤖 AI Study Coach")

if "student_data" not in st.session_state or not st.session_state.get("prediction_done"):
    st.info("Please go to the **Dashboard** page first and run a prediction to start your coaching session.")
    if st.button("← Go to Dashboard"):
        st.switch_page("pages/dashboard.py")
    st.stop()

student_data = st.session_state.student_data
prediction = student_data.get("prediction", "Unknown")
prob = student_data.get("probability", 0.0)
cluster = student_data.get("cluster", 0)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Outcome", prediction)
col2.metric("Pass Probability", f"{prob*100:.1f}%")
col3.metric("Student Group", f"Group {cluster + 1}")

st.markdown("---")
st.markdown("Ask me anything — study tips, a weekly plan, subject help, or just chat!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_history" not in st.session_state:
    st.session_state.session_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask your study coach..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, updated_history = agent.chat(
                user_message=prompt,
                student_data=student_data,
                session_history=st.session_state.session_history,
            )
        st.write(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.session_history = updated_history

if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.session_history = []
        st.rerun()
