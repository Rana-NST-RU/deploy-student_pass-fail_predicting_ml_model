import streamlit as st
from agents.study_coach_agent import StudyCoachAgent
from src.ui_components import UIBuilder


@st.cache_resource
def load_agent():
    return StudyCoachAgent()


agent = load_agent()
ui = UIBuilder()
ui.load_css()

# ── Guard: must predict first ──
if "student_data" not in st.session_state or not st.session_state.get("prediction_done"):
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:3rem; margin-bottom:12px;">🎓</div>
        <div style="font-size:1.2rem; font-weight:700; color:#e0e7ff; margin-bottom:8px;">No prediction found</div>
        <div style="font-size:0.9rem; color:rgba(255,255,255,0.5); margin-bottom:24px;">
            Run a prediction on the Dashboard first to start your coaching session.
        </div>
    </div>
    """, unsafe_allow_html=True)
    c = st.columns([1, 2, 1])[1]
    with c:
        if st.button("← Go to Dashboard", type="primary", use_container_width=True):
            st.switch_page("pages/dashboard.py")
    st.stop()

student_data = st.session_state.student_data
prediction = student_data.get("prediction", "Unknown")
prob = student_data.get("probability", 0.0)
cluster = student_data.get("cluster", 0)

# ── Header ──
ui.render_chat_profile(prediction, prob, cluster)

# ── Quick stats ──
m1, m2, m3, m4 = st.columns(4)
m1.metric("Math", f"{student_data.get('math_score', '—')}/100")
m2.metric("Reading", f"{student_data.get('reading_score', '—')}/100")
m3.metric("Writing", f"{student_data.get('writing_score', '—')}/100")
m4.metric("Attendance", f"{student_data.get('attendance_rate', 0)*100:.0f}%")

st.markdown("""
<div style="font-size:0.82rem; color:rgba(255,255,255,0.4); margin:10px 0 20px;">
    💬 Ask me about study tips, a weekly plan, subject help, or motivation.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Session init ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_history" not in st.session_state:
    st.session_state.session_history = []

# ── Chat history display ──
for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)

# ── Input ──
if prompt := st.chat_input("Ask your study coach anything about your studies..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            reply, updated_history = agent.chat(
                user_message=prompt,
                student_data=student_data,
                session_history=st.session_state.session_history,
            )
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.session_history = updated_history

# ── Clear chat ──
if st.session_state.chat_history:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.session_history = []
            st.rerun()
