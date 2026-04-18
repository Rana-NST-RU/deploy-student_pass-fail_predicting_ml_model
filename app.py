import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Student Study Coach", layout="wide", page_icon="🎓")

dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="🎓", default=True)
chat = st.Page("pages/chat_interface.py", title="Study Coach", icon="🤖")

pg = st.navigation([dashboard, chat])
pg.run()
