import os
import streamlit as st
from dotenv  import load_dotenv
from utils.db import Attendance

load_dotenv()
URI = os.getenv('DB_URI')

st.set_page_config(
    page_title="View Attendance",
    page_icon="🪟"
)

st.title("View Attendance")
attendance = Attendance(URI)
student_id = st.text_input("Enter student ID")


def handle_view_btn(student_id, n_days, key):
    view_btn = st.button('Get', key=key)
    info = st.empty()
    entries = st.empty()
    if view_btn and student_id:
        atd_data, atd_percent = attendance.fetch_attendance(student_id, n_days)
        if atd_data and atd_percent:
            info.info(f"Last Week Attendance: {atd_percent:.2f}%")
            entries.table(atd_data)
        else: 
            st.error('Student id does\'t exist')
        
    elif view_btn:
        st.error('Student id is empty', icon="🚨")


w, m = st.tabs(['Last week', 'Last month'])

with w:
    handle_view_btn(student_id, 7, key='view week')
with m:
    handle_view_btn(student_id, 30, key='view month')
