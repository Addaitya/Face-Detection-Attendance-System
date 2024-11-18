import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import cv2
import numpy as np
from utils.encoder import Encoder
from utils.db import PersonCollection, Attendance

load_dotenv()

URI = os.getenv('DB_URI')
COSIN = os.getenv('COSIN_INDEX')
# EUCLIDEAN = os.getenv('EUCLIDEAN_INDEX')
# DOT_PRODUCT = os.getenv('DOT_PRODUCT_INDEX')
SEARCH_FIELD = os.getenv('SEARCH_FIELD')

ecd = Encoder()
person = PersonCollection(URI)
atd = Attendance(URI)

st.set_page_config(
    "Take attendance", 
    page_icon='📸'
)



def handle_check(img_file_buffer):
    if img_file_buffer:
        progress_giver.text("Processing...")
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        embeddings = ecd.encode(img)
        person_info = []
        for e in embeddings:
            result = person.search(
                e,
                index_name=COSIN,
                field=SEARCH_FIELD
            )
            if result:
                person_info.append(result[0])
            else:
                person_info.append(None)
        
        progress_giver.empty()
        st.session_state['attendees'] = person_info
        st.session_state['submit_disable'] = False
        

    else:
        st.error("Please Take picture first")

def handle_camera():
    if st.session_state['camera']:
        bytes_data = st.session_state['camera'].getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_boxes = ecd.get_faceboxes(img)
        
        for x,y,w,h in face_boxes:
            cv2.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)

        st.session_state['canvas'] = img
    else:
        st.session_state['canvas']= None

def handle_submit():
    if st.session_state['attendees']:
        attendance_data = []
        for p in st.session_state['attendees']:
            if p and 'student_id' in p:
                attendance_data.append({
                    'student_id': p['student_id'],
                    'time_stamp': datetime.combine(st.session_state['date'], datetime.min.time()),
                })
        if attendance_data:
            atd.add_many(attendance_data)
        st.success("Attendance added successfully.")
    else:
        st.error("No Attendee detected")
    
    
if "canvas" not in st.session_state:
    st.session_state['canvas'] = None

if "attendees" not in st.session_state:
    st.session_state['attendees'] = []

if "submit_disable" not in st.session_state:
    st.session_state['submit_disable'] = True

if "date" not in st.session_state:
    st.session_state['date'] = datetime.today().date()

    

st.header("Take Attendance")

img_file_buffer = st.camera_input("Take a picture", on_change=handle_camera, key='camera')
canvas =  st.empty() if st.session_state['canvas'] is None else st.image(st.session_state['canvas'])
check_btn = st.button("Check", on_click=lambda : handle_check(img_file_buffer))
progress_giver = st.empty()
show_attendee = st.table(st.session_state['attendees']) if st.session_state['attendees'] else st.empty()
calender = st.date_input("Enter Date", key='date', max_value=datetime.today().date(), disabled=st.session_state['submit_disable'])
submit_btn = st.button("Submit", disabled=st.session_state['submit_disable'], on_click=handle_submit)





