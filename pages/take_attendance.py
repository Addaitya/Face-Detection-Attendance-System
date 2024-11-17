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

st.header("Take Attendance")

img_file_buffer = st.camera_input("Take a picture")
img_canvas1 = st.empty()
img1 = None
face_boxes = None

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    img1 = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    face_boxes = ecd.get_faceboxes(img1)
    

    for x,y,w,h in face_boxes:
        cv2.rectangle(img1, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    
    img_canvas1.image(img1, caption="Confirm faces")
else:
    img_canvas1.empty()

if "person_info" not in st.session_state:
    st.session_state.person_info = []
if "choose_date" not in st.session_state:
    st.session_state.choose_date = datetime.today().date()

# Check Button Logic
check_btn = st.button("Check")
if check_btn and img_file_buffer is not None:
    progress_giver = st.empty()
    progress_giver.text("Processing...")
    
    img_array = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
    img1 = cv2.imdecode(img_array, 1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    embeddings = ecd.encode(img1)
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
    
    st.session_state.person_info = person_info  # Save to session state

    for i in range(len(face_boxes)):
        x, y, w, h = face_boxes[i]
        if person_info[i]:
            cv2.putText(img1, person_info[i]['name'], (x, y), color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
        else:
            cv2.putText(img1, "Unknown", (x, y), color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
    
    img_canvas1.image(img1, caption="result")
    progress_giver.empty()
    st.table(person_info)

if st.session_state.person_info:
    st.session_state.choose_date = st.date_input("Enter date", value=st.session_state.choose_date, max_value=datetime.today().date())

    if st.button("Submit"):
        attendance_data = []
        for p in st.session_state.person_info:
            if p and 'student_id' in p:
                attendance_data.append({
                    'student_id': p['student_id'],
                    'time_stamp': datetime.combine(st.session_state.choose_date, datetime.min.time()),
                })
        if attendance_data:
            atd.add_many(attendance_data)
        st.success("Attendance added successfully.")

elif check_btn:
    st.error("Please take a picture first")