import os
import streamlit as st
import cv2
import numpy as np
from dotenv import load_dotenv
from utils.db import PersonCollection
from utils.encoder import Encoder

load_dotenv()
URI = os.getenv('DB_URI')

st.set_page_config(
    page_title="Register student",
    page_icon="👨‍🎓",
)

st.title("Register Students")
person = PersonCollection(URI)
ecd = Encoder()

student_name = st.text_input("Enter student name", key="name")
student_id = st.text_input("Enter student ID", key="id")
img_file_buffer = st.camera_input("Take a picture")
img_canvas = st.empty()

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_boxes = ecd.get_faceboxes(img)
    img_cp = np.copy(img)

    for x,y,w,h in face_boxes:
        cv2.rectangle(img_cp, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    
    img_canvas.image(img_cp, caption="Confirm faces")
else:
    img_canvas.empty()

if st.button("Register"):
    if  img_file_buffer is not None and student_name and student_id:
        if not person.check_person(student_id):
            progress_giver2 = st.empty()
            progress_giver2.text("Processing...")
            img_array = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
            new_img = cv2.imdecode(img_array, 1)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

            embeddings = ecd.encode(new_img)

            person_data = {
                "name": student_name,
                "student_id": student_id,
                "embedding": embeddings[0]
            }
            res = person.add_person(person_data)
            progress_giver2.success("Registeration Successful")

        else:
            st.error("You are already registered.")
    else:
        st.error("Please provide all the required information.")
