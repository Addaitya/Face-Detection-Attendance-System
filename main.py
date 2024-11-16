import os
from dotenv import load_dotenv
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import datetime
load_dotenv()
URI = os.getenv('DB_URI')
COSIN = os.getenv('COSIN_INDEX')
EUCLIDEAN = os.getenv('EUCLIDEAN_INDEX')
DOT_PRODUCT = os.getenv('DOT_PRODUCT_INDEX')
SEARCH_FIELD = os.getenv('SEARCH_FIELD')

from utils.db import PersonCollection
from utils.encoder import Encoder

person = PersonCollection(URI)
ecd = Encoder()
# Streamlit app
st.title("Attendance System")
# Sidebar options
option = st.sidebar.selectbox("Choose an option", ["Take Attendance", "Register Student", "View Attendance"])

if option == "Take Attendance":
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


    if st.button("Submit") and img_file_buffer is not None:
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
            person_info.append(result[0])
        
        for i in range(len(face_boxes)):
            x,y,w,h = face_boxes[i]
            cv2.putText(img1, person_info[i]['name'], (x,y), color=(0,255,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
        img_canvas1.image(img1, caption="result")
    
        progress_giver.empty()
        df = pd.DataFrame(person_info)
        st.table(df)
        if st.radio("Select the date", ["Today", "Custom"]) == "Custom":
            custom_date=st.date_input("Enter the date", value=datetime.datetime.now().date())
            df["date"] = custom_date
        else:
            df["date"] = datetime.datetime.now().date()
        if st.button("Save Attendance"):
            from utils.db import Attendence
            attendance= Attendence(URI)
            for i in range(len(df)):
                if not attendance.check_one(df["student_id"].values[i], df["date"].values[i]):
                    attendance.add_many(df["student_id"].values[i], df["date"].values[i])
                else:
                    st.write(f"Attendance already taken for {df['name'].values[i]}")
            st.write("Attendance saved successfully!")
    else:
        st.write("Please take a picture first")
        
elif option == "Register Student":
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
                progress_giver2.text("Registeration Successful")
            else:
                st.write("You are already registered.")
        else:
            st.write("Please provide all the required information.")
            
elif option == "View Attendance":
    st.header("View Attendance")
    student_id = st.text_input("Enter student ID")
    if st.radio("Select the number of days", ["7"]) == "7":
        n_days = 7
    if st.button("Search"):
        from utils.db import Attendence
        attendance = Attendence(URI)
        df,percent = attendance.fetch_attendence(student_id, n_days)
        st.table(df)
        st.write(f"Attendance percentage: {percent}")