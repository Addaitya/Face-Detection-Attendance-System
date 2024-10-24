import os
from dotenv import load_dotenv
import cv2
import streamlit as st
import numpy as np
load_dotenv()
URI = os.getenv('DB_URI')
COSIN = os.getenv('COSIN_INDEX')
EUCLIDEAN = os.getenv('EUCLIDEAN_INDEX')
DOT_PRODUCT = os.getenv('DOT_PRODUCT_INDEX')
SEARCH_FIELD = os.getenv('SEARCH_FIELD')

print(f"{URI}\n{COSIN}\n{EUCLIDEAN}\n{DOT_PRODUCT}\n{SEARCH_FIELD}")

from utils.db import PersonCollection
from utils.encoder import Encoder

person = PersonCollection(URI)
ecd = Encoder()
# Streamlit app
st.title("Attendance System")
# Sidebar options
option = st.sidebar.selectbox("Choose an option", ["Take Attendance", "Register Student"])

if option == "Take Attendance":
    st.header("Take Attendance")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        img_array = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        #img = cv2.imread('Trash/8.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # embed
        embeddings = ecd.encode(img)

        result = person.search(
            embeddings[0],
            index_name=COSIN,
            field=SEARCH_FIELD
        )
        # Display the result
        st.write("Search Result:", result)

elif option == "Register Student":
    st.header("Register Student")
     # Input fields for student information
    student_name = st.text_input("Enter student name")
    student_id = st.text_input("Enter student ID")
    img_file_buffer = st.camera_input("Take a picture")

    if st.button("Register"):
        if img_file_buffer is not None and student_name and student_id:
            # Convert the image to a format suitable for OpenCV
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
        else:
            st.write("Please provide all the required information.")