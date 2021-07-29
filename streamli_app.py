import cv2
from helper.compied import funcmain, draw_all
from helper.face_detector import get_face_detector, find_faces, draw_faces
from helper.face_landmarks import get_landmark_model
import streamlit as st
from helper.info import about, welcome

st.set_page_config(page_title='Face Features and Landmarks Detection', layout= 'wide')

st.title('Facial Landmarks Detection App')
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Select page:", options = ["Welcome", "App", "About"])


face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)


if page == 'Welcome':
    st.header("Welcome to the Neural Network based facial landmarks detection App!")
    welcome()



if page == 'App':
    val = st.slider('threshold', 0, 255, 120)
    hold = st.checkbox("show thresholded image")
    if st.button('start'):
        stop = st.button('stop')
        frame = st.empty()
        frame1 = st.empty()
        while(True):
            ret, img = cap.read()
            rects = find_faces(img, face_model)
            img = draw_faces(img, rects)
            for rect in rects:
                try:
                    cxl, cyl, cxr, cyr, points, points2, points3, points4, thresh = funcmain(img, landmark_model, rect, val)
                    draw_all(img, cxl, cyl, cxr, cyr, points, points2, points3, points4)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e: pass       
            try: 
                if hold:
                    frame.image(img)
                    frame1.image(thresh)
                else:
                    frame.image(img)
            except Exception as e: pass
                
            if stop:
                break

    cap.release()

if page == 'About':
    st.header("About section")
    about()