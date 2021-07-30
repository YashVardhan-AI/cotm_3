import cv2
from helper.compied import funcmain, draw_all
from helper.face_detector import get_face_detector, find_faces, draw_faces
from helper.face_landmarks import get_landmark_model
import streamlit as st
from helper.info import about, welcome
from PIL import Image
import numpy as np
from io import BytesIO
import base64

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    img = np.array(img)
    return img

def download_link(img):
    img = Image.fromarray(img)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =f'<a href="data:file/jpg;base64,{img_str}" download="{"stylized.jpg"}"><input type="button" value="Download"></a>'
    return href


st.set_page_config(
    page_title='Face Features and Landmarks Detection', layout='wide')

st.title('Facial Landmarks Detection App')
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Select page:", options=[
                            "Welcome", "Face Detection", "Edge Detection", "About"])


face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)


if page == 'Welcome':
    st.header("Welcome to the Neural Network based facial landmarks detection App!")
    welcome()


elif page == 'Face Detection':
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
                    cxl, cyl, cxr, cyr, points, points2, points3, points4, thresh = funcmain(
                        img, landmark_model, rect, val)
                    draw_all(img, cxl, cyl, cxr, cyr, points,
                             points2, points3, points4)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    pass
            try:
                if hold:
                    frame.image(img)
                    frame1.image(thresh)
                else:
                    frame.image(img)
            except Exception as e:
                pass

            if stop:
                break

    cap.release()

elif page == 'Edge Detection':
    thresh1 = st.slider('threshold1', 0, 255, 100)
    thresh2 = st.slider('threshold2', 0, 255, 220)
    option = st.sidebar.radio('What do you prefer?', options=[
                              'Custom images', 'real time'])

    if option == 'Custom images':
        content = st.sidebar.file_uploader(
            "Choose a content image", type=['png', 'jpg', 'jpeg'])
        if content != None:
            image = load_image(content)
            image = cv2.Canny(image, thresh1, thresh2)
            st.image(image, width=500)
            st.markdown(download_link(image), unsafe_allow_html=True)

    else:
        if st.button('start real time camera'):
            stop = st.button('stop camera')
            frame2 = st.empty()
            while(True):
                ret, img = cap.read()
                img = cv2.Canny(img, thresh1, thresh2)
                frame2.image(img)
                if stop:
                    break
    cap.release()


if page == 'About':
    st.header("About section")
    about()
