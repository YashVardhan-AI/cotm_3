import cv2
from helper.compied import funcmain, draw_all
from helper.face_detector import get_face_detector, find_faces, draw_faces
from helper.face_landmarks import get_landmark_model


face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    img = draw_faces(img, rects)
    for rect in rects:
        
        cxl, cyl, cxr, cyr, points, points2, points3, points4 = funcmain(img, landmark_model, rect)

        draw_all(img, cxl, cyl, cxr, cyr, points, points2, points3, points4)
    
    cv2.imshow('pls', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
