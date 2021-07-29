import os 
import cv2
from helper.compied import funcmain, draw_all
from helper.face_detector import get_face_detector, find_faces, draw_faces
from helper.face_landmarks import get_landmark_model

path = 'Face_detection/img'
face_model = get_face_detector()
landmark_model = get_landmark_model()

for image in os.listdir(path):
    img1 =  cv2.imread(path + "/" + image)
    img =  cv2.resize(img, (600, 490))
    rects = find_faces(img, face_model)
    img = draw_faces(img, rects)

    for rect in rects:
        try: 
            cxl, cyl, cxr, cyr, points, points2, points3, points4 = funcmain(img, landmark_model, rect)

            draw_all(img, cxl, cyl, cxr, cyr, points, points2, points3, points4)
            
            
        except:
            pass
    
    cv2.imwrite(f'Face_detection/processed/edited_{image}', img)
    
    cv2.waitKey(1000)

    

cv2.destroyAllWindows()
