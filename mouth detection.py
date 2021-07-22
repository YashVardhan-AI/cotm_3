import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3
font = cv2.FONT_HERSHEY_SIMPLEX 
cap = cv2.VideoCapture(0)
#make a function to make lines using the marks in opencv2
def line(img, marks):
    cv2.drawContours(img, [marks], 0, (255,255,255), 1)
    
def linemain(img,marks):
    for index, item in enumerate(marks): 
        if index == len(marks) -1:
            break
        cv2.line(img, item, marks[index + 1], [255, 255, 255], 1) 
while(True):
    try:
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            points = shape[30:36]
            draw_marks(img, points, color=(150,100,50))
            line(img, points)
            points2=shape[27:31]
            linemain(img, points2)
            print(points2)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()
