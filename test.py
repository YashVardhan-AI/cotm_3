import cv2
points = [(375, 193), (364, 113), (277, 20), (271, 16), (52, 106), (133, 266), (289, 296), (372, 282)]

cap = cv2.VideoCapture(0)
#make a function to make lines using the marks in opencv2
def line(img, marks):
    cv2.drawContours(img, [marks], 0, (255,255,255), 2)
    

def linemain(img,marks):
    for index, item in enumerate(marks): 
        if index == len(marks) -1:
            break
        cv2.line(img, item, marks[index + 1], [0, 255, 0], 2) 
while(True):
    ret, frame = cap.read()
    linemain(frame,points)
    cv2.imshow("Output", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
