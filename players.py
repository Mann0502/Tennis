import cv2
import numpy as np
from ultralytics import YOLO

model=YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model
cap=cv2.VideoCapture("Test.mp4")
court=np.load("court.npy")   # load the court polygon from file

def racket_hold(person,racket):
    px1, py1, px2, py2 = person
    w=abs(px2-px1)
    h=abs(py2-py1)
    tol_x=int(w*0.4)
    tol_y=int(h*0.4)

    for (rx1, ry1, rx2, ry2) in racket:
        rxc=((rx1+rx2)/2)
        ryc=((ry1+ry2)/2)
        if (px1-tol_x <= rxc <= px2+tol_x) and (py1-tol_y <= ryc <= py2+tol_y):
            return True
    return False

def score(person,racket):
    px1, py1, px2, py2 = person
    area=abs(px2-px1)*abs(py2-py1)
    score=area
    if racket_hold(person, racket):
        score =score +10000000  
    return score


while True:
    ret, frame=cap.read()
    if not ret:
        break
    results=model(frame , classes=[0,32,38] , verbose=False)  # Perform inference on the frame
    r=results[0]

    persons,rackets,balls=[],[],[]
    for box in r.boxes:
        cls=int(box.cls[0])
        x1,y1,x2,y2=box.xyxy[0]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        if cls == 0:  # person
            persons.append((x1,y1,x2,y2))
        elif cls == 32:  
            balls.append((x1,y1,x2,y2))
        elif cls == 38:  
            rackets.append((x1,y1,x2,y2))

    candidates=[]
    for person in persons:
        x1, y1, x2, y2 = person
        foot = (int((x1 + x2) / 2), y2)
        on_court=cv2.pointPolygonTest(court, foot, False) >= 0
        if on_court or racket_hold(person, rackets):
            candidates.append(person)

    candidates.sort(key=lambda p: score(p, rackets), reverse=True)
    players = candidates[:2]

    for (x1, y1, x2, y2) in players:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    # players green
    for (x1, y1, x2, y2) in balls:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)    # ball red
    for (x1, y1, x2, y2) in rackets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)    # racket blue

    cv2.polylines(frame, [court], True, (0, 255, 255), 2)           # court outline
    cv2.imshow("Tennis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

        
