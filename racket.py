import cv2
from ultralytics import YOLO

racket_model = YOLO("racket.pt")
cap = cv2.VideoCapture("TEST (2).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = racket_model(frame, imgsz=1920, conf=0.15, verbose=False)
    r = results[0]
    print("rackets detected:", len(r.boxes))

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow("racket test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()