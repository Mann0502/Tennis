import cv2
from ultralytics import YOLO

ball_model = YOLO("best.pt")      # the downloaded fine-tuned ball weights
cap = cv2.VideoCapture("TEST (2).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = ball_model(frame, imgsz=1280, conf=0.1, verbose=False)
    r = results[0]
    print("balls detected:", len(r.boxes))        # how many this frame

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)   # red box on ball

    cv2.imshow("ball test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()