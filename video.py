import cv2
import numpy as np

cap = cv2.VideoCapture("Tennis_file.mp4")
if not cap.isOpened():
    print("Could not open the video")



while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the video back to start
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    

    lower = np.array([22, 16, 0])    # these values are obtained by manual curve fitting using sliders on masked videos
    upper = np.array([50, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50 or area > 200:                     # still drop the tiny specks
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        circle_area = 3.1416 * radius * radius
        circularity = area / circle_area   # how much of its circle the blob fills

        if circularity < 0.5:              # not round enough -> skip
            continue

        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        

    cv2.imshow("Tennis Vision", frame)
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
