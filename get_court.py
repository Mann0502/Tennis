
import numpy as np
import cv2
cap = cv2.VideoCapture("Test.mp4")
ret, frame = cap.read()
cap.release()


points=[]

def click(event , x, y ,flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(x, y)

cv2.namedWindow("court")
cv2.setMouseCallback("court", click)

while True:
    cv2.imshow("court", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
np.save("court.npy", np.array(points, np.int32))   # save clicks to a file
print("saved court.npy")

      
