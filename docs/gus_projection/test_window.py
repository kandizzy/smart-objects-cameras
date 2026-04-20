import cv2
import numpy as np

img = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.putText(img, "PROJECTOR TEST", (200, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()