import cv2
import numpy as np
from utils import get_limits

yellow = [0, 255, 255]

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert frame from BGR to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the lower and upper limits of the color
    lower_limit, upper_limit = get_limits(yellow)

    # Create a mask of the color
    mask = cv2.inRange(frame_hsv, lower_limit, upper_limit)

    # Show the mask
    cv2.imshow("Frame", mask)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()