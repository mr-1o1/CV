import cv2
import numpy as np

# Get the lower and upper limits of a color in HSV space
# color: list of 3 integers representing the color in BGR space
# returns: tuple of 2 lists of 3 integers representing the lower and upper limits of the color in HSV space
def get_limits(color):
    # Convert to numpy array
    c = np.uint8([[color]])  # insert the bgr color into an array to convert to hsv
    # Convert to HSV
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    # Set the lower limit to the same value as the color
    lower_limit = hsvC[0][0][0] - 10, 100, 100
    # Set the upper limit to the same value as the color
    upper_limit = hsvC[0][0][0] + 10, 255, 255
    
    # Convert to numpy array
    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    
    return lower_limit, upper_limit
