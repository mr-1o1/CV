from utils import get_image_directory, read_image
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import convolve
import cv2

def sobel_operator(image, kernel_size=3):
    ##############################################################################
    # Sobel operator
    # The Sobel operator is a discrete differentiation operator that computes an
    # approximation of the gradient of the image intensity function.
    # It is used to find the edges of an image.
    # The Sobel operator consists of a pair of 3x3 convolution kernels, one for
    # detecting changes in the horizontal direction (Gx) and one for detecting
    # changes in the vertical direction (Gy).
    # The kernels are as follows:
    # Gx = [[-1, 0, 1],
    #       [-2, 0, 2],
    #       [-1, 0, 1]]
    # Gy = [[-1, -2, -1],
    #       [0, 0, 0],
    #       [1, 2, 1]]
    # The gradient magnitude is then calculated as:
    # G = sqrt(Gx^2 + Gy^2)
    # The direction of the gradient can also be calculated as:
    # theta = arctan(Gy/Gx)
    # The Sobel operator is less sensitive to noise than the simple difference
    # operator and is therefore more suitable for edge detection.
    # The Sobel operator is also used in image processing to find the edges of
    # objects in an image.
    # It is a simple and efficient way to detect edges in an image.
    # The Sobel operator is used in many applications, including image
    # segmentation, object detection, and image recognition.
    ##############################################################################
    """
    Applies the Sobel operator to an image to detect edges.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The gradient magnitude of the image.
    """

    kernel_size = 3

    # Apply sobel operator using cv2
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Calculate the gradient magnitude (cartesian to polar)
    gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_x, gradient_y)  # returns magnitude and angle
    print("****************")
    # print(len(gradient_combined))
    print(gradient_magnitude.shape)
    print(gradient_angle.shape)

    # plt.imshow(gradient_angle, cmap='gray')
    # plt.title('Gradient Angle')
    # plt.axis('off')
    # plt.show()

    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)  # apply only to the magnitude
    return gradient_magnitude, gradient_angle

# ----------------------------------------------------------------------------
def gradient_angle_in_color(gradient_angle, gradient_img):
    """
    Displays the gradient angle in color.
    """
    # Normalize angle to [0, 360] for hue
    hue = gradient_angle * 180 / np.pi  # Convert radians to degrees

    # Create HSV image
    hsv = np.zeros((gradient_angle.shape[0], gradient_angle.shape[1], 3), dtype=np.float32)

    hsv[..., 0] = hue / 2  # Hue in [0, 180] for OpenCV
    hsv[..., 1] = 1.0  # Full saturation
    hsv[..., 2] = cv2.normalize(gradient_img, None, 0, 1, cv2.NORM_MINMAX)  # Magnitude as value

    # Convert to RGB
    gradient_angle_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return gradient_angle_color


# ----------------------------------------------------------------------------
                                # Main Function
# ----------------------------------------------------------------------------
# image_path = read_image("/Users/hanzy/Documents/Tuni/ML_Learning/CV/images/standard_test_images/cameraman.tif")
# image_path = "/Users/hanzy/Documents/Tuni/ML_Learning/CV/images/standard_test_images/cameraman.tif"
# # does file exist? check with os.path.exists(image_path)
# if os.path.exists(image_path):
#     print(f"File exists: {image_path}")
#     # Read the image
#     image = read_image(image_path)
    
#     # Apply the Sobel operator
#     gradient_img, gradient_angle = sobel_operator(image)
    
#     # Print the gradient magnitude
#     print(f"Gradient magnitude shape: {gradient_img.shape}")
#     # print(f"Gradient magnitude: {gradient_img}")
    
#     # plt show the gradient magnitude 
#     plt.imshow(gradient_img, cmap='gray')
#     plt.title('Gradient Magnitude')
#     plt.axis('off')
#     plt.show()

#     gradient_angle_in_color(gradient_angle, gradient_img)

# else:
#     print(f"File does not exist: {image_path}")


