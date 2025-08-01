import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


DATA_DIR = './data'

index = 0
for dir in os.listdir(DATA_DIR):
    if '.DS_Store' == dir:
         continue
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        img_filepath = os.path.join(DATA_DIR, dir, img_path)

        img = cv2.imread(img_filepath)
        img_rgb = to_rgb(img)

        plt.imshow(img_rgb)
        plt.show()

        index += 1
        if index >= 3:
            break
    
    if index >= 5:
            break
