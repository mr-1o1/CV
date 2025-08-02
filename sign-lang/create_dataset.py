import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def printp(*args):
    print("="*80)
    print()
    print(*args)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

DATA_DIR = './data'
for dir in os.listdir(DATA_DIR)[:2]:
    if '.DS_Store' == dir:
        continue
    
    printp("Current Reading From: ", dir)

    # Reading images in the selected directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        img_filepath = os.path.join(DATA_DIR, dir, img_path)
        img = cv2.imread(img_filepath)
        img_rgb = to_rgb(img)

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue
        
        # For each hand detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Each hand shall contain 21 landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
            

