import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)


DATA_DIR = './data'

for dir in os.listdir(DATA_DIR):
    if '.DS_Store' == dir:
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        img_filepath = os.path.join(DATA_DIR, dir, img_path)

        img = cv2.imread(img_filepath)
        img_rgb = to_rgb(img)

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        image_height, image_width, _ = img_rgb.shape
        annotated_image = img_rgb.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        plt.figure()
        plt.imshow(annotated_image)

plt.show()

