import cv2
import mediapipe as mp
import pickle
import numpy as np


model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

labels_dict = {i: chr(i+65) for i in range(26)}

cap = cv2.VideoCapture(0)
while True:
    # print("Reading Frame...")
    ret, frame = cap.read()
    
    H, W, _ = frame.shape
    
    # print("Converting from BGR to RGB")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print("Processing Frame!")
    results = hands.process(frame_rgb)
    
    data_points = []  # would contain hand landmarks datapoints for each image
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # For each hand detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Each hand shall contain 21 landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                
                data_points.append(x)
                data_points.append(y)
                
                x_.append(x)
                y_.append(y)
        
        prediction = model.predict([np.asarray(data_points)])
        predicted_char = labels_dict[int(prediction[0])]
        # print(predicted_char)
    
        x1 = int(min(x_) * W) - 10
        x2 = int(max(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        y2 = int(max(y_) * H) - 10
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    # print("Showing Frame!")
    cv2.imshow('frame', frame)
    cv2.waitKey(25)


print("Releasing webcam")
cap.release()
print("Destroying all cv2 windows")
cv2.destroyAllWindows()
