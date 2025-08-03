import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
while True:
    print("Reading Frame...")
    ret, frame = cap.read()
    
    print("Converting from BGR to RGB")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Processing Frame!")
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    print("Showing Frame!")
    cv2.imshow('frame', frame)
    cv2.waitKey(25)


print("Releasing webcam")
cap.release()
print("Destroying all cv2 windows")
cv2.destroyAllWindows()
