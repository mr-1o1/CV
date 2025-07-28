import cv2
import mediapipe as mp

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_bbox_values(bbox, H, W, padding=0.01):
    x1 = int((bbox.xmin - padding * bbox.width) * W)
    y1 = int((bbox.ymin - padding * bbox.height) * H)
    w = int((bbox.width + 2 * padding * bbox.width) * W)
    h = int((bbox.height + 2 * padding * bbox.height) * H)
    # Ensure coordinates stay within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    w = min(w, W - x1)
    h = min(h, H - y1)

    return x1, y1, w, h


def blur_img_segment(img, x1, y1, w, h, ksize=55):
    img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], ksize=(ksize, ksize))
    return img

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
