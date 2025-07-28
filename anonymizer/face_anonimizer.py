import cv2
import mediapipe as mp
from utils import to_bgr, to_rgb, blur_img_segment, get_bbox_values


# Read the image
# img_path = "/Users/hanzy/Documents/Projects/py-projects/CV/images/rgb/boy_portrait.png"
img_path = "/Users/hanzy/Documents/Projects/py-projects/CV/images/rgb/four_people.jpg"
img = cv2.imread(img_path)  # image in BGR format
H, W, _ = img.shape
print(f"Image dimensions: H={H}, W={W}")

# Detect the face
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(img)

    if results.detections:
        print(results.detections)
        for detection in results.detections:
            x1, y1, w, h = get_bbox_values(detection.location_data.relative_bounding_box, H, W)

            # Blur Image
            img = blur_img_segment(img, x1, y1, w, h)

            # green_color = [0, 255, 0]
            # thickness = 3
            # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), green_color, thickness=thickness)

            # mp_drawing.draw_detection(img, detection)

            # img = cv2.blur(img, (10, 10))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






