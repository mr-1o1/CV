import cv2
import mediapipe as mp
from utils import to_bgr, to_rgb, blur_img_segment, get_bbox_values


def process_img(img, face_detection_obj):
    print("="*80)
    print("In da process_img function")
    img_rgb = to_rgb(img)
    H, W, _ = img_rgb.shape
    results = face_detection_obj.process(img_rgb)

    if results.detections:
        print(results.detections)
        for detection in results.detections:
            x1, y1, w, h = get_bbox_values(detection.location_data.relative_bounding_box, H, W)

            # Blur Image
            img_rgb = blur_img_segment(img_rgb, x1, y1, w, h)
    return to_bgr(img_rgb)

# Detect the face
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring Empty Camera Frame")
            continue

        image.flags.writeable = False
        image = to_rgb(image)
        results = face_detection.process(image)

        image.flags.writeable = True
        image = to_bgr(image)

        image = process_img(image, face_detection)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
