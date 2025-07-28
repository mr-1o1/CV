import cv2
import numpy as np

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_bbox_values(bbox, H, W, padding=0.01):
    print("="*80)
    print("in the bounding box")
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


