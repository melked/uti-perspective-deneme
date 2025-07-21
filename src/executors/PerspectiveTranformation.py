import cv2
import numpy as np

def auto_perspective(image_path, output_path="output.jpg", width=800, height=600):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx.reshape(4, 2)
            break

    if screen_cnt is None:
        h, w = img.shape[:2]
        screen_cnt = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

    dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.float32(screen_cnt), dst_points)
    warped = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imwrite(output_path, warped)
    return warped

