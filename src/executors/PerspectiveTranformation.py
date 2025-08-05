import os
import sys
import cv2
import numpy as np
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# ----------------------------------------
# Yardımcı Fonksiyonlar
# ----------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _unsharp_mask(image: np.ndarray, ksize=(5, 5), strength=1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def _auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean < 80:
        gamma = 1.8
    elif mean > 180:
        gamma = 0.6
    else:
        gamma = 1.0
    return _gamma_correction(image, gamma)


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.03) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * min_area_ratio

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [px, py]


def _point_in_bounds(pt, shape):
    x, y = pt
    h, w = shape[:2]
    return 0 <= x < w and 0 <= y < h


def _cluster_points(points, max_clusters=4, tolerance=50):
    # sklearn yok, basit kümeleme:
    # Noktaları birbirine yakın gruplar halinde topla
    clusters = []
    for pt in points:
        found_cluster = False
        for c in clusters:
            if np.linalg.norm(np.array(pt) - np.array(c[0])) < tolerance:
                c.append(pt)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([pt])

    # Her kümeden ortalama al
    centers = []
    for c in clusters:
        center = np.mean(c, axis=0)
        centers.append(center)

    # 4 küme yoksa, eksik küme varsa en yakın noktaları birleştir
    while len(centers) > max_clusters:
        # En yakın iki merkezi bul, birleştir
        min_dist = np.inf
        idx1, idx2 = 0, 1
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    min_dist = dist
                    idx1, idx2 = i, j
        # İki kümeyi birleştir
        new_center = (centers[idx1] + centers[idx2]) / 2
        centers.pop(max(idx1, idx2))
        centers.pop(min(idx1, idx2))
        centers.append(new_center)

    # Eğer 4'ten azsa, yine en yakın merkezleri çoğalt (burası nadir ama)
    while len(centers) < max_clusters:
        centers.append(centers[-1])  # son merkezi çoğalt

    return np.array(centers)


def _auto_detect_corners_hough(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return None

    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            pt = _line_intersection(line1, line2)
            if pt is not None and _point_in_bounds(pt, image.shape):
                points.append(pt)

    if len(points) < 4:
        return None

    clustered = _cluster_points(points, max_clusters=4)
    if clustered.shape[0] != 4:
        return None

    return clustered.astype(np.float32)


def _prepare_variations(image: np.ndarray):
    variations = []

    gamma_auto = _auto_gamma_correction(image)
    variations.append(("gamma_auto", gamma_auto))

    gray = cv2.cvtColor(gamma_auto, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    variations.append(("clahe_gamma_auto", clahe_bgr))

    sharpened = _unsharp_mask(clahe_bgr, strength=1.5)
    variations.append(("clahe_sharp_gamma_auto", sharpened))

    sharp = _unsharp_mask(gamma_auto, strength=1.5)
    variations.append(("sharp_gamma_auto", sharp))

    sharp_only = _unsharp_mask(image, strength=1.5)
    variations.append(("sharp_only", sharp_only))

    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray_orig, 9, 75, 75)
    sharpened_blur = _unsharp_mask(blur, strength=2.0)
    thresh = cv2.adaptiveThreshold(
        sharpened_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    variations.append(("aggressive_thresh", thresh_bgr))

    gamma_low = _gamma_correction(image, gamma=0.6)
    variations.append(("gamma_low", gamma_low))

    gamma_high = _gamma_correction(image, gamma=1.8)
    variations.append(("gamma_high", gamma_high))

    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bilateral_bgr = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
    bilateral_sharp = _unsharp_mask(bilateral_bgr, strength=1.8)
    variations.append(("clahe_bilateral_sharp", bilateral_sharp))

    gamma_12 = _gamma_correction(image, gamma=1.2)
    gamma_12_sharp = _unsharp_mask(gamma_12, strength=1.3)
    variations.append(("gamma_12_sharp", gamma_12_sharp))

    return variations


def _score_quad(quad: np.ndarray, image_shape) -> float:
    def angle(pt1, pt2, pt3):
        a = np.array(pt1) - np.array(pt2)
        b = np.array(pt3) - np.array(pt2)
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

    angles = []
    quad = _order_points(quad)
    for i in range(4):
        angles.append(angle(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]))

    rectness_score = np.mean([abs(90 - a) for a in angles])

    img_area = image_shape[0] * image_shape[1]
    quad_area = cv2.contourArea(quad.reshape(4, 1, 2))
    area_ratio = quad_area / img_area

    if area_ratio < 0.02:
        return -np.inf

    score = (area_ratio * 1000) - rectness_score * 2

    return score


def _combine_quads(quad1: Optional[np.ndarray], quad2: Optional[np.ndarray], image_shape):
    if quad1 is None and quad2 is None:
        return None
    if quad1 is None:
        return quad2
    if quad2 is None:
        return quad1

    dist = np.linalg.norm(np.sort(quad1, axis=0) - np.sort(quad2, axis=0))
    if dist < 50:
        return (quad1 + quad2) / 2
    else:
        score1 = _score_quad(quad1, image_shape)
        score2 = _score_quad(quad2, image_shape)
        return quad1 if score1 > score2 else quad2


def _detect_best_quad(image: np.ndarray) -> np.ndarray:
    variations = _prepare_variations(image)
    best_quad = None
    best_score = -np.inf

    for name, var_img in variations:
        gray = cv2.cvtColor(var_img, cv2.COLOR_BGR2GRAY)
        if "aggressive_thresh" in name:
            binary = gray
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)

        quad_contour = _find_quad_from_contours(binary, image)
        quad_hough = _auto_detect_corners_hough(var_img)
        quad = _combine_quads(quad_contour, quad_hough, image.shape)

        if quad is not None:
            score = _score_quad(quad, image.shape)
            if score > best_score:
                best_score = score
                best_quad = quad

    if best_quad is None:
        return _full_image_quad(image)
    return _order_points(best_quad)


# ----------------------------------------
# Ana Component Class
# ----------------------------------------

class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        pts = _detect_best_quad(src_img)
        warped = _four_point_transform(src_img, pts)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = pts.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
