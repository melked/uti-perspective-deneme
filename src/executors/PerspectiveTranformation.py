import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# === Yardımcı Fonksiyonlar ===

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # sol üst
    rect[2] = pts[np.argmax(s)]      # sağ alt
    rect[1] = pts[np.argmin(diff)]   # sağ üst
    rect[3] = pts[np.argmax(diff)]   # sol alt
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


def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def _auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean < 80:
        gamma = 1.8  # koyu fotoğraflar için parlaklık artır
    elif mean > 180:
        gamma = 0.6  # parlak fotoğraflar için koyultma
    else:
        gamma = 1.0  # normal
    return _gamma_correction(image, gamma)


# === Kümelendirme (KMeans Olmadan) ===

def _line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return np.array([px, py], dtype=np.float32)


def _calculate_intersections(lines):
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i][0], lines[j][0])
            if pt is not None:
                points.append(pt)
    return np.array(points) if points else np.array([])


def _cluster_points_without_kmeans(points: np.ndarray, eps=30) -> np.ndarray:
    if len(points) <= 4:
        return points.astype(np.float32)

    clusters = []
    for pt in points:
        assigned = False
        for cluster in clusters:
            if np.linalg.norm(pt - cluster[0]) < eps:
                cluster.append(pt)
                assigned = True
                break
        if not assigned:
            clusters.append([pt])

    if len(clusters) > 4:
        clusters = clusters[:4]

    cluster_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    return cluster_centers.astype(np.float32)


# === Kenar Tespit Fonksiyonları ===

def _contour_based_corners(image: np.ndarray, block_size=11, c=2) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    sharpened = _unsharp_mask(blur)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, c)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)


def _clahe_canny_corners(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return _find_quad_from_contours(morph, image)


def _bright_blur_corners(image: np.ndarray) -> np.ndarray:
    gamma_corrected = _gamma_correction(image, gamma=1.8)
    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    sharp = _unsharp_mask(clahe_img, ksize=(5, 5), strength=1.5)
    edges = cv2.Canny(sharp, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_from_contours(closed, image)


def _mask_background_lab_range(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask_a = cv2.inRange(A, 130, 170)
    mask_b = cv2.inRange(B, 120, 160)
    color_mask = cv2.bitwise_or(mask_a, mask_b)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
    edge_mask = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(color_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined


def _mask_background_complex(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    _, mask_a = cv2.threshold(A, 135, 255, cv2.THRESH_BINARY_INV)
    _, mask_b = cv2.threshold(B, 135, 255, cv2.THRESH_BINARY)
    color_mask = cv2.bitwise_and(mask_a, mask_b)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
    edge_mask = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, color_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined


def _auto_detect_document_corners_hough(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return _full_image_quad(image)

    intersections = _calculate_intersections(lines)
    if intersections.size == 0:
        return _full_image_quad(image)

    clustered = _cluster_points_without_kmeans(intersections)
    if len(clustered) < 4:
        return _full_image_quad(image)

    return _order_points(clustered[:4])


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * 0.05

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return _full_image_quad(ref_image)


# === Skorlama (Basit) ===

def _score_quad(quad: np.ndarray, img_shape) -> float:
    # Alan oranı ve dikdörtgenlik için basit skor
    h, w = img_shape[:2]
    img_area = h * w

    area = cv2.contourArea(quad.reshape(-1,1,2))
    if area < img_area * 0.01:
        return 0

    rect = _order_points(quad)
    edges = [np.linalg.norm(rect[i] - rect[(i+1)%4]) for i in range(4)]
    angle_scores = 0
    for i in range(4):
        v1 = rect[(i+1)%4] - rect[i]
        v2 = rect[(i+2)%4] - rect[(i+1)%4]
        angle = abs(np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))))
        angle_deg = np.degrees(angle)
        angle_scores += (90 - abs(90 - angle_deg))  # 90'a ne kadar yakınsa o kadar iyi

    angle_scores /= 4
    if angle_scores < 70:
        return 0  # Çok yamuk

    return area / img_area * (angle_scores / 90)


# === Dinamik Çoklu Deneme ===

def auto_detect_document_corners_multi(image: np.ndarray) -> np.ndarray:
    attempts = []

    # 1. Kırmızı arka plan maskesi
    mask_lab = _mask_background_lab_range(image)
    pts = _find_quad_from_contours(mask_lab, image)
    if not np.allclose(pts, _full_image_quad(image), atol=2):
        attempts.append(pts)

    # 2. CLAHE + Canny
    pts = _clahe_canny_corners(image)
    if not np.allclose(pts, _full_image_quad(image), atol=2):
        attempts.append(pts)

    # 3. Sharpen + Adaptive Threshold
    pts = _contour_based_corners(image)
    if not np.allclose(pts, _full_image_quad(image), atol=2):
        attempts.append(pts)

    # 4. Bright blur variant
    pts = _bright_blur_corners(image)
    if not np.allclose(pts, _full_image_quad(image), atol=2):
        attempts.append(pts)

    # 5. Complex background mask
    mask_complex = _mask_background_complex(image)
    pts = _find_quad_from_contours(mask_complex, image)
    if not np.allclose(pts, _full_image_quad(image), atol=2):
        attempts.append(pts)

    # 6. Hough lines + intersection clustering
    pts = _auto_detect_document_corners_hough(image)
    if not np.allclose(pts, _full_image_quad(image), atol=2):
        attempts.append(pts)

    # 7. Full image fallback
    if not attempts:
        return _full_image_quad(image)

    # Skorla ve en iyisini seç
    best_score = 0
    best_pts = _full_image_quad(image)
    for candidate in attempts:
        score = _score_quad(candidate, image.shape)
        if score > best_score:
            best_score = score
            best_pts = candidate

    return best_pts


# === Ana Component ===

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

        pts = auto_detect_document_corners_multi(src_img)

        warped = _four_point_transform(src_img, pts)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = pts.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


Executor(sys.argv[1]).run()
