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


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
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


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * 0.02  # %2 alan eşiği agresif yaptık

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)  # biraz daha agresif
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            # Açılar ile dörtgen doğruluğu kontrolü
            angles = []
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i+1) % 4]
                p0 = pts[(i-1) % 4]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = abs(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-10), -1.0, 1.0))))
                angles.append(angle)
            if all(60 < a < 120 for a in angles):  # biraz daha gevşek açı aralığı soft
                return pts.astype(np.float32)
    return _full_image_quad(ref_image)


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
        gamma = 1.8
    elif mean > 180:
        gamma = 0.6
    else:
        gamma = 1.0
    return _gamma_correction(image, gamma)


def _morph_ops(img: np.ndarray, close_iter=2, open_iter=1, kernel_size=(7, 7)) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return img


# ----------------------------
# Ön İşleme Varyasyonları
# ----------------------------

def _preprocess_variations(image: np.ndarray):
    """Birden fazla agresif ve soft ön işleme varyasyonu döner"""

    variations = []

    # Orijinal düz (soft)
    variations.append(image)

    # Agresif gamma artırımı (koyu görüntüler için)
    variations.append(_gamma_correction(image, 2.2))

    # Soft gamma azaltımı (parlak görüntüler için)
    variations.append(_gamma_correction(image, 0.7))

    # Keskinleştirme - Unsharp mask soft
    variations.append(_unsharp_mask(image, ksize=(3,3), strength=1.0))

    # Keskinleştirme - Unsharp mask agresif
    variations.append(_unsharp_mask(image, ksize=(5,5), strength=2.0))

    # Hafif bulanıklaştırma (soft)
    variations.append(cv2.GaussianBlur(image, (3,3), 0))

    # Orta bulanık (agresif)
    variations.append(cv2.GaussianBlur(image, (7,7), 0))

    # CLAHE (kontrast artırma)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    variations.append(clahe_bgr)

    return variations


# ----------------------------
# Maskeleme Fonksiyonları
# ----------------------------

def _mask_lab_red_background(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask_a = cv2.inRange(A, 130, 170)
    mask_b = cv2.inRange(B, 120, 160)
    color_mask = cv2.bitwise_or(mask_a, mask_b)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    edge_mask = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(color_mask))
    return _morph_ops(combined)


def _mask_lab_complex(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    _, mask_a = cv2.threshold(A, 135, 255, cv2.THRESH_BINARY_INV)
    _, mask_b = cv2.threshold(B, 135, 255, cv2.THRESH_BINARY)
    color_mask = cv2.bitwise_and(mask_a, mask_b)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    edge_mask = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, color_mask)
    return _morph_ops(combined)


# ----------------------------
# Kenar Tespiti Fonksiyonları
# ----------------------------

def _detect_edges_canny(image: np.ndarray, low=50, high=150) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


def _detect_contours_quad(image: np.ndarray, low=50, high=150) -> np.ndarray:
    edges = _detect_edges_canny(image, low, high)
    morph = _morph_ops(edges, close_iter=2, open_iter=1)
    return _find_quad_from_contours(morph, image)


def _adaptive_thresh_quad(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    sharpened = _unsharp_mask(blur)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    morph = _morph_ops(thresh, close_iter=1, open_iter=1)
    return _find_quad_from_contours(morph, image)


def _hough_fallback_quad(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return _full_image_quad(image)
    all_points = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)


# ----------------------------
# Sonuçları Gruplama & Seçme
# ----------------------------

def _compare_quads(quad1: np.ndarray, quad2: np.ndarray, thresh=25) -> bool:
    if quad1.shape != quad2.shape:
        return False
    dists = np.linalg.norm(quad1 - quad2, axis=1)
    return np.mean(dists) < thresh


def _average_quads(quads: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(quads, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _select_best_quad(quads: list[np.ndarray], ref_image: np.ndarray) -> np.ndarray:
    if not quads:
        return _full_image_quad(ref_image)

    filtered = [q for q in quads if not np.allclose(q, _full_image_quad(ref_image), atol=1)]
    if not filtered:
        return _full_image_quad(ref_image)

    groups = []
    for q in filtered:
        matched = False
        for g in groups:
            if _compare_quads(q, g[0]):
                g.append(q)
                matched = True
                break
        if not matched:
            groups.append([q])

    best_group = max(groups, key=len)
    return _average_quads(best_group)


# ----------------------------
# Dinamik Çoklu Aşamalı Algoritma
# ----------------------------

def auto_detect_document_corners_dynamic(image: np.ndarray) -> np.ndarray:
    corrected = _auto_gamma_correction(image)

    quads = []

    # Ön işlem varyasyonlarını uygula
    variations = _preprocess_variations(corrected)

    # Maskeleme sonuçları (her varyasyonla)
    for var_img in variations:
        # LAB maskeleme
        quad_red = _find_quad_from_contours(_mask_lab_red_background(var_img), var_img)
        quads.append(quad_red)

        quad_complex = _find_quad_from_contours(_mask_lab_complex(var_img), var_img)
        quads.append(quad_complex)

        # Adaptif threshold ve keskinleştirme
        quads.append(_adaptive_thresh_quad(var_img))

        # Canny + CLAHE
        quads.append(_auto_detect_clahe_canny(var_img))

    # Hough fallback (orijinal veya gamma düzeltme yapılmış)
    quads.append(_hough_fallback_quad(corrected))

    # En iyi sonucu seç
    best_quad = _select_best_quad(quads, corrected)

    if np.allclose(best_quad, _full_image_quad(corrected), atol=1):
        # Gerçekten hiç bulunamadıysa tüm görüntü
        return _full_image_quad(image)
    return best_quad


def _auto_detect_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    morph = _morph_ops(edges, close_iter=1, open_iter=1, kernel_size=(5, 5))
    return _find_quad_from_contours(morph, image)


# ====================
# Ana Component
# ====================
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
        pts = auto_detect_document_corners_dynamic(src_img)
        warped = _four_point_transform(src_img, pts)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = pts.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


Executor(sys.argv[1]).run()
