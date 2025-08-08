import os
import sys
import cv2
import numpy as np

from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts)
    if pts is None:
        raise ValueError("order_points: pts is None")
    if pts.size == 0:
        raise ValueError("order_points: pts is empty")
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
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

    # prevent zero dimension
    maxWidth = max(1, maxWidth)
    maxHeight = max(1, maxHeight)

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


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.05) -> np.ndarray:
    """
    Güvenli findContours wrapper:
    - OpenCV 3 ve 4 dönüş formatlarına dayanıklı
    - Konturlar yoksa tüm resmi döndürür
    - Bulunan konturlardan ilk uygun 4-köşe poligonu döner
    """
    if binary_img is None:
        return _full_image_quad(ref_image)

    # ensure binary_img is single channel uint8
    if binary_img.dtype != np.uint8:
        binary_img = binary_img.astype(np.uint8)
    if binary_img.ndim == 3:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

    contours_data = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours_data can be (contours, hierarchy) or (image, contours, hierarchy)
    if len(contours_data) == 2:
        contours, hierarchy = contours_data
    elif len(contours_data) == 3:
        _, contours, hierarchy = contours_data
    else:
        contours = []

    if not contours:
        return _full_image_quad(ref_image)

    contours = [c for c in contours if isinstance(c, (np.ndarray, list)) and len(c) > 0]
    if not contours:
        return _full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = float(ref_image.shape[0]) * float(ref_image.shape[1])
    min_area = img_area * float(min_area_ratio)

    for c in contours:
        try:
            area = float(cv2.contourArea(c))
        except Exception:
            continue
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # ensure approx is numpy array
        approx = np.asarray(approx)
        if approx is None or approx.size == 0:
            continue
        if len(approx) == 4:
            # flatten shape if needed
            a = approx.reshape(4, 2).astype(np.float32)
            if cv2.isContourConvex(a.astype(np.int32)):
                return a
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


def _mask_background_lab_range(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mask_a = cv2.inRange(A, 130, 170)
    mask_b = cv2.inRange(B, 120, 160)

    color_mask = cv2.bitwise_or(mask_a, mask_b)

    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(
        L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )

    edge_mask = cv2.Canny(L_blur, 40, 120)

    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(color_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined


def _auto_detect_document_corners_sharpen_adaptive(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    sharpened = _unsharp_mask(blur)
    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_bright_blur(image: np.ndarray) -> np.ndarray:
    gamma_corrected = _gamma_correction(image, gamma=1.8)

    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    sharp = _unsharp_mask(clahe_img, ksize=(5, 5), strength=1.5)

    edges = cv2.Canny(sharp, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    pts = _find_quad_from_contours(closed, image)
    return pts


def _mask_background_complex(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    _, mask_a = cv2.threshold(A, 135, 255, cv2.THRESH_BINARY_INV)
    _, mask_b = cv2.threshold(B, 135, 255, cv2.THRESH_BINARY)

    color_mask = cv2.bitwise_and(mask_a, mask_b)

    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(
        L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )

    edge_mask = cv2.Canny(L_blur, 40, 120)

    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, color_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined


def _filter_lines_by_angle(lines, angle_tol=10):
    if lines is None:
        return None
    filtered = []
    for line in lines:
        try:
            x1, y1, x2, y2 = line[0]
        except Exception:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if (abs(angle - 0) < angle_tol) or (abs(angle - 90) < angle_tol) or (abs(angle - 180) < angle_tol):
            filtered.append(line)
    return np.array(filtered) if filtered else None


def _auto_detect_document_corners_hough_improved(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    lines = _filter_lines_by_angle(lines, angle_tol=15)
    if lines is None or len(lines) < 4:
        return _full_image_quad(image)

    all_points = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)


def _texture_mask_gabor(image: np.ndarray, ksize=31, sigma=4.0, theta=np.pi/4, lambd=10.0, gamma=0.5) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8U, g_kernel)
    _, mask = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
    return mask


def _auto_detect_document_corners_lab_adaptive(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_L = clahe.apply(L)

    thresh_L = cv2.adaptiveThreshold(
        clahe_L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 2
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    combined = cv2.bitwise_or(thresh_L, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_color_segmentation(image: np.ndarray) -> np.ndarray:

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_light = np.array([0, 0, 180])
    upper_light = np.array([180, 30, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_inverse_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)


def _auto_detect_document_corners_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    # mag is float32; threshold must have same dtype
    mag_u8 = cv2.convertScaleAbs(mag)
    _, thresh = cv2.threshold(mag_u8, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)


def _score_quad(image: np.ndarray, quad: np.ndarray) -> float:
    # Basit scoring: alanına göre, merkezin görüntü merkezine uzaklığına göre ve
    # kenar yoğunluğuna göre puan veriyoruz.
    try:
        quad = np.asarray(quad, dtype=np.float32)
        if quad.size == 0:
            return -1.0
        # area (shoelace)
        x = quad[:, 0]
        y = quad[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # merkez uzaklığı
        cx = np.mean(x)
        cy = np.mean(y)
        h, w = image.shape[:2]
        img_cx, img_cy = w / 2.0, h / 2.0
        dist = np.hypot(cx - img_cx, cy - img_cy)
        max_dist = np.hypot(img_cx, img_cy)
        # kenar yoğunluğu: Canny ile ölç
        warped = _four_point_transform(image, quad)
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_w, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (warped.shape[0] * warped.shape[1] + 1e-9)

        # normalize and combine (area positive, smaller dist better, higher edge_density better)
        score = (area / (w * h + 1e-9)) * 0.6 + (1.0 - dist / (max_dist + 1e-9)) * 0.3 + edge_density * 0.1
        return float(score)
    except Exception:
        return -1.0


def _adaptive_contrast_enhancement(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_lum = np.mean(L)
    clip_limit = 2.0 if mean_lum < 100 else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(L)

    lab_clahe = cv2.merge((cl, A, B))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Gamma adaptif
    mean_gray = np.mean(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY))
    if mean_gray < 80:
        gamma = 1.8
    elif mean_gray > 180:
        gamma = 0.6
    else:
        gamma = 1.0

    return _gamma_correction(img_clahe, gamma=gamma)


def _preprocess_image_for_edges(image: np.ndarray) -> np.ndarray:
    # Bilateral filtre ile gürültü azaltma, kenar koruma
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # CLAHE kontrast iyileştirme (L kanalında)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    lab_clahe = cv2.merge((cl, A, B))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    # Unsharp mask ile keskinlik arttırma
    sharpened = _unsharp_mask(img_clahe)
    return sharpened


def _auto_canny(image: np.ndarray, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)
    return edges


# --------- Yeni ek: koyu arka plan için güçlendirme ve maske ----------
def _mask_dark_background(image: np.ndarray) -> np.ndarray:
    """LAB ve HSV kombinasyonu ile koyu arka planlarda belgeyi ayırmaya çalışır."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # L'de daha parlak bölgeleri seç
    _, light_mask = cv2.threshold(L, 60, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    _, not_dark_mask = cv2.threshold(v, 40, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_and(light_mask, not_dark_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined_mask


def _auto_detect_document_corners_dark_boost(image: np.ndarray) -> np.ndarray:
    """Koyu görüntüler için özel maske + kontur tabanlı tespit."""
    mask = _mask_dark_background(image)
    return _find_quad_from_contours(mask, image)


def _boost_dark_regions(image: np.ndarray) -> np.ndarray:
    """Düşük parlaklıktaki görüntüleri daha agresif CLAHE + gamma ile güçlendirir."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 90:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        L = clahe.apply(L)
        merged = cv2.merge((L, A, B))
        boosted = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        boosted = _gamma_correction(boosted, gamma=2.0)
        return boosted
    return image


# --------- Ana candidate detection (koyu entegrasyonu ile) ----------
def detect_document_candidates(image: np.ndarray) -> List[np.ndarray]:
    """
    Mevcut pipeline'ı bozmadan çalışır; koyu görüntülerde önce boost uygulayıp
    tüm metotları preprocessed görüntü üzerinde dener. Dönen candidate'lar
    numpy array listesi şeklindedir.
    """
    candidates: List[np.ndarray] = []
    img_corrected = _adaptive_contrast_enhancement(image)
    # koyu görüntülerde güçlendir
    if np.mean(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)) < 90:
        img_corrected = _boost_dark_regions(img_corrected)

    img_preprocessed = _preprocess_image_for_edges(img_corrected)

    variants = {
        "lab_range": lambda img: _find_quad_from_contours(_mask_background_lab_range(img), img),
        "color_segmentation": _auto_detect_document_corners_color_segmentation,
        "lab_adaptive": _auto_detect_document_corners_lab_adaptive,
        "sharpen_adaptive": _auto_detect_document_corners_sharpen_adaptive,
        "bright_blur": _auto_detect_document_corners_bright_blur,
        "clahe_canny": _auto_detect_document_corners_clahe_canny,
        "mask_background_complex": lambda img: _find_quad_from_contours(_mask_background_complex(img), img),
        "hough_improved": _auto_detect_document_corners_hough_improved,
        "inverse_threshold": _auto_detect_document_corners_inverse_threshold,
        "gradient_magnitude": _auto_detect_document_corners_gradient_magnitude,
        "dark_boost": _auto_detect_document_corners_dark_boost
    }

    for name, func in variants.items():
        try:
            # bazı fonksiyonlar ham image, bazıları preprocessed bekliyor; burada preprocessed veriyoruz
            quad = func(img_preprocessed) if callable(func) else None
            # güvenlik: tuple veya diğer tipleri ayıkla
            if isinstance(quad, tuple):
                # eğer (arr, ...) biçimindeyse ilk elemanı al
                quad = quad[0] if len(quad) > 0 else None
            if quad is None:
                continue
            quad = np.asarray(quad, dtype=np.float32)
            # eğer quad tüm görüntü ise atla (gerekiyorsa include etme)
            if quad.size == 0:
                continue
            try:
                full = _full_image_quad(image)
                if np.allclose(quad, full, atol=1):
                    continue
            except Exception:
                pass
            # reshape güvenliği: eğer 4x2 değil ise atla
            if quad.size == 8:
                quad = quad.reshape(4, 2)
            if quad.shape == (4, 2):
                candidates.append(quad.astype(np.float32))
        except Exception as e:
            print(f"Error in {name} variant: {e}")

    if not candidates:
        candidates.append(_full_image_quad(image))

    return candidates


def select_best_quad(image: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    best_quad = _full_image_quad(image)
    best_score = -1.0

    for quad in candidates:
        try:
            score = _score_quad(image, quad)
            if score > best_score:
                best_score = score
                best_quad = quad
        except Exception:
            continue

    return best_quad


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

        # Detect candidate quads from different methods
        candidates = detect_document_candidates(src_img)

        # Select the best quad based on scoring
        best_quad = select_best_quad(src_img, candidates)

        warped = _four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)



Executor(sys.argv[1]).run()
