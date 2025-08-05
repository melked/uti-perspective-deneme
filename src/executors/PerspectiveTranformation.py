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


# ========== Yardımcı Fonksiyonlar ==========

def _order_points(pts: np.ndarray) -> np.ndarray:
    """4 köşe noktasını: [üst-sol, üst-sağ, alt-sağ, alt-sol] şeklinde sırala."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # en küçük toplam = üst-sol
    rect[2] = pts[np.argmax(s)]  # en büyük toplam = alt-sağ
    rect[1] = pts[np.argmin(diff)]  # en küçük fark = üst-sağ
    rect[3] = pts[np.argmax(diff)]  # en büyük fark = alt-sol
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Verilen 4 köşeye göre perspektif düzeltmesi yap."""
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
    """Resmin tamamını kapsayan 4 köşe noktası döner."""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.05) -> np.ndarray:
    """Verilen binary görüntüden 4 köşe şeklinde büyük kontur bulmaya çalışır."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

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

    return _full_image_quad(ref_image)


def _unsharp_mask(image: np.ndarray, ksize=(5, 5), strength=1.5) -> np.ndarray:
    """Görüntüyü keskinleştir (unsharp mask)."""
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def _auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    """Otomatik gamma düzeltmesi yapar (ortalamaya göre ayarlanır)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean < 80:
        gamma = 1.8
    elif mean > 180:
        gamma = 0.6
    else:
        gamma = 1.0
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def _mask_background_lab_range(image: np.ndarray) -> np.ndarray:
    """LAB renk uzayına göre arka planı maskele."""
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


def _filter_lines_by_angle(lines, angle_tol=10):
    """Çizgileri, yatay veya dikey açılara yakın olanlarla filtrele."""
    if lines is None:
        return None
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if (abs(angle - 0) < angle_tol) or (abs(angle - 90) < angle_tol) or (abs(angle - 180) < angle_tol):
            filtered.append(line)
    return np.array(filtered) if filtered else None


def adaptive_canny(image_gray):
    """Otomatik eşik değerleri ile Canny kenar algılama."""
    median_val = np.median(image_gray)
    lower = int(max(0, 0.7 * median_val))
    upper = int(min(255, 1.3 * median_val))
    return cv2.Canny(image_gray, lower, upper)


def _line_intersection(line1, line2):
    """İki çizginin kesişim noktasını hesapla."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return [px, py]


def _get_intersections(lines, img_shape):
    """Çizgi çiftlerinin kesişim noktalarını bul, görüntü içinde olanları döndür."""
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = _line_intersection(lines[i][0], lines[j][0])
            if pt is None:
                continue
            x, y = pt
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                intersections.append([x, y])
    return np.array(intersections)


def _select_corners(points):
    """Kesişim noktalarından dışbükey 4 köşe seç."""
    if len(points) < 4:
        return None
    hull = cv2.convexHull(points.astype(np.float32))
    pts = hull.reshape(-1, 2)
    if len(pts) < 4:
        return None
    # Eğer 4'ten fazla nokta varsa ilk 4'ü al (daha gelişmiş seçme opsiyonel)
    if len(pts) > 4:
        pts = pts[:4]
    rect = _order_points(pts)
    return rect


def auto_detect_document_corners_dynamic(image: np.ndarray) -> np.ndarray:
    """
    Çok aşamalı belge köşe tespiti:
    - Gamma ve doku maskesi ile arka plan azaltma,
    - LAB renk segmentasyonu ile maskeleme,
    - Hough çizgileriyle kesişim bazlı köşe seçme,
    - Çoklu senaryolar için fallback algoritmalar
    """

    corrected = _auto_gamma_correction(image)

    # Doku maskesi ile arka planı azalt
    texture_mask = _texture_mask_gabor(corrected)
    reduced_texture_img = _reduce_background_texture(corrected, texture_mask)

    # 1) LAB renk maskesi
    mask = _mask_background_lab_range(reduced_texture_img)
    pts = _find_quad_from_contours(mask, reduced_texture_img)
    if not np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
        return pts

    # 2) Renk segmentasyonu
    pts = _auto_detect_document_corners_color_segmentation(reduced_texture_img)
    if not np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
         return pts

    # 3) LAB adaptif threshold + Canny
    pts = _auto_detect_document_corners_lab_adaptive(reduced_texture_img)
    if not np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
        return pts

    # 4) Hough lines ve kesişim noktalarından köşe seçimi (yeni eklenti)
    gray = cv2.cvtColor(reduced_texture_img, cv2.COLOR_BGR2GRAY)
    edges = adaptive_canny(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    filtered_lines = _filter_lines_by_angle(lines, angle_tol=15)
    if filtered_lines is not None and len(filtered_lines) >= 2:
        intersections = _get_intersections(filtered_lines, reduced_texture_img.shape)
        corners = _select_corners(intersections)
        if corners is not None:
            return corners

    gray = cv2.cvtColor(reduced_texture_img, cv2.COLOR_BGR2GRAY)
    contrast = gray.max() - gray.min()
    brightness = np.mean(gray)

    # 5) Parlaklık ve kontrast durumlarına göre fallback yöntemleri
    if contrast < 40:
        return _auto_detect_document_corners_sharpen_adaptive(reduced_texture_img)
    elif brightness > 200:
        pts = _auto_detect_document_corners_bright_blur(reduced_texture_img)
        if not np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
            return pts
        pts = _auto_detect_document_corners_clahe_canny(reduced_texture_img)
        if not np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
            return pts
        mask = _mask_background_complex(reduced_texture_img)
        pts = _find_quad_from_contours(mask, reduced_texture_img)
        if not np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
            return pts
        return _auto_detect_document_corners_hough_improved(reduced_texture_img)
    elif brightness > 180:
        return _auto_detect_document_corners_clahe_canny(reduced_texture_img)
    else:
        pts = _auto_detect_document_corners_clahe_canny(reduced_texture_img)
        if np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
            mask = _mask_background_complex(reduced_texture_img)
            pts = _find_quad_from_contours(mask, reduced_texture_img)
            if np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
                 pts = _auto_detect_document_corners_hough_improved(reduced_texture_img)
                 if np.allclose(pts, _full_image_quad(reduced_texture_img), atol=1):
                     pts = _find_quad_from_contours(_mask_background_lab_range(reduced_texture_img), reduced_texture_img, min_area_ratio=0.1)
        return pts


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
