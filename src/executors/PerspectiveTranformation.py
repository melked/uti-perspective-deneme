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


# Yardımcı: Noktaları sırala (üst sol, üst sağ, alt sağ, alt sol)
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # Üst sol
    rect[2] = pts[np.argmax(s)]      # Alt sağ
    rect[1] = pts[np.argmin(diff)]   # Üst sağ
    rect[3] = pts[np.argmax(diff)]   # Alt sol
    return rect


# Perspektif dönüşüm uygula
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


# Tam görüntü dörtgeni (fallback)
def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# Konturdan en uygun dörtgeni bul (en büyük 4 köşe ve konveks)
def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * 0.03  # Gürültü eleme için %3

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return _full_image_quad(ref_image)


# Unsharp mask ile keskinleştir
def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


# Gamma düzeltme fonksiyonu
def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


# Otomatik gamma seçimi (parlaklık bazlı)
def _auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean < 80:
        gamma = 1.8  # Karanlık artır
    elif mean > 180:
        gamma = 0.6  # Çok parlak azalt
    else:
        gamma = 1.0  # Normal
    return _gamma_correction(image, gamma)


# Laplacian varyansı ile bulanıklık ölçümü
def _measure_blur(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# LAB renk alanında arka planı maskele (kırmızı, desenli arka plan için)
def _mask_background_lab_range(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mask_a = cv2.inRange(A, 130, 180)  # Kırmızı tonları için A aralığı
    mask_b = cv2.inRange(B, 120, 180)  # Kırmızımsı arka plan için B aralığı
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


# Agresif (bulanık ve düşük kontrast) kenar tespiti
def _auto_detect_document_corners_aggressive(image: np.ndarray) -> np.ndarray:
    corrected = _gamma_correction(image, gamma=1.8)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    sharpened = _unsharp_mask(clahe_img, ksize=(5, 5), strength=1.5)

    edges = cv2.Canny(sharpened, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return _find_quad_from_contours(closed, image)


# Soft (net ve iyi kontrast) kenar tespiti
def _auto_detect_document_corners_soft(image: np.ndarray) -> np.ndarray:
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


# CLAHE + Canny kenar tespiti (parlak veya normal durumlar için)
def _auto_detect_document_corners_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return _find_quad_from_contours(morph, image)


# Hough Lines fallback
def _auto_detect_document_corners_hough(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return _full_image_quad(image)

    points = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)


# Dinamik mod seçici ve köşe bulucu
def auto_detect_document_corners_dynamic(image: np.ndarray) -> np.ndarray:
    # Öncelikle gamma düzeltme ile ışık optimize
    corrected = _auto_gamma_correction(image)
    blur_val = _measure_blur(corrected)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    contrast = gray.max() - gray.min()
    brightness = np.mean(gray)

    # 1) Kırmızı / desenli arka plan varsa maskeyi deneyelim
    mask = _mask_background_lab_range(corrected)
    pts = _find_quad_from_contours(mask, corrected)
    if not np.allclose(pts, _full_image_quad(corrected), atol=1):
        return pts

    # 2) Bulanıklık eşiği (Laplacian varyansına göre)
    BLUR_THRESHOLD = 80.0

    if blur_val < BLUR_THRESHOLD or contrast < 40:
        # Agresif mod: bulanık, düşük kontrast
        pts = _auto_detect_document_corners_aggressive(corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

    # 3) Parlak ve net durumlar için soft mod deneyelim
    if brightness > 200:
        # Çok parlak → agresif modla deneyip başarısızsa soft mod
        pts = _auto_detect_document_corners_aggressive(corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts
        pts = _auto_detect_document_corners_clahe_canny(corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        # Renk maskesi
        mask = _mask_background_lab_range(corrected)
        pts = _find_quad_from_contours(mask, corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        # Fallback hough
        return _auto_detect_document_corners_hough(corrected)

    elif brightness > 180:
        # Normal parlak → soft mod yeterli olabilir
        pts = _auto_detect_document_corners_clahe_canny(corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        # Renk maskesi deneyelim
        mask = _mask_background_lab_range(corrected)
        pts = _find_quad_from_contours(mask, corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        return _auto_detect_document_corners_hough(corrected)

    else:
        # Düşük parlaklık için önce soft mod, sonra agresif ve en son maskeler/fallback
        pts = _auto_detect_document_corners_soft(corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        pts = _auto_detect_document_corners_aggressive(corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        mask = _mask_background_lab_range(corrected)
        pts = _find_quad_from_contours(mask, corrected)
        if not np.allclose(pts, _full_image_quad(corrected), atol=1):
            return pts

        return _auto_detect_document_corners_hough(corrected)


# Ana Component
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
