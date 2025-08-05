import os
import sys
import cv2
import numpy as np
from typing import Optional, List

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


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.015, approx_epsilon_factor=0.01, angle_range=(50, 130)) -> Optional[np.ndarray]:
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
        approx = cv2.approxPolyDP(c, approx_epsilon_factor * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            angles = []
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i+1) % 4]
                p0 = pts[(i-1) % 4]
                v1 = p1 - p0
                v2 = p2 - p1
                dot_product = np.dot(v1, v2)
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm_product == 0:
                    continue
                cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
                angle = abs(np.degrees(np.arccos(cosine_angle)))
                angles.append(angle)
            if all(angle_range[0] < a < angle_range[1] for a in angles):
                return pts.astype(np.float32)
    return None


def _hough_quad_from_lines(lines: List[np.ndarray], image_shape) -> Optional[np.ndarray]:
    if not lines or len(lines) < 4:
        return None

    # Find intersection points (simplified approach)
    points = []
    lines = lines.reshape(-1, 4)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            xdiff = (line1[0] - line1[2], line2[0] - line2[2])
            ydiff = (line1[1] - line1[3], line2[1] - line2[3])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
               continue

            d = (det(*[line1[:2], line1[2:]]), det(*[line2[:2], line2[2:]]))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div

            # Basic check if the intersection is within image bounds
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                 points.append([x, y])

    if len(points) < 4:
        return None

    points = np.array(points, dtype=np.float32)

    # Try to find the four extreme points as corners (simple approach)
    # This is a simplification; a more robust method would cluster and select
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Find points closest to the corners of the bounding box
    tl = points[np.argmin(np.sum(points - [min_x, min_y], axis=1))]
    tr = points[np.argmin(np.sum(points - [max_x, min_y], axis=1))]
    br = points[np.argmin(np.sum(points - [max_x, max_y], axis=1))]
    bl = points[np.argmin(np.sum(points - [min_x, max_y], axis=1))]

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    # Validate quad (check angles, area) - simplified
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect

    width1 = np.linalg.norm(br - bl)
    width2 = np.linalg.norm(tr - tl)
    height1 = np.linalg.norm(tr - br)
    height2 = np.linalg.norm(tl - bl)

    if min(width1, width2, height1, height2) < 20: # Minimum size check
        return None

    return quad


def _unsharp_mask(image: np.ndarray, ksize=(5, 5), strength=1.5) -> np.ndarray:
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
# Ön İşleme Kombinasyonları
# ----------------------------

def _preprocess_combinations(image: np.ndarray):
    """Birden fazla agresif ve soft ön işleme kombinasyonu döner"""

    combinations = []

    # Orijinal
    combinations.append(("Original", image))

    # Gamma + Sharpening
    gamma_img_soft = _gamma_correction(image, 1.5)
    combinations.append(("Gamma_Soft", gamma_img_soft))
    combinations.append(("Gamma_Soft_Sharpened", _unsharp_mask(gamma_img_soft, ksize=(3,3), strength=1.0)))

    gamma_img_aggressive = _gamma_correction(image, 2.2)
    combinations.append(("Gamma_Aggressive", gamma_img_aggressive))
    combinations.append(("Gamma_Aggressive_Sharpened", _unsharp_mask(gamma_img_aggressive, ksize=(5,5), strength=2.0)))


    # CLAHE + Sharpening
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    combinations.append(("CLAHE", clahe_bgr))
    combinations.append(("CLAHE_Sharpened", _unsharp_mask(clahe_bgr, ksize=(3,3), strength=1.0)))

    # Blurring + Adaptive Threshold (requires grayscale)
    blur_soft_gray = cv2.cvtColor(cv2.GaussianBlur(image, (3,3), 0), cv2.COLOR_BGR2GRAY)
    combinations.append(("Blur_Soft_Gray", blur_soft_gray))

    blur_aggressive_gray = cv2.cvtColor(cv2.GaussianBlur(image, (7,7), 0), cv2.COLOR_BGR2GRAY)
    combinations.append(("Blur_Aggressive_Gray", blur_aggressive_gray))

    # Auto Gamma Corrected
    auto_gamma_img = _auto_gamma_correction(image)
    combinations.append(("Auto_Gamma", auto_gamma_img))
    combinations.append(("Auto_Gamma_Sharpened", _unsharp_mask(auto_gamma_img, ksize=(3,3), strength=1.0)))


    return combinations


# ----------------------------
# Köşe Tespiti Yöntemleri
# ----------------------------

def _detect_corners_canny_contour(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    morph = _morph_ops(edges, close_iter=2, open_iter=1)
    return _find_quad_from_contours(morph, image)

def _detect_corners_adaptive_thresh_contour(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # sharpened = _unsharp_mask(blur) # Adaptive threshold works better on slightly smoothed images
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    morph = _morph_ops(thresh, close_iter=1, open_iter=1)
    return _find_quad_from_contours(morph, image)

def _detect_corners_hough(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None
    return _hough_quad_from_lines(lines, image.shape)


# ----------------------------
# Skorlama ve Seçim
# ----------------------------

def _score_quad(quad: np.ndarray, image_shape: tuple) -> float:
    if quad is None:
        return 0.0

    # Ensure the quad is ordered for consistent scoring
    quad = _order_points(quad)
    (tl, tr, br, bl) = quad

    # 1. Aspect Ratio Score (closer to 1 is better for rectangles)
    width1 = np.linalg.norm(br - bl)
    width2 = np.linalg.norm(tr - tl)
    height1 = np.linalg.norm(tr - br)
    height2 = np.linalg.norm(tl - bl)

    avg_width = (width1 + width2) / 2.0
    avg_height = (height1 + height2) / 2.0

    if avg_height == 0 or avg_width == 0: # Prevent division by zero
        return 0.0

    aspect_ratio = avg_width / avg_height
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0)) # Penalize deviation from 1

    # 2. Area Score (larger area relative to image area is better, up to a point)
    quad_area = cv2.contourArea(quad)
    image_area = image_shape[0] * image_shape[1]
    area_ratio = quad_area / image_area
    area_score = min(1.0, area_ratio * 2.0) # Score increases with area ratio, capped at 1

    # 3. Angle Score (closer to 90 degrees is better)
    angles = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i+1) % 4]
        p0 = quad[(i-1) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
             angles.append(0) # Handle degenerate case
             continue
        cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        angle = abs(np.degrees(np.arccos(cosine_angle)))
        angles.append(angle)

    angle_deviations = [abs(a - 90) for a in angles]
    avg_angle_deviation = np.mean(angle_deviations)
    angle_score = max(0.0, 1.0 - (avg_angle_deviation / 45.0)) # Penalize deviation from 90

    # Combine scores (weights can be adjusted)
    combined_score = (aspect_score * 0.3) + (area_score * 0.4) + (angle_score * 0.3)

    return combined_score


def _select_best_quad_scored(quads: List[Tuple[np.ndarray, str]], ref_image_shape: tuple) -> np.ndarray:
    scored_quads = [(quad, score, method) for quad, score, method in [(q, _score_quad(q, ref_image_shape), method) for q, method in quads] if q is not None]

    if not scored_quads:
        return _full_image_quad(np.zeros(ref_image_shape, dtype=np.uint8)) # Return full image quad with correct shape

    # Sort by score in descending order
    scored_quads.sort(key=lambda item: item[1], reverse=True)

    # Simple selection: pick the highest-scoring quad
    best_quad, best_score, best_method = scored_quads[0]

    # Optional: Add a check for a minimum acceptable score, if the best score is too low, maybe fallback?
    # For now, we just return the best found, or the full image if none found.

    return best_quad


# ----------------------------
# Dinamik Çoklu Aşamalı Algoritma
# ----------------------------

def auto_detect_document_corners_dynamic(image: np.ndarray) -> np.ndarray:
    """
    Applies multiple preprocessing combinations and detection methods
    to find the best document corners.
    """
    processed_combinations = _preprocess_combinations(image)

    all_quads = []

    for method_name, processed_img in processed_combinations:
        if processed_img.ndim == 3: # Only apply color-based masks to color images
            # Masking methods (if applicable to the processed image)
            quad_lab_red = _find_quad_from_contours(_mask_lab_red_background(processed_img), processed_img)
            if quad_lab_red is not None:
                all_quads.append((quad_lab_red, f"{method_name}_LAB_Red"))

            quad_lab_complex = _find_quad_from_contours(_mask_lab_complex(processed_img), processed_img)
            if quad_lab_complex is not None:
                all_quads.append((quad_lab_complex, f"{method_name}_LAB_Complex"))

            # Contour-based detection methods
            quad_canny = _detect_corners_canny_contour(processed_img)
            if quad_canny is not None:
                all_quads.append((quad_canny, f"{method_name}_Canny_Contour"))

            quad_adaptive_thresh = _detect_corners_adaptive_thresh_contour(processed_img)
            if quad_adaptive_thresh is not None:
                 all_quads.append((quad_adaptive_thresh, f"{method_name}_AdaptiveThresh_Contour"))

            # Hough Line method (can work on grayscale too, apply to color first)
            quad_hough = _detect_corners_hough(processed_img)
            if quad_hough is not None:
                all_quads.append((quad_hough, f"{method_name}_Hough"))

        elif processed_img.ndim == 2: # For grayscale images (like outputs of blurring, CLAHE)
             # Contour-based detection methods on grayscale
            quad_canny_gray = _find_quad_from_contours(_detect_edges_canny(processed_img), cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)) # Need BGR for _find_quad_from_contours ref_image
            if quad_canny_gray is not None:
                all_quads.append((quad_canny_gray, f"{method_name}_Canny_Contour_Gray"))

            quad_adaptive_thresh_gray = _find_quad_from_contours(cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2), cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR))
            if quad_adaptive_thresh_gray is not None:
                 all_quads.append((quad_adaptive_thresh_gray, f"{method_name}_AdaptiveThresh_Contour_Gray"))

             # Hough Line method on grayscale
            quad_hough_gray = _detect_corners_hough(cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)) # Hough expects BGR input
            if quad_hough_gray is not None:
                all_quads.append((quad_hough_gray, f"{method_name}_Hough_Gray"))


    # Select the best quad based on scoring
    best_quad = _select_best_quad_scored(all_quads, image.shape)

    # Fallback to full image quad if no valid quad was found or score was too low
    if best_quad is None or _score_quad(best_quad, image.shape) < 0.4: # Example threshold
         return _full_image_quad(image)

    return best_quad


# ----------------------------
# Maskeleme Fonksiyonları (Updated to return binary masks)
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
# Kenar Tespiti Fonksiyonları (kept for internal use if needed)
# ----------------------------

def _detect_edges_canny(image: np.ndarray, low=50, high=150) -> np.ndarray:
    # Can be applied to BGR or Grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low, high)
    return edges


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