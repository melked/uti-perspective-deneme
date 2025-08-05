import os
import sys
import cv2
import numpy as np
from itertools import combinations

# Workaround for __file__ in Colab
try:
    _file_path = __file__
except NameError:
    _file_path = os.path.join(os.getcwd(), "placeholder_file.py")


sys.path.append(os.path.join(os.path.dirname(_file_path), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# Helper functions (copied and potentially modified from previous turns)
def order_points(pts):
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]  # Sol üst
    rect[2] = pts[np.argmax(s)]  # Sağ alt
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sağ üst
    rect[3] = pts[np.argmax(diff)]  # Sol alt
    return rect

def get_intersections(img, lines):
    height, width = img.shape[:2]
    intersections = []
    for i, line1 in enumerate(lines):
        x1, y1, x2, y2 = line1
        for j, line2 in enumerate(lines):
            if j <= i:
                continue
            x3, y3, x4, y4 = line2
            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if denom == 0:
                continue
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            if -width*0.5 <= px <= width*1.5 and -height*0.5 <= py <= height*1.5: # Kesişim noktalarını biraz geniş al
                intersections.append([int(px), int(py)])
    return np.array(intersections, dtype=np.float32)

def reorder_corners(corners):
    """Köşeleri: [üst sol, üst sağ, alt sağ, alt sol] olarak sırala"""
    if corners is None or len(corners) != 4:
        return None
    new_corners = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    new_corners[0] = corners[np.argmin(s)]      # üst sol
    new_corners[2] = corners[np.argmax(s)]      # alt sağ
    new_corners[1] = corners[np.argmin(diff)]   # üst sağ
    new_corners[3] = corners[np.argmax(diff)]   # sol alt
    return new_corners

def filter_perpendicular(lines, margin=np.pi/18):
    """İki çizginin açısı yaklaşık 90 derece ise tut"""
    perpendicular = np.pi / 2
    filtered = []
    for i, line1 in enumerate(lines):
        rho1, theta1 = line1
        count = 0
        for j, line2 in enumerate(lines):
            if i == j:
                continue
            rho2, theta2 = line2
            angle_diff = abs(theta1 - theta2)
            angle_diff = min(angle_diff, np.pi - angle_diff)
            if abs(angle_diff - perpendicular) < margin:
                count += 1
        if count >= 1:  # En az 1 çizgi ile dik açı yapıyor
            filtered.append(line1)
    return np.array(filtered)

def eliminate_duplicates(img, lines, threshold_distance=0.15, threshold_angle=np.pi/36):
    """Çizgilerden çok yakın ve paralel olanları eler"""
    eliminated = np.zeros(len(lines), dtype=bool)
    max_dim = max(img.shape[:2])
    for i, j in combinations(range(len(lines)), 2):
        if eliminated[i] or eliminated[j]:
            continue
        rho1, theta1 = lines[i]
        rho2, theta2 = lines[j]
        dist = abs(rho1 - rho2)
        angle_diff = abs(theta1 - theta2)
        angle_diff = min(angle_diff, np.pi - angle_diff)
        if dist < max_dim * threshold_distance and angle_diff < threshold_angle:
            eliminated[j] = True  # Daha yüksek indeksli çizgiyi eler
    return lines[~eliminated]

def to_cartesian(img, lines):
    """Polar koordinattan Kartezyen çizgi noktalarına çevir"""
    h, w = img.shape[:2]
    length = max(h, w)
    cartesian = []
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + length * (-b)), int(y0 + length * a)
        x2, y2 = int(x0 - length * (-b)), int(y0 - length * a)
        cartesian.append((x1, y1, x2, y2))
    return cartesian

def select_outermost_corners(points, k=4):
    """
    Verilen noktalardan en dıştaki k (varsayılan 4) köşeyi seçer.
    K-Means yerine basit geometrik yaklaşımla en uzak noktaları bulur.
    """
    if points is None or len(points) < k:
        return None

    # Noktaların merkezini bul
    centroid = np.mean(points, axis=0)

    # Merkezden her noktanın uzaklığını hesapla
    distances = np.linalg.norm(points - centroid, axis=1)

    # En uzak k noktayı seç
    # Argumenleri azalan sırada sırala ve ilk k indeksi al
    outermost_indices = np.argsort(distances)[-k:]

    return points[outermost_indices].astype(np.float32)


def find_document_contours(img_gray, area_threshold_ratio=0.05):
    contours, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    document_contours = []
    img_area = img_gray.shape[0] * img_gray.shape[1]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > img_area * area_threshold_ratio:
            document_contours.append(approx)
    return document_contours

def get_corners_from_contours(contours):
    """Konturlardan köşe noktalarını çıkarır."""
    corners = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        corners.extend(points)
    if corners:
        return np.array(corners, dtype=np.float32)
    return np.array([], dtype=np.float32)

def detect_corners_shi_tomasi(img_gray, maxCorners=100, qualityLevel=0.01, minDistance=10):
    """Shi-Tomasi köşe tespit yöntemi"""
    corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    if corners is not None:
        return np.float32(corners).reshape(-1, 2)
    return np.array([], dtype=np.float32)

def detect_corners_harris(img_gray, blockSize=2, ksize=3, k=0.04, threshold=0.01):
    """Harris köşe tespit yöntemi"""
    dst = cv2.cornerHarris(img_gray, blockSize, ksize, k)
    corners = np.argwhere(dst > threshold * dst.max())
    return np.float32(corners).reshape(-1, 2)

def find_lines_probabilistic_hough(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10):
    """Olasılıksal Hough Dönüşümü ile çizgi segmentlerini bulur."""
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        return lines.reshape(-1, 4)
    return np.array([], dtype=np.int32)

def get_intersections_from_segments(segments, img_shape, img=None):
    """Çizgi segmentlerinin kesişim noktalarını hesaplar."""
    height, width = img_shape[:2]
    intersections = []
    extended_lines = []
    if segments is None or len(segments) == 0:
        return np.array([], dtype=np.float32)

    for x1, y1, x2, y2 in segments:
        if x2 - x1 == 0: # Vertical line
            extended_lines.append((x1, 0, x1, height))
        elif y2 - y1 == 0: # Horizontal line
            extended_lines.append((0, y1, width, y1))
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            y_at_x0 = int(b)
            y_at_xw = int(m * width + b)
            x_at_y0 = int(-b / m) if m != 0 else -1
            x_at_yh = int((height - b) / m) if m != 0 else -1

            points_on_boundary = []
            # Check if points are within a reasonable range of the image boundaries
            if -height*0.5 <= y_at_x0 <= height*1.5: points_on_boundary.append((0, y_at_x0))
            if -height*0.5 <= y_at_xw <= height*1.5: points_on_boundary.append((width, y_at_xw))
            if -width*0.5 <= x_at_y0 <= width*1.5 and m != 0: points_on_boundary.append((x_at_y0, 0))
            if -width*0.5 <= x_at_yh <= width*1.5 and m != 0: points_on_boundary.append((x_at_yh, height))


            if len(points_on_boundary) >= 2:
                 p1, p2 = points_on_boundary[0], points_on_boundary[-1]
                 extended_lines.append((p1[0], p1[1], p2[0], p2[1]))
            elif len(points_on_boundary) == 1:
                 # Extend the single point towards the other end of the segment
                 dx = x2 - x1
                 dy = y2 - y1
                 # Determine which end of the segment is closer to the boundary point
                 dist1 = np.linalg.norm(np.array([x1, y1]) - np.array(points_on_boundary[0]))
                 dist2 = np.linalg.norm(np.array([x2, y2]) - np.array(points_on_boundary[0]))

                 if dist1 < dist2:
                     # Extend from (x1, y1) through the boundary point
                     extended_lines.append((x1, y1, points_on_boundary[0][0], points_on_boundary[0][1]))
                 else:
                     # Extend from (x2, y2) through the boundary point
                     extended_lines.append((x2, y2, points_on_boundary[0][0], points_on_boundary[0][1]))

            elif len(points_on_boundary) < 2 and len(segments) > 1:
                # If less than 2 boundary points, extend the segment significantly
                dx = x2 - x1
                dy = y2 - y1
                length = max(width, height) * 2 # Extend by a large factor
                p1_ext = (int(x1 - dx * length), int(y1 - dy * length))
                p2_ext = (int(x2 + dx * length), int(y2 + dy * length))
                extended_lines.append((p1_ext[0], p1_ext[1], p2_ext[0], p2_ext[1]))


    if extended_lines:
         # Filter extended lines to be nearly horizontal or vertical
         filtered_extended_lines = []
         for x1, y1, x2, y2 in extended_lines:
             angle = np.arctan2(y2 - y1, x2 - x1) # Angle in radians
             angle = np.abs(angle) # Take absolute value
             # Normalize angle to be between 0 and pi/2
             angle = min(angle, np.pi - angle)
             # Check if angle is close to 0 (horizontal) or pi/2 (vertical)
             if abs(angle) < np.pi/18 or abs(angle - np.pi/2) < np.pi/18:
                  filtered_extended_lines.append((x1, y1, x2, y2))


         if filtered_extended_lines:
             intersections = get_intersections(img, filtered_extended_lines)
             return intersections

    return np.array([], dtype=np.float32) # Return empty if no valid intersections found

def select_best_corners(points, img_shape):
    """
    Selects the best 4 corners from a set of points, prioritizing points near image corners
    and forming a convex quadrilateral. Improved selection based on distance from image corners.
    Adds a check for aspect ratio and area.
    """
    if points is None or len(points) < 4:
        # print("Not enough points to select 4 corners.") # Keep print for debugging if needed
        return None

    h, w = img_shape[:2]
    image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Try to find 4 points close to image corners first
    closest_to_image_corners = []
    # Use a threshold for distance to image corners
    corner_distance_threshold = min(h, w) * 0.1 # Example: 10% of the minimum dimension
    potential_corners_indices = set()

    for img_corner in image_corners:
        distances = np.linalg.norm(points - img_corner, axis=1)
        closest_idx = np.argmin(distances)
        # Check if the closest point is within the threshold distance to the image corner
        if distances[closest_idx] < corner_distance_threshold:
             # Add index to avoid duplicates if multiple image corners are close to the same detected point
             potential_corners_indices.add(closest_idx)


    potential_corners = points[list(potential_corners_indices)]

    # If we found exactly 4 points close to image corners, check if they form a reasonable quad
    if len(potential_corners) == 4:
        # Check convexity, area and aspect ratio
        reordered_potential = reorder_corners(potential_corners)
        if reordered_potential is not None:
            area = cv2.contourArea(reordered_potential)
            img_area = h * w
            area_ratio = area / img_area if img_area > 0 else 0

            side1 = np.linalg.norm(reordered_potential[0] - reordered_potential[1])
            side2 = np.linalg.norm(reordered_potential[1] - reordered_potential[2])
            aspect_ratio = max(side1, side2) / min(side1, side2) if min(side1, side2) > 0 else float('inf')

            min_area_ratio = 0.01
            max_aspect_ratio = 10.0

            v1 = reordered_potential[1] - reordered_potential[0]
            v2 = reordered_potential[2] - reordered_potential[1]
            v3 = reordered_potential[3] - reordered_potential[2]
            v4 = reordered_potential[0] - reordered_potential[3]

            cross_products = [
                np.cross(v1, v2),
                np.cross(v2, v3),
                np.cross(v3, v4),
                np.cross(v4, v1)
            ]

            signs = np.sign(cross_products)
            is_convex = np.all(signs >= 0) or np.all(signs <= 0)


            if is_convex and area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio:
                 print("Selected corners are convex, have sufficient area, and reasonable aspect ratio.")
                 return reordered_potential # Found good corners


    # Fallback: If initial corner-based selection fails or didn't find 4 points, try finding the outermost 4 points
    print("Initial corner selection failed or produced poor results. Falling back to outermost points.")
    fallback_corners = select_outermost_corners(points, k=4)

    if fallback_corners is not None and len(fallback_corners) == 4:
        reordered_fallback = reorder_corners(fallback_corners)
        if reordered_fallback is not None:
            area = cv2.contourArea(reordered_fallback)
            img_area = h * w
            area_ratio = area / img_area if img_area > 0 else 0

            side1 = np.linalg.norm(reordered_fallback[0] - reordered_fallback[1])
            side2 = np.linalg.norm(reordered_fallback[1] - reordered_fallback[2])
            aspect_ratio = max(side1, side2) / min(side1, side2) if min(side1, side2) > 0 else float('inf')

            min_area_ratio = 0.01
            max_aspect_ratio = 10.0

            # Re-check convexity and other properties for fallback corners
            v1 = reordered_fallback[1] - reordered_fallback[0]
            v2 = reordered_fallback[2] - reordered_fallback[1]
            v3 = reordered_fallback[3] - reordered_fallback[2]
            v4 = reordered_fallback[0] - reordered_fallback[3]

            cross_products = [
                np.cross(v1, v2),
                np.cross(v2, v3),
                np.cross(v3, v4),
                np.cross(v4, v1)
            ]

            signs = np.sign(cross_products)
            is_convex = np.all(signs >= 0) or np.all(signs <= 0)


            if is_convex and area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio:
                print("Fallback outermost corners are convex, have sufficient area, and reasonable aspect ratio.")
                return reordered_fallback # Use fallback if it looks reasonable
            else:
                 print("Fallback outermost corners do not meet criteria.")
                 return None # Fallback also failed


    # If both methods fail, return None
    print("Both initial and fallback corner selection methods failed.")
    return None


class PerspectiveTransformation(Component):
    """
    Auto perspective correction executor.
    Detects a document-like quadrilateral and warps it to a target size.
    Includes detailed preprocessing and detection options.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {} # Initialize context

        # self.request.data'nın bir sözlük olduğundan emin olun ve PackageModel'i güvenle başlatın
        if not isinstance(self.request.data, dict):
            # print(f"UYARI: self.request.data bir sözlük değil, tipi: {type(self.request.data)}. Boş bir sözlük kullanılıyor.")
            model_data = {}
        else:
            model_data = self.request.data

        try:
            self.request.model = PackageModel(**model_data)
        except TypeError as e:
            print(f"HATA: PackageModel başlatılırken TypeError oluştu: {e}")
            # print(f"self.request.data içeriği: {model_data}")
            raise RuntimeError("PackageModel başlatılamadı, lütfen request.data'yı kontrol edin.") from e
        except Exception as e:
            print(f"HATA: PackageModel başlatılırken beklenmeyen bir hata oluştu: {e}")
            # print(f"self.request.data içeriği: {model_data}")
            raise RuntimeError("PackageModel başlatılamadı.") from e

        # inputImage parametresini güvenle alın
        try:
            self.image = self.request.get_param("inputImage")
            if self.image is None:
                print("UYARI: 'inputImage' parametresi bulunamadı veya değeri None.")
        except AttributeError:
            print("HATA: 'request' nesnesinin 'get_param' metodu yok.")
            raise RuntimeError("Request nesnesi geçersiz, 'get_param' metodu eksik.")
        except Exception as e:
            print(f"HATA: 'inputImage' parametresi alınırken beklenmeyen bir hata oluştu: {e}")
            raise RuntimeError("'inputImage' parametresi alınamadı.") from e

        # PackageModel'den keep_side, output_width, output_height ve perspective_mode değerlerini alın
        # Assuming the structure of PackageModel based on the user's provided definition snippet
        configs = getattr(getattr(getattr(getattr(self.request.model, 'configs', None), 'executor', None), 'value', None), 'configs', None)

        self.keep_side = getattr(getattr(configs, 'drawBBox', None), 'value', False) if configs else False
        self.output_width = getattr(getattr(configs, 'outputWidth', None), 'value', 800) if configs else 800
        self.output_height = getattr(getattr(configs, 'outputHeight', None), 'value', 600) if configs else 600

        perspective_mode_obj = getattr(getattr(configs, 'PerspectiveTypeMode', None), 'value', None)
        self.perspective_mode = getattr(perspective_mode_obj, 'name', 'Auto') if perspective_mode_obj else 'Auto'

        self.warp_image_flag = True # Assuming warp_image is always desired


    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """Canny öncesi: dtype, channel ve değer aralığını düzelt."""
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    # ----- ÖN İŞLEME FONKSİYONLARI (Sınıf Metodları Olarak) -----

    def preprocess_image(self, img, method='default', deblur=False, remove_background=False, remove_shadows_flag=False):
        """Applies selected preprocessing steps."""
        if img is None or img.size == 0: return None

        preprocessed = img.copy()

        if method == 'aggressive':
            preprocessed = self._preprocess_image_aggressive_internal(preprocessed)
        elif method == 'small':
            preprocessed = self._preprocess_image_small_internal(preprocessed)
        elif method == 'advanced':
            preprocessed = self._preprocess_image_advanced_internal(preprocessed)
        else: # 'default'
            preprocessed = self._preprocess_image_default_internal(preprocessed)

        if preprocessed is None:
             raise RuntimeError(f"Preprocessing method '{method}' failed.")

        if deblur:
            preprocessed = self.sharpen_image(preprocessed)
            if preprocessed is None: raise RuntimeError("Deblurring failed.")

        if remove_background:
            preprocessed = self.remove_background_grabcut(preprocessed)
            if preprocessed is None: raise RuntimeError("Background removal failed.")

        if remove_shadows_flag:
            preprocessed = self.remove_shadows(preprocessed)
            if preprocessed is None: raise RuntimeError("Shadow removal failed.")

        return preprocessed

    def _preprocess_image_default_internal(self, img):
        """Orta seviye CLAHE ile kontrast artırma (normal)"""
        if img is None or img.size == 0: return None
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab_clahe = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return img

    def _preprocess_image_aggressive_internal(self, img):
        """Agresif CLAHE, histogram eşitleme ile güçlü kontrast artırma"""
        if img is None or img.size == 0: return None
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab_clahe = cv2.merge([l, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        return img_eq

    def _preprocess_image_small_internal(self, img):
        """Küçük resimler için daha hafif CLAHE ve keskinleştirme"""
        if img is None or img.size == 0: return None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab_clahe = cv2.merge([l, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img_sharp = cv2.filter2D(img_clahe, -1, kernel)
        return img_sharp

    def _preprocess_image_advanced_internal(self, img):
        """Daha gelişmiş ön işleme: CLAHE + Median Blur + Unsharp Mask"""
        if img is None or img.size == 0: return None
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10,10))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        img_blurred = cv2.medianBlur(img_clahe, 5)
        gaussian = cv2.GaussianBlur(img_blurred, (9, 9), 0)
        unsharp_mask = cv2.addWeighted(img_blurred, 1.5, gaussian, -0.5, 0)
        return unsharp_mask

    def sharpen_image(self, img):
        """Bulanıklık için keskinleştirme filtresi"""
        if img is None or img.size == 0: return None
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(img, -1, kernel)
        return sharp

    def remove_background_grabcut(self, img):
        """GrabCut ile arka planı kaldırma (başlangıç maskesi gerekebilir)"""
        if img is None or img.size == 0: return None
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, img.shape[1]-1, img.shape[0]-1)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        return img

    def remove_shadows(self, img):
        """Gölge kaldırma (basit)"""
        if img is None or img.size == 0: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        return result

    def _find_document_corners(self, img, detection_method, params):
        """
        Applies a specific detection method to find document corners.
        Returns the corners if found, otherwise None.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = gray
        if params.get('blur_method') == 'median' and params.get('median_blur_size') > 1 and params.get('median_blur_size') % 2 == 1:
             blurred = cv2.medianBlur(gray, params.get('median_blur_size'))
        elif params.get('blur_method') == 'bilateral':
             blurred = cv2.bilateralFilter(gray, 9, 75, 75)


        use_adaptive_thresholding = params.get('use_adaptive_thresholding', False)
        corners = None

        if detection_method == 'hough':
            edges = None
            if use_adaptive_thresholding:
                 block_size = params.get('adaptive_block_size', 11)
                 block_size = block_size if block_size % 2 == 1 and block_size > 1 else 11
                 edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, params.get('adaptive_c_value', 2))
            else:
                edge_detector = params.get('edge_detector', 'canny')
                if edge_detector == 'canny':
                     edges = cv2.Canny(blurred, params.get('canny_threshold_min', 30), params.get('canny_threshold_max', 140))
                elif edge_detector == 'sobel':
                     sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                     sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                     edges = cv2.magnitude(sobelx, sobely)
                     max_edge_val = edges.max()
                     if max_edge_val > 0: edges = np.uint8(edges * 255 / max_edge_val)
                     else: edges = np.zeros_like(edges, dtype=np.uint8)
                     _, edges = cv2.threshold(edges, params.get('sobel_threshold_min', 5), params.get('sobel_threshold_max', 255), cv2.THRESH_BINARY)
                else: # Simple thresholding
                     _, edges = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if edges is not None and np.sum(edges) > 0:
                lines = cv2.HoughLines(edges, params.get('rho', 1), params.get('theta', np.pi/180), params.get('threshold_intersect', 250))
                if lines is not None:
                    lines = filter_perpendicular(lines[:,0], params.get('perpendicular_margin', np.pi/18))
                    lines = eliminate_duplicates(img, lines, params.get('threshold_distance', 0.15))
                    if len(lines) >= 2:
                        cartesian = to_cartesian(img, lines)
                        intersections = get_intersections(img, cartesian)
                        if intersections is not None and len(intersections) >= 4:
                            corners = select_best_corners(intersections, img.shape)

        elif detection_method == 'contour':
            thresh = None
            if use_adaptive_thresholding:
                 block_size = params.get('adaptive_block_size', 11)
                 block_size = block_size if block_size % 2 == 1 and block_size > 1 else 11
                 thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, params.get('adaptive_c_value', 2))
            else:
                 _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if thresh is not None and np.sum(thresh) > 0:
                document_contours = find_document_contours(thresh, area_threshold_ratio=params.get('contour_area_threshold_ratio', 0.05))
                if document_contours:
                    potential_corners = get_corners_from_contours([document_contours[0]])
                    if potential_corners is not None and len(potential_corners) >= 4:
                        corners = select_best_corners(potential_corners, img.shape)

        elif detection_method == 'shi_tomasi':
            potential_corners = detect_corners_shi_tomasi(
                blurred,
                maxCorners=params.get('shi_tomasi_maxCorners', 100),
                qualityLevel=params.get('shi_tomasi_qualityLevel', 0.01),
                minDistance=params.get('shi_tomasi_minDistance', 10)
            )
            if potential_corners is not None and len(potential_corners) >= 4:
                 corners = select_best_corners(potential_corners, img.shape)

        elif detection_method == 'harris':
             potential_corners = detect_corners_harris(
                 blurred,
                 blockSize=params.get('harris_blockSize', 2),
                 ksize=params.get('harris_ksize', 3),
                 k=params.get('harris_k', 0.04),
                 threshold=params.get('harris_threshold', 0.01)
             )
             if potential_corners is not None and len(potential_corners) >= 4:
                 corners = select_best_corners(potential_corners, img.shape)

        elif detection_method == 'hough_lines_p':
            edges_for_hough_p = None
            if use_adaptive_thresholding:
                 block_size = params.get('adaptive_block_size', 11)
                 block_size = block_size if block_size % 2 == 1 and block_size > 1 else 11
                 edges_for_hough_p = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, params.get('adaptive_c_value', 2))
            else:
                edge_detector = params.get('edge_detector', 'canny')
                if edge_detector == 'canny':
                     edges_for_hough_p = cv2.Canny(blurred, params.get('canny_threshold_min', 30), params.get('canny_threshold_max', 140))
                elif edge_detector == 'sobel':
                     sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                     sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                     edges_for_hough_p = cv2.magnitude(sobelx, sobely)
                     max_edge_val = edges_for_hough_p.max()
                     if max_edge_val > 0: edges_for_hough_p = np.uint8(edges_for_hough_p * 255 / max_edge_val)
                     else: edges_for_hough_p = np.zeros_like(edges_for_hough_p, dtype=np.uint8)
                     _, edges_for_hough_p = cv2.threshold(edges_for_hough_p, params.get('sobel_threshold_min', 5), params.get('sobel_threshold_max', 255), cv2.THRESH_BINARY)
                else:
                     _, edges_for_hough_p = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if edges_for_hough_p is not None and np.sum(edges_for_hough_p) > 0:
                lines_p = find_lines_probabilistic_hough(
                    edges_for_hough_p,
                    rho=params.get('rho', 1),
                    theta=params.get('theta', np.pi/180),
                    threshold=params.get('hough_p_threshold', 50),
                    minLineLength=params.get('hough_p_minLineLength', 50),
                    maxLineGap=params.get('hough_p_maxLineGap', 10)
                )
                if lines_p is not None and len(lines_p) > 0:
                     intersections_p = get_intersections_from_segments(lines_p, img.shape)
                     if intersections_p is not None and len(intersections_p) >= 4:
                         corners = select_best_corners(intersections_p, img.shape)


        if corners is not None and len(corners) == 4:
             return corners # Return corners if found
        else:
             return None # Return None if 4 corners not found

    def _apply_perspective_transform(self, img, corners):
        """Applies the perspective transform to the image using the found corners."""
        if corners is None or len(corners) != 4:
            raise ValueError("Invalid corners provided for perspective transform.")

        h_img, w_img = img.shape[:2]

        # Determine output size
        if not self.keep_side:
            new_w, new_h = self.output_width, self.output_height
        else:
            # Original KeepSide logic based on A4 ratio
            min_dim = min(h_img, w_img)
            if h_img > w_img:
                new_h, new_w = int(min_dim / 0.707), int(min_dim)
                if new_h > h_img * 1.5: new_h = int(h_img * 1.5)
                if new_w > w_img * 1.5: new_w = int(w_img * 1.5)
            else:
                new_h, new_w = int(min_dim), int(min_dim / 0.707)
                if new_h > h_img * 1.5: new_h = int(h_img * 1.5)
                if new_w > w_img * 1.5: new_w = int(w_img * 1.5)

        output_size = (new_w, new_h)
        destination = np.array([[0,0], [new_w,0], [new_w,new_h], [0,new_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, destination)
        corrected = cv2.warpPerspective(img, M, output_size)

        return corrected, output_size


    def _process_image_with_params(self, img, params):
        """Applies preprocessing and attempts to find corners with given parameters."""
        preprocessed_img = self.preprocess_image(
            img,
            method=params.get('preprocess_method', 'default'),
            deblur=params.get('deblur', False),
            remove_background=params.get('remove_background', False),
            remove_shadows_flag=params.get('remove_shadows_flag', False)
        )
        if preprocessed_img is None:
            # Log a warning or handle the preprocessing failure more gracefully
            print("Warning: Preprocessing failed for a parameter set.")
            return None

        corners = self._find_document_corners(preprocessed_img, params.get('detection_method'), params)

        # Add a check for the quality of detected corners
        if corners is not None and len(corners) == 4:
            reordered_corners = reorder_corners(corners)
            if reordered_corners is not None:
                area = cv2.contourArea(reordered_corners)
                img_area = img.shape[0] * img.shape[1]
                area_ratio = area / img_area if img_area > 0 else 0

                side1 = np.linalg.norm(reordered_corners[0] - reordered_corners[1])
                side2 = np.linalg.norm(reordered_corners[1] - reordered_corners[2])
                aspect_ratio = max(side1, side2) / min(side1, side2) if min(side1, side2) > 0 else float('inf')

                min_area_ratio = 0.01 # Minimum acceptable area ratio
                max_aspect_ratio = 10.0 # Maximum acceptable aspect ratio

                # Check convexity - this is already done in select_best_corners, but can be re-verified
                v1 = reordered_corners[1] - reordered_corners[0]
                v2 = reordered_corners[2] - reordered_corners[1]
                v3 = reordered_corners[3] - reordered_corners[2]
                v4 = reordered_corners[0] - reordered_corners[3]
                cross_products = [np.cross(v1, v2), np.cross(v2, v3), np.cross(v3, v4), np.cross(v4, v1)]
                signs = np.sign(cross_products)
                is_convex = np.all(signs >= 0) or np.all(signs <= 0)


                if is_convex and area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio:
                    return reordered_corners # Return valid corners
                else:
                    # print("Detected corners did not pass quality checks (convexity, area, aspect ratio).")
                    return None
            else:
                 # print("Reordering detected corners failed.")
                 return None
        else:
            return None # Return None if 4 corners not found initially


    def _apply_perspective_auto(self, src_img: np.ndarray):
        """
        Görüntüye gelişmiş perspektif düzeltme uygular.
        Farklı parametre kombinasyonlarını dener ve başarılı olan ilkini döndürür.
        Orijinal ve negatif görüntü üzerinde denemeler yapar.
        Prioritizes parameter sets for faster detection.
        """
        if src_img is None or src_img.size == 0:
            raise ValueError("Input image is empty or None in _apply_perspective_auto.")

        # Define prioritized parameter sets to try
        # Start with commonly effective combinations
        parameter_sets = [
            # Priority 1: Basic attempts with default preprocessing
            {'preprocess_method': 'default', 'detection_method': 'hough', 'edge_detector': 'canny'},
            {'preprocess_method': 'default', 'detection_method': 'contour', 'use_adaptive_thresholding': True},
            {'preprocess_method': 'default', 'detection_method': 'hough_lines_p', 'edge_detector': 'canny'},

            # Priority 2: Basic attempts with aggressive preprocessing
            {'preprocess_method': 'aggressive', 'detection_method': 'hough', 'edge_detector': 'canny'},
            {'preprocess_method': 'aggressive', 'detection_method': 'contour', 'use_adaptive_thresholding': True},
            {'preprocess_method': 'aggressive', 'detection_method': 'hough_lines_p', 'edge_detector': 'canny'},


            # Priority 3: Add some common variations (blur, different edge detectors)
            {'preprocess_method': 'default', 'detection_method': 'hough', 'edge_detector': 'sobel', 'blur_method': 'median', 'median_blur_size': 9},
            {'preprocess_method': 'default', 'detection_method': 'contour', 'use_adaptive_thresholding': False, 'blur_method': 'bilateral'},
            {'preprocess_method': 'default', 'detection_method': 'shi_tomasi'},
            {'preprocess_method': 'default', 'detection_method': 'harris'},


            # Add more specific combinations here if needed, ordered by likelihood of success
            {'preprocess_method': 'advanced', 'detection_method': 'hough', 'edge_detector': 'canny', 'blur_method': 'median', 'median_blur_size': 5},
            {'preprocess_method': 'small', 'detection_method': 'contour', 'use_adaptive_thresholding': True, 'adaptive_block_size': 15},

        ]

        # Try with original image
        print("Trying perspective correction with original image...")
        for i, params in enumerate(parameter_sets):
            print(f"  Attempt {i+1}/{len(parameter_sets)} (Original) with params: {params}")
            try:
                corners = self._process_image_with_params(src_img, params)
                if corners is not None:
                    print("  ✅ Corners found!")
                    warped, output_size = self._apply_perspective_transform(src_img, corners)
                    print("  ✅ Perspective transform applied successfully!")
                    return warped, [], corners, output_size # Return empty corrected_boxes
                else:
                    print("  ❌ Corners not found or failed quality checks.")
            except Exception as e:
                print(f"  ⛔ Attempt {i+1}/{len(parameter_sets)} (Original) failed: {e}")
                # Log the error but continue trying other parameters
                continue

        # If original image failed, try with negative image
        print("Trying perspective correction with negative image...")
        img_neg = cv2.bitwise_not(src_img)
        for i, params in enumerate(parameter_sets):
             print(f"  Attempt {i+1}/{len(parameter_sets)} (Negative) with params: {params}")
             try:
                corners = self._process_image_with_params(img_neg, params)
                if corners is not None:
                    print("  ✅ Corners found!")
                    warped, output_size = self._apply_perspective_transform(src_img, corners) # Apply transform to original image with corners found on negative
                    print("  ✅ Perspective transform applied successfully!")
                    return warped, [], corners, output_size # Return empty corrected_boxes
                else:
                    print("  ❌ Corners not found or failed quality checks.")
             except Exception as e:
                print(f"  ⛔ Attempt {i+1}/{len(parameter_sets)} (Negative) failed: {e}")
                # Log the error but continue trying other parameters
                continue


        # If both attempts failed, raise a more informative error
        raise RuntimeError("Perspektif düzeltme hem orijinal hem de negatif görüntüde tüm denemelerde başarısız oldu.")


    def run(self):
        # self.image, __init__ içinde zaten ayarlanmıştır
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img is None or img.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img.value)

        # Apply perspective correction using the auto method
        warped, corrected_boxes, src_quad, (out_w, out_h) = self._apply_perspective_auto(src_img)


        img.value = warped
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        self.context = {
            "src_quad": src_quad.tolist(),
            "output_size": [out_w, out_h],
            "corrected_boxes": corrected_boxes,
            "keep_side": self.keep_side,
            "warp_image": self.warp_image_flag,
        }
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()