import os
import sys
import cv2
import numpy as np
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# Helper functions
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
    new_corners = np.zeros((4, 2), dtype="float32")
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
    if segments is None:
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
            if 0 <= y_at_x0 <= height: points_on_boundary.append((0, y_at_x0))
            if 0 <= y_at_xw <= height: points_on_boundary.append((width, y_at_xw))
            if 0 <= x_at_y0 <= width and m != 0: points_on_boundary.append((x_at_y0, 0))
            if 0 <= x_at_yh <= width and m != 0: points_on_boundary.append((x_at_yh, height))

            if len(points_on_boundary) >= 2:
                 p1, p2 = points_on_boundary[0], points_on_boundary[-1]
                 extended_lines.append((p1[0], p1[1], p2[0], p2[1]))
            elif len(points_on_boundary) == 1:
                 if np.linalg.norm(np.array([x1, y1]) - np.array(points_on_boundary[0])) < np.linalg.norm(np.array([x2, y2]) - np.array(points_on_boundary[0])):
                     extended_lines.append((x1, y1, points_on_boundary[0][0], points_on_boundary[0][1]))
                 else:
                     extended_lines.append((x2, y2, points_on_boundary[0][0], points_on_boundary[0][1]))
            elif len(points_on_boundary) < 2 and len(segments) > 1:
                dx = x2 - x1
                dy = y2 - y1
                p1_ext = (int(x1 - dx * max(width, height)), int(y1 - dy * max(width, height)))
                p2_ext = (int(x2 + dx * max(width, height)), int(y2 + dy * max(width, height)))
                extended_lines.append((p1_ext[0], p1_ext[1], p2_ext[0], p2_ext[1]))

    if extended_lines:
         intersections = get_intersections(img, extended_lines)
    return intersections

def select_best_corners(points, img_shape):
    """
    Selects the best 4 corners from a set of points, prioritizing points near image corners
    and forming a convex quadrilateral. Improved selection based on distance from image corners.
    Adds a check for aspect ratio and area.
    """
    if points is None or len(points) < 4:
        print("Not enough points to select 4 corners.")
        return None

    h, w = img_shape[:2]
    image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Try to find 4 points close to image corners first
    closest_to_image_corners = []
    for img_corner in image_corners:
        distances = np.linalg.norm(points - img_corner, axis=1)
        if len(distances) > 0: # Added check for empty distances array
            closest_idx = np.argmin(distances)
            closest_to_image_corners.append(points[closest_idx])
        else:
            print(f"No points found near image corner {img_corner}.") # Added debug print
            return None # Cannot find 4 corners if no points are near corners


    potential_corners = np.array(closest_to_image_corners, dtype=np.float32)

    # Check if these 4 points form a reasonable quadrilateral
    if len(potential_corners) == 4:
        # Check convexity
        try:
            reordered_potential = reorder_corners(potential_corners)
            if reordered_potential is None:
                print("Reordering potential corners failed.")
                pass # Continue to fallback

            else:
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

                # Check area (should be a significant portion of the image area)
                area = cv2.contourArea(reordered_potential)
                img_area = h * w
                area_ratio = area / img_area if img_area > 0 else 0

                # Check aspect ratio (should be somewhat close to 1 or the expected document aspect ratio)
                side1 = np.linalg.norm(reordered_potential[0] - reordered_potential[1])
                side2 = np.linalg.norm(reordered_potential[1] - reordered_potential[2])
                # Added check for division by zero
                aspect_ratio = max(side1, side2) / min(side1, side2) if min(side1, side2) > 0 else float('inf')


                # Define thresholds
                min_area_ratio = 0.01
                max_aspect_ratio = 10.0

                if is_convex and area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio:
                     print("Selected corners are convex, have sufficient area, and reasonable aspect ratio.")
                     return reordered_potential # Found good corners


        except Exception as e:
            print(f"Error during convexity/area/aspect ratio check (initial): {e}")
            # Continue to fallback if check fails


    # Fallback: If initial corner-based selection fails, try finding the outermost 4 points
    print("Initial corner selection failed or produced poor results. Falling back to outermost points.")
    fallback_corners = select_outermost_corners(points, k=4)

    if fallback_corners is not None and len(fallback_corners) == 4:
        try:
            reordered_fallback = reorder_corners(fallback_corners)
            if reordered_fallback is None:
                print("Reordering fallback corners failed.")
                return None # Fallback also failed

            area = cv2.contourArea(reordered_fallback)
            img_area = h * w
            area_ratio = area / img_area if img_area > 0 else 0

            side1 = np.linalg.norm(reordered_fallback[0] - reordered_fallback[1])
            side2 = np.linalg.norm(reordered_fallback[1] - reordered_fallback[2])
            # Added check for division by zero
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


        except Exception as e:
             print(f"Error during convexity/area/aspect ratio check (fallback): {e}")
             return None # Fallback check failed


    # If both methods fail, return None
    print("Both initial and fallback corner selection methods failed.")
    return None


class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}

        if not isinstance(self.request.data, dict):
            model_data = {}
        else:
            model_data = self.request.data

        try:
            self.request.model = PackageModel(**model_data)
        except TypeError as e:
            print(f"HATA: PackageModel başlatılırken TypeError oluştu: {e}")
            raise RuntimeError("PackageModel başlatılamadı, lütfen request.data'yı kontrol edin.") from e
        except Exception as e:
            print(f"HATA: PackageModel başlatılırken beklenmeyen bir hata oluştu: {e}")
            raise RuntimeError("PackageModel başlatılamadı.") from e

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

        self.keep_side = getattr(getattr(getattr(getattr(self.request.model, 'configs', None), 'executor', None), 'value', None), 'configs', None)
        self.keep_side = getattr(getattr(self.keep_side, 'drawBBox', None), 'value', False) if self.keep_side else False

        self.output_width = getattr(getattr(getattr(getattr(self.request.model, 'configs', None), 'executor', None), 'value', None), 'configs', None)
        self.output_width = getattr(getattr(self.output_width, 'outputWidth', None), 'value', 800) if self.output_width else 800

        self.output_height = getattr(getattr(getattr(getattr(self.request.model, 'configs', None), 'executor', None), 'value', None), 'configs', None)
        self.output_height = getattr(getattr(self.output_height, 'outputHeight', None), 'value', 600) if self.output_height else 600

        self.perspective_mode = getattr(getattr(getattr(getattr(self.request.model, 'configs', None), 'executor', None), 'value', None), 'configs', None)
        self.perspective_mode = getattr(getattr(self.perspective_mode, 'PerspectiveTypeMode', None), 'value', None)
        self.perspective_mode = getattr(self.perspective_mode, 'name', 'Auto') if self.perspective_mode else 'Auto'

        self.warp_image_flag = True


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

    def preprocess_image(self, img):
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

    def preprocess_image_aggressive(self, img):
        """Agresif CLAHE, histogram eşitleme ile güçlü kontrast artırma"""
        if img is None or img.size == 0: return None
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab_clahe = cv2.merge([l, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        return img_eq

    def preprocess_image_small(self, img):
        """Küçük resimler için daha hafif CLAHE ve keskinleştirme"""
        if img is None or img.size == 0: return None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab_clahe = cv2.merge([l, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img_sharp = cv2.filter2D(img_clahe, -1, kernel)
        return img_sharp

    def preprocess_image_advanced(self, img):
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

    # ----- ANA PERSPEKTİF DÜZELTME METODU (Tek Deneme) -----
    def _correct_perspective_single_try(self, img,
                                        preprocess_method='default',
                                        deblur=False,
                                        remove_background=False,
                                        remove_shadows_flag=False,
                                        use_adaptive_thresholding=False,
                                        adaptive_block_size=11,
                                        adaptive_c_value=2,
                                        edge_detector='canny',
                                        blur_method='median',
                                        median_blur_size=51,
                                        canny_threshold_max=140,
                                        canny_threshold_min=30,
                                        sobel_threshold_min=5,
                                        sobel_threshold_max=255,
                                        rho=1,
                                        theta=np.pi/180,
                                        threshold_intersect=250,
                                        threshold_distance=0.15,
                                        perpendicular_margin=np.pi/18,
                                        detection_method='hough',
                                        contour_area_threshold_ratio=0.05,
                                        shi_tomasi_maxCorners=100,
                                        shi_tomasi_qualityLevel=0.01,
                                        shi_tomasi_minDistance=10,
                                        harris_blockSize=2,
                                        harris_ksize=3,
                                        harris_k=0.04,
                                        harris_threshold=0.01,
                                        hough_p_threshold=50,
                                        hough_p_minLineLength=50,
                                        hough_p_maxLineGap=10
                                        ):

        # Ön işleme
        preprocessed = img.copy() # Start with a copy
        if preprocess_method == 'aggressive':
            preprocessed = self.preprocess_image_aggressive(preprocessed)
        elif preprocess_method == 'small':
            preprocessed = self.preprocess_image_small(preprocessed)
        elif preprocess_method == 'advanced':
            preprocessed = self.preprocess_image_advanced(preprocessed)
        else: # 'default'
            preprocessed = self.preprocess_image(preprocessed)

        if preprocessed is None:
             raise RuntimeError("Preprocessing failed.")


        if deblur:
            preprocessed = self.sharpen_image(preprocessed)
            if preprocessed is None: raise RuntimeError("Deblurring failed.")


        if remove_background:
            preprocessed = self.remove_background_grabcut(preprocessed)
            if preprocessed is None: raise RuntimeError("Background removal failed.")


        if remove_shadows_flag:
            preprocessed = self.remove_shadows(preprocessed)
            if preprocessed is None: raise RuntimeError("Shadow removal failed.")


        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)

        blurred = gray # Initialize blurred
        if blur_method == 'median' and median_blur_size > 1 and median_blur_size % 2 == 1: # Median blur size must be odd and > 1
             blurred = cv2.medianBlur(gray, median_blur_size)
        elif blur_method == 'bilateral':
             blurred = cv2.bilateralFilter(gray, 9, 75, 75)


        # Perform detection based on the selected method
        corners = None
        edges = None # Initialize edges to None for Hough methods
        thresh = None # Initialize thresh for contour method


        if detection_method == 'hough':
            if use_adaptive_thresholding:
                 # Adaptive thresholding parameters block_size must be odd and > 1
                 block_size = adaptive_block_size if adaptive_block_size % 2 == 1 and adaptive_block_size > 1 else 11
                 edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c_value)
            else:
                if edge_detector == 'canny':
                     edges = cv2.Canny(blurred, canny_threshold_min, canny_threshold_max)
                elif edge_detector == 'sobel':
                     sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                     sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                     edges = cv2.magnitude(sobelx, sobely)
                     # Avoid division by zero if edges.max() is 0
                     max_edge_val = edges.max()
                     if max_edge_val > 0:
                        edges = np.uint8(edges * 255 / max_edge_val) # Normalize to 0-255
                     else:
                        edges = np.zeros_like(edges, dtype=np.uint8)

                     _, edges = cv2.threshold(edges, sobel_threshold_min, sobel_threshold_max, cv2.THRESH_BINARY)
                else: # Simple thresholding
                     _, edges = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            if edges is not None and np.sum(edges) > 0: # Check if edges were successfully created and are not all zero
                lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
                if lines is not None:
                    lines = filter_perpendicular(lines[:,0], perpendicular_margin)
                    lines = eliminate_duplicates(img, lines, threshold_distance)
                    if len(lines) >= 2: # Need at least 2 lines to find an intersection
                        cartesian = to_cartesian(img, lines)
                        intersections = get_intersections(img, cartesian)
                        if intersections is not None and len(intersections) >= 4:
                            # Use improved corner selection method
                            corners = select_best_corners(intersections, img.shape)


        elif detection_method == 'contour':
            # Contour detection often works better on thresholded images
            if use_adaptive_thresholding:
                 # Adaptive thresholding parameters block_size must be odd and > 1
                 block_size = adaptive_block_size if adaptive_block_size % 2 == 1 and adaptive_block_size > 1 else 11
                 thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c_value)
            else:
                 _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Use Otsu's thresholding

            if thresh is not None and np.sum(thresh) > 0: # Check if thresholded image is valid
                document_contours = find_document_contours(thresh, area_threshold_ratio=contour_area_threshold_ratio)
                if document_contours:
                    # Assuming the largest contour is the document
                    potential_corners = get_corners_from_contours([document_contours[0]])
                    if potential_corners is not None and len(potential_corners) >= 4:
                        # Use improved corner selection method
                        corners = select_best_corners(potential_corners, img.shape)


        elif detection_method == 'shi_tomasi':
            # Apply preprocessing suitable for corner detection before Shi-Tomasi
            # Using blurred image as input for corner detectors often works well
            potential_corners = detect_corners_shi_tomasi(
                blurred,
                maxCorners=shi_tomasi_maxCorners,
                qualityLevel=shi_tomasi_qualityLevel,
                minDistance=shi_tomasi_minDistance
            )
            if potential_corners is not None and len(potential_corners) >= 4:
                 # Use improved corner selection method
                 corners = select_best_corners(potential_corners, img.shape)


        elif detection_method == 'harris':
             # Apply preprocessing suitable for corner detection before Harris
             # Using blurred image as input for corner detectors often works well
             potential_corners = detect_corners_harris(
                 blurred,
                 blockSize=harris_blockSize,
                 ksize=harris_ksize,
                 k=harris_k,
                 threshold=harris_threshold
             )
             if potential_corners is not None and len(potential_corners) >= 4:
                 # Use improved corner selection method
                 corners = select_best_corners(potential_corners, img.shape)


        elif detection_method == 'hough_lines_p':
            # Apply preprocessing suitable for probabilistic Hough
            # Using edges or thresholded image as input often works well
            edges_for_hough_p = None
            if use_adaptive_thresholding:
                 # Adaptive thresholding parameters block_size must be odd and > 1
                 block_size = adaptive_block_size if adaptive_block_size % 2 == 1 and adaptive_block_size > 1 else 11
                 edges_for_hough_p = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c_value)
            else:
                if edge_detector == 'canny':
                     edges_for_hough_p = cv2.Canny(blurred, canny_threshold_min, canny_threshold_max)
                elif edge_detector == 'sobel':
                     sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                     sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                     edges_for_hough_p = cv2.magnitude(sobelx, sobely)
                     # Avoid division by zero if edges_for_hough_p.max() is 0
                     max_edge_val = edges_for_hough_p.max()
                     if max_edge_val > 0:
                        edges_for_hough_p = np.uint8(edges_for_hough_p * 255 / max_edge_val)
                     else:
                         edges_for_hough_p = np.zeros_like(edges_for_hough_p, dtype=np.uint8)

                     _, edges_for_hough_p = cv2.threshold(edges_for_hough_p, sobel_threshold_min, sobel_threshold_max, cv2.THRESH_BINARY)
                else: # Fallback if no edge detector is specified but use_adaptive_thresholding is False
                     _, edges_for_hough_p = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Use Otsu's on blurred


            lines_p = None
            if edges_for_hough_p is not None and np.sum(edges_for_hough_p) > 0:
                lines_p = find_lines_probabilistic_hough(
                    edges_for_hough_p,
                    rho=rho, # Using general rho/theta for simplicity, can add specific ones if needed
                    theta=theta,
                    threshold=hough_p_threshold,
                    minLineLength=hough_p_minLineLength,
                    maxLineGap=hough_p_maxLineGap
                )
                if lines_p is not None and len(lines_p) > 0:
                     # Find intersections from these line segments and then corners
                     intersections_p = get_intersections_from_segments(lines_p, img.shape)
                     if intersections_p is not None and len(intersections_p) >= 4:
                         # Use improved corner selection method
                         corners = select_best_corners(intersections_p, img.shape)


        if corners is None or len(corners) != 4:
             raise ValueError(f"Detection method '{detection_method}' failed to find exactly 4 corners.")

        # No need to reorder here, select_best_corners already returns reordered corners if successful

        h_img, w_img = img.shape[:2]
        # Use output dimensions from configs if KeepSide is False
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

        # Return corrected image, source corners and output size
        return corrected, corners, output_size

    def _apply_perspective(self, src_img: np.ndarray):
        """
        Görüntüye gelişmiş perspektif düzeltme uygular.
        Farklı parametre kombinasyonlarını dener ve başarılı olan ilkini döndürür.
        Orijinal ve negatif görüntü üzerinde denemeler yapar.
        """
        if src_img is None or src_img.size == 0:
            raise ValueError("Input image is empty or None in _apply_perspective.")

        original_image_error = None
        negative_image_error = None

        def try_all_tries_internal(image_to_process):
            """İç yardımcı fonksiyon: Belirli bir görüntü üzerinde tüm denemeleri yapar."""
            # Denenecek parametre kombinasyonları (Genişletilmiş ve daha sistematik)
            tries = []

            # Define parameter ranges for more comprehensive testing
            preprocess_methods = ['default', 'aggressive', 'small', 'advanced']
            bool_options = [False, True]
            edge_detectors = ['canny', 'sobel', 'threshold'] # Added 'threshold' as a simple edge method
            blur_methods = ['none', 'median', 'bilateral'] # Added 'none' for no blur
            median_blur_sizes = [3, 5, 7, 9, 11, 21, 31, 51] # More median blur sizes
            canny_threshold_max_values = [100, 150, 200, 250]
            canny_threshold_min_values = [10, 20, 30, 40, 50]
            sobel_threshold_min_values = [5, 10, 20]
            sobel_threshold_max_values = [100, 150, 200, 255]
            rho_values = [1, 0.5] # Added a smaller rho
            theta_values = [np.pi/180, np.pi/360] # Added a smaller theta
            threshold_intersect_values = [100, 150, 200, 250, 300] # More Hough intersect thresholds
            detection_methods_to_try = ['hough', 'contour', 'shi_tomasi', 'harris', 'hough_lines_p']
            contour_area_threshold_ratios = [0.01, 0.03, 0.05, 0.1] # More contour area thresholds
            shi_tomasi_maxCorners_values = [50, 100, 150]
            shi_tomasi_qualityLevel_values = [0.005, 0.01, 0.05]
            shi_tomasi_minDistance_values = [5, 10, 20]
            harris_threshold_values = [0.001, 0.005, 0.01, 0.05]
            hough_p_threshold_values = [30, 50, 80, 100]
            hough_p_minLineLength_values = [30, 50, 80]
            hough_p_maxLineGap_values = [5, 10, 20]
            adaptive_block_size_values = [11, 15, 21, 25] # More adaptive thresholds
            adaptive_c_value_values = [1, 2, 5, 10]

            # Generate all combinations of parameters
            for preprocess_method in preprocess_methods:
                for deblur in bool_options:
                    for remove_background in bool_options:
                        for remove_shadows_flag in bool_options:
                            for use_adaptive_thresholding in bool_options:
                                for blur_method in blur_methods:
                                    for median_blur_size in median_blur_sizes:
                                        # Median blur size must be odd and > 1
                                        if blur_method == 'median' and (median_blur_size % 2 == 0 or median_blur_size <= 1):
                                            continue
                                        if blur_method != 'median' and median_blur_size != median_blur_sizes[0]: # Only iterate median_blur_size for median blur
                                            continue

                                        for current_detection_method in detection_methods_to_try:
                                            params = {
                                                "preprocess_method": preprocess_method,
                                                "deblur": deblur,
                                                "remove_background": remove_background,
                                                "remove_shadows_flag": remove_shadows_flag,
                                                "use_adaptive_thresholding": use_adaptive_thresholding,
                                                "blur_method": blur_method,
                                                "median_blur_size": median_blur_size,
                                                "detection_method": current_detection_method,
                                            }

                                            if use_adaptive_thresholding:
                                                 for adaptive_block_size in adaptive_block_size_values:
                                                     if adaptive_block_size % 2 == 0 or adaptive_block_size <= 1:
                                                          continue
                                                     for adaptive_c_value in adaptive_c_value_values:
                                                        current_params = params.copy()
                                                        current_params.update({
                                                            "adaptive_block_size": adaptive_block_size,
                                                            "adaptive_c_value": adaptive_c_value
                                                        })
                                                        if current_params not in tries:
                                                            tries.append(current_params)
                                            else: # Fixed thresholding or edge detection
                                                if current_detection_method == 'hough':
                                                    for edge_detector in edge_detectors:
                                                        for rho in rho_values:
                                                            for theta in theta_values:
                                                                for threshold_intersect in threshold_intersect_values:
                                                                    if edge_detector == 'canny':
                                                                        for canny_threshold_max in canny_threshold_max_values:
                                                                            for canny_threshold_min in canny_threshold_min_values:
                                                                                if canny_threshold_min < canny_threshold_max:
                                                                                    current_params = params.copy()
                                                                                    current_params.update({
                                                                                        "edge_detector": edge_detector,
                                                                                        "canny_threshold_max": canny_threshold_max,
                                                                                        "canny_threshold_min": canny_threshold_min,
                                                                                        "rho": rho,
                                                                                        "theta": theta,
                                                                                        "threshold_intersect": threshold_intersect,
                                                                                    })
                                                                                    if current_params not in tries:
                                                                                        tries.append(current_params)
                                                                    elif edge_detector == 'sobel':
                                                                        for sobel_threshold_min in sobel_threshold_min_values:
                                                                            for sobel_threshold_max in sobel_threshold_max_values:
                                                                                if sobel_threshold_min < sobel_threshold_max:
                                                                                     current_params = params.copy()
                                                                                     current_params.update({
                                                                                        "edge_detector": edge_detector,
                                                                                        "sobel_threshold_min": sobel_threshold_min,
                                                                                        "sobel_threshold_max": sobel_threshold_max,
                                                                                        "rho": rho,
                                                                                        "theta": theta,
                                                                                        "threshold_intersect": threshold_intersect,
                                                                                    })
                                                                                     if current_params not in tries:
                                                                                        tries.append(current_params)
                                                                    else: # Simple thresholding
                                                                         current_params = params.copy()
                                                                         current_params.update({
                                                                            "edge_detector": edge_detector, # 'threshold'
                                                                            "rho": rho,
                                                                            "theta": theta,
                                                                            "threshold_intersect": threshold_intersect,
                                                                        })
                                                                         if current_params not in tries:
                                                                            tries.append(current_params)


                                                elif current_detection_method == 'contour':
                                                     for contour_area_threshold_ratio in contour_area_threshold_ratios:
                                                        current_params = params.copy()
                                                        current_params.update({
                                                            "contour_area_threshold_ratio": contour_area_threshold_ratio
                                                        })
                                                        if current_params not in tries:
                                                             tries.append(current_params)
                                                elif current_detection_method == 'shi_tomasi':
                                                     for shi_tomasi_maxCorners in shi_tomasi_maxCorners_values:
                                                         for shi_tomasi_qualityLevel in shi_tomasi_qualityLevel_values:
                                                             for shi_tomasi_minDistance in shi_tomasi_minDistance_values:
                                                                 current_params = params.copy()
                                                                 current_params.update({
                                                                     "shi_tomasi_maxCorners": shi_tomasi_maxCorners,
                                                                     "shi_tomasi_qualityLevel": shi_tomasi_qualityLevel,
                                                                     "shi_tomasi_minDistance": shi_tomasi_minDistance
                                                                 })
                                                                 if current_params not in tries:
                                                                      tries.append(current_params)

                                                elif current_detection_method == 'harris':
                                                    for harris_threshold in harris_threshold_values:
                                                         current_params = params.copy()
                                                         current_params.update({
                                                             "harris_threshold": harris_threshold
                                                         })
                                                         if current_params not in tries:
                                                              tries.append(current_params)

                                                elif current_detection_method == 'hough_lines_p':
                                                    for edge_detector in edge_detectors: # HoughLinesP için de kenar dedektörü önemli
                                                        for rho in rho_values:
                                                            for theta in theta_values:
                                                                for hough_p_threshold in hough_p_threshold_values:
                                                                    for hough_p_minLineLength in hough_p_minLineLength_values:
                                                                        for hough_p_maxLineGap in hough_p_maxLineGap_values:
                                                                             current_params = params.copy()
                                                                             current_params.update({
                                                                                 "edge_detector": edge_detector,
                                                                                 "rho": rho,
                                                                                 "theta": theta,
                                                                                 "hough_p_threshold": hough_p_threshold,
                                                                                 "hough_p_minLineLength": hough_p_minLineLength,
                                                                                 "hough_p_maxLineGap": hough_p_maxLineGap
                                                                             })
                                                                             if current_params not in tries:
                                                                                  tries.append(current_params)
                                                else: # detection_method specified but no parameters handled
                                                    if params not in tries:
                                                        tries.append(params)


            print(f"Toplam {len(tries)} deneme yapılacak.")

            for i, params in enumerate(tries):
                try:
                    corrected_image, src_quad, output_size = self._correct_perspective_single_try(
                        image_to_process,
                        preprocess_method=params.get("preprocess_method", 'default'),
                        deblur=params.get("deblur", False),
                        remove_background=params.get("remove_background", False),
                        remove_shadows_flag=params.get("remove_shadows_flag", False),
                        use_adaptive_thresholding=params.get("use_adaptive_thresholding", False),
                        adaptive_block_size=params.get("adaptive_block_size", 11),
                        adaptive_c_value=params.get("adaptive_c_value", 2),
                        edge_detector=params.get("edge_detector", 'canny'),
                        blur_method=params.get("blur_method", 'median'),
                        median_blur_size=params.get("median_blur_size", 51),
                        canny_threshold_max=params.get("canny_threshold_max", 140),
                        canny_threshold_min=params.get("canny_threshold_min", 30),
                        sobel_threshold_min=params.get("sobel_threshold_min", 5),
                        sobel_threshold_max=params.get("sobel_threshold_max", 255),
                        rho=params.get("rho", 1),
                        theta=params.get("theta", np.pi/180),
                        threshold_intersect=params.get("threshold_intersect", 250),
                        detection_method=params.get("detection_method", 'hough'),
                        contour_area_threshold_ratio=params.get("contour_area_threshold_ratio", 0.05),
                        shi_tomasi_maxCorners=params.get("shi_tomasi_maxCorners", 100),
                        shi_tomasi_qualityLevel=params.get("shi_tomasi_qualityLevel", 0.01),
                        shi_tomasi_minDistance=params.get("shi_tomasi_minDistance", 10),
                        harris_threshold=params.get("harris_threshold", 0.01),
                        hough_p_threshold=params.get("hough_p_threshold", 50),
                        hough_p_minLineLength=params.get("hough_p_minLineLength", 50),
                        hough_p_maxLineGap=params.get("hough_p_maxLineGap", 10)
                    )
                    return corrected_image, src_quad, output_size
                except Exception as e:
                    continue
            raise RuntimeError("Tüm denemeler başarısız oldu.")

        try:
            warped, src_quad, (out_w, out_h) = try_all_tries_internal(src_img)
            return warped, [], src_quad, (out_w, out_h)
        except RuntimeError as e:
            original_image_error = str(e)

        img_neg = cv2.bitwise_not(src_img)
        try:
            warped, src_quad, (out_w, out_h) = try_all_tries_internal(img_neg)
            return warped, [], src_quad, (out_w, out_h)
        except RuntimeError as e:
            negative_image_error = str(e)

        error_message = "Perspektif düzeltme hem orijinal hem de negatif görüntüde başarısız oldu."
        if original_image_error:
            error_message += f"\nOrijinal görüntü hatası: {original_image_error}"
        if negative_image_error:
            error_message += f"\nNegatif görüntü hatası: {negative_image_error}"
        raise RuntimeError(error_message)


    # Added explicit self parameter and type hint for run method
    def run(self) -> Image:
        # self.image, __init__ içinde zaten ayarlanmıştır
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img is None or img.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img.value)

        warped, corrected_boxes, src_quad, (out_w, out_h) = self._apply_perspective(src_img)

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