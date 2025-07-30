import os
import sys
from itertools import combinations
import cv2
import numpy as np

# Sistem yolunu güncelleyin
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel
from components.PerspectiveTransformation.src.utils.response import build_response

# Yardımcı fonksiyonlar (sınıfın dışında kalacak)
def order_points(pts):
    # Noktaları numpy dizisine çevir
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
    new_corners = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    new_corners[0] = corners[np.argmin(s)]      # üst sol
    new_corners[2] = corners[np.argmax(s)]      # alt sağ
    new_corners[1] = corners[np.argmin(diff)]   # üst sağ
    new_corners[3] = corners[np.argmax(diff)]   # alt sol
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

def approximate_4_corners(points):
    if len(points) <= 4:
        return points.astype(np.float32)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    idxs = np.argsort(distances)[-4:]
    return points[idxs].astype(np.float32)

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
    return np.array(corners, dtype=np.float32)

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

class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}

        # self.request.data'nın bir sözlük olduğundan emin olun ve PackageModel'i güvenle başlatın
        if not isinstance(self.request.data, dict):
            print(f"UYARI: self.request.data bir sözlük değil, tipi: {type(self.request.data)}. Boş bir sözlük kullanılıyor.")
            model_data = {}
        else:
            model_data = self.request.data

        try:
            self.request.model = PackageModel(**model_data)
        except TypeError as e:
            print(f"HATA: PackageModel başlatılırken TypeError oluştu: {e}")
            print(f"self.request.data içeriği: {model_data}")
            raise RuntimeError("PackageModel başlatılamadı, lütfen request.data'yı kontrol edin.") from e
        except Exception as e:
            print(f"HATA: PackageModel başlatılırken beklenmeyen bir hata oluştu: {e}")
            print(f"self.request.data içeriği: {model_data}")
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

        # PackageModel'den keep_side ve warp_image_flag değerlerini alın
        # Eğer PackageModel'de bu özellikler yoksa varsayılan değerler atayın
        self.keep_side = getattr(self.request.model, 'keep_side', 'auto') # Varsayılan değer 'auto'
        self.warp_image_flag = getattr(self.request.model, 'warp_image', True) # Varsayılan değer True

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

    def preprocess_image(self, img): # self parametresi eklendi
        """Orta seviye CLAHE ile kontrast artırma (normal)"""
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

    def preprocess_image_aggressive(self, img): # self parametresi eklendi
        """Agresif CLAHE, histogram eşitleme ile güçlü kontrast artırma"""
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

    def preprocess_image_small(self, img): # self parametresi eklendi
        """Küçük resimler için daha hafif CLAHE ve keskinleştirme"""
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

    def preprocess_image_advanced(self, img): # self parametresi eklendi
        """Daha gelişmiş ön işleme: CLAHE + Median Blur + Unsharp Mask"""
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

    def sharpen_image(self, img): # self parametresi eklendi
        """Bulanıklık için keskinleştirme filtresi"""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(img, -1, kernel)
        return sharp

    def remove_background_grabcut(self, img): # self parametresi eklendi
        """GrabCut ile arka planı kaldırma (başlangıç maskesi gerekebilir)"""
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, img.shape[1]-1, img.shape[0]-1)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        return img

    def remove_shadows(self, img): # self parametresi eklendi
        """Gölge kaldırma (basit)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        return result

    def correct_perspective_enhanced(self, img,
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
                                      hough_p_maxLineGap=10): # intermediate parametresi kaldırıldı, her zaman nihai çıktıyı döndürecek

        if preprocess_method == 'aggressive':
            preprocessed = self.preprocess_image_aggressive(img)
        elif preprocess_method == 'small':
            preprocessed = self.preprocess_image_small(img)
        elif preprocess_method == 'advanced':
            preprocessed = self.preprocess_image_advanced(img)
        else: # 'default'
            preprocessed = self.preprocess_image(img)

        if deblur:
            preprocessed = self.sharpen_image(preprocessed)

        if remove_background:
            preprocessed = self.remove_background_grabcut(preprocessed)

        if remove_shadows_flag:
            preprocessed = self.remove_shadows(preprocessed)

        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)

        if blur_method == 'median':
             blurred = cv2.medianBlur(gray, median_blur_size)
        elif blur_method == 'bilateral':
             blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
             blurred = gray

        corners = None
        edges = None
        thresh = None

        if detection_method == 'hough':
            if use_adaptive_thresholding:
                 edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, adaptive_c_value)
            else:
                if edge_detector == 'canny':
                     edges = cv2.Canny(blurred, canny_threshold_min, canny_threshold_max)
                elif edge_detector == 'sobel':
                     sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                     sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                     edges = cv2.magnitude(sobelx, sobely)
                     edges = np.uint8(edges * 255 / edges.max())
                     _, edges = cv2.threshold(edges, sobel_threshold_min, sobel_threshold_max, cv2.THRESH_BINARY)

            if edges is not None:
                lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
                if lines is not None:
                    lines = filter_perpendicular(lines[:,0], perpendicular_margin)
                    lines = eliminate_duplicates(img, lines, threshold_distance)
                    if len(lines) >= 4:
                        cartesian = to_cartesian(img, lines)
                        intersections = get_intersections(img, cartesian)
                        if intersections is not None and len(intersections) > 4: # Kesişim noktaları bulunduysa
                            corners = approximate_4_corners(intersections) # Kesişim noktalarından 4 köşe seç
                        elif intersections is not None and len(intersections) == 4:
                            corners = intersections
                        else:
                            corners = None # Yeterli kesişim noktası bulunamadı

        elif detection_method == 'contour':
            if use_adaptive_thresholding:
                 thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, adaptive_c_value)
            else:
                 _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            document_contours = find_document_contours(thresh, area_threshold_ratio=contour_area_threshold_ratio)
            if document_contours:
                corners = get_corners_from_contours([document_contours[0]])
                if corners is not None and len(corners) != 4:
                    corners = None

        elif detection_method == 'shi_tomasi':
            corners = detect_corners_shi_tomasi(
                blurred,
                maxCorners=shi_tomasi_maxCorners,
                qualityLevel=shi_tomasi_qualityLevel,
                minDistance=shi_tomasi_minDistance
            )
            if corners is not None and len(corners) > 4:
                corners = approximate_4_corners(corners)
            elif corners is not None and len(corners) != 4:
                 corners = None

        elif detection_method == 'harris':
             corners = detect_corners_harris(
                 blurred,
                 blockSize=harris_blockSize,
                 ksize=harris_ksize,
                 k=harris_k,
                 threshold=harris_threshold
             )
             if corners is not None and len(corners) > 4:
                 corners = approximate_4_corners(corners)
             elif corners is not None and len(corners) != 4:
                  corners = None

        elif detection_method == 'hough_lines_p':
            if use_adaptive_thresholding:
                 edges_for_hough_p = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, adaptive_c_value)
            else:
                if edge_detector == 'canny':
                     edges_for_hough_p = cv2.Canny(blurred, canny_threshold_min, canny_threshold_max)
                elif edge_detector == 'sobel':
                     sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                     sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                     edges_for_hough_p = cv2.magnitude(sobelx, sobely)
                     edges_for_hough_p = np.uint8(edges_for_hough_p * 255 / edges_for_hough_p.max())
                     _, edges_for_hough_p = cv2.threshold(edges_for_hough_p, sobel_threshold_min, sobel_threshold_max, cv2.THRESH_BINARY)
                else:
                     _, edges_for_hough_p = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            lines_p = None
            if edges_for_hough_p is not None:
                lines_p = find_lines_probabilistic_hough(
                    edges_for_hough_p,
                    rho=rho,
                    theta=theta,
                    threshold=hough_p_threshold,
                    minLineLength=hough_p_minLineLength,
                    maxLineGap=hough_p_maxLineGap
                )
                if lines_p is not None and len(lines_p) > 0:
                     intersections_p = get_intersections_from_segments(lines_p, img.shape)
                     if len(intersections_p) >= 4:
                         corners = approximate_4_corners(intersections_p)
                     else:
                         corners = None # Yeterli kesişim noktası bulunamadı

        if corners is None or len(corners) < 4:
             raise ValueError(f"Detection method '{detection_method}' failed to find 4 corners.")

        src_quad = reorder_corners(corners) # Bu, kaynak köşeleriniz (src_quad)

        h_img, w_img = img.shape[:2]
        min_dim = min(h_img, w_img)
        if h_img > w_img:
            new_h, new_w = int(min_dim), int(min_dim * 0.707)
        else:
            new_h, new_w = int(min_dim * 0.707), int(min_dim)

        output_size = (new_w, new_h) # Bu, çıktı boyutunuz (out_w, out_h)
        destination = np.array([[0,0], [new_w,0], [new_w,new_h], [0,new_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_quad, destination)
        corrected = cv2.warpPerspective(img, M, output_size)

        # Düzeltilmiş görüntüyü, kaynak köşeleri ve çıktı boyutunu döndürün
        return corrected, src_quad, output_size

    def _apply_perspective(self, src_img: np.ndarray):
        """
        Görüntüye gelişmiş perspektif düzeltme uygular.
        Farklı parametre kombinasyonlarını dener ve başarılı olan ilkini döndürür.
        """
        h, w = src_img.shape[:2]
        max_dim = max(h, w)

        # Denenecek parametre kombinasyonları (orijinal dosyanızdaki geniş listeyi buraya kopyalayabilirsiniz)
        tries = [
            {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 51, "canny_threshold_max": 150, "canny_threshold_min": 50, "rho": 1, "theta": np.pi/180, "threshold_intersect": 150, "detection_method": 'hough'},
            {"preprocess_method": 'aggressive', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 31, "canny_threshold_max": 100, "canny_threshold_min": 30, "rho": 1, "theta": np.pi/180, "threshold_intersect": 100, "detection_method": 'hough'},
            {"preprocess_method": 'advanced', "deblur": True, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 61, "canny_threshold_max": 200, "canny_threshold_min": 80, "rho": 1, "theta": np.pi/180, "threshold_intersect": 200, "detection_method": 'hough'},
            {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": True, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 51, "canny_threshold_max": 150, "canny_threshold_min": 50, "rho": 1, "theta": np.pi/180, "threshold_intersect": 150, "detection_method": 'hough'},
            {"preprocess_method": 'default', "remove_shadows_flag": False, "use_adaptive_thresholding": False, "blur_method": 'median', "median_blur_size": 51, "detection_method": 'contour', "contour_area_threshold_ratio": 0.05},
            {"preprocess_method": 'aggressive', "remove_shadows_flag": False, "use_adaptive_thresholding": False, "blur_method": 'median', "median_blur_size": 31, "detection_method": 'contour', "contour_area_threshold_ratio": 0.03},
            {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 5, "detection_method": 'shi_tomasi', "shi_tomasi_maxCorners": 100, "shi_tomasi_qualityLevel": 0.01, "shi_tomasi_minDistance": 10},
            {"preprocess_method": 'advanced', "deblur": True, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 3, "detection_method": 'shi_tomasi', "shi_tomasi_maxCorners": 50, "shi_tomasi_qualityLevel": 0.05, "shi_tomasi_minDistance": 20},
            {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 5, "detection_method": 'harris', "harris_blockSize": 2, "harris_ksize": 3, "harris_k": 0.04, "harris_threshold": 0.01},
            {"preprocess_method": 'advanced', "deblur": True, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 3, "detection_method": 'harris', "harris_blockSize": 3, "harris_ksize": 5, "harris_k": 0.05, "harris_threshold": 0.005},
            {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 51, "canny_threshold_max": 150, "canny_threshold_min": 50, "rho": 1, "theta": np.pi/180, "hough_p_threshold": 50, "hough_p_minLineLength": 50, "hough_p_maxLineGap": 10, "detection_method": 'hough_lines_p'},
            {"preprocess_method": 'aggressive', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 31, "canny_threshold_max": 100, "canny_threshold_min": 30, "rho": 1, "theta": np.pi/180, "hough_p_threshold": 80, "hough_p_minLineLength": 80, "hough_p_maxLineGap": 20, "detection_method": 'hough_lines_p'},
        ]

        print(f"Toplam {len(tries)} deneme yapılacak.")

        for i, params in enumerate(tries):
            print(f"\n🔁 Deneme {i+1}/{len(tries)}: {params}")
            try:
                # self.correct_perspective_enhanced metodunu çağırın
                # Bu metod artık 3 değer döndürüyor: corrected_image, src_quad, output_size
                corrected_image, src_quad, output_size = self.correct_perspective_enhanced(
                    src_img,
                    preprocess_method=params.get("preprocess_method", 'default'),
                    deblur=params.get("deblur", False),
                    remove_background=params.get("remove_background", False),
                    remove_shadows_flag=params.get("remove_shadows_flag", False),
                    use_adaptive_thresholding=params.get("use_adaptive_thresholding", False),
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
                print("✅ Başarılı!")
                # corrected_boxes için bir mantık yoksa şimdilik boş bir liste döndürün
                # Eğer dönüştürülmüş bounding box'lar bekleniyorsa, burada hesaplanmalıdır.
                corrected_boxes = []
                return corrected_image, corrected_boxes, src_quad, output_size
            except Exception as e:
                print(f"⛔ Deneme {i+1}/{len(tries)} başarısız oldu: {e}")
                continue
        raise RuntimeError("Tüm denemeler başarısız oldu.")

    def run(self, image: Image) -> Image:
        img = Image.get_frame(img=image, redis_db=self.redis_db)
        if img is None or img.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img.value)

        # _apply_perspective metodunu çağırın. Bu metod artık run'ın beklediği tüm değerleri döndürüyor.
        warped, corrected_boxes, src_quad, (out_w, out_h) = self._apply_perspective(src_img)

        img.value = warped
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        self.context = {
            "src_quad": src_quad.tolist(),
            "output_size": [out_w, out_h],
            "corrected_boxes": corrected_boxes,
            "keep_side": self.keep_side, # __init__ içinde PackageModel'den alınacak
            "warp_image": self.warp_image_flag, # __init__ içinde PackageModel'den alınacak
        }
        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()