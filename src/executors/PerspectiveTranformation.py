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
    rect[1] = pts[np.argmin(diff)]   # Sağ üst
    rect[3] = pts[np.argmax(diff)]   # Sol alt
    return rect

def get_intersections(img, lines):
    """Çizgilerin kesişim noktalarını hesaplar"""
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

def select_outermost_corners(points, k=4):
    """
    Verilen noktalardan en dıştaki k (varsayılan 4) köşeyi seçer.
    """
    if len(points) <= k:
        return points.astype(np.float32)

    # Noktaların merkezini bul
    centroid = np.mean(points, axis=0)

    # Merkezden her noktanın uzaklığını hesapla
    distances = np.linalg.norm(points - centroid, axis=1)

    # En uzak k noktayı seç
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

        # PackageModel'den konfigürasyonları alın
        self.perspective_type_mode = getattr(self.request.model.configs.PerspectiveTypeMode.value, 'value', 'Auto')
        self.keep_side = getattr(self.request.model.configs.drawBBox.value, 'value', True) # drawBBox'tan keep_side
        self.output_width = getattr(self.request.model.configs.outputWidth, 'value', None)
        self.output_height = getattr(self.request.model.configs.outputHeight, 'value', None)


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

    # ----- ÖN İŞLEME FONKSİYONLARI (Sınıf Metodları Olarak) -----

    def preprocess_image(self, img):
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

    def preprocess_image_aggressive(self, img):
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

    def preprocess_image_small(self, img):
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

    def preprocess_image_advanced(self, img):
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

    def sharpen_image(self, img):
        """Bulanıklık için keskinleştirme filtresi"""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(img, -1, kernel)
        return sharp

    def remove_background_grabcut(self, img):
        """GrabCut ile arka planı kaldırma (başlangıç maskesi gerekebilir)"""
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        return result

    def _perform_single_perspective_correction(self, img,
                                        threshold_max=140,
                                        threshold_min=30,
                                        median_blur_size=51,
                                        rho=1,
                                        theta=np.pi/180,
                                        threshold_intersect=250,
                                        threshold_distance=0.15,
                                        perpendicular_margin=np.pi/18,
                                        aggressive_preprocess=False,
                                        small_image_preprocess=False,
                                        deblur=False,
                                        use_adaptive_thresholding=False,
                                        edge_detector='canny',
                                        blur_method='median',
                                        target_output_size=None): # Yeni parametre

        # Ön işleme
        if small_image_preprocess:
            preprocessed = self.preprocess_image_small(img)
        elif aggressive_preprocess:
            preprocessed = self.preprocess_image_aggressive(img)
        else:
            preprocessed = self.preprocess_image(img)

        if deblur:
            preprocessed = self.sharpen_image(preprocessed)

        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)

        if blur_method == 'median':
            blurred = cv2.medianBlur(gray, median_blur_size)
        elif blur_method == 'bilateral':
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
            blurred = gray

        # Kenar Tespiti
        if use_adaptive_thresholding:
            edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            if edge_detector == 'canny':
                edges = cv2.Canny(blurred, threshold_min, threshold_max)
            elif edge_detector == 'sobel':
                sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
                edges = cv2.magnitude(sobelx, sobely)
                edges = np.uint8(edges * 255 / edges.max())
                _, edges = cv2.threshold(edges, threshold_min, threshold_max, cv2.THRESH_BINARY)
            else: # Varsayılan olarak Canny'ye düş
                edges = cv2.Canny(blurred, threshold_min, threshold_max)


        # Hough çizgilerini al
        lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
        if lines is None:
            raise ValueError("HoughLines çizgi bulamadı, threshold değerlerini değiştir.")
        lines = filter_perpendicular(lines[:,0], perpendicular_margin)
        lines = eliminate_duplicates(img, lines, threshold_distance)

        # Yeterince çizgi yoksa threshold düşürerek dene
        while len(lines) < 4 and threshold_intersect > 10:
            threshold_intersect -= 5
            lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
            if lines is None:
                continue
            lines = filter_perpendicular(lines[:,0], perpendicular_margin)
            lines = eliminate_duplicates(img, lines, threshold_distance)

        if len(lines) < 4:
            raise ValueError("Yeterli çizgi tespiti yapılamadı.")

        cartesian = to_cartesian(img, lines)
        intersections = get_intersections(img, cartesian)

        if len(intersections) < 4:
            raise ValueError("Yeterli kesişim noktası bulunamadı.")

        corners = select_outermost_corners(intersections, k=4)

        # Köşeleri sırala
        src_quad = reorder_corners(corners)

        # Hedef boyutunu kullan veya dinamik olarak hesapla
        if target_output_size:
            new_w, new_h = target_output_size
        else:
            h_img, w_img = img.shape[:2]
            min_dim = min(h_img, w_img)
            if h_img > w_img:
                new_h, new_w = int(min_dim), int(min_dim * 0.707)
            else:
                new_h, new_w = int(min_dim * 0.707), int(min_dim)

        output_size = (new_w, new_h)
        destination = np.array([[0,0], [new_w,0], [new_w,new_h], [0,new_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_quad, destination)
        corrected = cv2.warpPerspective(img, M, output_size)

        return corrected, src_quad, output_size

    def _apply_perspective(self, src_img: np.ndarray):
        """
        Görüntüye gelişmiş perspektif düzeltme uygular.
        Seçilen moda göre farklı parametre kombinasyonlarını veya tek bir denemeyi kullanır.
        Orijinal ve negatif görüntü üzerinde denemeler yapar.
        """
        original_image_error = None
        negative_image_error = None

        target_output_size = None
        if self.output_width is not None and self.output_height is not None:
            target_output_size = (self.output_width, self.output_height)

        if self.perspective_type_mode == 'Auto':
            def try_all_tries_internal(image_to_process):
                """İç yardımcı fonksiyon: Belirli bir görüntü üzerinde tüm denemeleri yapar."""
                prioritized_tries = [
                    {"threshold_max": 150, "threshold_min": 50, "median_blur_size": 51, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "threshold_intersect": 150, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"threshold_max": 100, "threshold_min": 30, "median_blur_size": 31, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "threshold_intersect": 100, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"threshold_max": 200, "threshold_min": 80, "median_blur_size": 61, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": True, "use_adaptive_thresholding": False, "edge_detector": 'canny', "threshold_intersect": 200, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"median_blur_size": 51, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": True, "edge_detector": 'canny', "threshold_intersect": 150, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"median_blur_size": 31, "aggressive_preprocess": True, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": True, "edge_detector": 'canny', "threshold_intersect": 100, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"threshold_max": 150, "threshold_min": 50, "median_blur_size": 51, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": False, "edge_detector": 'sobel', "threshold_intersect": 150, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"threshold_max": 200, "threshold_min": 80, "median_blur_size": 61, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": True, "use_adaptive_thresholding": False, "edge_detector": 'sobel', "threshold_intersect": 200, "rho": 1, "theta": np.pi/180, "blur_method": 'median'},
                    {"median_blur_size": 0, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "threshold_intersect": 150, "rho": 1, "theta": np.pi/180, "blur_method": 'bilateral'},
                    {"median_blur_size": 0, "aggressive_preprocess": False, "small_image_preprocess": False, "deblur": False, "use_adaptive_thresholding": True, "edge_detector": 'canny', "threshold_intersect": 150, "rho": 1, "theta": np.pi/180, "blur_method": 'bilateral'},
                ]

                broad_tries = []
                for threshold_max in range(20, 201, 20):
                    for threshold_min in range(5, threshold_max // 2 + 1, 5):
                        for median_blur_size in [3, 5, 7, 11, 21, 31, 41, 51, 61]:
                            for aggressive_preprocess in [False, True]:
                                for small_image_preprocess in [False, True]:
                                    for deblur in [False, True]:
                                        for threshold_intersect in range(50, 301, 50):
                                            for rho in [1, 0.5]:
                                                for theta in [np.pi/180, np.pi/360]:
                                                    for blur_method in ['median', 'bilateral']:
                                                        params = {
                                                            "threshold_max": threshold_max,
                                                            "threshold_min": threshold_min,
                                                            "median_blur_size": median_blur_size,
                                                            "aggressive_preprocess": aggressive_preprocess,
                                                            "small_image_preprocess": small_image_preprocess,
                                                            "deblur": deblur,
                                                            "use_adaptive_thresholding": False,
                                                            "edge_detector": 'canny',
                                                            "threshold_intersect": threshold_intersect,
                                                            "rho": rho,
                                                            "theta": theta,
                                                            "blur_method": blur_method
                                                        }
                                                        if params not in prioritized_tries:
                                                            broad_tries.append(params)

                for median_blur_size in [3, 5, 7, 11, 21, 31, 41, 51, 61]:
                     for aggressive_preprocess in [False, True]:
                        for small_image_preprocess in [False, True]:
                            for deblur in [False, True]:
                                 for threshold_intersect in range(50, 301, 50):
                                    for rho in [1, 0.5]:
                                        for theta in [np.pi/180, np.pi/360]:
                                             for blur_method in ['median', 'bilateral']:
                                                params = {
                                                    "median_blur_size": median_blur_size,
                                                    "aggressive_preprocess": aggressive_preprocess,
                                                    "small_image_preprocess": small_image_preprocess,
                                                    "deblur": deblur,
                                                    "use_adaptive_thresholding": True,
                                                    "edge_detector": 'canny',
                                                    "threshold_intersect": threshold_intersect,
                                                    "rho": rho,
                                                    "theta": theta,
                                                    "blur_method": blur_method
                                                }
                                                if params not in prioritized_tries:
                                                    broad_tries.append(params)

                for threshold_min in range(5, 51, 5):
                    for threshold_max in range(threshold_min + 10, 256, 10):
                        for median_blur_size in [3, 5, 7, 11, 21, 31, 41, 51, 61]:
                            for aggressive_preprocess in [False, True]:
                                for small_image_preprocess in [False, True]:
                                    for deblur in [False, True]:
                                        for threshold_intersect in range(50, 301, 50):
                                            for rho in [1, 0.5]:
                                                for theta in [np.pi/180, np.pi/360]:
                                                    for blur_method in ['median', 'bilateral']:
                                                        params = {
                                                            "threshold_max": threshold_max,
                                                            "threshold_min": threshold_min,
                                                            "median_blur_size": median_blur_size,
                                                            "aggressive_preprocess": aggressive_preprocess,
                                                            "small_image_preprocess": small_image_preprocess,
                                                            "deblur": deblur,
                                                            "use_adaptive_thresholding": False,
                                                            "edge_detector": 'sobel',
                                                            "threshold_intersect": threshold_intersect,
                                                            "rho": rho,
                                                            "theta": theta,
                                                            "blur_method": blur_method
                                                        }
                                                        if params not in prioritized_tries:
                                                            broad_tries.append(params)

                all_tries = prioritized_tries + broad_tries

                for params in all_tries:
                    try:
                        corrected_image, src_quad, output_size = self._perform_single_perspective_correction(
                            image_to_process,
                            threshold_max=params.get("threshold_max", 140),
                            threshold_min=params.get("threshold_min", 30),
                            median_blur_size=params.get("median_blur_size", 51),
                            rho=params.get("rho", 1),
                            theta=params.get("theta", np.pi/180),
                            threshold_intersect=params.get("threshold_intersect", 250),
                            aggressive_preprocess=params.get("aggressive_preprocess", False),
                            small_image_preprocess=params.get("small_image_preprocess", False),
                            deblur=params.get("deblur", False),
                            use_adaptive_thresholding=params.get("use_adaptive_thresholding", False),
                            edge_detector=params.get("edge_detector", 'canny'),
                            blur_method=params.get("blur_method", 'median'),
                            target_output_size=target_output_size
                        )
                        return corrected_image, src_quad, output_size
                    except Exception:
                        continue
                raise RuntimeError("Tüm denemeler başarısız oldu.")

            # Orijinal görüntü ile deneme
            try:
                warped, src_quad, output_size = try_all_tries_internal(src_img)
                corrected_boxes = []
                return warped, corrected_boxes, src_quad, output_size
            except RuntimeError as e:
                original_image_error = str(e)

            # Negatif görüntü ile deneme
            img_neg = cv2.bitwise_not(src_img)
            try:
                warped, src_quad, output_size = try_all_tries_internal(img_neg)
                corrected_boxes = []
                return warped, corrected_boxes, src_quad, output_size
            except RuntimeError as e:
                negative_image_error = str(e)

            error_message = "Perspektif düzeltme hem orijinal hem de negatif görüntüde başarısız oldu."
            if original_image_error:
                error_message += f"\nOrijinal görüntü hatası: {original_image_error}"
            if negative_image_error:
                error_message += f"\nNegatif görüntü hatası: {negative_image_error}"
            raise RuntimeError(error_message)

        elif self.perspective_type_mode == 'Advanced':
            # Advanced mod için tek bir deneme
            try:
                warped, src_quad, output_size = self._perform_single_perspective_correction(
                    src_img,
                    # Advanced mod için varsayılan veya PackageModel'den alınabilecek parametreler
                    # Şu an PackageModel'de bu parametreler exposed olmadığı için varsayılanlar kullanılıyor.
                    # Eğer ek parametreler eklenirse, buradan okunabilir.
                    target_output_size=target_output_size
                )
                corrected_boxes = []
                return warped, corrected_boxes, src_quad, output_size
            except Exception as e:
                raise RuntimeError(f"Advanced modda perspektif düzeltme başarısız oldu: {e}")
        else:
            raise ValueError(f"Bilinmeyen perspektif tipi modu: {self.perspective_type_mode}")


    def run(self) -> Image:
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
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
            "keep_side": self.keep_side,
            "warp_image": self.warp_image_flag, # Bu bayrak hala kullanılıyor mu kontrol edilebilir
        }
        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
