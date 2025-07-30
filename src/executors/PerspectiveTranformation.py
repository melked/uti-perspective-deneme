import os
import sys
from itertools import combinations

import cv2
import numpy as np
from tensorflow.python.ops.clustering_ops import KMeans

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel
from components.PerspectiveTransformation.src.utils.response import build_response


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

def kmeans_corners(points, k=4):
    """K-means ile fazla köşeleri 4'e indir"""
    if len(points) <= k:
        return points
    kmeans = KMeans(n_clusters=k, n_init=10) # Add n_init
    kmeans.fit(points)
    centers = kmeans.cluster_centers_
    return centers.astype(np.float32)

def find_document_contours(img_gray, area_threshold_ratio=0.05):
    """Kontur tespiti ile olası belge kenarlarını bulur."""
    # Find contours
    contours, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep only the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contours = []
    img_area = img_gray.shape[0] * img_gray.shape[1]

    for cnt in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # If the approximated contour has 4 points and a significant area, consider it a potential document
        if len(approx) == 4 and cv2.contourArea(cnt) > img_area * area_threshold_ratio:
            document_contours.append(approx)

    return document_contours

def get_corners_from_contours(contours):
    """Konturlardan köşe noktalarını çıkarır."""
    corners = []
    for contour in contours:
        # Reshape the contour points to a list of points
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
    # Result is dilated for marking the corners, not important for the detection itself
    # dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    corners = np.argwhere(dst > threshold * dst.max())
    return np.float32(corners).reshape(-1, 2)

def find_lines_probabilistic_hough(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10):
    """Olasılıksal Hough Dönüşümü ile çizgi segmentlerini bulur."""
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        # Reshape to a list of lines (x1, y1, x2, y2)
        return lines.reshape(-1, 4)
    return np.array([], dtype=np.int32)

# Helper to get intersections from line segments (Probabilistic Hough)
def get_intersections_from_segments(segments, img_shape, img=None):
    """Çizgi segmentlerinin kesişim noktalarını hesaplar."""
    height, width = img_shape[:2]
    intersections = []
    # Convert segments to line equations or extend them for intersection
    # A simpler approach is to use the get_intersections function if the segments are long enough
    # Or extend segments to image boundaries
    extended_lines = []
    for x1, y1, x2, y2 in segments:
        # Extend the line segment to the image boundaries (conceptual)
        # This is a simplified approach and might not be accurate for all cases
        # A more robust method would involve calculating the line equation
        if x2 - x1 == 0: # Vertical line
            extended_lines.append((x1, 0, x1, height))
        elif y2 - y1 == 0: # Horizontal line
            extended_lines.append((0, y1, width, y1))
        else:
            # Calculate slope and y-intercept
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            # Points at image boundaries
            y_at_x0 = int(b)
            y_at_xw = int(m * width + b)
            x_at_y0 = int(-b / m) if m != 0 else -1 # Avoid division by zero
            x_at_yh = int((height - b) / m) if m != 0 else -1

            points_on_boundary = []
            if 0 <= y_at_x0 <= height: points_on_boundary.append((0, y_at_x0))
            if 0 <= y_at_xw <= height: points_on_boundary.append((width, y_at_xw))
            if 0 <= x_at_y0 <= width and m != 0: points_on_boundary.append((x_at_y0, 0))
            if 0 <= x_at_yh <= width and m != 0: points_on_boundary.append((x_at_yh, height))

            # Add the two most distinct points on the boundary as extended line
            if len(points_on_boundary) >= 2:
                 p1, p2 = points_on_boundary[0], points_on_boundary[-1]
                 extended_lines.append((p1[0], p1[1], p2[0], p2[1]))
            elif len(points_on_boundary) == 1: # Handle case where only one point is on boundary
                 # Extend from the closest segment endpoint to the boundary point
                 if np.linalg.norm(np.array([x1, y1]) - np.array(points_on_boundary[0])) < np.linalg.norm(np.array([x2, y2]) - np.array(points_on_boundary[0])):
                     extended_lines.append((x1, y1, points_on_boundary[0][0], points_on_boundary[0][1]))
                 else:
                     extended_lines.append((x2, y2, points_on_boundary[0][0], points_on_boundary[0][1]))
            # If less than 2 points on boundary, try extending segment endpoints
            elif len(points_on_boundary) < 2 and len(segments) > 1:
                # Simple extension based on segment direction
                dx = x2 - x1
                dy = y2 - y1
                # Extend from endpoints
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


    def preprocess_image(img):
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

    def preprocess_image_aggressive(img):
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

    def preprocess_image_small(img):
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

    def preprocess_image_advanced(img):
        """Daha gelişmiş ön işleme: CLAHE + Median Blur + Unsharp Mask"""
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10,10))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

        # Median Blur ile gürültü azaltma
        img_blurred = cv2.medianBlur(img_clahe, 5)

        # Unsharp Masking ile keskinleştirme
        gaussian = cv2.GaussianBlur(img_blurred, (9, 9), 0)
        unsharp_mask = cv2.addWeighted(img_blurred, 1.5, gaussian, -0.5, 0)

        return unsharp_mask


    def sharpen_image(img):
        """Bulanıklık için keskinleştirme filtresi"""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(img, -1, kernel)
        return sharp

    def remove_background_grabcut(img):
        """GrabCut ile arka planı kaldırma (başlangıç maskesi gerekebilir)"""
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # Dikdörtgen alanı (document'ı içeren) manuel olarak belirtmek gerekebilir
        # Şimdilik tüm görüntüyü foreground olarak işaretliyoruz, bu her zaman iyi sonuç vermeyebilir
        rect = (1, 1, img.shape[1]-1, img.shape[0]-1)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        return img

    def remove_shadows(img):
        """Gölge kaldırma (basit)"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to get a rough mask of highlights
        dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # Simple thresholding or Otsu's thresholding can be added here for a binary mask

        # Convert back to BGR to apply the mask
        result = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        return result


    def correct_perspective_enhanced(self,img,
                                      preprocess_method='default', # 'default', 'aggressive', 'small', 'advanced'
                                      deblur=False,
                                      remove_background=False, # GrabCut
                                      remove_shadows_flag=False,
                                      use_adaptive_thresholding=False,
                                      adaptive_block_size=11,
                                      adaptive_c_value=2,
                                      edge_detector='canny', # 'canny', 'sobel'
                                      blur_method='median', # 'median', 'bilateral'
                                      median_blur_size=51,
                                      canny_threshold_max=140,
                                      canny_threshold_min=30,
                                      sobel_threshold_min=5,
                                      sobel_threshold_max=255,
                                      rho=1,
                                      theta=np.pi/180,
                                      threshold_intersect=250, # For HoughLines
                                      threshold_distance=0.15,
                                      perpendicular_margin=np.pi/18,
                                      detection_method='hough', # 'hough', 'contour', 'shi_tomasi', 'harris', 'hough_lines_p'
                                      contour_area_threshold_ratio=0.05, # For contour detection
                                      shi_tomasi_maxCorners=100, # For Shi-Tomasi
                                      shi_tomasi_qualityLevel=0.01,
                                      shi_tomasi_minDistance=10,
                                      harris_blockSize=2, # For Harris
                                      harris_ksize=3,
                                      harris_k=0.04,
                                      harris_threshold=0.01,
                                      hough_p_threshold=50, # For Probabilistic Hough
                                      hough_p_minLineLength=50,
                                      hough_p_maxLineGap=10,
                                      intermediate=True):

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


        # Perform detection based on the selected method
        corners = None
        edges = None # Initialize edges to None for Hough methods
        thresh = None # Initialize thresh for contour method
        detection_output_img = img.copy() # Image to display for intermediate step, initialized with original image


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

            if edges is not None: # Check if edges were successfully created
                lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
                if lines is not None:
                    lines = filter_perpendicular(lines[:,0], perpendicular_margin)
                    lines = eliminate_duplicates(img, lines, threshold_distance)
                    if len(lines) >= 4:
                        cartesian = to_cartesian(img, lines)
                        intersections = get_intersections(img, cartesian)
                        if len(intersections) >= 4:
                            corners = kmeans_corners(intersections, k=4)
                detection_output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Convert edges to color for display


        elif detection_method == 'contour':
            # Contour detection often works better on thresholded images
            if use_adaptive_thresholding:
                 thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, adaptive_c_value)
            else:
                 _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Use Otsu's thresholding

            document_contours = find_document_contours(thresh, area_threshold_ratio=contour_area_threshold_ratio)
            if document_contours:
                # Assuming the largest contour is the document
                corners = get_corners_from_contours([document_contours[0]])
                # Ensure we have exactly 4 corners, if not, this attempt fails
                if corners is not None and len(corners) != 4:
                    corners = None

            detection_output_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) # Convert thresholded to color for display
            if document_contours: # Draw the detected contour on the output image
                cv2.drawContours(detection_output_img, document_contours, -1, (0, 255, 0), 3)


        elif detection_method == 'shi_tomasi':
            # Apply preprocessing suitable for corner detection before Shi-Tomasi
            # Using blurred image as input for corner detectors often works well
            corners = detect_corners_shi_tomasi(
                blurred,
                maxCorners=shi_tomasi_maxCorners,
                qualityLevel=shi_tomasi_qualityLevel,
                minDistance=shi_tomasi_minDistance
            )
            if corners is not None and len(corners) > 4:
                corners = kmeans_corners(corners, k=4)
            elif corners is not None and len(corners) != 4:
                 corners = None # Ensure exactly 4 corners are found or set to None

            # Visualize corners on the original image for intermediate display
            if corners is not None:
                 for corner in corners:
                     x, y = corner.ravel()
                     cv2.circle(detection_output_img, (int(x), int(y)), 5, (0, 255, 0), -1)


        elif detection_method == 'harris':
             # Apply preprocessing suitable for corner detection before Harris
             # Using blurred image as input for corner detectors often works well
             corners = detect_corners_harris(
                 blurred,
                 blockSize=harris_blockSize,
                 ksize=harris_ksize,
                 k=harris_k,
                 threshold=harris_threshold
             )
             if corners is not None and len(corners) > 4:
                 corners = kmeans_corners(corners, k=4)
             elif corners is not None and len(corners) != 4:
                  corners = None # Ensure exactly 4 corners are found or set to None

             # Visualize corners on the original image for intermediate display
             if corners is not None:
                 for corner in corners:
                     x, y = corner.ravel()
                     cv2.circle(detection_output_img, (int(x), int(y)), 5, (0, 255, 0), -1)


        elif detection_method == 'hough_lines_p':
            # Apply preprocessing suitable for probabilistic Hough
            # Using edges or thresholded image as input often works well
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
                else: # Fallback if no edge detector is specified but use_adaptive_thresholding is False
                     _, edges_for_hough_p = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Use Otsu's on blurred


            lines_p = None
            if edges_for_hough_p is not None:
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
                     if len(intersections_p) >= 4:
                         corners = kmeans_corners(intersections_p, k=4)

            # Visualize detected lines on the original image for intermediate display
            if lines_p is not None:
                for x1, y1, x2, y2 in lines_p:
                    cv2.line(detection_output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)


        if corners is None or len(corners) < 4:
             raise ValueError(f"Detection method '{detection_method}' failed to find 4 corners.")

        corners = reorder_corners(corners)

        h_img, w_img = img.shape[:2]
        min_dim = min(h_img, w_img)
        if h_img > w_img:
            new_h, new_w = int(min_dim), int(min_dim * 0.707)
        else:
            new_h, new_w = int(min_dim * 0.707), int(min_dim)

        destination = np.array([[0,0], [new_w,0], [new_w,new_h], [0,new_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, destination)
        corrected = cv2.warpPerspective(img, M, (new_w, new_h))

        if intermediate:
            # Return the relevant intermediate image based on the detection method
            return gray, blurred, detection_output_img, corrected
        else:
            return corrected


    def correct_perspective_auto_advanced_detection(img, intermediate=True):
        def try_all_tries(image):
            h, w = image.shape[:2]
            max_dim = max(h, w)

            # Define a wider range of parameters and methods to try (Further Reduced for faster execution)
            tries = []

            # Add a baseline set of parameters that are generally effective
            baseline_tries = [
                {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 51, "canny_threshold_max": 150, "canny_threshold_min": 50, "rho": 1, "theta": np.pi/180, "threshold_intersect": 150, "detection_method": 'hough'},
                {"preprocess_method": 'aggressive', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 31, "canny_threshold_max": 100, "canny_threshold_min": 30, "rho": 1, "theta": np.pi/180, "threshold_intersect": 100, "detection_method": 'hough'},
                 {"preprocess_method": 'advanced', "deblur": True, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 61, "canny_threshold_max": 200, "canny_threshold_min": 80, "rho": 1, "theta": np.pi/180, "threshold_intersect": 200, "detection_method": 'hough'},
                {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": True, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 51, "canny_threshold_max": 150, "canny_threshold_min": 50, "rho": 1, "theta": np.pi/180, "threshold_intersect": 150, "detection_method": 'hough'},

                 # Add baseline for contour detection
                {"preprocess_method": 'default', "remove_shadows_flag": False, "use_adaptive_thresholding": False, "blur_method": 'median', "median_blur_size": 51, "detection_method": 'contour', "contour_area_threshold_ratio": 0.05},
                 {"preprocess_method": 'aggressive', "remove_shadows_flag": False, "use_adaptive_thresholding": False, "blur_method": 'median', "median_blur_size": 31, "detection_method": 'contour', "contour_area_threshold_ratio": 0.03},

                # Add baseline for Shi-Tomasi
                {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 5, "detection_method": 'shi_tomasi', "shi_tomasi_maxCorners": 100, "shi_tomasi_qualityLevel": 0.01, "shi_tomasi_minDistance": 10},
                {"preprocess_method": 'advanced', "deblur": True, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 3, "detection_method": 'shi_tomasi', "shi_tomasi_maxCorners": 50, "shi_tomasi_qualityLevel": 0.05, "shi_tomasi_minDistance": 20},

                # Add baseline for Harris
                {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 5, "detection_method": 'harris', "harris_blockSize": 2, "harris_ksize": 3, "harris_k": 0.04, "harris_threshold": 0.01},
                 {"preprocess_method": 'advanced', "deblur": True, "remove_background": False, "remove_shadows_flag": False, "blur_method": 'median', "median_blur_size": 3, "detection_method": 'harris', "harris_blockSize": 3, "harris_ksize": 5, "harris_k": 0.05, "harris_threshold": 0.005},

                 # Add baseline for Probabilistic Hough
                {"preprocess_method": 'default', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 51, "canny_threshold_max": 150, "canny_threshold_min": 50, "rho": 1, "theta": np.pi/180, "hough_p_threshold": 50, "hough_p_minLineLength": 50, "hough_p_maxLineGap": 10, "detection_method": 'hough_lines_p'},
                 {"preprocess_method": 'aggressive', "deblur": False, "remove_background": False, "remove_shadows_flag": False, "use_adaptive_thresholding": False, "edge_detector": 'canny', "blur_method": 'median', "median_blur_size": 31, "canny_threshold_max": 100, "canny_threshold_min": 30, "rho": 1, "theta": np.pi/180, "hough_p_threshold": 80, "hough_p_minLineLength": 80, "hough_p_maxLineGap": 20, "detection_method": 'hough_lines_p'},


            ]
            tries.extend(baseline_tries)

            # Add variations for each parameter and detection method (Further Reduced for faster execution)
            preprocess_methods = ['default', 'advanced'] # Further reduced preprocess methods
            bool_options = [False, True]
            edge_detectors = ['canny', 'sobel']
            blur_methods = ['median', 'bilateral']
            median_blur_sizes = [5, 51] # Further reduced blur sizes
            canny_threshold_max_values = [100, 200] # Further reduced Canny thresholds
            canny_threshold_min_values = [20, 40]
            sobel_threshold_min_values = [10] # Further reduced Sobel thresholds
            sobel_threshold_max_values = [150]
            rho_values = [1]
            theta_values = [np.pi/180]
            threshold_intersect_values = [150] # Further reduced Hough intersect thresholds
            detection_methods = ['hough', 'contour', 'shi_tomasi', 'harris', 'hough_lines_p']
            contour_area_threshold_ratios = [0.05] # Further reduced contour area thresholds
            shi_tomasi_maxCorners_values = [100] # Further reduced Shi-Tomasi parameters
            shi_tomasi_qualityLevel_values = [0.01]
            shi_tomasi_minDistance_values = [10]
            harris_threshold_values = [0.01] # Further reduced Harris parameters
            hough_p_threshold_values = [80] # Further reduced Probabilistic Hough parameters
            hough_p_minLineLength_values = [50]
            hough_p_maxLineGap_values = [10]


            for preprocess_method in preprocess_methods:
                for deblur in bool_options:
                    for remove_background in bool_options:
                        for remove_shadows_flag in bool_options:
                            for use_adaptive_thresholding in bool_options:
                                for blur_method in blur_methods:
                                    for median_blur_size in median_blur_sizes:
                                        for detection_method in detection_methods:
                                            if detection_method == 'hough':
                                                for edge_detector in edge_detectors:
                                                    for rho in rho_values:
                                                        for theta in theta_values:
                                                            for threshold_intersect in threshold_intersect_values:
                                                                if edge_detector == 'canny' and not use_adaptive_thresholding:
                                                                    for canny_threshold_max in canny_threshold_max_values:
                                                                        for canny_threshold_min in canny_threshold_min_values:
                                                                            if canny_threshold_min < canny_threshold_max:
                                                                                params = {
                                                                                    "preprocess_method": preprocess_method,
                                                                                    "deblur": deblur,
                                                                                    "remove_background": remove_background,
                                                                                    "remove_shadows_flag": remove_shadows_flag,
                                                                                    "use_adaptive_thresholding": use_adaptive_thresholding,
                                                                                    "edge_detector": edge_detector,
                                                                                    "blur_method": blur_method,
                                                                                    "median_blur_size": median_blur_size,
                                                                                    "canny_threshold_max": canny_threshold_max,
                                                                                    "canny_threshold_min": canny_threshold_min,
                                                                                    "rho": rho,
                                                                                    "theta": theta,
                                                                                    "threshold_intersect": threshold_intersect,
                                                                                    "detection_method": detection_method
                                                                                }
                                                                                if params not in tries:
                                                                                    tries.append(params)
                                                                elif edge_detector == 'sobel' and not use_adaptive_thresholding:
                                                                    for sobel_threshold_min in sobel_threshold_min_values:
                                                                        for sobel_threshold_max in sobel_threshold_max_values:
                                                                            if sobel_threshold_min < sobel_threshold_max:
                                                                                 params = {
                                                                                    "preprocess_method": preprocess_method,
                                                                                    "deblur": deblur,
                                                                                    "remove_background": remove_background,
                                                                                    "remove_shadows_flag": remove_shadows_flag,
                                                                                    "use_adaptive_thresholding": use_adaptive_thresholding,
                                                                                    "edge_detector": edge_detector,
                                                                                    "blur_method": blur_method,
                                                                                    "median_blur_size": median_blur_size,
                                                                                    "sobel_threshold_min": sobel_threshold_min,
                                                                                    "sobel_threshold_max": sobel_threshold_max,
                                                                                    "rho": rho,
                                                                                    "theta": theta,
                                                                                    "threshold_intersect": threshold_intersect,
                                                                                    "detection_method": detection_method
                                                                                }
                                                                                 if params not in tries:
                                                                                    tries.append(params)
                                                                elif use_adaptive_thresholding:
                                                                     params = {
                                                                        "preprocess_method": preprocess_method,
                                                                        "deblur": deblur,
                                                                        "remove_background": remove_background,
                                                                        "remove_shadows_flag": remove_shadows_flag,
                                                                        "use_adaptive_thresholding": use_adaptive_thresholding,
                                                                        "edge_detector": edge_detector, # Ignored for adaptive
                                                                        "blur_method": blur_method,
                                                                        "median_blur_size": median_blur_size,
                                                                        "rho": rho,
                                                                        "theta": theta,
                                                                        "threshold_intersect": threshold_intersect,
                                                                        "detection_method": detection_method
                                                                    }
                                                                     if params not in tries:
                                                                        tries.append(params)
                                            elif detection_method == 'contour':
                                                 for contour_area_threshold_ratio in contour_area_threshold_ratios:
                                                    params = {
                                                        "preprocess_method": preprocess_method,
                                                        "deblur": deblur,
                                                        "remove_background": remove_background,
                                                        "remove_shadows_flag": remove_shadows_flag,
                                                        "use_adaptive_thresholding": use_adaptive_thresholding,
                                                        "blur_method": blur_method,
                                                        "median_blur_size": median_blur_size,
                                                        "detection_method": detection_method,
                                                        "contour_area_threshold_ratio": contour_area_threshold_ratio
                                                    }
                                                    if params not in tries:
                                                         tries.append(params)
                                            elif detection_method == 'shi_tomasi':
                                                 for shi_tomasi_maxCorners in shi_tomasi_maxCorners_values:
                                                     for shi_tomasi_qualityLevel in shi_tomasi_qualityLevel_values:
                                                         for shi_tomasi_minDistance in shi_tomasi_minDistance_values:
                                                             params = {
                                                                 "preprocess_method": preprocess_method,
                                                                 "deblur": deblur,
                                                                 "remove_background": remove_background,
                                                                 "remove_shadows_flag": remove_shadows_flag,
                                                                 "use_adaptive_thresholding": use_adaptive_thresholding,
                                                                 "blur_method": blur_method,
                                                                 "median_blur_size": median_blur_size,
                                                                 "detection_method": detection_method,
                                                                 "shi_tomasi_maxCorners": shi_tomasi_maxCorners,
                                                                 "shi_tomasi_qualityLevel": shi_tomasi_qualityLevel,
                                                                 "shi_tomasi_minDistance": shi_tomasi_minDistance
                                                             }
                                                             if params not in tries:
                                                                  tries.append(params)

                                            elif detection_method == 'harris':
                                                for harris_threshold in harris_threshold_values:
                                                     params = {
                                                         "preprocess_method": preprocess_method,
                                                         "deblur": deblur,
                                                         "remove_background": remove_background,
                                                         "remove_shadows_flag": remove_shadows_flag,
                                                         "use_adaptive_thresholding": use_adaptive_thresholding,
                                                         "blur_method": blur_method,
                                                         "median_blur_size": median_blur_size,
                                                         "detection_method": detection_method,
                                                         "harris_threshold": harris_threshold
                                                     }
                                                     if params not in tries:
                                                          tries.append(params)

                                            elif detection_method == 'hough_lines_p':
                                                for hough_p_threshold in hough_p_threshold_values:
                                                    for hough_p_minLineLength in hough_p_minLineLength_values:
                                                        for hough_p_maxLineGap in hough_p_maxLineGap_values:
                                                             params = {
                                                                 "preprocess_method": preprocess_method,
                                                                 "deblur": deblur,
                                                                 "remove_background": remove_background,
                                                                 "remove_shadows_flag": remove_shadows_flag,
                                                                 "use_adaptive_thresholding": use_adaptive_thresholding,
                                                                 "edge_detector": edge_detector,
                                                                 "blur_method": blur_method,
                                                                 "median_blur_size": median_blur_size,
                                                                 "rho": rho,
                                                                 "theta": theta,
                                                                 "detection_method": detection_method,
                                                                 "hough_p_threshold": hough_p_threshold,
                                                                 "hough_p_minLineLength": hough_p_minLineLength,
                                                                 "hough_p_maxLineGap": hough_p_maxLineGap
                                                             }
                                                             if params not in tries:
                                                                  tries.append(params)


            print(f"Toplam {len(tries)} deneme yapılacak.")

            for i, params in enumerate(tries):
                print(f"\n🔁 Deneme {i+1}/{len(tries)}: {params}")
                try:
                    result = image.correct_perspective_enhanced(
                        image,
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
                        hough_p_maxLineGap=params.get("hough_p_maxLineGap", 10),
                        intermediate=intermediate
                    )
                    print("✅ Başarılı!")
                    return result
                except Exception as e:
                    print(f"⛔ Deneme {i+1}/{len(tries)} başarısız oldu: {e}")
                    continue
            raise RuntimeError("Tüm denemeler başarısız oldu.")


        def run(self):
            img = Image.get_frame(img=self.image, redis_db=self.redis_db)
            if img is None or img.value is None:
                raise ValueError("No input image provided or failed to load.")

            src_img = self._prepare_image(img.value)

            # Perspektif düzeltmeyi uygula
            warped, corrected_boxes, src_quad, (out_w, out_h) = self._apply_perspective(src_img)

            # Güncellenen görüntüyü pakete set et
            img.value = warped
            self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)


            # Yanıt için context hazırla
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