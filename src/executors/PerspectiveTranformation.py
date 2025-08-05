import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


class PerspectiveTransformation(Component):
    """
    Auto perspective correction executor.
    Detects a document-like quadrilateral and warps it to a target size.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.perspective_mode = self.request.get_param("PerspectiveTypeMode") or "Auto"
        self.keep_side = self.request.get_param("KeepSide") or False
        self.output_width = self.request.get_param("OutputWidth") or 800
        self.output_height = self.request.get_param("OutputHeight") or 600
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _resize_for_detection(self, img: np.ndarray, max_dim: int = 800):
        h, w = img.shape[:2]
        scale = 1.0
        m = max(h, w)
        if m > max_dim:
            scale = max_dim / float(m)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img_small, scale
        return img, scale

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def _largest_quad_from_contours(self, contours):
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")
        return None

    def _quad_from_min_area_rect(self, c):
        rot_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rot_rect)
        return np.array(box, dtype="float32")

    def _aspect_ratio(self, quad):
        (tl, tr, br, bl) = quad
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        width = (widthA + widthB) / 2.0
        height = (heightA + heightB) / 2.0
        if height == 0:
            return 1
        return width / height

    def _line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        return [px, py]

    def _detect_document_corners(self, img: np.ndarray) -> np.ndarray:
        img_prep = self._prepare_image(img)
        small, scale = self._resize_for_detection(img_prep)
        img_area = img_prep.shape[0] * img_prep.shape[1]

        def preprocess_variant(gray_img, aggressive=False):
            gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
            if aggressive:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gray = np.clip((gray / 255.0) ** 0.8 * 255, 0, 255).astype(np.uint8)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)
            else:
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                thresh = cv2.bitwise_not(thresh)
                edges = cv2.Canny(thresh, 30, 120)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            return edges

        edge_maps = [
            preprocess_variant(small, aggressive=False),
            preprocess_variant(small, aggressive=True)
        ]

        best_quad = None
        best_score = 0
        last_contours = []

        # 1) Kontur tabanlı tespit
        for edges in edge_maps:
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            last_contours = contours if contours else last_contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in contours[:10]:
                quad_small = self._largest_quad_from_contours([c])
                if quad_small is None:
                    quad_small = self._quad_from_min_area_rect(c)
                if quad_small is not None:
                    quad = quad_small / scale
                    quad = self._order_points(quad.astype("float32"))
                    area_score = cv2.contourArea(quad) / img_area

                    # Tam resmi alma engeli
                    if area_score > 0.90:
                        continue

                    desired_ar = self.output_width / self.output_height
                    quad_ar = self._aspect_ratio(quad)
                    ratio_score = 1 - min(abs(quad_ar - desired_ar), 1)
                    score = area_score * 0.7 + ratio_score * 0.3
                    if score > best_score:
                        best_score = score
                        best_quad = quad

        if best_quad is not None:
            return best_quad

        # 2) Çizgi tabanlı fallback
        def hough_lines_corners(edges_img):
            lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
            if lines is None:
                return None
            lines = lines[:, 0, :]
            intersections = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    ang1 = np.arctan2(lines[i][3] - lines[i][1], lines[i][2] - lines[i][0])
                    ang2 = np.arctan2(lines[j][3] - lines[j][1], lines[j][2] - lines[j][0])
                    if abs(abs(ang1 - ang2) - np.pi / 2) < np.deg2rad(15):
                        pt = self._line_intersection(lines[i], lines[j])
                        if pt is not None:
                            intersections.append(pt)
            if len(intersections) >= 4:
                pts = np.array(intersections, dtype="float32")
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                return np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype="float32")
            return None

        for edges in edge_maps:
            quad_lines = hough_lines_corners(edges)
            if quad_lines is not None:
                quad = quad_lines / scale
                return self._order_points(quad.astype("float32"))

        # 3) minAreaRect fallback
        if last_contours:
            quad_small = self._quad_from_min_area_rect(last_contours[0])
            quad = quad_small / scale
            return self._order_points(quad.astype("float32"))

        # 4) Tam resim fallback
        h, w = img_prep.shape[:2]
        return np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")

    def _apply_perspective_auto(self, img: np.ndarray):
        src = self._detect_document_corners(img)
        dst = np.array([
            [0, 0],
            [self.output_width - 1, 0],
            [self.output_width - 1, self.output_height - 1],
            [0, self.output_height - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (self.output_width, self.output_height))
        return warped

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img is None or img.value is None:
            raise ValueError("No input image provided or failed to load.")
        warped = self._apply_perspective_auto(img.value)
        img.value = warped
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        package_model = build_response(context=self)
        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
