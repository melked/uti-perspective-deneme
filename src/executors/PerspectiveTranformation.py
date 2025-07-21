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
    Auto perspective correction with edge enhancement.
    Detects document-like quadrilateral and warps it.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.output_width = self.request.get_param("OutputWidth") or 800
        self.output_height = self.request.get_param("OutputHeight") or 600
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """BGR formatına ve uint8 aralığına getir."""
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
        """Hızlı kontur tespiti için küçült."""
        h, w = img.shape[:2]
        scale = 1.0
        m = max(h, w)
        if m > max_dim:
            scale = max_dim / float(m)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img_small, scale
        return img, scale

    def _highlight_edges(self, img: np.ndarray) -> np.ndarray:
        """Kenarları daha belirgin hale getirir."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Gradyan tabanlı kenar belirginleştirme
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))

        edges = cv2.Canny(gradient, 50, 150)

        # Kenarları kalınlaştır
        kernel = np.ones((5, 5), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=2)
        return thick_edges

    def _largest_quad_from_contours(self, contours):
        """En büyük 4 köşe konturu bul."""
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")
        return None

    def _quad_from_min_area_rect(self, c):
        """Fallback: 4 nokta çıkaramazsak min area rect."""
        rot_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rot_rect)
        return np.array(box, dtype="float32")

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Köşeleri TL, TR, BR, BL sıralar."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def _detect_document_corners(self, img: np.ndarray) -> np.ndarray:
        img_prep = self._prepare_image(img)
        small, scale = self._resize_for_detection(img_prep)

        thick_edges = self._highlight_edges(small)

        contours, _ = cv2.findContours(thick_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        quad_small = self._largest_quad_from_contours(contours)
        if quad_small is None and contours:
            quad_small = self._quad_from_min_area_rect(contours[0])

        if quad_small is not None:
            quad = quad_small / scale
            return self._order_points(quad.astype("float32"))

        # fallback: tüm resmi al
        h, w = img_prep.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

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
