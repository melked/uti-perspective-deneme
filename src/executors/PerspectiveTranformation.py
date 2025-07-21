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
    Gelişmiş Auto Perspective Correction Executor.
    Çeşitli threshold ve kontur skorlaması ile en uygun dörtgeni bulur.
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
        """BGR formatına ve uint8 aralığına normalize eder."""
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
        """Kontur araması için resmi küçült, ölçek faktörünü döndür."""
        h, w = img.shape[:2]
        scale = 1.0
        m = max(h, w)
        if m > max_dim:
            scale = max_dim / float(m)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img_small, scale
        return img, scale

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Köşe noktalarını TL, TR, BR, BL şeklinde sırala."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def _quad_from_min_area_rect(self, c):
        """4 nokta çıkmazsa MinimumAreaRect'ten köşe üret."""
        rot_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rot_rect)
        return np.array(box, dtype="float32")

    def _binarize_variants(self, gray_small):
        """Normal ve invert threshold çıktıları döndür."""
        th = cv2.adaptiveThreshold(
            gray_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return [th, cv2.bitwise_not(th)]

    def _score_contour(self, contour, img_area, target_aspect=None):
        """Konturu alan, dörtgen yakınlığı ve aspect'e göre puanlar."""
        area = cv2.contourArea(contour)
        area_score = area / img_area
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        quad_score = 1.0 if len(approx) == 4 else 0.5 if len(approx) < 10 else 0.1
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h) if h > 0 else 0.0
        aspect_score = 1.0
        if target_aspect:
            aspect_score = 1.0 / (1.0 + abs(aspect - target_aspect))
        return area_score * 0.6 + quad_score * 0.3 + aspect_score * 0.1

    def _compute_output_size(self, src_quad):
        """keep_side=True ise, hedef boyut kaynak oranına göre ayarlanır."""
        if not self.keep_side:
            return self.output_width, self.output_height
        (tl, tr, br, bl) = src_quad
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        src_w = max(width_top, width_bottom)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        src_h = max(height_left, height_right)
        if src_w == 0 or src_h == 0:
            return self.output_width, self.output_height
        scale = self.output_width / src_w
        out_w = int(round(src_w * scale))
        out_h = int(round(src_h * scale))
        return out_w, out_h


    def _detect_document_corners(self, img: np.ndarray) -> np.ndarray:
        img_prep = self._prepare_image(img)
        small, scale = self._resize_for_detection(img_prep)
        h_s, w_s = small.shape[:2]
        img_area_s = float(h_s * w_s)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_variants = self._binarize_variants(gray)

        best_quad = None
        best_score = -1
        target_aspect = self.output_width / float(self.output_height)

        for th in thresh_variants:
            edges = cv2.Canny(th, 50, 150)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in contours:
                score = self._score_contour(c, img_area_s, target_aspect)
                if score > best_score:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        best_quad = approx.reshape(4, 2).astype("float32")
                    else:
                        best_quad = self._quad_from_min_area_rect(c)
                    best_score = score

        if best_quad is not None:
            quad = best_quad / scale
            return self._order_points(quad.astype("float32"))

        H, W = img_prep.shape[:2]
        return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")


    def _apply_perspective_auto(self, img: np.ndarray):
        src = self._detect_document_corners(img)
        out_w, out_h = self._compute_output_size(src)
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (out_w, out_h))
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
