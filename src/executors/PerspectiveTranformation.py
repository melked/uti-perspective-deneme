import os
import sys
import cv2
import numpy as np
from typing import Any, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect

def _quad_from_min_area_rect(c: np.ndarray) -> np.ndarray:
    rot_rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rot_rect)
    return np.array(box, dtype=np.float32)


class PerspectiveTransformation(Component):
    """
    Tek dörtgen otomatik perspektif düzeltme.

    - Tespit için downscale (hız), warp için *orijinal çözünürlük* (kalite).
    - CLAHE + Canny + morfolojiyle kenar güçlendirme.
    - Skor tabanlı kontur seçimi: alan yüzdesi + dörtgen approx + hedef aspect.
    - Küçük konturlar (logo vb.) elenir; fallback tüm görüntü.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))

        ow = self.request.get_param("OutputWidth")
        oh = self.request.get_param("OutputHeight")
        self.output_width = int(ow) if ow is not None else 800
        self.output_height = int(oh) if oh is not None else 600

        self.image = self.request.get_param("inputImage")

        self.min_area_rel_initial = 0.05
        self.min_area_rel_relaxed = 0.01
        self.aspect_tolerance = 0.35

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

    def _resize_for_detection(self, img: np.ndarray, max_dim: int = 1000) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > max_dim:
            scale = max_dim / float(m)
            small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return small, scale
        return img, 1.0

    def _preprocess_for_edges(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(blur)

    def _edges_from_gray(self, g: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(g, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return edges

    def _score_contour(self, c: np.ndarray, img_area: float, target_aspect: float) -> float:
        area = cv2.contourArea(c)
        if area <= 0:
            return -1.0
        area_score = area / img_area

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        quad_score = 1.0 if len(approx) == 4 else 0.3

        x, y, w, h = cv2.boundingRect(c)
        if h <= 0:
            aspect_score = 0.0
        else:
            c_aspect = w / float(h)
            aspect_score = 1.0 / (1.0 + abs(c_aspect - target_aspect))

        return area_score * 0.6 + quad_score * 0.3 + aspect_score * 0.1

    def _choose_best_quad(self, contours, img_area, min_area_rel, target_aspect) -> Optional[np.ndarray]:
        best_score = -1.0
        best_quad = None

        min_area_abs = img_area * min_area_rel

        for c in contours:
            if cv2.contourArea(c) < min_area_abs:
                continue

            score = self._score_contour(c, img_area, target_aspect)
            if score <= best_score:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
            else:
                quad = _quad_from_min_area_rect(c)

            if target_aspect > 0:
                (tl, tr, br, bl) = _order_points(quad)
                w_top = np.linalg.norm(tr - tl)
                h_left = np.linalg.norm(bl - tl)
                if h_left > 0:
                    src_aspect = w_top / h_left
                    if abs(src_aspect - target_aspect) > (target_aspect * self.aspect_tolerance):
                        score *= 0.5

            if score > best_score:
                best_score = score
                best_quad = quad

        return best_quad

    def _detect_document_corners(self, img: np.ndarray) -> np.ndarray:
        """
        Belge köşelerini döndürür (orijinal boyut koordinatlarında).
        """
        img_prep = self._prepare_image(img)

        small, scale = self._resize_for_detection(img_prep)
        h_s, w_s = small.shape[:2]
        img_area_s = float(h_s * w_s)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        enhanced = self._preprocess_for_edges(gray)
        edges = self._edges_from_gray(enhanced)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        target_aspect = self.output_width / float(self.output_height) if self.output_height > 0 else 1.0

        quad_small = self._choose_best_quad(contours, img_area_s, self.min_area_rel_initial, target_aspect)

        if quad_small is None:
            quad_small = self._choose_best_quad(contours, img_area_s, self.min_area_rel_relaxed, target_aspect)

        if quad_small is None and contours:
            quad_small = _quad_from_min_area_rect(contours[0])

        if quad_small is not None:
            quad_orig = quad_small / scale
            return _order_points(quad_orig.astype(np.float32))

        H, W = img_prep.shape[:2]
        return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    @staticmethod
    def _choose_interp(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
        """Büyütmede LANCZOS4, küçültmede INTER_AREA."""
        if dst_w > src_w or dst_h > src_h:
            return cv2.INTER_LANCZOS4
        else:
            return cv2.INTER_AREA

    def _apply_perspective_auto(self, img: np.ndarray) -> np.ndarray:
        """
        Orijinal çözünürlükte warp.
        """
        src_quad = self._detect_document_corners(img)

        dst_quad = np.array([
            [0, 0],
            [self.output_width - 1, 0],
            [self.output_width - 1, self.output_height - 1],
            [0, self.output_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_quad, dst_quad)

        interp = self._choose_interp(
            img.shape[1], img.shape[0],
            self.output_width, self.output_height
        )

        warped = cv2.warpPerspective(img, M, (self.output_width, self.output_height), flags=interp)
        self.context["src_quad"] = src_quad.tolist()
        return warped

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        warped = self._apply_perspective_auto(src_img)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)

    Executor(sys.argv[1]).run()
