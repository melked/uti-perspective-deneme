import os
import sys
import cv2
import numpy as np
from typing import Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel
from components.PerspectiveTransformation.src.utils.response import build_response

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Dört köşeyi TL, TR, BR, BL sıralamasına sok."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _quad_from_min_area_rect(contour: np.ndarray) -> np.ndarray:
    """Konturden döndürülmüş (rotated) dikdörtgen kutu çıkar."""
    rot_rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rot_rect)
    return np.array(box, dtype="float32")

class PerspectiveTransformation(Component):
    """
    Tek dörtgen, tam otomatik perspektif düzeltme.

    - CLAHE + adaptif threshold + Canny + morphology ile zor sahalarda belge tespiti.
    - En iyi kontur alan / dörtgene benzerlik / hedef aspect skoruyla seçilir.
    - keep_side=True → kaynak oranı korunur (tarayıcı davranışı).
    - keep_side=False → OutputWidth/Height'e warp (gerekirse stretch).
    - Dörtgen bulunamazsa fallback: tüm görüntü (keep_side=True → kendi oranı, False → resize).
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.mode = self.request.get_param("PerspectiveTypeMode") or "Auto"

        raw_keep = self.request.get_param("KeepSide")
        if raw_keep is None:
            raw_keep = self.request.get_param("drawBBox")
        self.keep_side = self._resolve_bool(raw_keep)

        self.output_width = int(self.request.get_param("OutputWidth") or 800)
        self.output_height = int(self.request.get_param("OutputHeight") or 600)

        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    @staticmethod
    def _resolve_bool(v: Any) -> bool:
        """
        UI katmanından gelebilecek farklı tipleri gerçek bool'a çevir.
        Örnek: True, "True", 1, {"value": {"value": True}}, vs.
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "enable", "enabled")
        if isinstance(v, dict):

            inner = v.get("value", None)
            return PerspectiveTransformation._resolve_bool(inner)
        return False

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """Girdiyi BGR uint8'e normalize et."""
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _resize_for_detection(self, img: np.ndarray, max_dim: int = 1000) -> tuple[np.ndarray, float]:
        """
        Kontur tespiti hızlansın diye büyük görüntüleri küçült.
        Orijinal koordinata dönmek için scale döndür.
        """
        h, w = img.shape[:2]
        m = max(h, w)
        if m > max_dim:
            scale = max_dim / float(m)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img_small, scale
        return img, 1.0

    def _preprocess_for_edges(self, gray: np.ndarray) -> np.ndarray:
        """Blur + CLAHE."""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        return enhanced

    def _threshold_variants(self, enhanced_gray: np.ndarray) -> list[np.ndarray]:
        """
        Hem normal hem invert adaptif threshold üret.
        Böylece açık zemin / açık belge durumlarında şans artar.
        """
        th = cv2.adaptiveThreshold(
            enhanced_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return [th, cv2.bitwise_not(th)]

    def _score_contour(self, contour: np.ndarray, img_area: float, target_aspect: float) -> tuple[float, Optional[np.ndarray]]:
        """
        Konturu puanla: alan (önemli), dörtgene yakınlık, hedef aspect'e yakınlık.
        Dörtgen approx dönebilir.
        """
        area = cv2.contourArea(contour)
        if area <= 0:
            return -1.0, None
        area_score = area / img_area
        if area_score < 0.01:  # çok küçükleri at
            return -1.0, None

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        quad_score = 1.0 if len(approx) == 4 else 0.3

        x, y, w, h = cv2.boundingRect(contour)
        aspect = (w / float(h)) if h > 0 else 0.0
        aspect_score = 1.0 / (1.0 + abs(aspect - target_aspect))

        score = area_score * 0.6 + quad_score * 0.3 + aspect_score * 0.1
        return score, approx if len(approx) == 4 else None

    def _detect_document_corners(self, img: np.ndarray) -> np.ndarray:
        """
        En uygun dörtgeni bul; yoksa tüm görüntüyü döndür.
        """
        img_prep = self._prepare_image(img)
        small, scale = self._resize_for_detection(img_prep)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        enhanced = self._preprocess_for_edges(gray)
        thresh_list = self._threshold_variants(enhanced)

        area_small = float(small.shape[0] * small.shape[1])
        target_aspect = self.output_width / float(self.output_height)

        best_score = -1.0
        best_quad = None

        for th in thresh_list:
            edges = cv2.Canny(th, 50, 150)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in contours:
                score, approx = self._score_contour(c, area_small, target_aspect)
                if score > best_score:
                    best_score = score
                    if approx is not None:
                        best_quad = approx.reshape(4, 2).astype("float32")
                    else:
                        best_quad = _quad_from_min_area_rect(c)

        if best_quad is not None:
            quad = best_quad / scale
            return _order_points(quad.astype("float32"))

        H, W = img_prep.shape[:2]
        return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")


    def _compute_output_size(self, src_quad: np.ndarray) -> tuple[int, int]:
        """
        keep_side=True → kaynak dörtgen oranına göre ölçekli çıktı (genişliği user config'ine göre referans alabiliriz).
        keep_side=False → config'teki width/height'i kullan.
        """
        if not self.keep_side:
            return self.output_width, self.output_height

        (tl, tr, br, bl) = src_quad
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        src_w = max(w_top, w_bot)

        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        src_h = max(h_left, h_right)

        if src_w <= 0 or src_h <= 0:
            return self.output_width, self.output_height

        scale = self.output_width / src_w
        out_w = int(round(src_w * scale))
        out_h = int(round(src_h * scale))
        return out_w, out_h

    def _apply_perspective(self, img: np.ndarray) -> np.ndarray:
        src = self._detect_document_corners(img)
        out_w, out_h = self._compute_output_size(src)

        H, W = img.shape[:2]
        full_src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
        if (not self.keep_side) and np.allclose(src, full_src, atol=1.0):
            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC)

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img is None or img.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img.value)
        warped = self._apply_perspective(src_img)

        img.value = warped
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
