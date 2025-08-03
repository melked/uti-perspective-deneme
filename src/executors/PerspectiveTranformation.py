import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        
        self.image = self.request.get_param("inputImage")

        # Güvenli config alma
        self.perspective_mode = self._safe_get_config("PerspectiveTypeMode", key_chain=["value", "name"])
        self.keep_side = self._safe_get_config("KeepSide", key_chain=["value"])
        self.output_width = self.request.get_param("OutputWidth")
        self.output_height = self.request.get_param("OutputHeight")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _safe_get_config(self, key, key_chain=None):
        value = self.request.get_param(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass
        if key_chain:
            for k in key_chain:
                value = value[k]
        return value

    def correct_perspective_auto(self, image):
        # 1. Gri + CLAHE + gamma düzeltme + sharp
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        gamma = 1.5
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
        clahe_gamma = cv2.LUT(clahe, table)

        blurred = cv2.GaussianBlur(clahe_gamma, (3, 3), 0)
        sharp = cv2.addWeighted(clahe_gamma, 1.5, blurred, -0.5, 0)

        # 2. Canny kenar tespiti
        edges = cv2.Canny(sharp, 50, 150)

        # 3. Contour bulma
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                break
        else:
            return image  # dört köşe bulunamadıysa orijinali döndür

        # 4. Köşeleri sırala
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # 5. Perspektif düzeltme
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)

        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def resize_output(self, image):
        if self.keep_side:
            return image
        else:
            return cv2.resize(image, (self.output_width, self.output_height))

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        original = img.value

        if self.perspective_mode == "Auto":
            corrected = self.correct_perspective_auto(original)
        else:
            corrected = self.correct_perspective_auto(original)  # şimdilik aynı, ileride advanced ayrılabilir

        final = self.resize_output(corrected)
        img.value = final

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()