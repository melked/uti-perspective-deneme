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
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**self.request.data)

        self.perspective_mode = self.request.get_param("PhotoTypeMode") or "Auto"
        self.keep_side = self.request.get_param("KeepSide") or False
        self.output_width = self.request.get_param("OutputWidth") or 800
        self.output_height = self.request.get_param("OutputHeight") or 600
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _auto_detect_corners(self, img: np.ndarray):
        """
        Görüntüde en büyük dörtgen alanı bulur ve köşe noktalarını döndürür.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        h, w = img.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    def _apply_perspective(self, img: np.ndarray):
        src_points = self._auto_detect_corners(img)
        dst_points = np.array([
            [0, 0],
            [self.output_width - 1, 0],
            [self.output_width - 1, self.output_height - 1],
            [0, self.output_height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
        warped = cv2.warpPerspective(img, matrix, (self.output_width, self.output_height))
        return warped

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self._apply_perspective(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        package_model = build_response(context=self)
        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
