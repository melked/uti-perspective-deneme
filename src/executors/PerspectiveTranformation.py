import cv2
import numpy as np
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.output_width = self.request.get_param("OutputWidth") or 400
        self.output_height = self.request.get_param("OutputHeight") or 600
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):

        img = Image.get_frame(img=self.image, redis_db=self.redis_db)

        pts1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
        pts2 = np.float32(
            [[0, 0], [self.output_width, 0], [0, self.output_height], [self.output_width, self.output_height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        warped = cv2.warpPerspective(img.value, matrix, (self.output_width, self.output_height))

        img.value = warped
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        package_model = build_response(context=self)
        return package_model


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
