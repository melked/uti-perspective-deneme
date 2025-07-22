from sdks.novavision.src.helper.package import PackageHelper
from components.PerspectiveTransformation.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    PerspectiveTransformationOutputs,
    PerspectiveTransformationResponse,
    PerspectiveTransformationExecutor,
    OutputImage
)

def build_response(context):
    output_image = OutputImage(value=context.image)
    outputs = PerspectiveTransformationOutputs(outputImage=output_image)
    perspective_response = PerspectiveTransformationResponse(outputs=outputs)
    perspective_executor = PerspectiveTransformationExecutor(value=perspective_response)
    executor = ConfigExecutor(value=perspective_executor)
    package_configs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    return package.build_model(context)