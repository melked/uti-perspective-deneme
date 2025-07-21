
from sdks.novavision.src.helper.package import PackageHelper
from components.PerspectiveTransformatio.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PerspectiveTransformationOutputs, PerspectiveTransformationResponse, PerspectiveTransformationExecutor, OutputImage


def build_response(context):
    outputImage = OutputImage(value=context.image)
    Outputs = PerspectiveTransformationOutputs(outputImage=outputImage)
    perspectiveTransformationeResponse = PerspectiveTransformationResponse(outputs=Outputs)
    perspectiveTransformationExecutor = PerspectiveTransformationExecutor(value=perspectiveTransformationResponse)
    executor = ConfigExecutor(value=perspectiveTransformationExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel