from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class KeepSideFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class KeepSideTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class KeepSideBBox(Config):
    """
    output ölçüleri için.
    """
    name: Literal["KeepSide"] = "KeepSide"
    value: Union[KeepSideTrue, KeepSideFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Keep Sides"


class OutputWidth(Config):
    """
    Output image width in pixels.
    Minimum 100, maximum 4096.
    """
    name: Literal["OutputWidth"] = "OutputWidth"
    value: int = Field(default=800, ge=100, le=4096)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Output Width (px)"


class OutputHeight(Config):
    """
    Output image height in pixels.
    Minimum 100, maximum 4096.
    """
    name: Literal["OutputHeight"] = "OutputHeight"
    value: int = Field(default=600, ge=100, le=4096)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Output Height (px)"


class PerspectiveTransformationInputs(Inputs):
    inputImage: InputImage


class PerspectiveTransformationConfigs(Configs):
    drawBBox: KeepSideBBox
    outputWidth: OutputWidth
    outputHeight: OutputHeight


class PerspectiveTransformationOutputs(Outputs):
    outputImage: OutputImage


class PerspectiveTransformationRequest(Request):
    inputs: Optional[PerspectiveTransformationInputs]
    configs: PerspectiveTransformationConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PerspectiveTransformationResponse(Response):
    outputs: PerspectiveTransformationOutputs


class PerspectiveTransformationExecutor(Config):
    name: Literal["PerspectiveTransformation"] = "PerspectiveTransformation"
    value: Union[PerspectiveTransformationRequest, PerspectiveTransformationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "PerspectiveTransformation"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: PerspectiveTransformationExecutor
    type: Literal["executor"] = "executor"
    field: Literal["option"] = "option"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["PerspectiveTransformation"] = "PerspectiveTransformation"
