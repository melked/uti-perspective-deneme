from pydantic import Field, validator
from typing import List, Optional, Literal, Union
from sdks.novavision.src.base.model import (
    Package, Image, Inputs, Configs, Outputs,
    Response, Request, Output, Input, Config
)


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


class KeepSideBBox(Config):
    name: Literal["KeepSide"] = "KeepSide"
    value: bool = False  # Auto modda genelde kapalı olabilir
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Keep Sides"


class OutputWidth(Config):
    name: Literal["OutputWidth"] = "OutputWidth"
    value: int = Field(default=800, ge=100, le=4096)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Output Width (px)"


class OutputHeight(Config):
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
    value: PerspectiveTransformationExecutor  # Union değil artık
    type: Literal["executor"] = "executor"
    field: Literal["option"] = "option"  # ✅ en kritik değişiklik burada

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
