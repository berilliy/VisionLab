from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ObjectClass = Literal["book", "bottle", "clock", "keyboard"]
ModelId = Literal["resnet18", "dinov2"]
ModelSelection = Literal["resnet18", "dinov2", "both"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ClassProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book: Probability
    bottle: Probability
    clock: Probability
    keyboard: Probability


class PredictionResult(BaseModel):
    predicted_class: ObjectClass
    confidence: Probability
    probabilities: ClassProbabilities
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_scores(self):
        values = self.probabilities.model_dump()
        if abs(sum(values.values()) - 1) > 1e-5:
            raise ValueError("Model probabilities must sum to 1.")
        if abs(values[self.predicted_class] - self.confidence) > 1e-6:
            raise ValueError("Confidence does not match the predicted class.")
        if abs(max(values.values()) - self.confidence) > 1e-6:
            raise ValueError("Confidence must be the largest class probability.")
        return self


class NearestNeighbor(BaseModel):
    image_url: str
    class_name: ObjectClass
    similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    results: dict[ModelId, PredictionResult]
    nearest_neighbors: list[NearestNeighbor] = Field(min_length=3, max_length=3)
    device: Literal["cpu"] = "cpu"
    runtime: str


class HealthResponse(BaseModel):
    status: Literal["ok", "error", "loading"]
    device: Literal["cpu"] = "cpu"
    resnet18_loaded: bool
    dinov2_loaded: bool
    classifier_loaded: bool
    neighbors_ready: bool
    training_images: int = 0
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelId] = ["resnet18", "dinov2"]
    selections: list[ModelSelection] = ["resnet18", "dinov2", "both"]
    classes: list[ObjectClass] = ["book", "bottle", "clock", "keyboard"]
    readiness: HealthResponse
