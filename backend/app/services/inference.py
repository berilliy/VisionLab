import logging
import threading
from time import perf_counter

import numpy as np
import torch

from ..config import CLASSES, Settings
from ..schemas import HealthResponse, PredictionResult, PredictionResponse
from .model_loader import ModelBundle
from .neighbors import NeighborIndex

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = ModelBundle(settings)
        self.neighbors = NeighborIndex(settings, self.models)
        self.lock = threading.Lock()
        self.error: str | None = None
        self.initialized = False

    def initialize(self):
        try:
            self.models.load()
            self.neighbors.initialize()
            self.initialized = True
            logger.info("VisionLab ready on CPU with %d training neighbors", len(self.neighbors.entries))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok" if self.initialized else "error" if self.error else "loading",
                              resnet18_loaded=self.models.resnet is not None,
                              dinov2_loaded=self.models.dino is not None,
                              classifier_loaded=self.models.classifier is not None,
                              neighbors_ready=self.neighbors.ready,
                              training_images=len(self.neighbors.entries), error=self.error)

    @staticmethod
    def _result(probabilities, elapsed_ms: float) -> PredictionResult:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise RuntimeError("Model returned invalid probabilities.")
        best = int(values.argmax())
        return PredictionResult(predicted_class=CLASSES[best], confidence=float(values[best]),
                                probabilities=dict(zip(CLASSES, values.tolist())), latency_ms=elapsed_ms)

    def predict(self, image, model: str) -> PredictionResponse:
        if not self.initialized:
            raise RuntimeError(self.error or "Models and neighbor index are not ready.")
        if model not in {"resnet18", "dinov2", "both"}:
            raise ValueError("Select resnet18, dinov2, or both.")
        results = {}
        if model in {"resnet18", "both"}:
            start = perf_counter()
            with torch.inference_mode():
                tensor = self.models.resnet_transform(image).unsqueeze(0)
                probabilities = torch.softmax(self.models.resnet(tensor), dim=1)[0].cpu().numpy()
            results["resnet18"] = self._result(probabilities, (perf_counter() - start) * 1000)

        # Required for retrieval in ALL modes. For DINOv2 classification, reuse
        # this exact query embedding rather than calling the backbone twice.
        start = perf_counter()
        features = self.models.embeddings([image])
        if model in {"dinov2", "both"}:
            raw = self.models.classifier.predict_proba(features)[0]
            probabilities = [raw[self.models.class_order.index(label)] for label in CLASSES]
            results["dinov2"] = self._result(probabilities, (perf_counter() - start) * 1000)
        neighbors = self.neighbors.nearest(features)
        return PredictionResponse(results=results, nearest_neighbors=neighbors,
                                  runtime=f"PyTorch {torch.__version__}")
