import hashlib
import logging
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModel

from ..config import CLASSES, DINO_REVISION, MODEL_NAME, Settings

logger = logging.getLogger(__name__)


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ModelBundle:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.resnet = None
        self.dino = None
        self.processor = None
        self.classifier = None
        self.class_order: list[str] = []
        self.embedding_metadata: dict = {}
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self):
        torch.set_num_threads(self.settings.torch_threads)
        for label, path in [("ResNet18 checkpoint", self.settings.resnet_path),
                            ("DINOv2 classifier", self.settings.classifier_path)]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        logger.info("Loading trained ResNet18 on CPU")
        resnet = models.resnet18(weights=None)
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(CLASSES))
        try:
            state = torch.load(self.settings.resnet_path, map_location="cpu", weights_only=True)
            resnet.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(f"Incompatible ResNet18 checkpoint: {self.settings.resnet_path}: {exc}") from exc
        self.resnet = resnet.cpu().eval()

        # Trusted, fixed local artifact only. No request can choose a pickle path.
        classifier = joblib.load(self.settings.classifier_path)
        if not isinstance(classifier, Pipeline) or len(classifier.steps) != 2:
            raise RuntimeError("Expected a StandardScaler -> LogisticRegression pipeline.")
        if not isinstance(classifier.steps[0][1], StandardScaler) or not isinstance(classifier.steps[1][1], LogisticRegression):
            raise RuntimeError("Classifier pipeline must contain StandardScaler then LogisticRegression.")
        if getattr(classifier, "n_features_in_", None) != 384:
            raise RuntimeError("DINOv2 classifier must accept 384-dimensional CLS embeddings.")
        actual = classifier.classes_.tolist()
        if actual == [0, 1, 2, 3]:
            self.class_order = list(CLASSES)
        elif len(actual) == 4 and set(actual) == set(CLASSES):
            self.class_order = actual
        else:
            raise RuntimeError(f"Classifier has incompatible classes: {actual}; expected [0,1,2,3] or {list(CLASSES)}.")
        self.classifier = classifier

        snapshot = self._snapshot()
        logger.info("Loading %s at revision %s", MODEL_NAME, DINO_REVISION)
        # AutoImageProcessor matches the notebook's default processor. Its exact
        # class, config, and library version are included in the embedding cache.
        processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        dino = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True, trust_remote_code=False)
        if dino.config.hidden_size != 384 or dino.config.model_type != "dinov2":
            raise RuntimeError("Expected facebook/dinov2-small with hidden_size=384.")
        self.processor = processor
        self.dino = dino.cpu().eval()
        self.embedding_metadata = {
            "schema": 1, "model": MODEL_NAME, "revision": DINO_REVISION,
            "weights_sha256": file_digest(snapshot / "model.safetensors"),
            "model_config_sha256": file_digest(snapshot / "config.json"),
            "processor_class": type(processor).__name__, "processor": processor.to_dict(),
            "transformers": version("transformers"), "torch": version("torch"),
            "torchvision": version("torchvision"), "pillow": version("pillow"),
            "embedding": "last_hidden_state[:,0,:]", "dimension": 384,
            "decode_policy": "exif_transpose-white_alpha-RGB-v1",
        }

    def _snapshot(self) -> Path:
        patterns = ["config.json", "preprocessor_config.json", "model.safetensors"]
        for cache in [None, str(self.settings.cache_dir / "huggingface")]:
            try:
                folder = Path(snapshot_download(MODEL_NAME, revision=DINO_REVISION,
                                               cache_dir=cache, local_files_only=True))
                if all((folder / name).is_file() for name in patterns):
                    return folder
            except (OSError, ValueError):
                pass
        if self.settings.offline:
            raise RuntimeError(f"DINOv2 snapshot {DINO_REVISION} is not cached. Start once with VISIONLAB_OFFLINE=0 to download {MODEL_NAME}.")
        try:
            return Path(snapshot_download(MODEL_NAME, revision=DINO_REVISION,
                        cache_dir=str(self.settings.cache_dir / "huggingface"), allow_patterns=patterns))
        except Exception as exc:
            raise RuntimeError(f"Cannot download {MODEL_NAME}. Check connectivity or populate the Hugging Face cache. {exc}") from exc

    def embeddings(self, images) -> np.ndarray:
        if self.dino is None or self.processor is None:
            raise RuntimeError("DINOv2 is not initialized.")
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            outputs = self.dino(**{key: value.to("cpu") for key, value in inputs.items()})
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        if features.shape != (len(images), 384) or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid DINOv2 embedding shape or values: {features.shape}")
        return features
