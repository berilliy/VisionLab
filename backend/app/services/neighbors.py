import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CLASSES, Settings
from ..schemas import NearestNeighbor
from ..utils.image_utils import decode_image
from .model_loader import ModelBundle, file_digest

logger = logging.getLogger(__name__)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class TrainingImage:
    relative_path: str
    class_name: str
    image_id: str
    sha256: str


class NeighborIndex:
    def __init__(self, settings: Settings, models: ModelBundle):
        self.settings = settings
        self.models = models
        self.entries: list[TrainingImage] = []
        self.mapping: dict[str, TrainingImage] = {}
        self.normalized: np.ndarray | None = None
        self.cache_loaded = False

    @property
    def ready(self) -> bool:
        return self.normalized is not None and len(self.entries) >= 3

    def _scan(self) -> list[TrainingImage]:
        root = self.settings.train_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Missing training dataset: {root}")
        entries = []
        for label in CLASSES:
            folder = root / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing training class directory: {folder}")
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise RuntimeError(f"No training images found in {folder}")
            for path in paths:
                if not path.resolve().is_relative_to(root):
                    raise RuntimeError(f"Training file points outside the training directory: {path}")
                relative = path.relative_to(root).as_posix()
                digest = file_digest(path)
                image_id = hashlib.sha256(f"{relative}:{digest}".encode()).hexdigest()[:32]
                entries.append(TrainingImage(relative, label, image_id, digest))
        if len(entries) < 3:
            raise RuntimeError("Nearest-neighbor retrieval requires at least three training images.")
        return entries

    def initialize(self):
        self.entries = self._scan()
        self.mapping = {entry.image_id: entry for entry in self.entries}
        metadata = json.dumps({"extractor": self.models.embedding_metadata,
                               "dataset": [entry.__dict__ for entry in self.entries]},
                              sort_keys=True, default=str)
        cache_path = self.settings.cache_dir / "train_embeddings.npz"
        features = self._load_cache(cache_path, metadata)
        if features is None:
            logger.info("Building DINOv2 index from %d training images", len(self.entries))
            batches = []
            for offset in range(0, len(self.entries), self.settings.batch_size):
                images = []
                try:
                    for entry in self.entries[offset:offset + self.settings.batch_size]:
                        path = self.settings.train_dir / entry.relative_path
                        images.append(decode_image(path.read_bytes(), enforce_upload_limit=False))
                    batches.append(self.models.embeddings(images))
                except Exception as exc:
                    raise RuntimeError(f"Failed to build training neighbor index near batch offset {offset}: {exc}") from exc
                finally:
                    for image in images:
                        image.close()
                logger.info("Training embeddings: %d/%d", min(offset + self.settings.batch_size, len(self.entries)), len(self.entries))
            features = np.vstack(batches).astype(np.float32)
            self._validate_features(features)
            self._save_cache(cache_path, features, metadata)
        self.normalized = features / np.linalg.norm(features, axis=1, keepdims=True)

    def _validate_features(self, features):
        if features.shape != (len(self.entries), 384) or features.dtype.kind != "f":
            raise ValueError("Embedding cache has incompatible dimensions or dtype.")
        if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) < 1e-10):
            raise ValueError("Embedding cache contains invalid feature vectors.")

    def _load_cache(self, path: Path, metadata: str):
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as cache:
                if str(cache["metadata"].item()) != metadata:
                    logger.info("Training embedding cache is stale; rebuilding")
                    return None
                for key, expected in [("labels", [e.class_name for e in self.entries]),
                                      ("paths", [e.relative_path for e in self.entries]),
                                      ("ids", [e.image_id for e in self.entries])]:
                    if cache[key].tolist() != expected:
                        raise ValueError(f"Embedding cache {key} do not match dataset ordering.")
                features = cache["embeddings"].copy()
                self._validate_features(features)
            self.cache_loaded = True
            logger.info("Loaded %d cached training embeddings", len(features))
            return features
        except Exception as exc:
            logger.warning("Invalid embedding cache; rebuilding: %s", exc)
            return None

    def _save_cache(self, path: Path, features, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, embeddings=features,
                                    labels=np.array([e.class_name for e in self.entries]),
                                    paths=np.array([e.relative_path for e in self.entries]),
                                    ids=np.array([e.image_id for e in self.entries]),
                                    metadata=np.array(metadata))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def nearest(self, features: np.ndarray) -> list[NearestNeighbor]:
        if not self.ready:
            raise RuntimeError("Training neighbor index is not initialized.")
        if features.shape != (1, 384) or not np.isfinite(features).all():
            raise ValueError("Invalid query embedding.")
        norm = np.linalg.norm(features[0])
        if norm < 1e-10:
            raise ValueError("Cannot retrieve neighbors for a zero embedding.")
        similarities = np.clip(self.normalized @ (features[0] / norm), -1, 1)
        indices = np.argsort(-similarities, kind="stable")[:3]
        return [NearestNeighbor(image_url=f"/api/images/{self.entries[i].image_id}",
                                class_name=self.entries[i].class_name,
                                similarity=float(similarities[i])) for i in indices]

    def image_path(self, image_id: str) -> Path:
        entry = self.mapping.get(image_id)
        if entry is None:
            raise KeyError(image_id)
        root = self.settings.train_dir.resolve()
        path = (root / entry.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != entry.sha256:
            raise KeyError(image_id)
        return path
