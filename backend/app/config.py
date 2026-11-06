"""Paths are resolved from this file, independent of the shell's working directory."""
import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
CLASSES = ("book", "bottle", "clock", "keyboard")
MODEL_NAME = "facebook/dinov2-small"
# Exact revision found in the notebook's local Hugging Face cache.
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 25_000_000


@dataclass(frozen=True)
class Settings:
    resnet_path: Path = PROJECT_ROOT / "models/resnet18_household.pth"
    classifier_path: Path = PROJECT_ROOT / "models/dinov2_classifier.pkl"
    train_dir: Path = PROJECT_ROOT / "dataset/train"
    cache_dir: Path = BACKEND_ROOT / "cache"
    batch_size: int = field(default_factory=lambda: int(os.getenv("VISIONLAB_BATCH_SIZE", "8")))
    torch_threads: int = field(default_factory=lambda: int(os.getenv("VISIONLAB_TORCH_THREADS", "4")))
    offline: bool = field(default_factory=lambda: os.getenv("VISIONLAB_OFFLINE", "0") == "1")
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        origin.strip() for origin in os.getenv(
            "VISIONLAB_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",") if origin.strip()
    ))

    def __post_init__(self):
        if not 1 <= self.batch_size <= 32:
            raise ValueError("VISIONLAB_BATCH_SIZE must be between 1 and 32.")
        if not 1 <= self.torch_threads <= 32:
            raise ValueError("VISIONLAB_TORCH_THREADS must be between 1 and 32.")
        if "*" in self.cors_origins:
            raise ValueError("Configure explicit frontend CORS origins, not '*'.")
