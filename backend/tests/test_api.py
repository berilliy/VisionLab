"""API tests use synthetic fixtures; no model downloads or fake production mode."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import HealthResponse, PredictionResponse
from app.utils.image_utils import decode_image
from tests.test_core import image_bytes


class FakeService:
    def __init__(self, _settings):
        self.lock = threading.Lock()
        self.initialized = False
        self.error = None
        self.neighbors = SimpleNamespace(image_path=self.image_path)

    def initialize(self):
        self.initialized = True

    def health(self):
        return HealthResponse(status="ok" if self.initialized else "error", resnet18_loaded=self.initialized,
                              dinov2_loaded=self.initialized, classifier_loaded=self.initialized,
                              neighbors_ready=self.initialized, training_images=280 if self.initialized else 0,
                              error=self.error)

    def predict(self, _image, model):
        selected = ["resnet18", "dinov2"] if model == "both" else [model]
        return PredictionResponse(results={key: {"predicted_class": "book", "confidence": 0.7,
                  "probabilities": {"book": 0.7, "bottle": 0.1, "clock": 0.1, "keyboard": 0.1}, "latency_ms": 1.0}
                  for key in selected}, nearest_neighbors=[{"image_url": f"/api/images/{i:032x}", "class_name": "book", "similarity": 0.9-i*0.1} for i in range(3)], runtime="test fixture")

    def image_path(self, image_id):
        if image_id != "0" * 32:
            raise KeyError(image_id)
        return self.test_image


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(service_factory=FakeService)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(lambda: self.client.__exit__(None, None, None))

    def test_health_and_metadata(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/api/models").json()["classes"], ["book", "bottle", "clock", "keyboard"])

    def test_modes_and_aliases(self):
        for field in ["file", "image"]:
            for model in ["resnet18", "dinov2", "both"]:
                response = self.client.post("/api/predict", data={"model": model}, files={field: ("input.png", image_bytes(), "image/png")})
                self.assertEqual(response.status_code, 200, response.text)
                expected = {"resnet18", "dinov2"} if model == "both" else {model}
                self.assertEqual(set(response.json()["results"]), expected)
                self.assertEqual(len(response.json()["nearest_neighbors"]), 3)

    def test_invalid_input(self):
        for model, data in [("unknown", image_bytes()), ("both", b"invalid"), ("both", b"")]:
            response = self.client.post("/api/predict", data={"model": model}, files={"image": ("test.png", data, "image/png")})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("detail", response.json())
        self.assertEqual(self.client.post("/api/predict", data={"model": "both"}).status_code, 415)

    def test_busy_returns_429(self):
        lock = self.app.state.service.lock
        lock.acquire()
        try:
            self.assertEqual(self.client.post("/api/predict").status_code, 429)
        finally:
            lock.release()

    def test_thumbnails_and_path_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.png"
            path.write_bytes(image_bytes())
            self.app.state.service.test_image = path
            response = self.client.get("/api/images/" + "0" * 32)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "image/jpeg")
            with decode_image(response.content) as image:
                self.assertEqual(image.mode, "RGB")
        self.assertEqual(self.client.get("/api/images/" + "f" * 32).status_code, 404)
        self.assertNotEqual(self.client.get("/api/images/%2e%2e%2fmodels%2fresnet18_household.pth").status_code, 200)

    def test_cors(self):
        headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
        response = self.client.options("/api/predict", headers=headers)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")
        headers["Origin"] = "https://unapproved.example"
        self.assertEqual(self.client.options("/api/predict", headers=headers).status_code, 400)

    def test_initialization_failure_is_visible(self):
        class FailedService(FakeService):
            def initialize(self):
                self.error = "Missing checkpoint: test fixture"
                raise FileNotFoundError(self.error)
        with TestClient(create_app(service_factory=FailedService)) as client:
            self.assertEqual(client.get("/api/health").status_code, 503)
            self.assertIn("Missing checkpoint", client.get("/api/health").json()["error"])
            self.assertEqual(client.post("/api/predict").status_code, 503)


if __name__ == "__main__":
    unittest.main()
