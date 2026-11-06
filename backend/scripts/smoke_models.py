"""Run real-model smoke checks; no synthetic predictions and no model changes.

python scripts/smoke_models.py                 # direct service checks
python scripts/smoke_models.py --api http://127.0.0.1:8000  # actual HTTP checks
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import CLASSES, PROJECT_ROOT, Settings
from app.schemas import PredictionResponse
from app.utils.image_utils import decode_image, thumbnail_bytes


def upload(base, path, mode):
    boundary = "visionlab-real-smoke-boundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{mode}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.jpg"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request(base + "/api/predict", data=body,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=120) as response:
        return PredictionResponse.model_validate_json(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="Running backend origin; omit to test the service directly")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = None
    if args.api:
        base = args.api.rstrip("/")
        with urlopen(base + "/api/health", timeout=10) as response:
            assert json.load(response)["status"] == "ok"
    else:
        from app.services.inference import InferenceService
        service = InferenceService(Settings())
        service.initialize()
        assert service.health().status == "ok"
        print("Training embeddings:", len(service.neighbors.entries), "cache reused:", service.neighbors.cache_loaded)
    count = 0
    for label in CLASSES:
        path = next(p for p in sorted((PROJECT_ROOT / "dataset/test" / label).iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        for mode in ["resnet18", "dinov2", "both"]:
            if service:
                with decode_image(path.read_bytes()) as image:
                    result = service.predict(image, mode)
            else:
                result = upload(base, path, mode)
            assert set(result.results) == ({"resnet18", "dinov2"} if mode == "both" else {mode})
            assert len(result.nearest_neighbors) == 3
            scores = [n.similarity for n in result.nearest_neighbors]
            assert scores == sorted(scores, reverse=True)
            for prediction in result.results.values():
                assert set(prediction.probabilities.model_dump()) == set(CLASSES)
                assert prediction.latency_ms > 0
            for neighbor in result.nearest_neighbors:
                if service:
                    training = service.neighbors.image_path(neighbor.image_url.rsplit("/", 1)[1])
                    assert training.is_relative_to((PROJECT_ROOT / "dataset/train").resolve())
                    data = thumbnail_bytes(training)
                else:
                    with urlopen(base + neighbor.image_url, timeout=10) as response:
                        assert response.headers["Content-Type"] == "image/jpeg"
                        data = response.read()
                with decode_image(data) as thumbnail:
                    assert max(thumbnail.size) <= 384
            print(json.dumps({"input_class": label, "mode": mode, **result.model_dump()}, ensure_ascii=True), flush=True)
            count += 1
    print(f"PASS: {count} real prediction requests; all modes, four input classes, three valid training thumbnails per request.")
    print("This is an integration smoke test, not a new accuracy benchmark.")


if __name__ == "__main__":
    main()
