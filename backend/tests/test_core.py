import asyncio
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from starlette.requests import Request

from app.config import CLASSES, MAX_UPLOAD_BYTES, Settings
from app.schemas import PredictionResult
from app.services.neighbors import NeighborIndex
from app.utils.image_utils import ImageValidationError, decode_image
from app.utils.uploads import read_upload


def image_bytes(fmt="PNG", mode="RGB", color="red", size=(16, 12), **kwargs):
    with Image.new(mode, size, color) as image:
        output = io.BytesIO()
        image.save(output, format=fmt, **kwargs)
        return output.getvalue()


class ImageTests(unittest.TestCase):
    def test_supported_content_ignores_browser_mime(self):
        for fmt in ["JPEG", "PNG", "WEBP"]:
            with self.subTest(fmt=fmt), decode_image(image_bytes(fmt)) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (16, 12))

    def test_empty_invalid_and_unsupported(self):
        for data in [b"", b"not an image", image_bytes("GIF")]:
            with self.subTest(data=data[:8]), self.assertRaises(ImageValidationError):
                decode_image(data)

    def test_byte_and_pixel_limits(self):
        with self.assertRaises(ImageValidationError) as error:
            decode_image(b"x" * (MAX_UPLOAD_BYTES + 1))
        self.assertEqual(error.exception.status_code, 413)
        with patch("app.utils.image_utils.MAX_IMAGE_PIXELS", 10):
            with self.assertRaises(ImageValidationError) as error:
                decode_image(image_bytes())
            self.assertEqual(error.exception.status_code, 413)

    def test_transparency_composited_on_white(self):
        with decode_image(image_bytes(mode="RGBA", color=(0, 0, 0, 0))) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_exif_orientation(self):
        exif = Image.Exif()
        exif[274] = 6
        with decode_image(image_bytes("JPEG", exif=exif)) as image:
            self.assertEqual(image.size, (12, 16))

    def test_probabilities_enforced(self):
        args = dict(predicted_class="book", confidence=0.7,
                    probabilities=dict(zip(CLASSES, [0.7, 0.1, 0.1, 0.1])), latency_ms=1.2)
        self.assertEqual(PredictionResult(**args).predicted_class, "book")
        with self.assertRaises(ValueError):
            PredictionResult(**{**args, "confidence": 0.8})


class TestExtractor:
    """Deterministic, synthetic embeddings for cache mechanics tests only."""
    embedding_metadata = {"test_fixture": True, "dimension": 384}
    calls = 0

    def embeddings(self, images):
        self.calls += 1
        return np.vstack([np.arange(1, 385, dtype=np.float32) + np.asarray(im).mean() for im in images])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(Settings(), train_dir=root / "train", cache_dir=root / "cache", batch_size=2)
        for i, name in enumerate(CLASSES):
            folder = self.settings.train_dir / name
            folder.mkdir(parents=True)
            (folder / "sample.png").write_bytes(image_bytes(color=(i * 40, 30, 50)))

    def test_cache_reuse_and_three_sorted_training_neighbors(self):
        extractor = TestExtractor()
        index = NeighborIndex(self.settings, extractor)
        index.initialize()
        self.assertEqual(extractor.calls, 2)
        neighbors = index.nearest(np.arange(1, 385, dtype=np.float32)[None, :])
        self.assertEqual(len(neighbors), 3)
        self.assertEqual([n.similarity for n in neighbors], sorted([n.similarity for n in neighbors], reverse=True))
        for item in neighbors:
            self.assertTrue(index.image_path(item.image_url.rsplit("/", 1)[1]).is_relative_to(self.settings.train_dir))
        another = NeighborIndex(self.settings, extractor)
        another.initialize()
        self.assertTrue(another.cache_loaded)
        self.assertEqual(extractor.calls, 2)
        with self.assertRaises(KeyError):
            index.image_path("../../models/resnet18_household.pth")

    def test_changed_dataset_and_processor_rebuild_cache(self):
        extractor = TestExtractor()
        first = NeighborIndex(self.settings, extractor)
        first.initialize()
        (self.settings.train_dir / "book/sample.png").write_bytes(image_bytes(color="blue"))
        changed = NeighborIndex(self.settings, extractor)
        changed.initialize()
        self.assertFalse(changed.cache_loaded)
        self.assertEqual(extractor.calls, 4)
        extractor.embedding_metadata = {"test_fixture": True, "processor": "changed"}
        again = NeighborIndex(self.settings, extractor)
        again.initialize()
        self.assertFalse(again.cache_loaded)
        self.assertEqual(extractor.calls, 6)

    def test_corrupt_cache_rebuilt_and_changed_image_not_served(self):
        extractor = TestExtractor()
        index = NeighborIndex(self.settings, extractor)
        index.initialize()
        (self.settings.cache_dir / "train_embeddings.npz").write_bytes(b"corrupt")
        rebuilt = NeighborIndex(self.settings, extractor)
        rebuilt.initialize()
        self.assertTrue(rebuilt.ready)
        self.assertFalse(rebuilt.cache_loaded)
        entry = index.entries[0]
        (self.settings.train_dir / entry.relative_path).write_bytes(image_bytes(color="green"))
        with self.assertRaises(KeyError):
            index.image_path(entry.image_id)

    def test_missing_class_fails_clearly(self):
        (self.settings.train_dir / "book/sample.png").unlink()
        with self.assertRaisesRegex(RuntimeError, "No training images"):
            NeighborIndex(self.settings, TestExtractor()).initialize()


class MultipartTests(unittest.TestCase):
    def request(self, body, content_type="multipart/form-data; boundary=visionlab"):
        sent = False
        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return Request({"type": "http", "method": "POST", "path": "/api/predict",
                        "headers": [(b"content-type", content_type.encode())]}, receive)

    def body(self, field="image", model="both", data=b"test"):
        return (b'--visionlab\r\nContent-Disposition: form-data; name="model"\r\n\r\n' + model.encode() +
                b'\r\n--visionlab\r\nContent-Disposition: form-data; name="' + field.encode() +
                b'"; filename="image.png"\r\nContent-Type: image/png\r\n\r\n' + data + b'\r\n--visionlab--\r\n')

    def test_both_upload_aliases(self):
        for field in ["image", "file"]:
            self.assertEqual(asyncio.run(read_upload(self.request(self.body(field)))), (b"test", "both"))

    def test_bad_model_empty_and_content_type(self):
        for request in [self.request(self.body(model="fake")), self.request(self.body(data=b"")), self.request(b"x", "application/json")]:
            with self.assertRaises(ImageValidationError):
                asyncio.run(read_upload(request))

    def test_uploads_do_not_spill_to_disk(self):
        from tempfile import SpooledTemporaryFile
        created = []
        def tracked(*args, **kwargs):
            stream = SpooledTemporaryFile(*args, **kwargs)
            created.append(stream)
            return stream
        # Larger than Starlette's default 1 MiB spill threshold.
        data = b"x" * (2 * 1024 * 1024)
        with patch("starlette.formparsers.SpooledTemporaryFile", side_effect=tracked):
            actual, _ = asyncio.run(read_upload(self.request(self.body(data=data))))
        self.assertEqual(len(actual), len(data))
        self.assertTrue(all(stream.closed and not stream._rolled for stream in created))

    def test_stream_limit_even_without_content_length(self):
        with self.assertRaises(ImageValidationError) as error:
            asyncio.run(read_upload(self.request(self.body(data=b"x" * (MAX_UPLOAD_BYTES + 100_000)))))
        self.assertEqual(error.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
