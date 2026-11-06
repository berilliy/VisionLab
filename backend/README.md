# VisionLab FastAPI backend

Real CPU inference using the existing `resnet18_household.pth`, `dinov2_classifier.pkl`, and the pretrained `facebook/dinov2-small` backbone. The notebook, dataset, and trained artifacts are not modified. There is no mock prediction mode.

## Windows setup

Run from the project root. Python 3.12 is required. The existing machine already has the matching CPU PyTorch, Torchvision, Transformers, and scikit-learn packages. Reusing them avoids reinstalling large libraries:

```powershell
cd backend
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The `.venv` has already been created in this workspace. Re-creating it is unnecessary if it exists. `requirements-dev.txt` includes HTTPX for FastAPI TestClient; production only needs `requirements.txt`. No activation is necessary when using the explicit executable paths below.

For a completely isolated installation on another machine, create the venv without `--system-site-packages`. Install the CPU builds before the remaining dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The pinned ML versions in `requirements.txt` reflect the versions actually installed in this project environment. In particular, scikit-learn 1.9.1 matches the loaded serialized pipeline. Do not upgrade it casually: pickle compatibility is not guaranteed across versions. Only load this trusted local classifier; no endpoint accepts model artifacts or arbitrary pickle paths.

## Start the backend

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

From any directory you can instead pass an absolute `--app-dir` pointing to `backend`. Model and dataset paths are always resolved relative to `app/config.py`, not the process working directory.

One process loads each model once during the FastAPI lifespan. Initial startup builds or validates the training embedding cache before the app accepts requests. `--reload` is optional for development, but every reload loads models again; do not combine it with multiple workers. CPU inference is limited to one request at a time; concurrent prediction attempts return 429 rather than accumulating large image tensors.

Open:

- Health: `http://127.0.0.1:8000/api/health`
- Metadata: `http://127.0.0.1:8000/api/models`
- Interactive API docs: `http://127.0.0.1:8000/docs`

Initialization failures are logged, reported by health with HTTP 503, and prevent inference. Diagnostic endpoints remain available after startup failure. Fix the reported path/dependency/cache problem and restart.

## Start the existing frontend

In a second PowerShell window from the project root:

```powershell
cd frontend
npm install
npm run dev
```

`frontend/.env.local` is configured as:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_API_TIMEOUT_MS=60000
```

Restart Vite after environment changes. The frontend checks `/api/health` once on mounting and also updates connection status after predictions. If the backend starts later, reload the browser or submit an image. The existing simplified interface remains intact: upload, model selection, real result cards, and a collapsible training-neighbor gallery. It displays no fallback predictions.

## Required files

```text
<project>/models/resnet18_household.pth
<project>/models/dinov2_classifier.pkl
<project>/dataset/train/book/*
<project>/dataset/train/bottle/*
<project>/dataset/train/clock/*
<project>/dataset/train/keyboard/*
```

The inspected dataset has 70 training images per class (280 total), plus 60 validation and 60 test images. Retrieval scans **only** the four class directories under `dataset/train`. Missing classes/files produce explicit errors.

ResNet18 uses `weights=None`, replaces the FC layer with four outputs, and loads the complete trained state dict strictly with `torch.load(..., weights_only=True, map_location='cpu')`. It never downloads ImageNet weights. Test preprocessing matches notebook cell 2: resize to 224×224, ToTensor, ImageNet normalization, no random augmentation.

DINOv2 uses `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, the revision found in the existing local Hugging Face cache. The existing snapshot is tried first. If unavailable, that exact snapshot is downloaded to `backend/cache/huggingface`. The official AutoImageProcessor configuration and `last_hidden_state[:, 0, :]` match the notebook. The loader validates hidden size 384 and the local StandardScaler → LogisticRegression pipeline, feature count, and class mapping. The exact public class order is `book, bottle, clock, keyboard`.

## Endpoints

### `POST /api/predict`

Multipart fields:

| Field | Meaning |
| --- | --- |
| `image` | The current frontend's upload field |
| `file` | Supported alias for other clients; send only one file field |
| `model` | `resnet18`, `dinov2`, or `both` |

Example using a real image (replace the path with an existing image):

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict -F "image=@C:/path/to/book.jpg" -F "model=both"
```

The response follows `frontend/src/lib/types.ts` and `app/schemas.py`:

```text
results.<selected model>:
  predicted_class: one of the four class names
  confidence: largest actual model probability in [0,1]
  probabilities: { book: number, bottle: number, clock: number, keyboard: number }
  latency_ms: measured milliseconds for this request/model
nearest_neighbors: exactly three { image_url, class_name, similarity } records
device: "cpu"
runtime: actual PyTorch version
```

Only the requested model result(s) are returned. All modes include three DINOv2 training neighbors. Confidence is not calibrated certainty or test accuracy. Cosine similarity is in [-1,1], not a probability, and does not prove the classifier's causal decision mechanism. When ResNet18 is selected, its result is accompanied by a **DINOv2** similarity view, not a ResNet18 attribution map.

Examples intentionally omit fabricated scores. To see actual JSON, run the curl command or the real smoke script below.

### `GET /api/images/{image_id}`

Returns a JPEG thumbnail, at most 384×384, from a validated mapping of opaque 32-character hexadecimal identifiers to training files. It never accepts filesystem paths. Path containment and content hashes are checked before serving; symlinks pointing outside the training directory and changed/missing files are rejected. No project files or test/validation images are mounted as static directories.

### `GET /api/health`

Reports actual readiness: `status`, `device`, `resnet18_loaded`, `dinov2_loaded`, `classifier_loaded`, `neighbors_ready`, `training_images`, and `error`. HTTP 503 indicates unavailability; no flags are fabricated.

### `GET /api/models`

Returns the supported model identifiers, selection modes, exact class order, and the same readiness object.

## Upload validation and privacy

- JPEG, PNG, and WEBP content only, regardless of browser MIME or filename extension.
- Nonempty files up to 10 MiB; a streaming total-body bound also handles missing Content-Length.
- Maximum 25 megapixels; animated and malformed files rejected.
- EXIF orientation applied; transparency composited onto white; RGB conversion.
- Training/query/thumbnail decoding uses the same orientation/transparency policy.
- Multipart parsing is bounded and stays in memory, including files above Starlette's usual 1 MiB spool threshold. Uploaded photographs are never saved to disk, cached, or logged.
- The frontend still performs its own type/extension/size checks for usability; backend validation is independent.

Errors use JSON `detail` with statuses 400/415/422 for malformed requests, 413 for size limits, 408 for an upload timeout, 429 for a busy CPU, 503 for missing initialized components, and 500 for inference failure. A user cancelling a browser request may not stop native CPU computation; concurrency remains limited while it finishes.

## Embedding cache

`backend/cache/train_embeddings.npz` stores raw float32 CLS embeddings, class labels, relative training paths, opaque IDs, and JSON metadata. It is loaded with `allow_pickle=False`. Model identity/revision, weight/config hashes, processor class/config, Torch/Transformers/Torchvision/Pillow versions, decoding policy, and every training image's relative path/content hash are checked. Dataset or preprocessing changes, corrupt arrays, incorrect dimensions/order, or nonfinite/zero vectors trigger a rebuild.

Initial extraction runs in batches of eight by default. Saving uses an atomic replacement. Cache entries contain no uploaded user photos. Rebuilding this index does not retrain or modify either classifier. The first index creation can take longer than ordinary inference; subsequent startups reuse it.

## Timing boundaries

`time.perf_counter()` measures preprocessing + forward pass + probability computation per model. DINOv2 includes the logistic-regression pipeline. Its query embedding is shared with nearest-neighbor search, not recomputed.

Timing excludes upload, initial byte decoding/orientation/alpha conversion, request queueing, model loading, cache creation, similarity ranking, thumbnail retrieval, and HTTP serialization. Consequently the total HTTP request takes longer than the displayed per-model latency. ResNet-only requests also run DINOv2 for required retrieval, outside the ResNet latency field. All inference is under `torch.inference_mode()`.

The historical CPU values 45.95 ms and 164.20 ms are preserved only in frontend experiment data. Live measurements are independent and can differ with CPU, thread count, library versions, and warmup. Original benchmark accuracy concerns 60 test images; the additional ten images are books only.

## Configuration

Set environment variables in PowerShell; `.env.example` documents them but is not automatically loaded:

```powershell
$env:VISIONLAB_OFFLINE = "1"       # Require an already cached DINOv2 snapshot
$env:VISIONLAB_TORCH_THREADS = "4"
$env:VISIONLAB_BATCH_SIZE = "8"
$env:VISIONLAB_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
```

Use the actual Vite origin/port. No wildcard origins or credentials are enabled. Bind to localhost for the local university demo. The runtime is CPU-only; GPU is not required.

## Tests

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core -v
.\.venv\Scripts\python.exe -m unittest tests.test_api -v
.\.venv\Scripts\python.exe scripts/smoke_models.py
# With the backend running:
.\.venv\Scripts\python.exe scripts/smoke_models.py --api http://127.0.0.1:8000
```

Core tests cover decoding, orientation, transparency, limits, memory-only multipart handling, probability validation, cache reuse/invalidation, and safe retrieval paths. API tests inject a lightweight fake service **only in tests** to verify HTTP contracts, errors, CORS, busy handling, and safe image serving. The separate real smoke script loads actual trained artifacts and tests all three modes on one test image from each class, checking real probabilities, measured latency, and three valid training thumbnails. It is an integration check, not a new benchmark.

Frontend checks:

```powershell
cd ../frontend
npm run build
npm test
```

## Troubleshooting

- **Missing checkpoint/classifier/dataset:** health and logs name the exact required path. Restore the original files and restart. Nothing is replaced with random weights.
- **DINOv2 unavailable:** allow the first download or populate the exact revision's local Hugging Face snapshot. `VISIONLAB_OFFLINE=1` forbids downloads explicitly.
- **`WinError 10013` from pip:** this authoring environment can deny external sockets. Run installation in a regular PowerShell terminal with PyPI access.
- **Pipeline version errors:** use the scikit-learn version recorded in requirements; do not retrain merely to resolve installation issues.
- **CORS/network error:** confirm `/api/health`, frontend origin, and Vite environment URL. Restart Vite after `.env.local` changes.
- **Changed training file:** restart to rebuild the validated index. Image URLs contain a content-derived ID and are deliberately invalidated when a training file changes.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [FastAPI multipart forms/files](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/), and [Hugging Face DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2).
