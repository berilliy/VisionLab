# VisionLab AI — Household Object Recognition

Astana IT University · Computer Vision · Assignment 2

A React frontend and Python backend for classifying **book, bottle, clock, and keyboard** using trained ResNet18 and DINOv2-small + StandardScaler/LogisticRegression models. DINOv2 embeddings retrieve three similar training images as a visual explanation.

## Project contents

- `frontend/`: React, TypeScript, Vite, Tailwind CSS, Recharts; classification and comparison pages.
- `backend/`: FastAPI, real CPU inference, validated image uploads, training-neighbor retrieval and caching.
- `models/`: trained ResNet18 weights and DINOv2 classifier; no retraining is required.
- `dataset/`: 280 training, 60 validation, and 60 test images across four classes.
- `new_test/`: ten additional book images, evaluated separately.
- `household_object_recognition.ipynb`: original experiments and training methodology.

## Run on Windows

Python 3.12 and Node.js 20.19+ or 22 LTS are recommended. See the backend README for the inspected ML dependency versions and CPU installation details.

From the repository root:

```powershell
python -m venv --system-site-packages backend/.venv
backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

The frontend connects to `http://127.0.0.1:8000`. The backend first uses an existing DINOv2 cache or downloads the pinned pretrained snapshot. It builds a training-only embedding index once and reuses it. Model weights are not modified; uploaded photos are processed in memory and are not saved.

## API

- `POST /api/predict`: multipart `image` (or `file`) plus `model=resnet18|dinov2|both`.
- `GET /api/health`: actual component readiness.
- `GET /api/models`: models and class order.
- `GET /api/images/{image_id}`: validated training-image thumbnails.
- `/docs`: interactive API documentation.

## Evaluation and interpretation

| Model | Original test accuracy | Weighted test F1 |
| --- | --- | --- |
| HOG + SVM | 68.33% | 67.16% |
| ResNet18 | 100% | 100% |
| DINOv2 | 100% | 100% |

The original test set contains only 60 images. The separate ten-image book evaluation gave ResNet18 70% and DINOv2 100%; it is not a general all-class robustness benchmark. Historical CPU latency averages of ten runs including preprocessing were 45.95 ms and 164.20 ms, respectively. Live latencies and predictions are computed independently.

Confidence is not test accuracy or a guarantee. Cosine neighbors illustrate feature similarity, not the exact cause of a prediction. Performance on new images may be lower.

## Documentation and verification

- [Backend setup, model loading, caching, privacy, and tests](backend/README.md)
- [Frontend setup](frontend/README.md)
- [API contract](frontend/docs/API.md)
- [Executed verification and remaining environment limitations](backend/VERIFICATION.md)

Virtual environments, dependencies, generated caches, build output, local environment files, and temporary checks are excluded from version control.
