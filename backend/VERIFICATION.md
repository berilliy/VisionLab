# Executed verification — 2026-11-07

## Passed

- Python source compilation: `python -m compileall -q backend/app backend/tests backend/scripts`.
- Real ResNet18 checkpoint loaded strictly: no missing or unexpected state-dict keys; FC output size 4.
- Real cached DINOv2-small snapshot loaded; embedding shape `(1, 384)`.
- Trusted local classifier loaded: StandardScaler → LogisticRegression, 384 input features, classes `[0, 1, 2, 3]` mapped to book/bottle/clock/keyboard.
- Training index built from all 280 images under `dataset/train` and saved to `backend/cache/train_embeddings.npz`.
- Real-model smoke script passed 12 requests: one original test image per class, each in resnet18/dinov2/both modes. It validated selected model keys, four actual probabilities, positive measured latency, three descending cosine neighbors, training-only path membership, and real JPEG thumbnails.
- Subsequent real startup reused the saved 280-image cache.
- `python -m unittest tests.test_core -v`: **14 tests passed**. Includes cache freshness/corruption handling, path mapping, image byte/pixel limits, transparency, EXIF, both multipart field names, invalid input, and no disk-spooling of a 2 MiB upload.
- Frontend TypeScript: `tsc -b` passed.
- Frontend Vite production build passed: 2,032 modules transformed; output in `frontend/dist`.
- Frontend Vitest: **22 tests passed**, covering request contract, response validation, server details, readiness checks, network errors, cancellation, timeout, and uploads.

The real-model smoke results are integration checks, not additional accuracy or latency benchmark claims. Synthetic data appears only in automated test fixtures.

## Environment-specific invocation

The sandbox prevents some tools from resolving parent directories under `C:/Users/pc`. TypeScript used Node symlink-preservation flags. Vite and Vitest ran through their installed programmatic APIs with config discovery disabled and `resolve.preserveSymlinks=true`; the production build used the same React/Tailwind plugins and Recharts chunk settings as `vite.config.ts`. Source configuration was not changed merely to accommodate the sandbox.

## Not verified yet

- `app.main` import and FastAPI TestClient tests.
- Actual HTTP health/prediction/image endpoints and CORS in a running server.
- Browser upload → HTTP inference → rendered prediction/gallery end to end.

Reason: **FastAPI and HTTPX are not installed**, and pip cannot connect to `pypi.org:443` in this environment (`WinError 10013`, socket access denied). A project virtual environment with access to the existing ML packages was created. No substitute web framework or fake API was used.

To complete verification from a normal PowerShell terminal with network access, from the project root:

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
cd backend
.\.venv\Scripts\python.exe -m unittest tests.test_api -v
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal, run `backend\.venv\Scripts\python.exe backend/scripts/smoke_models.py --api http://127.0.0.1:8000`, then start/restart the frontend and analyze a real image. The frontend local environment already points at that backend origin.
