# VisionLab AI frontend

The existing React + TypeScript application is connected to the Python backend in [../backend](../backend/README.md). The interface has two pages: classification and model comparison. Model weights and historical experiment results are preserved.

## Run

Start the backend using its README, then in a separate terminal from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Use Node.js 20.19+ or 22 LTS. Open the URL Vite prints, normally http://127.0.0.1:5173.

```powershell
npm run build
npm test
npm run preview
```

Production output is `dist/`. Static hosting needs an index.html fallback for client-side routes.

## Backend connection

`.env.local` has been created with:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_API_TIMEOUT_MS=60000
```

Restart Vite after changes. The URL is the backend origin without /api/predict. Vite variables are public; never put secrets in them. Both localhost:5173 and 127.0.0.1:5173 are allowed by the backend by default; configure VISIONLAB_CORS_ORIGINS for another port.

The client checks GET /api/health when mounted. A successful health response or prediction shows “Backend ready”; this is not a continuously polled monitor. If the backend starts later, reload the page or analyze an image.

POST /api/predict sends `image` and `model` in multipart form data. Responses are validated before rendering. Backend error details, unavailable-server states, timeouts, and cancellation are supported. Selected input/model changes discard stale predictions. See [docs/API.md](docs/API.md).

## Pages

- `/`: image upload, model selection, actual results, and a collapsible nearest-training-image gallery after prediction.
- `/comparison`: fixed historical experiment metrics, accuracy/F1 switch, CPU timings, and the separate book-only test.
- `/classify` and removed page URLs redirect to classification.

There is no sidebar, dashboard, About page, or separate Explainability page. The UI was not replaced during backend integration.

Files and results remain in browser memory only. Uploads go to the configured backend when Analyze image is pressed. The provided backend processes uploaded photographs in memory and does not store them.

## Scientific interpretation

Historical values in `src/lib/experiments.ts` never populate live results. The original test set has 60 images across four classes. The separate experiment has ten unseen book images only (ResNet18: 7/10; DINOv2: 10/10). Historical CPU averages of ten runs were 45.95 ms and 164.20 ms including preprocessing; live timings are independently measured.

The notebook uses weighted precision, recall, and F1 for the deep learning models. A small test set's 100% score does not establish perfect real-world accuracy. Model confidence is not test accuracy or a calibrated guarantee. DINOv2 cosine neighbors illustrate feature similarity, not causal attribution, including when ResNet18 is selected.

## Structure

```text
src/components/   # Existing simplified layout, uploads, predictions, charts, gallery
src/context/      # Inference state, one-time health request, cleanup and cancellation
src/lib/          # API client, types, upload checks, fixed experiment data
src/pages/        # Classification and comparison
tests/            # Synthetic contract fixtures; never imported into the application
docs/API.md       # Implemented backend contract
```

## Verification

With the installed project dependencies, strict TypeScript checking, the Vite production build, and all 22 Vitest API/upload/health tests passed during backend integration. To accommodate this authoring sandbox's restricted parent-directory access, Vite/Vitest were invoked programmatically with the same plugins/build settings, config-file discovery disabled, and symlink preservation enabled. Ordinary local terminals use the npm commands above.

The backend README and verification report distinguish real-model checks from HTTP and browser integration checks. No mocked values are presented as live predictions.
