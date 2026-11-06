# Implemented API contract

Backend implementation: [../../backend/app/main.py](../../backend/app/main.py). Setup and model/cache details: [backend README](../../backend/README.md).

## POST /api/predict

Multipart form data with exactly one file and one model field:

| Field | Meaning |
| --- | --- |
| `image` | Upload field used by this frontend |
| `file` | Backend-supported alias; never send both file fields |
| `model` | `resnet18`, `dinov2`, or `both` |

JPEG, PNG, WEBP content only; nonempty and at most 10 MiB. The backend verifies actual decoding, applies EXIF orientation, composites transparency on white, converts to RGB, and limits images to 25 megapixels. Multipart parsing stays in memory. Frontend validation additionally checks extension and MIME for convenience.

## Successful response

HTTP 200 JSON. These are types, not example predictions:

```typescript
type ObjectClass = 'book' | 'bottle' | 'clock' | 'keyboard';
type ModelId = 'dinov2' | 'resnet18';

interface PredictionResult {
  predicted_class: ObjectClass;
  confidence: number;
  probabilities: Record<ObjectClass, number>;
  latency_ms: number;
}
interface PredictionResponse {
  results: Partial<Record<ModelId, PredictionResult>>;
  nearest_neighbors: Array<{
    image_url: string;
    class_name: ObjectClass;
    similarity: number;
  }>;
  device: 'cpu';
  runtime: string;
}
```

Only requested model keys are returned. All four probabilities are actual numeric model outputs in [0,1], approximately summing to one. Confidence is the maximum probability and matches predicted_class. The frontend tolerates minor rounding; backend schemas validate more strictly.

Exactly three training neighbors are returned, sorted by descending DINOv2 cosine similarity in [-1,1], in every mode. The DINOv2 query embedding is shared between its classification and retrieval. A missing/incompatible neighbor index makes the endpoint unavailable rather than returning fake examples.

Per-model latency includes model preprocessing, forward pass, and probability computation; DINOv2 includes its classifier pipeline. It excludes upload, initial image decoding/orientation, startup, cache construction, retrieval ranking, and HTTP transport. ResNet18-only requests additionally run DINOv2 for retrieval outside the ResNet18 timer.

## GET /api/images/{image_id}

Returns real training-image JPEG thumbnails using an opaque 32-character hexadecimal ID. URLs are resolved against the configured backend origin. No arbitrary filesystem paths, test images, or validation images are exposed. Broken image loads show a visible fallback.

## GET /api/health

Reports status (`ok`, `loading`, `error`), device, resnet18_loaded, dinov2_loaded, classifier_loaded, neighbors_ready, training_images, and an optional error. Unavailable components return HTTP 503. The frontend checks this once on mount.

## GET /api/models

Returns supported model IDs, selection modes, exact class order, and readiness.

## Errors and lifecycle

Errors use JSON `detail`. The client displays bounded string details as text, never HTML. Common statuses: 400 malformed multipart, 408 upload timeout, 413 size limit, 415 wrong content type, 422 invalid file/model, 429 CPU busy, 503 unavailable model/index, 500 inference failure.

Network/CORS failures and unavailable endpoints show a disconnected state. The client uses a configurable abortable timeout, clears results when the image or selected model changes, and discards stale responses. Browser cancellation does not guarantee cancellation of native CPU computation.

No mock outputs, historical benchmark values, synthetic neighbors, or paid inference services are used.
