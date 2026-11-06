import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .schemas import HealthResponse, ModelsResponse, PredictionResponse
from .services.inference import InferenceService
from .utils.image_utils import ImageValidationError, decode_image, thumbnail_bytes
from .utils.uploads import read_upload

logger = logging.getLogger("visionlab")

UPLOAD_SCHEMA = {
    "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["model"],
        "properties": {
            "image": {"type": "string", "format": "binary", "description": "Frontend image upload"},
            "file": {"type": "string", "format": "binary", "description": "Alias for image; send only one"},
            "model": {"type": "string", "enum": ["resnet18", "dinov2", "both"]},
        }, "oneOf": [{"required": ["image"]}, {"required": ["file"]}],
    }}}},
}


def create_app(settings: Settings | None = None, *, service_factory=InferenceService) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = service_factory(settings)
        application.state.service = service
        try:
            await run_in_threadpool(service.initialize)
        except Exception:
            # Keep diagnostic endpoints available, but never serve incomplete or
            # random models. A restart is required after fixing initialization.
            logger.exception("Model initialization failed; prediction endpoint is unavailable")
        yield

    application = FastAPI(title="VisionLab AI", version="1.0.0", lifespan=lifespan,
                          description="CPU inference with trained ResNet18 and DINOv2. Uploaded images are processed in memory and not stored.")
    application.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                               allow_credentials=False, allow_methods=["GET", "POST"],
                               allow_headers=["Content-Type"])

    @application.exception_handler(ImageValidationError)
    async def invalid_image(_request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    def service_for(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Backend is starting. Retry after initialization.")
        return service

    def require_ready(service):
        health = service.health()
        if health.status != "ok":
            raise HTTPException(503, health.error or "Models and neighbor index are still loading.")

    @application.get("/api/health", response_model=HealthResponse)
    def health(request: Request, response: Response):
        result = service_for(request).health()
        if result.status != "ok":
            response.status_code = 503
        return result

    @application.get("/api/models", response_model=ModelsResponse)
    def models(request: Request):
        return ModelsResponse(readiness=service_for(request).health())

    @application.post("/api/predict", response_model=PredictionResponse, openapi_extra=UPLOAD_SCHEMA,
                      responses={413: {"description": "Image too large"}, 422: {"description": "Invalid input"},
                                 429: {"description": "Inference busy"}, 503: {"description": "Backend unavailable"}})
    async def predict(request: Request):
        service = service_for(request)
        require_ready(service)
        # One upload/inference at a time. Reject overload rather than queueing
        # decoded full-resolution images and duplicating model memory.
        if not service.lock.acquire(blocking=False):
            raise HTTPException(429, "The CPU is processing another image. Please retry shortly.", headers={"Retry-After": "1"})
        try:
            try:
                async with asyncio.timeout(60):
                    data, selected = await read_upload(request)
            except TimeoutError as exc:
                raise HTTPException(408, "Image upload timed out.") from exc

            def infer():
                image = decode_image(data)
                try:
                    return service.predict(image, selected)
                finally:
                    image.close()
            # Starlette/AnyIO waits for the worker even after request cancellation,
            # so the lock is not released while native CPU inference is running.
            return await run_in_threadpool(infer)
        except (ImageValidationError, HTTPException):
            raise
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(500, "Model inference failed. Check backend logs and retry.") from exc
        finally:
            service.lock.release()

    @application.get("/api/images/{image_id}", responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "Training image not found"}})
    def training_image(request: Request, image_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]):
        service = service_for(request)
        require_ready(service)
        try:
            path = service.neighbors.image_path(image_id)
            data = thumbnail_bytes(path)
        except (KeyError, OSError, ImageValidationError) as exc:
            raise HTTPException(404, "Training image not found or changed. Restart the backend to rebuild the index.") from exc
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})

    return application


app = create_app()
