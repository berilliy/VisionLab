"""Bounded, memory-only multipart parsing (UploadFile normally spills to disk)."""
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

from ..config import MAX_BODY_BYTES, MAX_UPLOAD_BYTES
from .image_utils import ImageValidationError


class MemoryMultipartParser(MultiPartParser):
    # Every accepted body is smaller than this threshold, so no uploaded photo
    # is rolled out from SpooledTemporaryFile's BytesIO to a filesystem file.
    spool_max_size = MAX_BODY_BYTES + 1


async def read_upload(request: Request) -> tuple[bytes, str]:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise ImageValidationError("Use multipart/form-data with image (or file) and model fields.", 415)
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise ImageValidationError("Invalid Content-Length header.", 400) from exc
    if length < 0:
        raise ImageValidationError("Invalid Content-Length header.", 400)
    if length > MAX_BODY_BYTES:
        raise ImageValidationError("Request exceeds the 10 MiB image upload limit.", 413)

    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                raise ImageValidationError("Request exceeds the 10 MiB image upload limit.", 413)
            yield chunk

    parser = MemoryMultipartParser(request.headers, bounded_stream(), max_files=1, max_fields=1, max_part_size=256)
    form = None
    try:
        form = await parser.parse()
        if set(form.keys()) not in ({"image", "model"}, {"file", "model"}):
            raise ImageValidationError("Provide exactly one image (or file) and a model field.")
        model = form.get("model")
        if model not in {"resnet18", "dinov2", "both"}:
            raise ImageValidationError("Invalid model. Choose resnet18, dinov2, or both.")
        image = form.get("image") or form.get("file")
        if not isinstance(image, UploadFile):
            raise ImageValidationError("An image file is required.")
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise ImageValidationError("The uploaded image is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ImageValidationError("Image exceeds the 10 MiB upload limit.", 413)
        return data, model
    except MultiPartException as exc:
        raise ImageValidationError(f"Invalid multipart upload: {exc.message}", 400) from exc
    finally:
        if form is not None:
            await form.close()
        # Also close partially constructed forms on malformed or truncated input.
        for file in parser._files_to_close_on_error:
            file.close()
