import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from ..config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES


class ImageValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def decode_image(data: bytes, *, enforce_upload_limit: bool = True) -> Image.Image:
    """Validate actual content, orient, composite transparency onto white, return RGB.

    The returned pixels and uploaded bytes are only kept in memory. This policy is
    shared by training-index extraction, query decoding, and served thumbnails.
    """
    if not data:
        raise ImageValidationError("The uploaded image is empty.")
    if enforce_upload_limit and len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Image exceeds the 10 MiB upload limit.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageValidationError("Only JPEG, PNG, and WEBP image content is supported.")
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise ImageValidationError("Image exceeds the 25 megapixel limit.", 413)
                if getattr(probe, "n_frames", 1) != 1:
                    raise ImageValidationError("Animated images are not supported. Upload a still image.")
                probe.verify()
            with Image.open(io.BytesIO(data)) as source:
                oriented = ImageOps.exif_transpose(source)
                try:
                    oriented.load()
                    if "A" in oriented.getbands() or "transparency" in oriented.info:
                        rgba = oriented.convert("RGBA")
                        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                        result = Image.alpha_composite(background, rgba).convert("RGB")
                        rgba.close()
                        background.close()
                        return result
                    return oriented.convert("RGB")
                finally:
                    oriented.close()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("Image exceeds the safe pixel limit.", 413) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("Invalid or damaged image. Upload a decodable JPEG, PNG, or WEBP.") from exc


def thumbnail_bytes(path) -> bytes:
    image = decode_image(path.read_bytes(), enforce_upload_limit=False)
    try:
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()
    finally:
        image.close()
