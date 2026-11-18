"""Model loading, image preprocessing, and prediction utilities."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emnist_cnn.keras"
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "emnist_balanced_preprocessed.npz"

EMNIST_BALANCED_CLASSES = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abdefghnqrt")
)


def load_class_names() -> list[str]:
    return EMNIST_BALANCED_CLASSES


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained CNN model found at {model_path}. "
            "Train it with: .venv/bin/python src/train_model.py"
        )

    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def _pil_stage(image: np.ndarray, mode: str = "L") -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode=mode)


def _choose_character_components(mask: np.ndarray, mode: str) -> np.ndarray:
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        raise ValueError("No clear handwritten character was found.")

    image_height, image_width = mask.shape
    image_area = image_height * image_width
    components: list[tuple[int, int, int, int, int, float, float]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < max(8, int(image_area * 0.00001)):
            continue

        touches_border = x == 0 or y == 0 or x + width >= image_width or y + height >= image_height
        is_paper_edge = touches_border and (
            width > image_width * 0.28
            or height > image_height * 0.28
            or area > image_area * 0.025
        )
        if is_paper_edge:
            continue
        components.append(
            (label, x, y, width, height, float(area), float(centers[label][0]))
        )

    if not components:
        raise ValueError(
            "The foreground looks like a page boundary or background, not a character."
        )

    primary = max(components, key=lambda component: component[5])
    _, px, py, pw, ph, primary_area, _ = primary
    proximity = max(18, int(max(pw, ph) * 0.16))
    selected_labels = [primary[0]]

    for component in components:
        label, x, y, width, height, area, _ = component
        if label == primary[0]:
            continue
        horizontal_gap = max(px - (x + width), x - (px + pw), 0)
        vertical_gap = max(py - (y + height), y - (py + ph), 0)
        near_character = max(horizontal_gap, vertical_gap) <= proximity
        likely_mark = area >= max(8, primary_area * 0.015)
        if near_character and likely_mark:
            selected_labels.append(label)

    selected = np.isin(labels, selected_labels)
    if mode == "photo":
        bounds_y, bounds_x = np.where(selected)
        if bounds_x.size == 0:
            raise ValueError("No character foreground remains after removing the page.")
        x0, x1 = int(bounds_x.min()), int(bounds_x.max()) + 1
        y0, y1 = int(bounds_y.min()), int(bounds_y.max()) + 1
        box_area = (x1 - x0) * (y1 - y0)
        selected_area = int(selected.sum())
        if box_area == 0 or selected_area / box_area < 0.012:
            raise ValueError(
                "The detected foreground is too sparse to be a readable character."
            )
        if (x1 - x0) > image_width * 0.92 and (y1 - y0) > image_height * 0.92:
            raise ValueError(
                "The detected foreground fills the photograph; move closer to the character."
            )

    return selected


def _prepare_rgb_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white_background.alpha_composite(image)
    return white_background.convert("RGB")


def _prepare_canvas_input(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Image.Image]]:
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    mask = _choose_character_components(binary > 0, mode="canvas")
    return _finish_character(gray, mask, {
        "Corrected illumination": gray,
        "Thresholded foreground mask": (mask.astype(np.uint8) * 255),
    })


def _prepare_photo_input(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Image.Image]]:
    height, width = gray.shape
    minimum_dimension = min(height, width)
    if minimum_dimension < 48:
        raise ValueError("The photograph is too small to isolate handwriting.")

    kernel_size = max(31, int(minimum_dimension * 0.12))
    kernel_size = min(kernel_size, 151)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, min(height, width) - (1 - min(height, width) % 2))
    kernel_size = max(3, kernel_size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(3, minimum_dimension / 18))
    corrected = cv2.divide(gray, np.maximum(background, 1), scale=255)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    block_size = max(31, int(minimum_dimension * 0.12))
    if block_size % 2 == 0:
        block_size += 1
    block_size = min(block_size, 151)
    if block_size >= minimum_dimension:
        block_size = minimum_dimension - 1 if minimum_dimension % 2 == 0 else minimum_dimension
    if block_size < 3:
        raise ValueError("The photograph is too small for adaptive thresholding.")

    binary = cv2.adaptiveThreshold(
        corrected,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )

    mask = _choose_character_components(binary > 0, mode="photo")
    return _finish_character(corrected, mask, {
        "Corrected illumination": corrected,
        "Thresholded foreground mask": (mask.astype(np.uint8) * 255),
    })


def _finish_character(
    intensity: np.ndarray,
    mask: np.ndarray,
    stages: dict[str, Image.Image | np.ndarray],
) -> tuple[np.ndarray, dict[str, Image.Image]]:
    y_coordinates, x_coordinates = np.where(mask)
    if x_coordinates.size < 12:
        raise ValueError("The foreground mask does not contain enough ink pixels.")

    x0, x1 = int(x_coordinates.min()), int(x_coordinates.max()) + 1
    y0, y1 = int(y_coordinates.min()), int(y_coordinates.max()) + 1
    cropped_mask = mask[y0:y1, x0:x1]
    cropped_intensity = intensity[y0:y1, x0:x1]
    foreground_values = cropped_intensity[cropped_mask]
    if foreground_values.size < 12:
        raise ValueError("The extracted character is too small or unreadable.")

    # Keep anti-aliased stroke pixels close to the binary mask while dropping
    # the photographed paper and its residual texture.
    soft_mask = cv2.dilate(
        cropped_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    darkness = np.clip(255.0 - cropped_intensity.astype(np.float32), 0.0, 255.0)
    intensity_scale = float(np.percentile(darkness[cropped_mask], 99))
    if intensity_scale < 8:
        raise ValueError("The detected character has insufficient ink contrast.")
    crop = np.clip(darkness * (255.0 / intensity_scale), 0, 255).astype(np.uint8)
    crop[~soft_mask] = 0
    crop_image = Image.fromarray(crop, mode="L")

    crop_height, crop_width = crop.shape
    if crop_height < 2 or crop_width < 2:
        raise ValueError("The extracted character is too small to recognize.")
    scale = min(20.0 / crop_width, 20.0 / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )

    final = np.zeros((28, 28), dtype=np.uint8)
    left = (28 - resized_width) // 2
    top = (28 - resized_height) // 2
    final[top:top + resized_height, left:left + resized_width] = resized

    if np.count_nonzero(final > 24) < 8 or int(final.max()) < 48:
        raise ValueError("The character is too faint or unreadable after normalization.")

    debug_stages: dict[str, Image.Image] = {}
    for name, value in stages.items():
        debug_stages[name] = (
            value if isinstance(value, Image.Image) else _pil_stage(value)
        )
    debug_stages["Cropped character"] = crop_image
    debug_stages["Final 28x28 CNN input"] = Image.fromarray(final, mode="L")
    normalized = final.astype(np.float32)[np.newaxis, ..., np.newaxis] / 255.0
    if normalized.shape != (1, 28, 28, 1):
        raise RuntimeError(f"Unexpected CNN input shape: {normalized.shape}")
    return normalized, debug_stages


def preprocess_uploaded_image(
    image: Image.Image,
    mode: str = "photo",
) -> tuple[np.ndarray, Image.Image, dict[str, Image.Image]]:
    """Preprocess photographed handwriting or a drawing-canvas image."""
    if image is None:
        raise ValueError("No image was provided.")
    if mode not in {"photo", "canvas"}:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    oriented_rgb = _prepare_rgb_image(image)
    width, height = oriented_rgb.size
    if width < 16 or height < 16:
        raise ValueError("The image is too small or unreadable.")
    scale = min(1600.0 / max(width, height), 1.0)
    if scale < 1.0:
        oriented_rgb = oriented_rgb.resize(
            (int(round(width * scale)), int(round(height * scale))),
            Image.Resampling.LANCZOS,
        )

    rgb_array = np.asarray(oriented_rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    base_stages: dict[str, Image.Image | np.ndarray] = {
        "Original image (EXIF corrected)": oriented_rgb,
        "Grayscale image": gray,
    }

    if mode == "canvas":
        input_array, extra_stages = _prepare_canvas_input(gray)
    else:
        input_array, extra_stages = _prepare_photo_input(gray)
    base_stages.update(extra_stages)
    final_preview = extra_stages["Final 28x28 CNN input"]
    if not isinstance(final_preview, Image.Image):
        raise RuntimeError("The final preprocessing preview was not generated.")
    return input_array, final_preview, {
        name: stage if isinstance(stage, Image.Image) else _pil_stage(stage)
        for name, stage in base_stages.items()
    }


def predict_character(
    model: Any,
    input_image: np.ndarray,
    class_names: list[str],
) -> list[tuple[str, float]]:
    probabilities = np.asarray(model.predict(input_image, verbose=0)[0], dtype=np.float32)
    if len(probabilities) != len(class_names):
        raise RuntimeError(
            f"The model produced {len(probabilities)} scores, but "
            f"{len(class_names)} class names are available."
        )

    top_indices = np.argsort(probabilities)[-3:][::-1]
    return [
        (class_names[int(index)], float(probabilities[index] * 100.0))
        for index in top_indices
    ]
