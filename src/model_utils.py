"""Model loading, image preprocessing, and prediction utilities."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emnist_cnn.keras"
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "emnist_balanced_preprocessed.npz"

EMNIST_BALANCED_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
]


def load_class_names() -> list[str]:
    if PROCESSED_DATA_PATH.exists():
        with np.load(PROCESSED_DATA_PATH, allow_pickle=False) as data:
            if "class_names" in data:
                names = [str(name) for name in data["class_names"].tolist()]
                if names:
                    return names
    return EMNIST_BALANCED_CLASSES


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained CNN model found at {model_path}. "
            "Train it with: .venv/bin/python src/train_model.py"
        )

    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def preprocess_uploaded_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Convert an uploaded image to the 28x28, normalized CNN input format."""
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        image = background.convert("RGB")

    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.float32)
    if pixels.size == 0:
        raise ValueError("The uploaded image is empty.")

    # EMNIST is black-on-white. Convert dark text on light backgrounds to
    # white-on-black by inverting when the page is predominantly light.
    if float(pixels.mean()) > 127:
        grayscale = ImageOps.invert(grayscale)
        pixels = np.asarray(grayscale, dtype=np.float32)

    # Remove low-level noise and crop to the character's bounding box.
    threshold = pixels > 35
    if not threshold.any():
        raise ValueError(
            "The image appears blank. Upload a clear handwritten character."
        )

    coordinates = np.argwhere(threshold)
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    crop = grayscale.crop((int(x0), int(y0), int(x1 + 1), int(y1 + 1)))

    canvas = Image.new("L", (28, 28), 0)
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    left = (28 - crop.width) // 2
    top = (28 - crop.height) // 2
    canvas.paste(crop, (left, top))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array[..., np.newaxis]
    return array[np.newaxis, ...], canvas


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
