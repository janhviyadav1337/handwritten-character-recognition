"""Download, preprocess, and preview the EMNIST Balanced dataset.

Run this file from the project virtual environment. TensorFlow Datasets keeps
its downloaded copy under ``dataset/tfds``; the normalized NumPy arrays are
saved as ``dataset/emnist_balanced_preprocessed.npz``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
TFDS_DATA_DIR = DATASET_DIR / "tfds"
PROCESSED_DATA_PATH = DATASET_DIR / "emnist_balanced_preprocessed.npz"
DATASET_NAME = "emnist/balanced"
SAMPLE_GRID_SIZE = 16


def convert_split_to_numpy(
    dataset: object,
    split_name: str,
    expected_count: int,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Correct image orientation and convert one TFDS split to NumPy arrays."""
    if len(image_shape) != 3 or image_shape[0] != image_shape[1]:
        raise ValueError(
            f"Expected square grayscale images with a channel dimension; "
            f"received shape {image_shape}."
        )
    if expected_count <= 0:
        raise ValueError(f"The {split_name} split has no examples.")

    images = np.empty((expected_count, *image_shape), dtype=np.uint8)
    labels = np.empty(expected_count, dtype=np.int64)
    loaded_count = 0

    # TFDS documents EMNIST images as rotated and horizontally inverted.
    # Transposing the two image axes corrects the orientation for display.
    for loaded_count, (image, label) in enumerate(tfds.as_numpy(dataset), start=1):
        if loaded_count > expected_count:
            raise ValueError(
                f"The {split_name} split contains more examples than its "
                f"metadata reports ({expected_count})."
            )

        image_array = np.asarray(image, dtype=np.uint8)
        if image_array.shape != image_shape:
            raise ValueError(
                f"Unexpected image shape in {split_name}: {image_array.shape}; "
                f"expected {image_shape}."
            )

        images[loaded_count - 1] = np.transpose(image_array, (1, 0, 2))
        labels[loaded_count - 1] = int(label)

    if loaded_count != expected_count:
        raise ValueError(
            f"The {split_name} split yielded {loaded_count} examples, but its "
            f"metadata reports {expected_count}."
        )

    # CNNs expect a channel dimension. TFDS already provides (height, width, 1);
    # reshape explicitly to keep that format clear before converting to floats.
    images = images.reshape((expected_count, *image_shape))
    images = images.astype(np.float32) / 255.0
    return images, labels


def display_samples(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> None:
    """Show up to 16 training images with labels from TFDS metadata."""
    sample_count = min(SAMPLE_GRID_SIZE, len(images))
    rows = int(np.ceil(sample_count / 4))
    figure, axes = plt.subplots(rows, 4, figsize=(9, 2.4 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        if index >= sample_count:
            axis.axis("off")
            continue

        class_index = int(labels[index])
        axis.imshow(images[index].squeeze(axis=-1), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"Label: {class_names[class_index]}")
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Load both dataset splits, preprocess them, save them, and preview samples."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {DATASET_NAME}. The first run downloads the dataset.")
    builder = tfds.builder(DATASET_NAME, data_dir=str(TFDS_DATA_DIR))
    builder.download_and_prepare()
    info = builder.info

    class_names = list(info.features["label"].names)
    number_of_classes = info.features["label"].num_classes
    if len(class_names) != number_of_classes:
        raise ValueError(
            "The label names from TFDS do not match its reported number of classes."
        )

    train_count = info.splits["train"].num_examples
    test_count = info.splits["test"].num_examples
    image_shape = tuple(info.features["image"].shape)

    print(f"Total training images: {train_count}")
    print(f"Total testing images: {test_count}")
    print(f"Number of character classes: {number_of_classes}")
    print("Available class labels:")
    print(", ".join(f"{index}: {name}" for index, name in enumerate(class_names)))

    train_dataset = builder.as_dataset(split="train", as_supervised=True)
    test_dataset = builder.as_dataset(split="test", as_supervised=True)

    x_train, y_train = convert_split_to_numpy(
        train_dataset, "train", train_count, image_shape
    )
    x_test, y_test = convert_split_to_numpy(
        test_dataset, "test", test_count, image_shape
    )

    if np.any(y_train < 0) or np.any(y_train >= number_of_classes):
        raise ValueError("A training label is outside the TFDS class-label range.")
    if np.any(y_test < 0) or np.any(y_test >= number_of_classes):
        raise ValueError("A testing label is outside the TFDS class-label range.")

    print(f"Training image array shape: {x_train.shape}")
    print(f"Training label array shape: {y_train.shape}")
    print(f"Testing image array shape: {x_test.shape}")
    print(f"Testing label array shape: {y_test.shape}")
    print(f"Normalized pixel range: {x_train.min():.1f} to {x_train.max():.1f}")

    try:
        np.savez_compressed(
            PROCESSED_DATA_PATH,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            class_names=np.asarray(class_names, dtype=str),
        )
    except OSError as error:
        raise OSError(
            f"Could not save the preprocessed dataset to {PROCESSED_DATA_PATH}."
        ) from error

    print(f"Preprocessed arrays saved to: {PROCESSED_DATA_PATH}")
    display_samples(x_train, y_train, class_names)


if __name__ == "__main__":
    main()
