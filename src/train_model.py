"""Train, compare, and evaluate a full-data EMNIST Balanced CNN candidate."""

import csv
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "emnist_balanced_preprocessed.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "emnist_cnn.keras"
CANDIDATE_PATH = PROJECT_ROOT / "models" / "emnist_cnn_candidate.keras"
BACKUP_DIR = PROJECT_ROOT / "models" / "backups"
REPORT_DIR = PROJECT_ROOT / "reports" / "full_training"
SEED = 42
TRAIN_PER_CLASS = 2160
VALIDATION_PER_CLASS = 240
MAX_EPOCHS = 30
BATCH_SIZE = 128

EMNIST_BALANCED_MAPPING = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abdefghnqrt")
)


def stratified_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []

    for class_index in range(len(EMNIST_BALANCED_MAPPING)):
        class_indices = np.flatnonzero(y == class_index)
        if len(class_indices) != TRAIN_PER_CLASS + VALIDATION_PER_CLASS:
            raise ValueError(
                f"Class {class_index} has {len(class_indices)} records; "
                f"expected {TRAIN_PER_CLASS + VALIDATION_PER_CLASS}."
            )
        shuffled = rng.permutation(class_indices)
        train_indices.append(shuffled[:TRAIN_PER_CLASS])
        validation_indices.append(shuffled[TRAIN_PER_CLASS:])

    train = rng.permutation(np.concatenate(train_indices))
    validation = rng.permutation(np.concatenate(validation_indices))
    if len(train) != 101520 or len(validation) != 11280:
        raise RuntimeError("The stratified split has unexpected sample counts.")
    if np.intersect1d(train, validation).size:
        raise RuntimeError("Training and validation indices overlap.")
    return train, validation


def augment_example(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)
    angle = tf.random.uniform([], -0.10, 0.10)
    zoom = tf.random.uniform([], 0.90, 1.10)
    shear = tf.random.uniform([], -0.06, 0.06)
    shift_x = tf.random.uniform([], -2.0, 2.0)
    shift_y = tf.random.uniform([], -2.0, 2.0)

    cosine = tf.cos(angle) / zoom
    sine = tf.sin(angle) / zoom
    center = 13.5
    transform = tf.stack([
        cosine,
        sine + shear,
        center - (cosine + sine + shear) * center + shift_x,
        -sine,
        cosine,
        center - (cosine - sine) * center + shift_y,
        0.0,
        0.0,
    ])
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transform[tf.newaxis, ...],
        output_shape=tf.constant([28, 28]),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )[0]

    stroke_chance = tf.random.uniform([])
    dilated = tf.nn.max_pool2d(image[tf.newaxis, ...], 2, 1, "SAME")[0]
    eroded = 1.0 - tf.nn.max_pool2d(
        (1.0 - image)[tf.newaxis, ...], 2, 1, "SAME"
    )[0]
    image = tf.cond(
        stroke_chance < 0.07,
        lambda: dilated,
        lambda: tf.cond(stroke_chance < 0.14, lambda: eroded, lambda: image),
    )

    contrast = tf.random.uniform([], 0.90, 1.10)
    brightness = tf.random.uniform([], -0.025, 0.025)
    mean = tf.reduce_mean(image)
    image = (image - mean) * contrast + mean + brightness
    image += tf.random.normal(tf.shape(image), stddev=0.015)
    return tf.clip_by_value(image, 0.0, 1.0), label


def create_datasets(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x[train_indices], y[train_indices].astype(np.int32))
    )
    train_ds = train_ds.shuffle(
        16384,
        seed=SEED,
        reshuffle_each_iteration=True,
    ).map(
        augment_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    validation_ds = tf.data.Dataset.from_tensor_slices(
        (x[validation_indices], y[validation_indices].astype(np.int32))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, validation_ds


def residual_block(
    inputs: tf.Tensor,
    filters: int,
    pool: bool,
    dropout: float,
) -> tf.Tensor:
    shortcut = inputs
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if int(shortcut.shape[-1]) != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, padding="same", use_bias=False
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)
    if pool:
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    if dropout:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    return x


def build_model(number_of_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = residual_block(inputs, 32, pool=True, dropout=0.08)
    x = residual_block(x, 64, pool=True, dropout=0.12)
    x = residual_block(x, 128, pool=False, dropout=0.18)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="emnist_residual_cnn_candidate")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
    predicted = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predicted == y))
    loss = float(
        -np.log(
            np.clip(probabilities[np.arange(len(y)), y.astype(np.int64)], 1e-8, 1.0)
        ).mean()
    )
    print(f"{name}: {len(y)} examples, accuracy={accuracy:.6f}, loss={loss:.6f}")
    return {"count": len(y), "accuracy": accuracy, "loss": loss}, predicted


def calculate_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    class_count = len(EMNIST_BALANCED_MAPPING)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)

    report: list[dict[str, Any]] = []
    for index, character in enumerate(EMNIST_BALANCED_MAPPING):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report.append({
            "class_index": index,
            "character": character,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        })
    return matrix, report


def save_evaluation_artifacts(
    history: tf.keras.callbacks.History,
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "epoch": epoch + 1,
            "accuracy": history.history["accuracy"][epoch],
            "val_accuracy": history.history["val_accuracy"][epoch],
            "loss": history.history["loss"][epoch],
            "val_loss": history.history["val_loss"][epoch],
            "learning_rate": history.history["learning_rate"][epoch],
        }
        for epoch in range(len(history.history["accuracy"]))
    ]
    history_path = REPORT_DIR / "candidate_training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, len(rows) + 1)
    axes[0].plot(epochs, [row["accuracy"] for row in rows], marker="o")
    axes[0].plot(epochs, [row["val_accuracy"] for row in rows], marker="o")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(["Training", "Validation"])
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["loss"] for row in rows], marker="o")
    axes[1].plot(epochs, [row["val_loss"] for row in rows], marker="o")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(["Training", "Validation"])
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "candidate_training_curves.png", dpi=160)
    plt.close(figure)

    confusion, class_report = calculate_class_report(y_test, predictions)
    confusion_path = REPORT_DIR / "candidate_confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *EMNIST_BALANCED_MAPPING])
        for character, row in zip(EMNIST_BALANCED_MAPPING, confusion):
            writer.writerow([character, *row.tolist()])

    with (REPORT_DIR / "candidate_classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(class_report[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(class_report)

    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(confusion, cmap="Blues", interpolation="nearest")
    axis.set_xticks(range(len(EMNIST_BALANCED_MAPPING)), EMNIST_BALANCED_MAPPING)
    axis.set_yticks(range(len(EMNIST_BALANCED_MAPPING)), EMNIST_BALANCED_MAPPING)
    axis.set_xlabel("Predicted character")
    axis.set_ylabel("Actual character")
    axis.set_title("EMNIST Balanced test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "candidate_confusion_matrix.png", dpi=160)
    plt.close(figure)

    similar_confusions = []
    for actual_index, row in enumerate(confusion):
        row_copy = row.copy()
        row_copy[actual_index] = 0
        for predicted_index in np.argsort(row_copy)[-2:][::-1]:
            if row_copy[predicted_index]:
                similar_confusions.append({
                    "actual": EMNIST_BALANCED_MAPPING[actual_index],
                    "predicted": EMNIST_BALANCED_MAPPING[int(predicted_index)],
                    "count": int(row_copy[predicted_index]),
                })
    similar_confusions.sort(key=lambda item: item["count"], reverse=True)

    with (REPORT_DIR / "candidate_performance.json").open("w", encoding="utf-8") as file:
        json.dump({
            "completed_epochs": len(rows),
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "training_examples": 101520,
            "validation_examples": 11280,
            "test_examples": 18800,
            "class_mapping": EMNIST_BALANCED_MAPPING,
            "top_confusions": similar_confusions[:25],
            "class_report": class_report,
        }, file, indent=2)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Preprocessed EMNIST data not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_backup = BACKUP_DIR / f"emnist_cnn_baseline_{timestamp}.keras"
    shutil.copy2(MODEL_PATH, baseline_backup)
    if CANDIDATE_PATH.exists():
        shutil.copy2(
            CANDIDATE_PATH,
            BACKUP_DIR / f"emnist_cnn_previous_candidate_{timestamp}.keras",
        )
    print(f"Baseline model backed up to {baseline_backup}", flush=True)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        x = data["x_train"]
        y = data["y_train"].astype(np.int32)
        x_test = data["x_test"]
        y_test = data["y_test"].astype(np.int32)

    if x.shape[1:] != (28, 28, 1) or x_test.shape[1:] != (28, 28, 1):
        raise ValueError(f"Unexpected image shapes: {x.shape}, {x_test.shape}")
    if len(EMNIST_BALANCED_MAPPING) != 47 or set(np.unique(y)) != set(range(47)):
        raise ValueError("EMNIST label mapping or 47-class coverage is invalid.")
    if np.bincount(y, minlength=47).min() != 2400:
        raise ValueError("Expected exactly 2,400 training examples per class.")

    train_indices, validation_indices = stratified_split(y)
    print(
        f"Split verified: {len(train_indices)} training, "
        f"{len(validation_indices)} validation, {len(y_test)} untouched test images.",
        flush=True,
    )
    print(
        f"Class mapping: {', '.join(EMNIST_BALANCED_MAPPING)}",
        flush=True,
    )
    print(
        f"TensorFlow devices: {tf.config.list_physical_devices()}",
        flush=True,
    )
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_validation = x[validation_indices]
    y_validation = y[validation_indices]

    print("Evaluating baseline on the same stratified split and test set.", flush=True)
    baseline_model = tf.keras.models.load_model(baseline_backup)
    baseline_scores: dict[str, dict[str, Any]] = {}
    for name, images, labels in (
        ("training", x_train, y_train),
        ("validation", x_validation, y_validation),
        ("testing", x_test, y_test),
    ):
        baseline_scores[name], _ = evaluate_model(baseline_model, images, labels, f"Baseline {name}")
    del baseline_model
    tf.keras.backend.clear_session()

    train_ds, validation_ds = create_datasets(x, y, train_indices, validation_indices)
    model = build_model(len(EMNIST_BALANCED_MAPPING))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CANDIDATE_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(REPORT_DIR / "candidate_epoch_log.csv"),
            append=False,
        ),
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting full-data CNN training with online training-only augmentation.", flush=True)
    training_started = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    training_seconds = time.perf_counter() - training_started

    if not CANDIDATE_PATH.exists():
        raise RuntimeError("Training finished without producing a candidate model.")
    del model, train_ds, validation_ds
    tf.keras.backend.clear_session()
    candidate = tf.keras.models.load_model(CANDIDATE_PATH)

    candidate_scores: dict[str, dict[str, Any]] = {}
    predictions: np.ndarray | None = None
    for name, images, labels in (
        ("training", x_train, y_train),
        ("validation", x_validation, y_validation),
        ("testing", x_test, y_test),
    ):
        candidate_scores[name], predicted = evaluate_model(
            candidate,
            images,
            labels,
            f"Candidate {name}",
        )
        if name == "testing":
            predictions = predicted
    if predictions is None:
        raise RuntimeError("Candidate test predictions were not generated.")

    save_evaluation_artifacts(history, y_test, predictions)
    comparison: dict[str, Any] = {
        "training_duration_seconds": training_seconds,
        "completed_epochs": len(history.history["accuracy"]),
        "max_epochs": MAX_EPOCHS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "baseline": baseline_scores,
        "candidate": candidate_scores,
        "candidate_model_path": str(CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
        "baseline_backup_path": str(baseline_backup.relative_to(PROJECT_ROOT)),
        "promoted_to_website": False,
    }

    configured_photo_path = os.environ.get("HANDWRITTEN_EVAL_IMAGE")
    photo_path = (
        Path(configured_photo_path).expanduser()
        if configured_photo_path
        else Path.home() / "Downloads" / "Image.jpeg"
    )
    if configured_photo_path and not photo_path.is_file():
        raise FileNotFoundError(f"Handwritten evaluation image not found: {photo_path}")
    if photo_path.exists():
        from PIL import Image
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from model_utils import load_class_names, preprocess_uploaded_image, predict_character

        photo_input, _, _ = preprocess_uploaded_image(
            Image.open(photo_path),
            mode="photo",
        )
        comparison["handwritten_photo"] = {"filename": photo_path.name}
        for label, model_path in (
            ("baseline", baseline_backup),
            ("candidate", CANDIDATE_PATH),
        ):
            photo_model = tf.keras.models.load_model(model_path)
            comparison["handwritten_photo"][label] = predict_character(
                photo_model,
                photo_input,
                load_class_names(),
            )
            del photo_model

    if candidate_scores["testing"]["accuracy"] > baseline_scores["testing"]["accuracy"]:
        shutil.copy2(CANDIDATE_PATH, MODEL_PATH)
        comparison["promoted_to_website"] = True
        print(f"Candidate promoted to website model: {MODEL_PATH}", flush=True)
    else:
        print(
            "Candidate did not beat baseline test accuracy; website model remains unchanged.",
            flush=True,
        )

    comparison_path = REPORT_DIR / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
