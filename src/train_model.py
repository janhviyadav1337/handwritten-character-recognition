"""Train a real CNN on the preprocessed EMNIST Balanced dataset."""

from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "emnist_balanced_preprocessed.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "emnist_cnn.keras"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {DATA_PATH}. "
            "Run .venv/bin/python src/data_preprocessing.py first."
        )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    number_of_classes = int(max(y_train.max(), y_test.max()) + 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1.0),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
