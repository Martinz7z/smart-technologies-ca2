import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


DATA_PATH = "data/driving_log_exp4_balanced.csv"
MODEL_PATH = "models/experiment_4_model.keras"
RESULTS_PATH = "results"

COLUMNS = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]

STEERING_CORRECTION = 0.20
BATCH_SIZE = 64
EPOCHS = 20


def preprocess_image(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0

    return image


def load_image(path):
    path = path.strip()

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def choose_camera(row):
    """
    Randomly choose centre, left or right camera.

    A steering correction is applied to the side cameras so that
    they teach the model how to recover towards the centre.
    """

    camera = random.choice(["center", "left", "right"])

    steering = float(row["steering"])

    if camera == "left":
        steering += STEERING_CORRECTION
    elif camera == "right":
        steering -= STEERING_CORRECTION

    image = load_image(row[camera])

    return image, steering


def augment_image(image, steering):
    """
    Apply simple augmentation to improve generalisation.
    """

    # Random horizontal flip.
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering

    # Random brightness adjustment.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    brightness = random.uniform(0.6, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image, steering


def generator(data, batch_size=64, training=True):
    while True:
        batch_images = []
        batch_steering = []

        for _ in range(batch_size):
            row = data.sample(1).iloc[0]

            if training:
                image, steering = choose_camera(row)
                image, steering = augment_image(image, steering)
            else:
                image = load_image(row["center"])
                steering = float(row["steering"])

            image = preprocess_image(image)

            batch_images.append(image)
            batch_steering.append(steering)

        yield (
            np.array(batch_images, dtype=np.float32),
            np.array(batch_steering, dtype=np.float32),
        )


def build_model():
    model = Sequential(
        [
            Conv2D(
                24,
                (5, 5),
                strides=(2, 2),
                activation="elu",
                input_shape=(66, 200, 3),
            ),
            Conv2D(
                36,
                (5, 5),
                strides=(2, 2),
                activation="elu",
            ),
            Conv2D(
                48,
                (5, 5),
                strides=(2, 2),
                activation="elu",
            ),
            Conv2D(
                64,
                (3, 3),
                activation="elu",
            ),
            Conv2D(
                64,
                (3, 3),
                activation="elu",
            ),
            Flatten(),
            Dense(100, activation="elu"),
            Dropout(0.5),
            Dense(50, activation="elu"),
            Dense(10, activation="elu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="mse",
    )

    return model


def save_training_plot(history):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Experiment 4 Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            "experiment_4_training_history.png",
        )
    )

    plt.close()


def main():
    os.makedirs("models", exist_ok=True)

    data = pd.read_csv(
        DATA_PATH,
        names=COLUMNS,
    )

    train_data, validation_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")

    model = build_model()

    model.summary()

    train_generator = generator(
        train_data,
        batch_size=BATCH_SIZE,
        training=True,
    )

    validation_generator = generator(
        validation_data,
        batch_size=BATCH_SIZE,
        training=False,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=max(
            1,
            len(train_data) // BATCH_SIZE,
        ),
        validation_data=validation_generator,
        validation_steps=max(
            1,
            len(validation_data) // BATCH_SIZE,
        ),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_training_plot(history)

    print("\nTraining complete.")
    print(f"Best model saved to: {MODEL_PATH}")
    print(
    "Training plot saved to: "
    "results/experiment_4_training_history.png"
)


if __name__ == "__main__":
    main()  