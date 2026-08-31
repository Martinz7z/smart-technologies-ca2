import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = "data/driving_log_exp4.csv"
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


def load_data():
    data = pd.read_csv(DATA_PATH, names=COLUMNS)
    return data


def balance_data(data, bins=25, max_samples_per_bin=400):
    """
    Reduce over-represented steering-angle ranges.

    The simulator produces many zero-steering samples because
    much of the track consists of straight sections.
    """
    histogram, bin_edges = np.histogram(data["steering"], bins=bins)

    remove_indices = []

    for i in range(bins):
        indices = data[
            (data["steering"] >= bin_edges[i])
            & (data["steering"] <= bin_edges[i + 1])
        ].index.tolist()

        random.shuffle(indices)

        if len(indices) > max_samples_per_bin:
            remove_indices.extend(indices[max_samples_per_bin:])

    balanced_data = data.drop(index=remove_indices).reset_index(drop=True)

    return balanced_data


def preprocess_image(image):
    """
    Prepare an image for the NVIDIA behavioural-cloning CNN.
    """

    # Remove unnecessary sky and lower car/bonnet area.
    image = image[60:135, :, :]

    # NVIDIA's behavioural cloning architecture commonly uses YUV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Reduce image noise.
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # NVIDIA model input size: 200 x 66.
    image = cv2.resize(image, (200, 66))

    # Normalise pixels to the range 0-1.
    image = image / 255.0

    return image


def save_distribution(data, filename, title):
    plt.figure(figsize=(10, 6))
    plt.hist(data["steering"], bins=25, edgecolor="black")
    plt.title(title)
    plt.xlabel("Steering Angle")
    plt.ylabel("Number of Samples")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.close()


def save_preprocessing_example(data):
    sample_path = data.iloc[len(data) // 2]["center"].strip()

    # OpenCV loads as BGR, convert to RGB for our pipeline.
    original = cv2.imread(sample_path)

    if original is None:
        raise FileNotFoundError(f"Could not load image: {sample_path}")

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(original)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original Simulator Image")
    axes[0].axis("off")

    # Convert YUV back to RGB only for displaying the processed example.
    display_image = cv2.cvtColor(
        (processed * 255).astype(np.uint8),
        cv2.COLOR_YUV2RGB,
    )

    axes[1].imshow(display_image)
    axes[1].set_title("Preprocessed Image (66 x 200)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "preprocessing_example.png"))
    plt.close()


def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    data = load_data()

    print(f"Samples before balancing: {len(data)}")

    save_distribution(
        data,
        "steering_distribution_unbalanced.png",
        "Steering Distribution Before Balancing",
    )

    balanced_data = balance_data(data)

    print(f"Samples after balancing: {len(balanced_data)}")
    print(f"Samples removed: {len(data) - len(balanced_data)}")

    print(
        f"Straight samples after balancing: "
        f"{(balanced_data['steering'] == 0).sum()}"
    )
    print(
        f"Left samples after balancing: "
        f"{(balanced_data['steering'] < 0).sum()}"
    )
    print(
        f"Right samples after balancing: "
        f"{(balanced_data['steering'] > 0).sum()}"
    )

    save_distribution(
        balanced_data,
        "steering_distribution_balanced.png",
        "Steering Distribution After Balancing",
    )

    save_preprocessing_example(balanced_data)

    balanced_data.to_csv(
        "data/driving_log_exp4_balanced.csv",
        index=False,
        header=False,
    )

    print("\nSaved:")
    print("data/driving_log_exp4_balanced.csv")
    print("results/steering_distribution_unbalanced.png")
    print("results/steering_distribution_balanced.png")
    print("results/preprocessing_example.png")


if __name__ == "__main__":
    main()