import os
import pandas as pd


COLUMNS = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]

TRACK1_LOG = "data/driving_log.csv"
TRACK2_LOG = "data_track2/driving_log.csv"

OUTPUT = "data/driving_log_combined.csv"


def fix_paths(df, image_folder):
    """
    Replace the original absolute simulator paths with paths
    relative to this project.
    """

    for camera in ["center", "left", "right"]:
        df[camera] = df[camera].apply(
            lambda path: os.path.join(
                image_folder,
                "IMG",
                os.path.basename(str(path).strip()),
            )
        )

    return df


def main():
    track1 = pd.read_csv(
        TRACK1_LOG,
        names=COLUMNS,
    )

    track2 = pd.read_csv(
        TRACK2_LOG,
        names=COLUMNS,
    )

    track1 = fix_paths(track1, "data")
    track2 = fix_paths(track2, "data_track2")

    print("Track 1 records:", len(track1))
    print("Track 2 records:", len(track2))

    combined = pd.concat(
        [track1, track2],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT,
        index=False,
        header=False,
    )

    print("Combined records:", len(combined))
    print("Saved:", OUTPUT)

    print("\nSteering summary:")
    print(combined["steering"].describe())

    straight = (combined["steering"] == 0).sum()
    left = (combined["steering"] < 0).sum()
    right = (combined["steering"] > 0).sum()

    print("\nSteering counts:")
    print("Straight:", straight)
    print("Left:", left)
    print("Right:", right)


if __name__ == "__main__":
    main()