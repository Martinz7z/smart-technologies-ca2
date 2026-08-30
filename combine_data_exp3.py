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

OUTPUT = "data/driving_log_exp3.csv"


def fix_paths(df, image_folder):
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

    # Use only 50% of Track 2 data.
    track2_sample = track2.sample(
        frac=0.5,
        random_state=42,
    )

    combined = pd.concat(
        [track1, track2_sample],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT,
        index=False,
        header=False,
    )

    print("Track 1 records:", len(track1))
    print("Track 2 records available:", len(track2))
    print("Track 2 records used:", len(track2_sample))
    print("Experiment 3 combined records:", len(combined))
    print("Saved:", OUTPUT)

    print("\nSteering counts:")
    print("Straight:", (combined["steering"] == 0).sum())
    print("Left:", (combined["steering"] < 0).sum())
    print("Right:", (combined["steering"] > 0).sum())


if __name__ == "__main__":
    main()