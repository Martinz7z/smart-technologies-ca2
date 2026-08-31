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
RECOVERY_LOG = "data_recovery/driving_log.csv"

OUTPUT = "data/driving_log_exp4.csv"


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

    recovery = pd.read_csv(
        RECOVERY_LOG,
        names=COLUMNS,
    )

    track1 = fix_paths(track1, "data")
    recovery = fix_paths(recovery, "data_recovery")

    combined = pd.concat(
        [track1, recovery],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT,
        index=False,
        header=False,
    )

    print("Track 1 records:", len(track1))
    print("Recovery records:", len(recovery))
    print("Experiment 4 combined records:", len(combined))
    print("Saved:", OUTPUT)

    print("\nSteering counts:")
    print("Straight:", (combined["steering"] == 0).sum())
    print("Left:", (combined["steering"] < 0).sum())
    print("Right:", (combined["steering"] > 0).sum())


if __name__ == "__main__":
    main()