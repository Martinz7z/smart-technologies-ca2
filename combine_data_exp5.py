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
TRACK1_RECOVERY_LOG = "data_recovery/driving_log.csv"
TRACK2_RECOVERY_LOG = "data_track2_recovery/driving_log.csv"

OUTPUT = "data/driving_log_exp5.csv"


def fix_paths(df, folder):
    for camera in ["center", "left", "right"]:
        df[camera] = df[camera].apply(
            lambda path: os.path.join(
                folder,
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

    track1_recovery = pd.read_csv(
        TRACK1_RECOVERY_LOG,
        names=COLUMNS,
    )

    track2_recovery = pd.read_csv(
        TRACK2_RECOVERY_LOG,
        names=COLUMNS,
    )

    track1 = fix_paths(track1, "data")
    track1_recovery = fix_paths(track1_recovery, "data_recovery")
    track2_recovery = fix_paths(
        track2_recovery,
        "data_track2_recovery",
    )

    combined = pd.concat(
        [track1, track1_recovery, track2_recovery],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT,
        index=False,
        header=False,
    )

    print("Track 1 records:", len(track1))
    print("Track 1 recovery records:", len(track1_recovery))
    print("Track 2 recovery records:", len(track2_recovery))
    print("Experiment 5 combined records:", len(combined))
    print("Saved:", OUTPUT)

    print("\nSteering counts:")
    print("Straight:", (combined["steering"] == 0).sum())
    print("Left:", (combined["steering"] < 0).sum())
    print("Right:", (combined["steering"] > 0).sum())


if __name__ == "__main__":
    main()