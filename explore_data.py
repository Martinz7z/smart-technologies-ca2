import os

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "data/driving_log.csv"

columns = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]

# The simulator CSV does not contain a header row.
data = pd.read_csv(DATA_PATH, names=columns)

print("Dataset summary")
print("----------------")
print(f"Driving records: {len(data)}")
print(f"Camera images available: {len(data) * 3}")

print("\nSteering statistics:")
print(data["steering"].describe())

print(f"\nStraight samples (steering = 0): {(data['steering'] == 0).sum()}")
print(f"Left steering samples: {(data['steering'] < 0).sum()}")
print(f"Right steering samples: {(data['steering'] > 0).sum()}")

# Plot steering-angle distribution.
plt.figure(figsize=(10, 6))
plt.hist(data["steering"], bins=25, edgecolor="black")
plt.title("Steering Angle Distribution Before Balancing")
plt.xlabel("Steering Angle")
plt.ylabel("Number of Samples")
plt.grid(axis="y", alpha=0.3)

os.makedirs("results", exist_ok=True)
plt.savefig("results/steering_distribution_before.png")
plt.show()