import pandas as pd
import json
from pathlib import Path

df = pd.read_csv("../data/raw/berlin_data.csv")

# 1. Pick the target column
# Look at df.columns and set this correctly:
target = ["strava_total_trip_count"]  # adjust if the actual name differs

# 2. Columns to ignore as features
ignore_cols = set(target + ["counter_name", "date"])

feature_cols = [c for c in df.columns if c not in ignore_cols]

numerical = []
categorical = []
binary = []

for c in feature_cols:
    dtype = df[c].dtype

    if dtype == "object":
        categorical.append(c)
    else:
        # treat numeric columns with only {0,1} as binary
        unique_vals = set(df[c].dropna().unique())
        if unique_vals <= {0, 1}:
            binary.append(c)
        else:
            numerical.append(c)

feature_list = {
    "target": target,
    "numerical": numerical,
    "binary": binary,
    "categorical": categorical,
}

out_path = Path("../data/berlin_features_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(feature_list, f, indent=2)
print("Wrote", out_path)
