import pandas as pd

df = pd.read_parquet("../data/raw/berlin_data.parquet")
df.to_csv("../data/raw/berlin_data.csv", index=False)