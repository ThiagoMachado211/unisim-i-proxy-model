import pandas as pd

df = pd.read_parquet("data/extracted/summary_timeseries.parquet")

print(df.head())
print(df.shape)
print(df["variable"].unique())
print(df["entity_name"].unique())