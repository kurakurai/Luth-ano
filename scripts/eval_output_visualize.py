import pandas as pd

path = "results_evals/details/...."
df = pd.read_parquet(path)

row = df.iloc[1]

print("=== INPUT ===")
print(row["example"])
print("\n=== PREDICTION ===")
print(row["predictions"][0])
print("\n=== GROUND TRUTH ===")
print(row["gold"][0])