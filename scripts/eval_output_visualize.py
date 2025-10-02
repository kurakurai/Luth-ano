import pandas as pd

path = "results/details//Luth-1.7B-Instruct/2025-08-20T01-33-27.422289/details_community|scholar_100_fr|0_2025-08-20T01-33-27.422289.parquet"
df = pd.read_parquet(path)
for i in range(30):
    row = df.iloc[i]

    # print("=== INPUT ===")
    # print(row["example"])
    print("\n=== PREDICTION ===")
    print(row["predictions"][0])
    print("\n=== GROUND TRUTH ===")
    print(row["gold"][0])
