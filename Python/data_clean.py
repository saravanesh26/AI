import pandas as pd

# Load your data
df = pd.read_csv("Soil data.csv")

# Group by district and take average
result = df.groupby("District").mean().reset_index()

print(result)

# Optional: save it
result.to_csv("district_avg.csv", index=False)