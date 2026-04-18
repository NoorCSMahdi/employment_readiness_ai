import pandas as pd

df = pd.read_csv("data/Coursera.csv")

df["Gained Skills"] = df["Gained Skills"].fillna("")
df.to_csv("data/courses_clean.csv", index=False)

print("Courses cleaned successfully!")
