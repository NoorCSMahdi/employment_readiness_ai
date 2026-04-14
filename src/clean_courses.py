import pandas as pd

try:
    df = pd.read_csv("data/Coursera.csv")
except FileNotFoundError:
    df = pd.read_csv("data/coursera.csv")

df["Gained Skills"] = df["Gained Skills"].fillna("")
df.to_csv("data/courses_clean.csv", index=False)

print("Courses cleaned successfully!")
