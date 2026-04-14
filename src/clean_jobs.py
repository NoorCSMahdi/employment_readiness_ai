import ast

import pandas as pd

# Load the raw jobs data
df = pd.read_csv("data/jobs.csv")


def clean_skills(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except (ValueError, SyntaxError, TypeError):
        return []


df["job_skills"] = df["job_skills"].apply(clean_skills)

column_mapping = {}
if "company" in df.columns:
    column_mapping["company"] = "company_name"
if "location" in df.columns:
    column_mapping["location"] = "job_location"
df = df.rename(columns=column_mapping)

keep_columns = ["job_title", "job_skills"]
if "company_name" in df.columns:
    keep_columns.append("company_name")
if "job_location" in df.columns:
    keep_columns.append("job_location")

df = df[keep_columns]
df.to_csv("data/jobs_clean.csv", index=False)

print("Jobs cleaned successfully!")
