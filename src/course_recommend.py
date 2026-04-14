import re

import pandas as pd


def courses_matching_skill(courses_df, skill):
    pattern = r"\b" + re.escape(skill.strip()) + r"\b"
    return courses_df[
        courses_df["Gained Skills"].fillna("").str.contains(
            pattern, case=False, na=False, regex=True
        )
    ]


def recommend_courses_from_missing_skills(courses_df, missing_skills, top_n=10):
    results = []
    seen_titles = set()

    for skill in missing_skills:
        for _, row in courses_matching_skill(courses_df, skill).iterrows():
            title = str(row.get("Title", "")).strip()

            if not title or title in seen_titles:
                continue

            seen_titles.add(title)
            results.append({
                "skill": skill,"Title": title,
                "Institution": row.get("Institution", "N/A"),
                "Rate": row.get("Rate", "N/A"),
            })

            if len(results) >= top_n:
                return pd.DataFrame(results)

    return pd.DataFrame(results)