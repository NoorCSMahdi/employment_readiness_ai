import ast

from sklearn.metrics.pairwise import cosine_similarity


def parse_job_skills(job_skills_text):
    try:
        skills = ast.literal_eval(job_skills_text)
        return [str(s).strip().lower() for s in skills]
    except (ValueError, SyntaxError, TypeError):
        return []


def compute_skill_overlap(user_text, job_skills_text):
    job_skills = set(parse_job_skills(job_skills_text))
    user_skills = set(user_text.lower().split())

    if not user_skills:
        return 0.0

    return len(user_skills & job_skills) / len(user_skills)


def explain_match(user_text, job_skills_text):
    job_skills = set(parse_job_skills(job_skills_text))
    user_skills = set(user_text.lower().split())

    matched = sorted(user_skills & job_skills)
    missing = sorted(job_skills - user_skills)[:5]

    return matched, missing


def rank_jobs(user_text, jobs_df, job_texts, vectorizer, job_matrix, top_n=10):
    user_text = user_text.strip().lower()

    if not user_text:
        return jobs_df.iloc[0:0].copy()

    user_vec = vectorizer.transform([user_text])
    tfidf_scores = cosine_similarity(user_vec, job_matrix)[0]
    overlap_scores = jobs_df["job_skills"].apply(
        lambda x: compute_skill_overlap(user_text, x)
    )

    results = jobs_df.copy()
    results["score"] = (0.6 * tfidf_scores + 0.4 * overlap_scores).round(3)

    skill_columns = results["job_skills"].apply(
        lambda x: explain_match(user_text, x)
    )
    results["matched_skills"] = skill_columns.apply(lambda x: x[0])
    results["missing_skills"] = skill_columns.apply(lambda x: x[1])

    return results.sort_values("score", ascending=False).head(top_n).copy()