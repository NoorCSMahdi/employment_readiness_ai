#Use the trained classifier to predict career path from a user's skill text.
def predict_career_path(skills_text, classifier):
    
    if not skills_text.strip():
        return "Not enough data"
    return classifier.predict([skills_text])[0]


def build_career_summary(cv_score, career_path, missing_skills, top_jobs_df):
    best_match = top_jobs_df.iloc[0]["job_title"] if not top_jobs_df.empty else "No job match found"
    main_missing = ", ".join(missing_skills[:3]) if missing_skills else "No major missing skills"

    return {
        "cv_score": cv_score,
        "career_path": career_path,
         "best_match": best_match,
         "main_missing": main_missing,
    }