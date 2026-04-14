def analyze_job_fit(cv_text, extracted_skills, job_description, vectorizer, skill_vocab, extract_skills_from_text):
    job_text = job_description.strip().lower()

    if not job_text:
        return 0.0, [], [], []
    

    job_skills = set(extract_skills_from_text(job_text, skill_vocab))
    cv_skills = set(extracted_skills)

    matched_skills = sorted(cv_skills & job_skills)
    missing_skills = sorted(job_skills - cv_skills)

    cv_vec = vectorizer.transform([cv_text.lower()])
    job_vec = vectorizer.transform(
        [job_text])

    similarity_score = (cv_vec * job_vec.T).toarray()[0][0]
    overlap_score = len(matched_skills) / len(job_skills) if job_skills else 0.0
    fit_score = round(0.6 * similarity_score + 0.4 * overlap_score, 3)

    suggestions = [
        f"Add or strengthen {skill} in your CV if you have experience with it"
        for skill in missing_skills[:5]
        ]

    return fit_score, matched_skills, missing_skills, suggestions