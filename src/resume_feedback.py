import re

EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_PATTERN = r"(\+?\d[\d\s\-\(\)]{7,})"

CV_CHECKS = [
    {
        "passed": lambda text, text_lower: re.search(EMAIL_PATTERN, text),
        "strength": "Email address found",
        "issue": "No email address found",
        "suggestion": "Add a professional email address",
        "score": 12,
    },
    {
        "passed": lambda text, text_lower: re.search(PHONE_PATTERN, text),
        "strength": "Phone number found",
        "issue": "No phone number found",
        "suggestion": "Add a phone number",
        "score": 12,
    },
    {
        "passed": lambda text, text_lower: "education" in text_lower,
        "strength": "Education section found",
        "issue": "Education section not found",
        "suggestion": "Add an Education section",
        "score": 12,
    },
    {
        "passed": lambda text, text_lower: "experience" in text_lower or "work experience" in text_lower,
        "strength": "Experience section found",
        "issue": "Experience section not found",
        "suggestion": "Add an Experience section",
        "score": 12,
    },
    {
        "passed": lambda text, text_lower: "skills" in text_lower,
        "strength": "Skills section found",
        "issue": "Skills section not found",
        "suggestion": "Add a clear Skills section",
        "score": 12,
    },
    {
        "passed": lambda text, text_lower: "linkedin.com" in text_lower,
        "strength": "LinkedIn profile found",
        "issue": "LinkedIn profile not found",
        "suggestion": "Add your LinkedIn profile link",
        "score": 10,
    },
    {
        "passed": lambda text, text_lower: "github.com" in text_lower,
        "strength": "GitHub profile found",
        "issue": "GitHub profile not found",
        "suggestion": "Add your GitHub profile link",
        "score": 10,
    },
]


def skill_score(extracted_skills):
    skill_count = len(extracted_skills)

    if skill_count >= 8:
        return 10
    if skill_count >= 5:
        return 7
    if skill_count >= 3:
        return 4

    return 0


def run_cv_checks(text, text_lower):
    strengths = []
    issues = []
    suggestions = []
    score = 0

    for check in CV_CHECKS:
        if check["passed"](text, text_lower):
            strengths.append(check["strength"])
            score += check["score"]
        else:
            issues.append(check["issue"])
            suggestions.append(check["suggestion"])

    return score, strengths, issues, suggestions


def calculate_cv_score(text, extracted_skills):
    text_lower = text.lower()
    score, _, _, _ = run_cv_checks(text, text_lower)

    if len(text.split()) >= 120:
        score += 10

    score += skill_score(extracted_skills)

    return min(score, 100)

def analyze_cv(text, extracted_skills, all_job_skills):
    text_lower = text.lower()
    _, strengths, issues, suggestions = run_cv_checks(text, text_lower)

    word_count = len(text.split())
    if word_count >= 120:
        strengths.append("CV has a reasonable amount of content")
    else:
        issues.append("CV looks too short")
        suggestions.append("Add more project, education, or experience details")

    if len(extracted_skills) >= 5:
        strengths.append("A good number of skills were detected")
    else:
        issues.append("Very few skills were detected")
        suggestions.append("Mention more technical skills clearly in the CV")

    extracted_skill_set = set(extracted_skills)
    missing_skills = [skill for skill in all_job_skills[:50] if skill not in extracted_skill_set][:15]

    return strengths, issues, suggestions, missing_skills