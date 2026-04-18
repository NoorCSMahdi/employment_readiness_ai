import ast
import html
import os
import pickle
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from src.career_utils import build_career_summary, predict_career_path
from src.course_recommend import recommend_courses_from_missing_skills
from src.cv_parser import extract_text_from_uploaded_file
from src.job_fit import analyze_job_fit
from src.job_matcher import rank_jobs
from src.resume_feedback import analyze_cv, calculate_cv_score
from src.skills import extract_skills_from_text
from src.theme_layout import apply_theme, render_header

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_PATH = os.path.join(BASE_DIR, "data", "jobs_clean.csv")
COURSES_PATH = os.path.join(BASE_DIR, "data", "courses_clean.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

st.set_page_config(
    page_title="AI Career Assistant",
    layout="wide"
    )

apply_theme(st)
render_header(st)

@st.cache_resource
def load_vectorizer():
    with open(VECTORIZER_PATH, "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_jobs():
    jobs = pd.read_csv(JOBS_PATH)
    jobs = jobs.drop_duplicates(subset=["job_title"]).copy()
    jobs = jobs.reset_index(drop=True)
    return jobs

@st.cache_data
def load_courses():
    return pd.read_csv(COURSES_PATH)

@st.cache_data
def build_job_texts(jobs_df):
    return (
        jobs_df["job_title"].fillna("").astype(str) + " " +
        jobs_df["job_skills"].fillna("").astype(str)
    )

@st.cache_resource
def build_job_matrix(job_texts, _vectorizer):
    return _vectorizer.transform(job_texts)

def parse_skills_value(skills_value):
    # The cleaned jobs dataset stores skills as stringified Python lists.
    try:
        parsed = ast.literal_eval(skills_value)
    except (ValueError, SyntaxError, TypeError):
        return []

    if not isinstance(parsed, list):
        return []

    return [str(skill).strip().lower() for skill in parsed if str(skill).strip()]

@st.cache_data
def build_skill_vocab(job_skills_column):
    vocab = set()

    for skills in job_skills_column.dropna():
        vocab.update(parse_skills_value(skills))

    return sorted(vocab)

@st.cache_data
def get_all_job_skills(job_skills_column):
    all_skills = set()

    for skills in job_skills_column.dropna():
        all_skills.update(parse_skills_value(skills))

    return sorted(all_skills)

def show_job_results(results_df):
    for _, row in results_df.iterrows():
        title = html.escape(str(row.get("job_title", "Untitled Role")))
        company = html.escape(str(row.get("company_name", "N/A")))
        location = html.escape(str(row.get("job_location", "N/A")))

        matched_skills = row.get("matched_skills", [])
        if not isinstance(matched_skills, list):
            matched_skills = []

        missing_skills = row.get("missing_skills", [])
        if not isinstance(missing_skills, list):
            missing_skills = []

        matched_html = "".join(
            f'<span class="skill-chip">{html.escape(str(skill))}</span>'
            for skill in matched_skills
        ) or '<span class="empty-note">None</span>'

        missing_html = "".join(
            f'<span class="skill-chip missing">{html.escape(str(skill))}</span>'
            for skill in missing_skills
        ) or '<span class="empty-note">None</span>'

        score = row.get("score", 0)
        try:
            score_value = f"{float(score):.3f}"
        except (TypeError, ValueError):
            score_value = "0.000"

        job_card_html = (
            f'<div class="job-card">'
            f'<div class="job-left">'
            f'<p class="job-title">{title}</p>'
            f'<div class="job-meta">{company} • {location}</div>'
            f'<p class="job-skill-title">Matched Skills</p>'
            f'<div class="skill-row">{matched_html}</div>'
            f'<p class="job-skill-title">Missing Skills</p>'
            f'<div class="skill-row">{missing_html}</div>'
            f'</div>'
            f'<div class="score-box">'
            f'<div class="score-label">Score</div>'
            f'<div class="score-value">{score_value}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(job_card_html, unsafe_allow_html=True)

def show_course_results(results_df):
    for _, row in results_df.iterrows():
        title = html.escape(str(row.get("Title", "Untitled Course")))
        institution = html.escape(str(row.get("Institution", "Unknown Institution")))
        related_skill = html.escape(str(row.get("skill", "N/A")))

        raw_rating = row.get("Rate", "N/A")
        rating_text = "N/A"
        try:
            rating_text = f"{float(raw_rating):.1f}"
        except (TypeError, ValueError):
            if str(raw_rating).strip():
                rating_text = html.escape(str(raw_rating))

        course_link = ""
        for key in ["URL", "url", "course_url", "Course URL", "link"]:
            value = row.get(key)
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                course_link = value.strip()
                break

        if not course_link:
            course_link = f"https://www.google.com/search?q={quote_plus(str(row.get('Title', 'online course')))}"

        course_link = html.escape(course_link)

        st.markdown(
            f'<div class="course-card">'
            f'<p class="course-title">{title}</p>'
            f'<div class="course-meta">{institution}</div>'
            f'<div class="course-row"><span class="course-row-label">Related Skill:</span>'
            f'<span class="skill-chip missing">{related_skill}</span></div>'
            f'<div class="course-row"><span class="course-row-label">Rating:</span>'
            f'<span class="rating-badge">★ {rating_text}</span></div>'
            f'<div class="course-action"><a class="course-btn" href="{course_link}" target="_blank" rel="noopener noreferrer">View Course</a></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

CAREER_CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "career_classifier.pkl")


@st.cache_resource
def load_career_classifier():
    if not os.path.exists(CAREER_CLASSIFIER_PATH):
        return None
    with open(CAREER_CLASSIFIER_PATH, "rb") as f:
        return pickle.load(f)


def load_app_resources():
    # Load shared artifacts once so the tab renderers stay focused on UI work
    vectorizer = load_vectorizer()
    jobs = load_jobs()
    courses = load_courses()
    job_texts = build_job_texts(jobs)

    return {
        "vectorizer": vectorizer,
        "jobs": jobs,
        "courses": courses,
        "job_texts": job_texts,
        "job_matrix": build_job_matrix(tuple(job_texts), vectorizer),
        "skill_vocab": build_skill_vocab(jobs["job_skills"]),
        "all_job_skills": get_all_job_skills(jobs["job_skills"]),
        "career_classifier": load_career_classifier(),
    }


def build_default_state():
    # Keep one predictable state shape even before a CV is uploaded.
    return {
        "cv_text": "",
        "extracted_skills": [],
        "top_jobs": pd.DataFrame(),
        "strengths": [],
        "issues": [],
        "suggestions": [],
        "missing_skills": [],
        "recommended_courses": pd.DataFrame(),
        "cv_score": 0,
        "career_path": "Not enough data",
        "career_summary": {},
    }


def analyze_uploaded_cv(uploaded_file, resources):
    # Centralize the pipeline so each tab can render from the same analysis output.
    state = build_default_state()
    state["cv_text"] = extract_text_from_uploaded_file(uploaded_file)

    if not state["cv_text"].strip():
        return state

    state["extracted_skills"] = extract_skills_from_text(
        state["cv_text"],
        resources["skill_vocab"]
    )
    state["cv_score"] = calculate_cv_score(
        state["cv_text"],
        state["extracted_skills"]
    )

    if state["extracted_skills"]:
        cv_input = " ".join(state["extracted_skills"]).lower()
        state["top_jobs"] = rank_jobs(
            user_text=cv_input,
            jobs_df=resources["jobs"],
            job_texts=resources["job_texts"],
             vectorizer=resources["vectorizer"],
            job_matrix=resources["job_matrix"],
            top_n=10,
        )

    (
        state["strengths"],
        state["issues"],
         state["suggestions"],
        state["missing_skills"],
    ) = analyze_cv(
        state["cv_text"],
        state["extracted_skills"],
        resources["all_job_skills"],
    )

    state["recommended_courses"] = recommend_courses_from_missing_skills(
        resources["courses"],
        state["missing_skills"],
        top_n=10,
    )
    classifier = resources.get("career_classifier")
    if classifier and state["extracted_skills"]:
        cv_input = " ".join(state["extracted_skills"]).lower()
        state["career_path"] = predict_career_path(cv_input, classifier)
    else:
        state["career_path"] = "General Technology Roles"
    state["career_summary"] = build_career_summary(
        state["cv_score"],
        state["career_path"],
        state["missing_skills"],
        state["top_jobs"],
    )

    return state


def render_upload_message(uploaded_file, success_message):
    if uploaded_file is None:
        st.info(success_message)
        return False

    return True


def render_text_extraction_warning(state):
    if state["cv_text"].strip():
        return False

    st.warning("Could not extract text from this file.")
    return True


def render_skill_tiles(skills, status="success", limit=None):
    visible_skills = skills[:limit] if limit is not None else skills

    if not visible_skills:
        return False

    chip_class = "skill-chip"
    if status in {"warning", "warn", "improve"}:
        chip_class = "skill-chip warn"
    elif status == "info":
        chip_class = "skill-chip info"

    chips = "".join(
        f'<span class="{chip_class}">{html.escape(str(skill))}</span>'
        for skill in visible_skills
    )
    st.markdown(
        f'<div class="skills-chip-list">{chips}</div>',
        unsafe_allow_html=True,
    )

    return True


def render_skill_section_card(title, skills, status="success", limit=None, empty_message="No skills found."):
    visible_skills = skills[:limit] if limit is not None else skills

    chip_class = "skill-chip"
    if status in {"warning", "warn", "improve"}:
        chip_class = "skill-chip improve"
    elif status == "info":
        chip_class = "skill-chip info"

    if visible_skills:
        chips = "".join(
            f'<span class="{chip_class}">{html.escape(str(skill))}</span>'
            for skill in visible_skills
        )
        body = f'<div class="skills-chip-list">{chips}</div>'
    else:
        body = f'<p class="empty-note">{html.escape(empty_message)}</p>'

    st.markdown(
        f'<section class="section-card"><p class="section-title">{html.escape(title)}</p>{body}</section>',
        unsafe_allow_html=True,
    )

    return bool(visible_skills)


def build_ats_item_list(items, item_type, icon):
    if not items:
        return '<p class="empty-note">No items to show.</p>'

    return "".join(
        f'<div class="ats-item {item_type}">{icon} {html.escape(str(item))}</div>'
        for item in items
    )


def build_suggestion_list(items):
    if not items:
        return '<p class="empty-note">No suggestions right now. Your CV looks solid.</p>'

    return "".join(
        f'<div class="suggestion-item"><span class="suggestion-dot">Tip</span><span>{html.escape(str(item))}</span></div>'
        for item in items
    )


def render_missing_skills_card(skills, title="Missing Skills", limit=12):
    visible_skills = skills[:limit]

    if visible_skills:
        chips = "".join(
            f'<span class="skill-chip missing{' missing-priority' if index < 3 else ''}">{html.escape(str(skill))}</span>'
            for index, skill in enumerate(visible_skills)
        )
        body = f'<div class="skills-chip-list">{chips}</div>'
    else:
        body = '<p class="empty-note">No missing skills detected.</p>'

    remaining = max(0, len(skills) - limit)
    more = f'<p class="missing-more">+ {remaining} more skills</p>' if remaining else ''

    st.markdown(
        f'<section class="section-card"><p class="section-title">{html.escape(title)}</p>{body}{more}</section>',
        unsafe_allow_html=True,
    )


def render_pipeline_tab(uploaded_file, state):
    st.header("CV Preview")
    if not render_upload_message(uploaded_file, "Upload a CV above to run the full pipeline."):
        return
    if render_text_extraction_warning(state):
        return

    with st.expander("Show CV text"):
        st.write(state["cv_text"])

    render_skill_section_card(
        title="Extracted Skills",
        skills=sorted(state["extracted_skills"]),
        status="success",
        empty_message="No skills found in the uploaded CV.",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown('<p class="section-title" style="color:var(--navy);font-weight:700;font-size:1rem;margin-bottom:0.6rem">Skill Gap Summary</p>', unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        render_skill_section_card(
            title="Your Strong Skills",
            skills=sorted(state["extracted_skills"]),
            status="success",
            limit=10,
            empty_message="No skills detected.",
        )

    with right_col:
        render_skill_section_card(
            title="Skills to Improve",
            skills=state["missing_skills"],
            status="improve",
            limit=10,
            empty_message="No missing skills detected.",
        )

    st.header("Recommended Career Path")
    st.info(state["career_path"])

    st.header("Matched Jobs")
    if state["top_jobs"].empty:
        st.warning("No matching jobs found.")
    else:
        show_job_results(state["top_jobs"])

    st.header("Career Summary")
    if state["career_summary"]:
        st.write(f"**CV Score:** {state['cv_score']}/100")
        st.write(f"**Best Career Path:** {state['career_summary']['career_path']}")
        st.write(f"**Best Job Match:** {state['career_summary']['best_match']}")
        st.write(f"**Main Missing Skills:** {state['career_summary']['main_missing']}")


def render_resume_tab(uploaded_file, state):
    st.header("Resume & ATS Review")
    if not render_upload_message(uploaded_file, "Upload a CV first to review resume quality."):
        return
    if render_text_extraction_warning(state):
        return

    score = max(0, min(int(state["cv_score"]), 100))
    st.markdown(
        f"""
        <section class="ats-score-card">
            <p class="ats-score-label">ATS Score</p>
            <div class="ats-score-value">{score}/100</div>
            <div class="ats-score-bar"><div class="ats-score-fill" style="width: {score}%;"></div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    strengths_html = build_ats_item_list(state["strengths"], "success", "✓")
    issues_html = build_ats_item_list(state["issues"], "issue", "⚠")

    st.markdown(
        f"""
        <div class="ats-grid">
            <section class="section-card">
                <p class="section-title">Strengths</p>
                {strengths_html}
            </section>
            <section class="section-card">
                <p class="section-title">Issues</p>
                {issues_html}
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions_html = build_suggestion_list(state["suggestions"])
    st.markdown(
        f"""
        <section class="section-card">
            <p class="section-title">Suggestions</p>
            {suggestions_html}
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_missing_skills_card(state["missing_skills"], title="Missing Skills", limit=12)


def render_learning_tab(uploaded_file, state):
    st.header("Learning Path")
    if not render_upload_message(uploaded_file, "Upload a CV first to get course recommendations."):
        return
    if render_text_extraction_warning(state):
        return

    st.subheader("Skills to Improve")
    if not render_skill_tiles(state["missing_skills"], status="warning", limit=9):
        st.success("No missing skills detected.")

    st.subheader("Recommended Courses")
    if state["recommended_courses"].empty:
        st.warning("No course recommendations found.")
    else:
        show_course_results(state["recommended_courses"])


def render_job_fit_tab(uploaded_file, state, resources):
    st.header("Job Fit Analyzer")
    if not render_upload_message(uploaded_file, "Upload a CV first, then paste a job description."):
        return
    if render_text_extraction_warning(state):
        return

    job_description = st.text_area(
        "Paste a job description here",
        height=220,
        placeholder="Paste the full job description here...",
    )

    if not st.button("Analyze Job Fit"):
        return

    if not job_description.strip():
        st.warning("Please paste a job description.")
        return

    fit_score, matched_job_skills, missing_job_skills, fit_suggestions = analyze_job_fit(
        state["cv_text"],
        state["extracted_skills"],
        job_description,
        resources["vectorizer"],
        resources["skill_vocab"],
        extract_skills_from_text,
    )

    st.metric("Job Fit Score", f"{fit_score:.3f}")

    st.subheader("Matched Skills")
    if not render_skill_tiles(matched_job_skills):
        st.warning("No matched skills found.")

    st.subheader("Missing Skills")
    if not render_skill_tiles(missing_job_skills, status="warning"):
        st.success("No major missing skills found.")

    st.subheader("Suggestions")
    if fit_suggestions:
        for item in fit_suggestions:
            st.info(item)
    else:
        st.success("Your CV already aligns well with this job.")


try:
    resources = load_app_resources()
except Exception as error:
    st.error(f"Startup error: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload CV (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
)
analysis_state = build_default_state()

if uploaded_file is not None:
    analysis_state = analyze_uploaded_cv(uploaded_file, resources)

pipeline_tab, resume_tab, learning_tab, fit_tab = st.tabs([
    "Pipeline",
    "Resume & ATS",
    "Learning Path",
    "Job Fit Analyzer",
])

with pipeline_tab:
    render_pipeline_tab(uploaded_file, analysis_state)

with resume_tab:
    render_resume_tab(uploaded_file, analysis_state)

with learning_tab:
    render_learning_tab(uploaded_file, analysis_state)

with fit_tab:
    render_job_fit_tab(uploaded_file, analysis_state, resources)