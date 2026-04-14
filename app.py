import ast
import os
import pickle

import pandas as pd
import streamlit as st

from src.career_utils import build_career_summary, predict_career_path
from src.course_recommend import recommend_courses_from_missing_skills
from src.cv_parser import extract_text_from_uploaded_file
from src.job_fit import analyze_job_fit
from src.job_matcher import rank_jobs
from src.resume_feedback import analyze_cv, calculate_cv_score
from src.skills import extract_skills_from_text

JOBS_PATH = "data/jobs_clean.csv"
COURSES_PATH = "data/courses_clean.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"

st.set_page_config(
    page_title="AI Career Assistant",
    layout="wide"
    )

st.title("AI Career Assistant")
st.write("Upload a CV, match jobs, review ATS quality, and get learning recommendations.")

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
    for row_index, row in results_df.iterrows():
        st.subheader(row["job_title"])

        col1, col2 = st.columns([4, 1])

        with col1:
            st.write("**Company:**", row.get("company_name", "N/A"))
            st.write("**Location:**", row.get("job_location", "N/A"))

            matched_skills = row.get("matched_skills", [])
            missing_skills = row.get("missing_skills", [])

            if matched_skills:
                st.write("**Matched Skills:**", ", ".join(matched_skills))
            else:
                st.write("**Matched Skills:** None")

            if missing_skills:
                st.write("**Missing Skills for This Job:**", ", ".join(missing_skills))
            else:
                st.write("**Missing Skills for This Job:** None")

        with col2:
            st.metric("Score", f"{row['score']:.3f}")

        st.divider()

def show_course_results(results_df):
    for row_index, row in results_df.iterrows():
        st.subheader(row.get("Title", "N/A"))
        st.write("**Institution:**", row.get("Institution", "N/A"))
        st.write("**Related Missing Skill:**", row.get("skill", "N/A"))
        st.write("**Rating:**", row.get("Rate", "N/A"))
        st.divider()

CAREER_CLASSIFIER_PATH = "models/career_classifier.pkl"


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

    # Reuse the same compact grid pattern for skills across the app.
    cols = st.columns(3)
    for index, skill in enumerate(visible_skills):
        getattr(cols[index % 3], status)(skill)

    return True


def render_pipeline_tab(uploaded_file, state):
    st.header("CV Preview")
    if not render_upload_message(uploaded_file, "Upload a CV above to run the full pipeline."):
        return
    if render_text_extraction_warning(state):
        return

    with st.expander("Show CV text"):
        st.write(state["cv_text"])

    st.header("Extracted Skills")
    if not render_skill_tiles(sorted(state["extracted_skills"])):
        st.warning("No skills found in the uploaded CV.")

    st.header("Skill Gap Summary")
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Your Strong Skills")
        if not render_skill_tiles(sorted(state["extracted_skills"][:10])):
            st.write("No skills detected.")

    with right_col:
        st.subheader("Skills to Improve")
        if not render_skill_tiles(state["missing_skills"], status="warning", limit=10):
            st.success("No missing skills detected.")

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

    st.metric("CV Score", f"{state['cv_score']}/100")
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Strengths")
        if state["strengths"]:
            for item in state["strengths"]:
                st.success(item)
        else:
            st.write("No strengths detected.")

    with right_col:
        st.subheader("Issues")
        if state["issues"]:
            for item in state["issues"]:
                st.warning(item)
        else:
            st.write("No major issues found.")

    st.subheader("Suggestions")
    if state["suggestions"]:
        for item in state["suggestions"]:
            st.info(item)
    else:
        st.success("Your CV looks good for a basic ATS check.")

    st.subheader("Missing Skills")
    if not render_skill_tiles(state["missing_skills"], status="info"):
        st.success("No missing skills detected.")


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