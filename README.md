# AI Career Assistant

My capstone project: upload a CV and get job matches, skill gaps, ATS feedback, and a suggested career path.

Why I Built It?
I wanted one tool that gives practical resume feedback, not just a score.

What This App Does
- Upload CV (PDF, DOCX, or TXT)
- Extract skills from CV text
- Show top matched jobs
- Show missing skills and ATS-style feedback
- Recommend online courses for skill gaps
- Predict a career category from CV skills
- Analyze CV fit for a pasted job description

Data
- data/jobs_clean.csv
- data/courses_clean.csv

Main Tabs
- Pipeline: CV preview, extracted skills, matched jobs, career path, summary
- Resume & ATS: ATS score, strengths, issues, suggestions, missing skills
- Learning Path: skills to improve and recommended courses
- Job Fit Analyzer: compare CV with a specific job description

Model
- Training notebook: train_career_classifier.ipynb
- Saved model: models/career_classifier.pkl
- Job matching vectorizer: models/vectorizer.pkl

How Training Works (Simple)
- Build skills text from job data
- Discover category labels using KMeans clustering
- Train/test split with stratification
- Compare Logistic Regression vs Random Forest
- Tune Logistic Regression with GridSearchCV
- Evaluate with macro F1, confusion matrix, and error analysis
- Save classifier and vectorizer into models/

Results
From my latest run:
- Macro F1 is around 0.99
- Includes confusion matrix and error analysis in the notebook

Tech Stack
- Python
- Streamlit
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- python-docx, pypdf

Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

How to Use
1. Start the app.
2. Upload your CV.
3. Open Pipeline tab to see extracted skills and matched jobs.
4. Open Resume & ATS tab to see score and suggestions.
5. Open Learning Path tab to see courses.
6. Open Job Fit Analyzer tab and paste a job description.

Train the Model Again
```bash
jupyter notebook
```
Then run all cells in train_career_classifier.ipynb.
It will update:
- models/career_classifier.pkl
- models/vectorizer.pkl

 Files
- app.py
- src/
- data/
- models/
- train_career_classifier.ipynb

Project Structure
- src/cv_parser.py: reads PDF, DOCX, TXT CV files
- src/skills.py: skill extraction from text
- src/job_matcher.py: ranks jobs using TF-IDF + overlap score
- src/resume_feedback.py: ATS checks and CV score
- src/course_recommend.py: recommends courses from missing skills
- src/job_fit.py: CV vs job description fit score
- src/career_utils.py: career prediction and summary helpers

Limitations
- Labels are discovered from data, not manually annotated
- Some roles overlap heavily in skills
- Missing skills in main tabs are global, not fully role-specific
- Skill extraction depends on known skills in the dataset vocabulary

Future Improvements
- Add role-specific missing skills in all tabs
- Add better parsing for multi-word skills and synonyms
- Add optional manual target role selection
- Add model retraining automation and experiment tracking

Demo and Slides
- Live demo link: https://employmentreadinessai-hrxxzmv8u4hffdrzfdsvm.streamlit.app
- Demo video link: https://www.youtube.com/watch?v=ww8_mk3M4_4
- Slides: docs/Job Search Strategy Presentation.pptx

Author
- Noor (Capstone Project)


