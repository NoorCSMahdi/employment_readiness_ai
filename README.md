# AI Career Assistant

My capstone project: upload a CV and get job matches, skill gaps, ATS feedback, and a suggested career path.

Why I Built It?
I wanted one tool that gives practical resume feedback, not just a score.

Data
- data/jobs_clean.csv
- data/courses_clean.csv

Model
- Training notebook: train_career_classifier.ipynb
- Saved model: models/career_classifier.pkl
- Job matching vectorizer: models/vectorizer.pkl

Results
From my latest run:
- Macro F1 is around 0.99
- Includes confusion matrix and error analysis in the notebook

Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

 Files
- app.py
- src/
- data/
- models/
- train_career_classifier.ipynb

Limitations
- Labels are discovered from data, not manually annotated
- Some roles overlap heavily in skills

Demo and Slides
- Demo video link: TODO


