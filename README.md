# AI Resume Screening System

I built an NLP-based system that ranks resumes against a job description using 
TF-IDF and Cosine Similarity, with a live Streamlit app for screening candidates.

## Project files

task3_resume_screening/
├── Resume_Screening_Colab.ipynb # Full pipeline, run on Colab
├── resume_screener.py # Core NLP logic (used by app.py)
├── app.py # Streamlit web app
├── requirements.txt
├── sample_resumes/ # 5 sample resumes I used for testing
└── README.md


## 1. Development on Google Colab
I built and tested the full pipeline in `Resume_Screening_Colab.ipynb` on 
Google Colab:
1. Uploaded sample resumes (PDF/TXT) into the Colab session — also tested with 
   the built-in `sample_resumes` dictionary for quick iteration.
2. Ran the notebook top to bottom — it cleaned the resume text, vectorized 
   everything with TF-IDF, computed cosine similarity against a job description, 
   and produced a ranked table with a bar chart.
3. Saved the results as `resume_screening_results.csv` and downloaded it from 
   the Colab session.

## 2. Local development in VS Code
I set up and ran the project locally as follows:
```bash
# Created & activated a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Installed dependencies
pip install -r requirements.txt

# Launched the app
streamlit run app.py
```
I tested the app at http://localhost:8501 by pasting a job description, uploading 
resumes from `sample_resumes/`, and reviewing the ranked scores, shortlist, and 
CSV download.

## 3. Version control — pushed to GitHub
```bash
git init
git add .
git commit -m "AI Resume Screening System"
git branch -M main
git remote add origin https://github.com/Areeba-Najam/resume-screening-app.git
git push -u origin main
```
I also added a `.gitignore` to keep `venv/` and `__pycache__/` out of the repo.

## 4. Deployment — live on Streamlit Community Cloud
I deployed the app for free on Streamlit Community Cloud:
1. Signed in to https://share.streamlit.io with GitHub.
2. Created a new app, pointed it at my repo/branch, and set the main file path 
   to `app.py`.
3. Deployed — Streamlit installed everything from `requirements.txt` and gave 
   me a public URL.
4. Along the way, I ran into a build failure where Pillow (a Streamlit 
   dependency) failed to compile on Python 3.14 due to a missing zlib library. 
   I fixed it by unpinning the exact versions in `requirements.txt` and adding 
   a `runtime.txt` file pinning the environment to Python 3.11, then redeployed.
5. Confirmed that pushing further changes to `main` auto-redeploys the app.

## How the matching works
1. **Preprocessing** — lowercased text, stripped punctuation/URLs/emails, 
   removed stopwords, and lemmatized tokens.
2. **Feature extraction** — vectorized the cleaned job description and all 
   resumes together using TF-IDF.
3. **Matching** — computed cosine similarity between the job description vector 
   and each resume vector to produce a match percentage.
4. **Ranking & shortlisting** — sorted candidates by score and applied an 
   adjustable shortlist threshold (default 40%) in the app's sidebar.

## Notes
- Candidate names are derived from resume filenames (e.g. `ali_raza.txt` → 
  "Ali Raza").
- PDF text extraction quality depends on how the PDF was generated — scanned/ 
  image PDFs would need OCR, which wasn't part of this project's scope.
