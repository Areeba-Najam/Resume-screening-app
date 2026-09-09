# AI Resume Screening System

NLP-based resume ranking: TF-IDF + Cosine Similarity matches resumes against a job description.

## Project files
```
task3_resume_screening/
├── Task3_Resume_Screening_Colab.ipynb   
├── resume_screener.py                  
├── app.py
├── requirements.txt
├── sample_resumes/                      # 5 sample .txt resumes to test with
└── README.md
```

## 1. Run on Google Colab
1. Open Colab → File → Upload notebook → select `Task3_Resume_Screening_Colab.ipynb`.
2. Run cells top to bottom.
3. Either upload your own resumes (PDF/TXT) via the Colab upload widget, or use the built-in
   `sample_resumes` dict.
4. The notebook prints a ranked table, a bar chart of match scores, and saves
   `resume_screening_results.csv`.

## 2. Run locally in VS Code
```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501). Paste a job description,
upload resumes from `sample_resumes/` (or your own PDFs), and click **Screen Resumes**.

## 3. Push to GitHub
```bash
git init
git add .
git commit -m "AI Resume Screening System"
git branch -M main
git remote add origin https://github.com/<your-username>/resume-screening-app.git
git push -u origin main
```
Create a `.gitignore` (already included) so `venv/`, `__pycache__/`, etc. aren't committed.

## 4. Deploy live on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app** → pick your repo/branch → set **Main file path** to `app.py`.
3. Click **Deploy**. Streamlit installs `requirements.txt` automatically and gives you a
   public URL like `https://your-app-name.streamlit.app`.
4. Any future `git push` to `main` auto-redeploys the app.

## How the matching works
1. **Preprocessing** — lowercase, strip punctuation/URLs/emails, remove stopwords, lemmatize.
2. **Feature extraction** — TF-IDF vectorizes the cleaned job description + all resumes together.
3. **Matching** — cosine similarity between the JD vector and each resume vector → match % score.
4. **Ranking & shortlisting** — sorted descending by score; a sidebar slider sets the
   shortlist threshold (default 40%).

## Notes / extensions
- To add real candidate names, rename resume files as `firstname_lastname.txt/.pdf`
  — the app title-cases the filename as the candidate name.
- For advanced matching, swap TF-IDF for sentence embeddings
  (e.g. `sentence-transformers` + cosine similarity)  same ranking logic applies.
- PDF text extraction quality depends on how the PDF was generated; scanned/image PDFs need OCR
  (not included here).
