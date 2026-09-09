
import streamlit as st
import pandas as pd
from resume_screener import load_resume, rank_resumes, shortlist

st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screening System")
st.caption("Upload resumes + a job description → get ranked, shortlisted candidates using TF-IDF + Cosine Similarity NLP matching.")

with st.sidebar:
    st.header("⚙️ Settings")
    min_score = st.slider("Shortlist threshold (match %)", 0, 100, 40, 5)
    top_n = st.number_input("Show top N candidates (0 = all)", min_value=0, value=0, step=1)
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Text is cleaned (lowercase, stopwords removed, lemmatized)\n"
        "2. TF-IDF vectorizes resumes + job description\n"
        "3. Cosine similarity scores each resume against the JD\n"
        "4. Candidates are ranked and filtered by threshold"
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣ Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="e.g. Looking for a Python developer with experience in machine learning, NLP, SQL and REST APIs...",
    )

with col2:
    st.subheader("2️⃣ Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes (.pdf or .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

run = st.button("🚀 Screen Resumes", type="primary", use_container_width=True)

if run:
    if not jd_text.strip():
        st.error("Please paste a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        resumes = {}
        with st.spinner("Extracting and processing resumes..."):
            for f in uploaded_files:
                text = load_resume(f, f.name)
                candidate_name = f.name.rsplit(".", 1)[0].replace("_", " ").title()
                resumes[candidate_name] = text

            ranked = rank_resumes(
                jd_text,
                resumes,
                top_n=top_n if top_n > 0 else None,
            )
            shortlisted = shortlist(ranked, min_score=min_score)

        st.success(f"Processed {len(resumes)} resume(s).")

        st.subheader("📊 Ranking Results")
        df = pd.DataFrame(
            [
                {
                    "Rank": i + 1,
                    "Candidate": r["name"],
                    "Match Score (%)": r["score"],
                    "Shortlisted": "✅" if r["score"] >= min_score else "❌",
                    "Matched Keywords": ", ".join(r["matched_keywords"][:15]),
                }
                for i, r in enumerate(ranked)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.bar_chart(df.set_index("Candidate")["Match Score (%)"])

        st.subheader(f"✅ Shortlisted Candidates (score ≥ {min_score}%)")
        if shortlisted:
            for r in shortlisted:
                with st.expander(f"{r['name']} — {r['score']}% match"):
                    st.write("**Matched keywords:**", ", ".join(r["matched_keywords"]) or "—")
        else:
            st.info("No candidates met the shortlist threshold. Try lowering it in the sidebar.")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv, "resume_screening_results.csv", "text/csv")
else:
    st.info("Paste a job description, upload resumes, then click **Screen Resumes**.")
