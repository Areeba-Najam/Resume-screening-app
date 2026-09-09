import re
import string
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def extract_text_from_pdf(file_path_or_buffer) -> str:
    """Extract raw text from a PDF file path or a file-like buffer."""
    reader = PyPDF2.PdfReader(file_path_or_buffer)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def read_txt(file_path_or_buffer) -> str:
    """Read a plain text resume from a path or file-like buffer."""
    if hasattr(file_path_or_buffer, "read"):
        content = file_path_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return content
    with open(file_path_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_resume(file_path_or_buffer, filename: str) -> str:
    """Dispatch to the right loader based on file extension."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path_or_buffer)
    return read_txt(file_path_or_buffer)

def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers/special chars, tokenize,
    remove stopwords, lemmatize. Returns a cleaned, space-joined string."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        
    text = re.sub(r"\S+@\S+", " ", text)                  
    text = re.sub(r"[^a-z\s]", " ", text)                 
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def rank_resumes(job_description: str, resumes: dict, top_n: int = None,
                  threshold: float = 0.0):
    """
    resumes: dict {candidate_name: raw_resume_text}
    Returns a sorted list of dicts: [{name, score, matched_keywords}, ...]
    """
    candidate_names = list(resumes.keys())
    cleaned_jd = clean_text(job_description)
    cleaned_resumes = [clean_text(resumes[name]) for name in candidate_names]

    corpus = [cleaned_jd] + cleaned_resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    jd_keywords = set(cleaned_jd.split())

    results = []
    for name, score, cleaned in zip(candidate_names, scores, cleaned_resumes):
        resume_keywords = set(cleaned.split())
        matched = sorted(jd_keywords.intersection(resume_keywords))
        results.append({
            "name": name,
            "score": round(float(score) * 100, 2),  # match %
            "matched_keywords": matched,
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    if threshold:
        results = [r for r in results if r["score"] >= threshold]
    if top_n:
        results = results[:top_n]

    return results


def shortlist(ranked_results: list, min_score: float = 30.0):
    """Filter candidates whose score is >= min_score."""
    return [r for r in ranked_results if r["score"] >= min_score]


if __name__ == "__main__":

    jd = "Looking for a Python developer with experience in machine learning, NLP and SQL."
    sample = {
        "Ali": "Python developer with Django, Flask, SQL experience.",
        "Sara": "Data scientist skilled in Python, machine learning, NLP, SQL, TensorFlow.",
    }
    for r in rank_resumes(jd, sample):
        print(r)
