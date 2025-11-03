import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease
from wordcloud import WordCloud
import io
import nltk

st.set_page_config(page_title="SEO Quality Analyzer", layout="centered")

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_resources():
    data = pd.read_csv("final_quality_scores.csv")
    model = joblib.load("quality_model.pkl")
    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        words = [w for w in text.split() if w not in stop_words]
        return " ".join(words)

    data["clean_text"] = data["body_text"].astype(str).apply(clean_text)
    tfidf = TfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf.fit_transform(data["clean_text"])
    return data, model, tfidf, tfidf_matrix

data, model, tfidf, tfidf_matrix = load_resources()

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return title, text

def safe_readability(text):
    try:
        clean_text = re.sub(r'\s+', ' ', text)
        if len(clean_text.split()) < 50 or '.' not in clean_text:
            return 0
        score = flesch_reading_ease(clean_text)
        return max(0, min(100, score))
    except:
        return 0

def analyze_url(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        title, text = extract_text_from_html(response.text)
        clean = re.sub(r"[^a-z\s]", " ", text.lower())
        wc = len(clean.split())
        sc = len(sent_tokenize(text))
        read = safe_readability(text)
        features = [[wc, sc, read]]
        pred = model.predict(features)[0]
        new_vec = tfidf.transform([clean])
        sims = cosine_similarity(new_vec, tfidf_matrix)[0]
        similar_idx = np.where(sims > 0.75)[0]
        similar_pages = []
        for i in similar_idx[:3]:
            if "url" in data.columns:
                similar_pages.append(data.iloc[i]["url"])
        return {
            "url": url,
            "title": title,
            "text": text,
            "word_count": wc,
            "readability": read,
            "predicted_quality": pred,
            "is_thin": wc < 500,
            "similar_pages": similar_pages
        }
    except Exception as e:
        return {"error": str(e)}

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f3f4f6, #e0f2fe);
        }
        .main-title {
            text-align: center;
            font-size: 2.3em;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #0072ff, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.1em;
            margin-bottom: 25px;
        }
        .result-card {
            background-color: white;
            border-radius: 14px;
            padding: 25px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-top: 15px;
        }
        .badge {
            display: inline-block;
            padding: 6px 15px;
            border-radius: 20px;
            font-weight: 600;
            color: white;
        }
        .badge-high { background-color: #16a34a; }
        .badge-medium { background-color: #facc15; color: black; }
        .badge-low { background-color: #dc2626; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🌐 SEO Content Quality & Duplicate Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze any live webpage URL for readability, quality, and keyword insights</p>", unsafe_allow_html=True)

url_input = st.text_input("Enter a webpage URL")

if st.button("Analyze URL"):
    if not url_input:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing webpage..."):
            result = analyze_url(url_input)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            quality_class = "badge-low"
            if result["predicted_quality"].lower() == "high":
                quality_class = "badge-high"
            elif result["predicted_quality"].lower() == "medium":
                quality_class = "badge-medium"

            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", result['word_count'])
            col2.metric("Readability", round(result['readability'], 2))
            col3.metric("Quality", result['predicted_quality'].capitalize())

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>{result['title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>URL:</strong> {result['url']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Thin Content:</strong> {'Yes' if result['is_thin'] else 'No'}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Quality:</strong> <span class='badge {quality_class}'>{result['predicted_quality'].capitalize()}</span></p>", unsafe_allow_html=True)

            if result["similar_pages"]:
                st.markdown("<p><strong>Possible Duplicate URLs:</strong></p>", unsafe_allow_html=True)
                for page in result["similar_pages"]:
                    st.markdown(f"- [{page}]({page})", unsafe_allow_html=True)
            else:
                st.markdown("<p><em>No close duplicates found.</em></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Keyword Insights (Word Cloud)")
            clean_text = " ".join(result["text"].split())
            wordcloud = WordCloud(width=900, height=450, background_color="white").generate(clean_text)
            st.image(wordcloud.to_array())

            st.subheader("📥 Download Analysis Report")
            buffer = io.BytesIO()
            pd.DataFrame([{
                "Title": result["title"],
                "URL": result["url"],
                "Word Count": result["word_count"],
                "Readability": result["readability"],
                "Quality": result["predicted_quality"],
                "Thin Content": "Yes" if result["is_thin"] else "No"
            }]).to_csv(buffer, index=False)
            st.download_button("⬇️ Download CSV Report", data=buffer.getvalue(), file_name="seo_report.csv", mime="text/csv")
