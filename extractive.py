import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from docx import Document
import fitz  # PyMuPDF

nltk.download('punkt')

# --- Fungsi untuk memuat teks dari berbagai sumber ---

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def summarize_text(text, n_clusters=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= n_clusters:
        return sentences

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    summary = []
    for i in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        centroid = kmeans.cluster_centers_[i]
        closest, min_dist = -1, float('inf')
        for idx in cluster_indices:
            vec = X[idx].toarray().flatten()
            dist = np.linalg.norm(vec - centroid)
            if dist < min_dist:
                closest = idx
                min_dist = dist
        summary.append(sentences[closest])

    summary = sorted(summary, key=lambda s: sentences.index(s))
    return summary

st.title("Text Summarization dengan K-Means Clustering")
st.write("Pilih sumber teks untuk diringkas:")

input_method = st.radio("Sumber Teks", ["Manual", "Upload File (.txt, .pdf, .docx)"])

text = ""

if input_method == "Manual":
    text = st.text_area("Masukkan teks:", height=300)

else:
    uploaded_file = st.file_uploader("Unggah file:", type=["txt", "pdf", "docx"])
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if ext == "txt":
                text = read_txt(uploaded_file)
            elif ext == "docx":
                text = read_docx(uploaded_file)
            elif ext == "pdf":
                text = read_pdf(uploaded_file)
        except:
            st.error("Gagal membaca file.")

if text:
    n_clusters = st.slider("Jumlah kalimat ringkasan:", 1, 10, 3)
    if st.button("Ringkas Teks"):
        summary = summarize_text(text, n_clusters)
        st.subheader("Hasil Ringkasan:")
        for i, s in enumerate(summary, 1):
            st.write(f"{i}. {s}")
