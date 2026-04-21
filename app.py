```python
import os
import re
import json
import warnings
import numpy as np
import streamlit as st
import gdown

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentLens",
    page_icon="◈",
    layout="centered",
)


# ── Google Drive links ────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?id=1RljcbiTOBeSRp9K-9NiXlrZnPGhqLcgQ"
WI_URL    = "https://drive.google.com/uc?id=17FPPfSp4Wb4Qrd4MFYusqHP9MIrPF6P4"

MODEL_PATH = "best_model.keras"
WI_PATH    = "word_index.json"


# ── Custom CSS (UNCHANGED) ────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #111110 !important;
    color: #e8e4dc !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #111110 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    padding: 4rem 2rem 6rem !important;
    max-width: 720px !important;
}

* { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.site-header {
    border-top: 1px solid #e8c97a;
    padding-top: 2rem;
    margin-bottom: 3.5rem;
}
.site-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #e8c97a;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.site-title {
    font-size: 2.6rem;
    color: #f0ece4;
    margin-bottom: 0.5rem;
}
.site-sub {
    font-size: 0.85rem;
    color: #7a7670;
}

textarea {
    background: #1a1917 !important;
    border: 1px solid #2e2d2a !important;
    color: #e8e4dc !important;
    padding: 1rem !important;
}

[data-testid="stButton"] button {
    border: 1px solid #e8c97a !important;
    color: #e8c97a !important;
    width: 100% !important;
}

.result-wrap {
    margin-top: 2rem;
}
.verdict.pos { color: #a8d8a0; }
.verdict.neg { color: #d98c8c; }

.err-box {
    border: 1px solid #7a3030;
    padding: 1rem;
    color: #d98c8c;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────
VOCAB_SIZE = 20000
MAX_LEN    = 400


# ── Load model & word index ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    # Download only if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(WI_PATH):
        with st.spinner("Downloading tokenizer..."):
            gdown.download(WI_URL, WI_PATH, quiet=False)

    model = load_model(MODEL_PATH)

    with open(WI_PATH) as f:
        word_index = json.load(f)

    return model, word_index


# ── Helpers ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def encode_review(text: str, word_index: dict) -> list:
    words = clean_text(text).split()
    return [min(word_index.get(w, 2) + 3, VOCAB_SIZE - 1) for w in words]

def predict(text: str, model, word_index: dict) -> float:
    enc = encode_review(text, word_index)
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding='pre')
    return float(model.predict(padded, verbose=0)[0][0])


# ── State ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <div class="site-label">◈ Sentiment Analysis</div>
    <div class="site-title">SentimentLens</div>
    <div class="site-sub">Deep learning · IMDB-trained · BiLSTM</div>
</div>
""", unsafe_allow_html=True)


# ── Load assets ───────────────────────────────────────────────
model_loaded = False
try:
    model, word_index = load_assets()
    model_loaded = True
except Exception as e:
    st.markdown(f'<div class="err-box">Error loading model: {e}</div>', unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────
review_text = st.text_area("Review text", placeholder="Type review here...", height=130)
run = st.button("Analyse →", disabled=not model_loaded)


# ── Prediction ────────────────────────────────────────────────
if run:
    text = review_text.strip()

    if not text:
        st.markdown('<div class="err-box">Enter text first</div>', unsafe_allow_html=True)
    else:
        prob = predict(text, model, word_index)

        label = "Positive" if prob > 0.5 else "Negative"
        cls   = "pos" if prob > 0.5 else "neg"

        st.markdown(f"""
        <div class="result-wrap">
            <div class="verdict {cls}">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("SentimentLens · BiLSTM")
```
