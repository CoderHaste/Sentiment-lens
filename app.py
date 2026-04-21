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


# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentLens",
    page_icon="◈",
    layout="centered",
)


# ── Google Drive links (converted to direct download) ──────
MODEL_URL = "https://drive.google.com/uc?id=1RljcbiTOBeSRp9K-9NiXlrZnPGhqLcgQ"
WI_URL    = "https://drive.google.com/uc?id=17FPPfSp4Wb4Qrd4MFYusqHP9MIrPF6P4"

MODEL_PATH = "best_model.keras"
WI_PATH    = "word_index.json"


# ── Download model if not present ──────────────────────────
def download_files():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(WI_PATH):
        with st.spinner("Downloading tokenizer..."):
            gdown.download(WI_URL, WI_PATH, quiet=False)


# ── Load model & word index ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    download_files()

    model = load_model(MODEL_PATH)
    with open(WI_PATH) as f:
        word_index = json.load(f)

    return model, word_index


# ── Constants ──────────────────────────────────────────────
VOCAB_SIZE = 20000
MAX_LEN    = 400


# ── Text helpers ───────────────────────────────────────────
def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def encode_review(text: str, word_index: dict) -> list:
    words = clean_text(text).split()
    return [min(word_index.get(w, 2) + 3, VOCAB_SIZE - 1) for w in words]

def predict(text: str, model, word_index: dict) -> float:
    enc    = encode_review(text, word_index)
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding='pre', truncating='pre')
    return float(model.predict(padded, verbose=0)[0][0])


# ── Session state ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #e8c97a; padding-top:2rem; margin-bottom:2rem;">
    <div style="font-size:0.7rem; color:#e8c97a;">◈ Sentiment Analysis</div>
    <h1 style="margin:0;">SentimentLens</h1>
    <p style="color:#888;">BiLSTM · IMDB Dataset</p>
</div>
""", unsafe_allow_html=True)


# ── Load assets safely ─────────────────────────────────────
model_loaded = False
try:
    model, word_index = load_assets()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model: {e}")


# ── Input ─────────────────────────────────────────────────
review_text = st.text_area(
    "Enter review",
    placeholder="Type a movie review...",
    height=120
)

run = st.button("Analyse", disabled=not model_loaded)


# ── Prediction ─────────────────────────────────────────────
if run:
    text = review_text.strip()

    if not text:
        st.warning("Please enter some text")
    else:
        prob = predict(text, model, word_index)

        label = "Positive" if prob > 0.5 else "Negative"
        confidence = prob if prob > 0.5 else 1 - prob

        st.subheader(label)
        st.progress(int(confidence * 100))

        # Save history
        st.session_state.history.insert(0, {
            "text": text,
            "label": label,
            "confidence": confidence
        })
        st.session_state.history = st.session_state.history[:5]


# ── History ────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("### Recent Predictions")
    for item in st.session_state.history:
        st.write(f"{item['label']} ({item['confidence']:.0%}) — {item['text'][:60]}...")


# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("SentimentLens · BiLSTM · 20k vocab")
