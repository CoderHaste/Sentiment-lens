import os
import re
import json
import warnings
import numpy as np
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentLens",
    page_icon="◈",
    layout="centered",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ── */
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

/* ── Typography ── */
* { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* ── Header block ── */
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
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0ece4;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.site-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #7a7670;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Divider ── */
.rule {
    border: none;
    border-top: 1px solid #2a2926;
    margin: 2rem 0;
}

/* ── Text area ── */
textarea {
    background: #1a1917 !important;
    border: 1px solid #2e2d2a !important;
    border-radius: 2px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    caret-color: #e8c97a !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s ease !important;
}
textarea:focus {
    border-color: #e8c97a !important;
    outline: none !important;
    box-shadow: none !important;
}
textarea::placeholder { color: #4a4845 !important; }

/* ── Label above textarea ── */
[data-testid="stTextArea"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    color: #7a7670 !important;
    text-transform: uppercase !important;
    margin-bottom: 0.5rem !important;
}

/* ── Button ── */
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid #e8c97a !important;
    border-radius: 2px !important;
    color: #e8c97a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    background: #e8c97a !important;
    color: #111110 !important;
}

/* ── Result card ── */
.result-wrap {
    margin-top: 2.5rem;
    border-top: 1px solid #2a2926;
    padding-top: 2rem;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #4a4845;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    line-height: 1;
    margin-bottom: 1.5rem;
}
.verdict.pos { color: #a8d8a0; }
.verdict.neg { color: #d98c8c; }

/* ── Confidence meter ── */
.meter-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.meter-track {
    flex: 1;
    height: 2px;
    background: #2a2926;
    border-radius: 1px;
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 1px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
.meter-fill.pos { background: #a8d8a0; }
.meter-fill.neg { background: #d98c8c; }
.meter-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #7a7670;
    min-width: 3.5rem;
    text-align: right;
}

/* ── History section ── */
.history-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #4a4845;
    text-transform: uppercase;
    margin-bottom: 1rem;
    margin-top: 3rem;
    border-top: 1px solid #2a2926;
    padding-top: 2rem;
}
.hist-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.85rem 0;
    border-bottom: 1px solid #1e1d1b;
    gap: 1rem;
}
.hist-text {
    font-size: 0.85rem;
    color: #7a7670;
    font-weight: 300;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 480px;
}
.hist-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 1px;
    flex-shrink: 0;
}
.hist-badge.pos { color: #a8d8a0; border: 1px solid #a8d8a033; }
.hist-badge.neg { color: #d98c8c; border: 1px solid #d98c8c33; }

/* ── Footer ── */
.site-footer {
    margin-top: 5rem;
    border-top: 1px solid #1e1d1b;
    padding-top: 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: #3a3835;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
}

/* ── Error box ── */
.err-box {
    border: 1px solid #7a3030;
    border-radius: 2px;
    background: #1e1414;
    padding: 1rem 1.2rem;
    color: #d98c8c;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    margin-top: 1.5rem;
}

/* Hide Streamlit clutter */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE = 20000
MAX_LEN    = 400
MODEL_PATH = "best_model.keras"
WI_PATH    = "word_index.json"


# ── Load model & word index (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    model      = load_model(MODEL_PATH)
    with open(WI_PATH) as f:
        word_index = json.load(f)
    return model, word_index


# ── Text helpers ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def encode_review(text: str, word_index: dict) -> list:
    words = clean_text(text).split()
    return [min(word_index.get(w, 2) + 3, VOCAB_SIZE - 1) for w in words]

def predict(text: str, model, word_index: dict) -> float:
    enc    = encode_review(text, word_index)
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding='pre', truncating='pre')
    return float(model.predict(padded, verbose=0)[0][0])


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <div class="site-label">◈ Sentiment Analysis</div>
    <div class="site-title">SentimentLens</div>
    <div class="site-sub">Deep learning · IMDB-trained · Bidirectional LSTM</div>
</div>
""", unsafe_allow_html=True)


# ── Load assets ───────────────────────────────────────────────────────────────
model_loaded = False
try:
    model, word_index = load_assets()
    model_loaded = True
except Exception as e:
    st.markdown(f"""
    <div class="err-box">
        ✗ &nbsp; Could not load model — {e}<br>
        <span style="color:#4a4845">Ensure <code>best_model.keras</code> and
        <code>word_index.json</code> are in the same directory.</span>
    </div>
    """, unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
review_text = st.text_area(
    "Review text",
    placeholder="Paste or type a movie review…",
    height=130,
    label_visibility="visible",
)

run = st.button("Analyse →", disabled=not model_loaded)


# ── Inference & result ────────────────────────────────────────────────────────
if run:
    text = review_text.strip()
    if not text:
        st.markdown('<div class="err-box">✗ &nbsp; Please enter some text first.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner(""):
            prob = predict(text, model, word_index)

        is_pos    = prob > 0.5
        cls       = "pos" if is_pos else "neg"
        label     = "Positive" if is_pos else "Negative"
        emoji     = "◉" if is_pos else "◎"
        pct       = prob if is_pos else 1 - prob
        fill_pct  = int(pct * 100)

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-label">Result</div>
            <div class="verdict {cls}">{emoji}&ensp;{label}</div>
            <div class="meter-row">
                <div class="meter-track">
                    <div class="meter-fill {cls}" style="width:{fill_pct}%"></div>
                </div>
                <div class="meter-pct">{pct:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Save to history (most recent first, cap at 10)
        st.session_state.history.insert(0, {
            "text": text,
            "label": label,
            "cls": cls,
            "pct": pct,
        })
        st.session_state.history = st.session_state.history[:10]


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="history-header">Recent analyses</div>', unsafe_allow_html=True)
    for item in st.session_state.history:
        short = item["text"][:90] + ("…" if len(item["text"]) > 90 else "")
        st.markdown(f"""
        <div class="hist-item">
            <span class="hist-text">{short}</span>
            <span class="hist-badge {item['cls']}">{item['label']} · {item['pct']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    <span>SentimentLens</span>
    <span>BiLSTM · IMDB · 20k vocab</span>
</div>
""", unsafe_allow_html=True)