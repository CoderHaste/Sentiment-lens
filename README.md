# 🎬 SentimentLens
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

A modern **Sentiment Analysis Web App** built using **Deep Learning (Bidirectional LSTM)** and **Streamlit**, capable of classifying movie reviews as **Positive** or **Negative** with confidence scores.

---

## 🚀 Overview

SentimentLens uses a trained neural network on the IMDB dataset to analyze user-input text and predict sentiment in real time.
The project combines **Natural Language Processing (NLP)** with a **clean, minimal UI** for an interactive experience.

---

## ✨ Features

* 🧠 Deep Learning model (BiLSTM)
* ⚡ Real-time sentiment prediction
* 📊 Confidence score visualization
* 🖥️ Modern Streamlit UI
* 📝 Custom text preprocessing pipeline
* 📜 Recent prediction history tracking

---

## 🧠 Model Details

* **Dataset:** IMDB Movie Reviews (Keras)
* **Vocabulary Size:** 20,000
* **Max Sequence Length:** 400
* **Architecture:**

  * Embedding Layer + Spatial Dropout
  * 2× Bidirectional LSTM
  * Dense layers for classification

---

## 📂 Project Structure

```bash
sentiment-lens/
│
├── src/
│   └── sentiment_model.py      # Training script
│
├── model/                      # (Optional) Saved model files
│   ├── best_model.keras
│   └── word_index.json
│
├── app.py                      # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-lens.git
cd sentiment-lens
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧪 Training the Model

```bash
python src/sentiment_model.py
```

---

<h2 align="center">📸 Demo</h2>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f182ebb7-1e75-4050-a57c-49cffed4b51b" width="45%" />
  <img src="https://github.com/user-attachments/assets/874cf35a-b48e-484a-8722-ca66e15f7215" width="45%" />
</p>

---

## ⚠️ Notes

* Model files (`.keras`) may not be included due to size limitations.
* If missing, run the training script before launching the app.

---

## 💡 Future Improvements

* 🔄 Transformer-based model (BERT)
* 🌐 Deploy online (Streamlit Cloud / HuggingFace)
* 📦 Use pretrained embeddings (GloVe)
* 📊 Add model performance visualization

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit

---

## 👨‍💻 Author

**Prateek Manjunath**

---

## ⭐ Show your support

If you like this project, consider giving it a ⭐ on GitHub!
