import os
import re
import json
import warnings
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Bidirectional,
    Dropout, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# Basic setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Configuration
VOCAB_SIZE = 20000
MAX_LEN = 400
EMBED_DIM = 256
LSTM_UNITS = 128

BATCH_SIZE = 64
EPOCHS = 20


# Load data
print("Loading dataset...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='pre', truncating='pre')
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding='pre', truncating='pre')

print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")


# Build model
def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.3),

        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)),
        Dropout(0.4),

        Bidirectional(LSTM(64, dropout=0.2)),
        Dropout(0.4),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_model()
model.summary()


# Training setup
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        "best_model.keras",
        monitor='val_accuracy',
        save_best_only=True
    )
]


# Train
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*40)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss    : {loss:.4f}")
print("="*40 + "\n")


# Text processing
word_index = imdb.get_word_index()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def encode_review(text: str) -> list:
    words = clean_text(text).split()
    return [min(word_index.get(w, 2) + 3, VOCAB_SIZE - 1) for w in words]


# Demo predictions
sample_reviews = [
    "This movie was absolutely fantastic I loved every moment",
    "Worst movie ever it was boring and a waste of time",
    "The plot was interesting but the acting was not good",
    "A masterpiece of storytelling with breathtaking performances"
]

encoded = [encode_review(r) for r in sample_reviews]
padded  = pad_sequences(encoded, maxlen=MAX_LEN, padding='pre')

preds = model.predict(padded, verbose=0)

print("Predictions:\n")
for text, pred in zip(sample_reviews, preds):
    prob = float(pred[0])
    label = "Positive" if prob > 0.5 else "Negative"
    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))

    print(text)
    print(f"→ {label} [{bar}] ({prob:.2%})\n")


# Save outputs
model.save("sentiment_model.keras")

with open("word_index.json", "w") as f:
    json.dump(word_index, f)

print("Model and tokenizer saved.")
