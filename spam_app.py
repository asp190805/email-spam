import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="📧 Spam Classifier", layout="centered")
st.title("📨 Email Spam Classifier")

MODEL_FILE = "spam_classifier.pkl"

# Load the pipeline model
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file 'spam_classifier_model.pkl' not found in app directory.")
    st.stop()

model = joblib.load(MODEL_FILE)

# User input
subject = st.text_input("✉️ Enter Email Subject", placeholder="E.g., WIN a FREE trip NOW!!!")

if subject:
    # Feature extraction
    subject_length = len(subject)
    num_uppercase_words = sum(1 for w in subject.split() if w.isupper())
    num_exclamations = subject.count("!")
    percent_uppercase = sum(1 for c in subject if c.isupper()) / (len(subject) + 1)

    # Input for model
    input_df = pd.DataFrame([{
        "Subject": subject,
        "subject_length": subject_length,
        "num_uppercase_words": num_uppercase_words,
        "num_exclamations": num_exclamations,
        "percent_uppercase": percent_uppercase
    }])

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        label = "📢 Spam" if pred == 1 else "✅ Ham"
        st.success(f"**{label}** — Confidence: {prob * 100:.2f}%")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
