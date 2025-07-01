import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📨 Email Spam Classifier")

# Load model
MODEL_FILE = "spam_classifier.pkl"
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error("❌ Model file not found.")
    st.stop()

# User input
subject = st.text_input("✉️ Enter Email Subject", placeholder="e.g., WIN A FREE PRIZE!!!")

if subject:
    features = {
        "Subject": subject,
        "subject_length": len(subject),
        "num_uppercase_words": sum(1 for w in subject.split() if w.isupper()),
        "num_exclamations": subject.count("!"),
        "percent_uppercase": sum(1 for c in subject if c.isupper()) / (len(subject) + 1)
    }

    input_df = pd.DataFrame([features])

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        label = "📢 Spam" if pred == 1 else "✅ Ham"
        st.success(f"**{label}** — Confidence: {prob * 100:.2f}%")
    except Exception as e:
        st.error(f"❌ Error: {e}")
