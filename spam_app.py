import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📨 Email Spam Classifier")

MODEL_FILE = "spam_classifier.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file 'spam_classifier_model.pkl' not found. Please upload it to the root directory.")
    st.stop()

# Load the trained pipeline
model = joblib.load(MODEL_FILE)

subject = st.text_input("✉️ Enter Email Subject", placeholder="e.g., Exclusive DEALS just for YOU!")

if subject:
    # Feature engineering
    subject_length = len(subject)
    num_uppercase_words = sum(1 for w in subject.split() if w.isupper())
    num_exclamations = subject.count("!")
    percent_uppercase = sum(1 for c in subject if c.isupper()) / (len(subject) + 1)

    # Input DataFrame for model
    input_df = pd.DataFrame([{
        "Subject": subject,
        "subject_length": subject_length,
        "num_uppercase_words": num_uppercase_words,
        "num_exclamations": num_exclamations,
        "percent_uppercase": percent_uppercase
    }])

    try:
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction]
        label = "📢 Spam" if prediction == 1 else "✅ Ham"
        st.success(f"Prediction: **{label}**\n\nConfidence: **{confidence*100:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
