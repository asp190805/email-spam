import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="📧 Email Spam Classifier", layout="centered")

st.title("📨 Email Spam Classifier")
st.write("This app predicts whether an email subject is spam or not based on text and structure.")

MODEL_PATH = "spam_classifier.pkl"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file `spam_classifier_model.pkl` not found. Please upload or include it.")
    st.stop()

# Load the model
model = joblib.load(MODEL_PATH)

# Input
subject_input = st.text_input("✉️ Enter Email Subject", placeholder="E.g., WIN a FREE trip NOW!!!")

if subject_input:
    # Feature Engineering
    subject_length = len(subject_input)
    num_uppercase_words = sum(1 for w in subject_input.split() if w.isupper())
    num_exclamations = subject_input.count("!")
    percent_uppercase = sum(1 for c in subject_input if c.isupper()) / (len(subject_input) + 1)

    # Create input DataFrame
    input_df = pd.DataFrame([{
        "Subject": subject_input,
        "subject_length": subject_length,
        "num_uppercase_words": num_uppercase_words,
        "num_exclamations": num_exclamations,
        "percent_uppercase": percent_uppercase
    }])

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction]
        label = "📢 Spam" if prediction == 1 else "✅ Ham"
        st.success(f"**{label}** — Confidence: {proba*100:.2f}%")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
