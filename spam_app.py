import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📧 Email Spam Classifier")
st.write("Enter the email subject line below. The app will analyze and classify it as either **Spam** or **Ham** based on content and formatting.")

# Check if model file exists
MODEL_PATH = "spam_classifier_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file `{MODEL_PATH}` not found. Please upload the trained model.")
    st.stop()

# Load the model
model = joblib.load(MODEL_PATH)

# Input field
subject_input = st.text_input("✉️ Email Subject", placeholder="e.g., WIN a FREE vacation NOW!!!")

if subject_input:
    # Feature engineering
    subject_length = len(subject_input)
    num_uppercase_words = sum(1 for w in subject_input.split() if w.isupper())
    num_exclamations = subject_input.count("!")
    percent_uppercase = sum(1 for c in subject_input if c.isupper()) / (len(subject_input) + 1)

    # Create input dataframe
    input_df = pd.DataFrame([{
        "Subject": subject_input,
        "subject_length": subject_length,
        "num_uppercase_words": num_uppercase_words,
        "num_exclamations": num_exclamations,
        "percent_uppercase": percent_uppercase
    }])

    # Prediction
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction]
        label = "📢 Spam" if prediction == 1 else "✅ Ham"
        st.success(f"Result: **{label}** ({proba * 100:.2f}% confidence)")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
