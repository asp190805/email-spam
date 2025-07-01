
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="📨 Spam Classifier", layout="centered")
st.title("📨 Email Spam Classifier")

MODEL_FILE = "spam_regression_model.pkl"

# Load model
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file 'spam_regression_model.pkl' not found.")
    st.stop()

model = joblib.load(MODEL_FILE)

# Input from user
subject = st.text_input("✉️ Enter Email Subject", placeholder="e.g., WIN a FREE trip NOW!!!")

if subject:
    # Prepare feature row
    input_df = pd.DataFrame([{
        "Subject": subject,
        "subject_length": len(subject),
        "num_uppercase_words": sum(1 for w in subject.split() if w.isupper()),
        "num_exclamations": subject.count("!"),
        "percent_uppercase": sum(1 for c in subject if c.isupper()) / (len(subject) + 1)
    }])

    # Predict
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        label = "📢 SPAM" if pred else "✅ HAM"

        # Output with confidence interpretation
        if prob >= 0.75:
            st.success(f"**{label}** — Confident ({prob * 100:.2f}%)")
        elif prob >= 0.5:
            st.info(f"**{label}** — Moderate confidence ({prob * 100:.2f}%)")
        else:
            st.warning(f"**{label}** — Low confidence ({prob * 100:.2f}%)")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

