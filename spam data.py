import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="📨 Spam Classifier", layout="centered")
st.title("📨 Email Spam Classifier")

MODEL_FILE = "spam_classifier_model.pkl"

# Load model
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file 'spam_classifier_model.pkl' not found.")
    st.stop()

model = joblib.load(MODEL_FILE)

# Input
subject = st.text_input("✉️ Enter Email Subject", placeholder="e.g., WIN a FREE trip NOW!!!")

if subject:
    # Match structure used during training
    input_df = pd.DataFrame([{
        "Subject": subject,
        "subject_length": len(subject),
        "num_uppercase_words": sum(1 for w in subject.split() if w.isupper()),
        "num_exclamations": subject.count("!"),
        "percent_uppercase": sum(1 for c in subject if c.isupper()) / (len(subject) + 1)
    }])

    try:
        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        label = "📢 SPAM" if pred else "✅ HAM"
        st.success(f"Prediction: **{label}** ({prob * 100:.2f}% confidence)")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

if prob >= 0.75:
    st.success(f"✅ Confident: **{label}** ({prob*100:.2f}%)")
elif prob >= 0.5:
    st.info(f"🤔 Possibly: **{label}** ({prob*100:.2f}%)")
else:
    st.warning(f"⚠️ Not confident: **{label}** ({prob*100:.2f}%)")

