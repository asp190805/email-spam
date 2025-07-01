import streamlit as st
import pandas as pd
import joblib

st.title("📨 Spam Classifier")

try:
    model = joblib.load("spam_classifier.pkl")
except:
    st.error("❌ Model file not found.")
    st.stop()

subject = st.text_input("✉️ Enter email subject")

if subject:
    data = pd.DataFrame([{
        "Subject": subject,
        "subject_length": len(subject),
        "num_uppercase_words": sum(1 for w in subject.split() if w.isupper()),
        "num_exclamations": subject.count("!"),
        "percent_uppercase": sum(1 for c in subject if c.isupper()) / (len(subject) + 1)
    }])

    try:
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][pred]
        label = "📢 Spam" if pred else "✅ Ham"
        st.success(f"Prediction: **{label}** ({prob*100:.2f}%)")
    except Exception as e:
        st.error(f"❌ Error: {e}")
