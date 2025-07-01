
# app.py
import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("spam_classifier.pkl")  # Update with your model filename

st.title("Email Spam Classifier")
st.write("Enter the email subject below to classify it as spam or ham.")

# Input field
subject_input = st.text_input("Email Subject")

if subject_input:
    # Feature engineering
    subject_length = len(subject_input)
    num_uppercase_words = sum(1 for w in subject_input.split() if w.isupper())
    num_exclamations = subject_input.count("!")
    percent_uppercase = sum(1 for c in subject_input if c.isupper()) / (len(subject_input) + 1)

    # Create dataframe
    input_df = pd.DataFrame([{
        "Subject": subject_input,
        "subject_length": subject_length,
        "num_uppercase_words": num_uppercase_words,
        "num_exclamations": num_exclamations,
        "percent_uppercase": percent_uppercase
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]

    label = "Spam" if prediction == 1 else "Ham"
    st.success(f"The email is classified as **{label}** with {proba*100:.2f}% confidence.")
