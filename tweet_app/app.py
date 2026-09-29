import streamlit as st
import joblib
import re
import pandas as pd

st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

@st.cache_resource
def load_models():
    try:
        model = joblib.load("tweet_classifier.joblib")
        vectorizer = joblib.load("tweet_vectorizer.joblib")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found!")
        st.stop()

model, vectorizer = load_models()

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_sentiment(text):
    cleaned_text = clean_tweet(text)
    if not cleaned_text:
        return None, None, "Empty"
    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    results = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
    return prediction, results, cleaned_text

st.markdown("# 🐦 Tweet Sentiment Analyzer")
st.markdown("Classify tweets as Positive, Negative, or Neutral")

tweet_input = st.text_area("Enter your tweet:", height=100)

if tweet_input.strip():
    prediction, probabilities, cleaned = predict_sentiment(tweet_input)
    if prediction is not None:
        st.subheader("Result")
        st.write(f"Sentiment: **{prediction.upper()}**")
        if probabilities:
            st.write(f"Confidence: **{max(probabilities.values()):.1%}**")
            for sentiment in sorted(probabilities.keys()):
                st.write(f"{sentiment}: {probabilities[sentiment]:.1%}")
else:
    st.info("Enter a tweet above")
