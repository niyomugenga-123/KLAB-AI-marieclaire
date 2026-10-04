import os
import re
import joblib
import streamlit as st

LEAK = re.compile(r"#(?!ironing\b)\w*(?:sarcas|iron[iy])\w*", flags=re.IGNORECASE)

def remove_label_tags(text):
    return LEAK.sub("", text)

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", " and ")
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_model():
    folder = os.path.dirname(__file__)
    model = joblib.load(os.path.join(folder, "tweet_classifier.joblib"))
    vec = joblib.load(os.path.join(folder, "tweet_vectorizer.joblib"))
    return vec, model

vec, model = load_model()

st.title("Tweet Figurative-Language Classifier")
st.write("Paste a tweet. The model predicts whether it is **regular**, **irony**, "
         "**sarcasm** or **figurative**.")

tweet = st.text_area("Tweet", height=120)

if st.button("Classify") and tweet.strip():
    cleaned = clean_tweet(remove_label_tags(tweet))
    if not cleaned:
        st.warning("Nothing left to analyse after cleaning. Try a tweet with more words.")
    else:
        probs = model.predict_proba(vec.transform([cleaned]))[0]
        order = probs.argsort()[::-1]
        st.subheader(f"Prediction: {model.classes_[order[0]]} ({probs[order[0]]:.0%})")
        for i in order:
            st.progress(float(probs[i]), text=f"{model.classes_[i]}: {probs[i]:.0%}")

st.caption("Test accuracy is about 58%. The model is very reliable for regular tweets "
           "but often confuses irony, sarcasm and figurative language.")