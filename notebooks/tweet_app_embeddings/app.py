import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix, hstack

MODELS = Path(__file__).parent

# Same label-hashtag removal and cleaning as in training
LEAK = re.compile(r"#(?!ironing\b)\w*(?:sarcas|iron[iy])\w*", flags=re.IGNORECASE)


def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", " and ")
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_assets():
    bundle = joblib.load(MODELS / "tweet_classifier_glove.joblib")
    data = np.load(MODELS / "glove100_subset.npz")
    vectors = data["vectors"]
    index = {w: i for i, w in enumerate(data["words"].tolist())}
    return bundle, vectors, index


def pool(text, vectors, index):
    """Mean + max pooling of GloVe word vectors (same as in training)."""
    ids = [index[w] for w in text.split() if w in index]
    dim = vectors.shape[1]
    if not ids:
        return np.zeros(2 * dim)
    v = vectors[ids]
    return np.concatenate([v.mean(axis=0), v.max(axis=0)])


def predict(tweet, bundle, vectors, index):
    cleaned = clean_tweet(LEAK.sub("", tweet))
    if not cleaned:
        return None, None
    x_tfidf = bundle["tfidf"].transform([cleaned])
    x_emb = bundle["scaler"].transform(pool(cleaned, vectors, index).reshape(1, -1))
    x = hstack([x_tfidf, csr_matrix(x_emb * bundle["embed_weight"])]).tocsr()
    model = bundle["model"]
    scores = model.decision_function(x)[0]
    rel = np.exp(scores - scores.max())
    rel = rel / rel.sum()
    return model.predict(x)[0], dict(zip(model.classes_, rel))


st.set_page_config(page_title="Irony & Sarcasm Detector", page_icon="🙃")
st.title("Irony & Sarcasm Detector")
st.write(
    "Paste a tweet and the model predicts whether it is **irony**, **sarcasm**, "
    "**figurative** or **regular**. It uses TF-IDF features plus pre-trained "
    "GloVe-Twitter-100 word embeddings."
)

bundle, vectors, index = load_assets()

tweet = st.text_area("Tweet", value="Oh great, another Monday. Just what I needed.", height=120)

if st.button("Classify"):
    label, rel = predict(tweet, bundle, vectors, index)
    if label is None:
        st.warning("Nothing left to classify after cleaning. Try a longer tweet.")
    else:
        st.subheader(f"Prediction: {label}")
        st.caption("Relative scores between the four classes (not calibrated probabilities).")
        st.bar_chart(rel)

with st.expander("About this model"):
    st.markdown(
        """
- **Data:** about 81,000 labelled tweets. Hashtags that give away the label
  (`#irony`, `#sarcasm`, ...) are removed before prediction, so adding them to your tweet changes nothing.
- **Test accuracy:** 0.591 (TF-IDF + GloVe-Twitter-100, Linear SVM) vs 0.584 for TF-IDF alone.
- **Known weakness:** the `figurative` class overlaps with irony and sarcasm and is
  predicted poorly (F1 about 0.15). `regular` tweets are detected reliably.
"""
    )
