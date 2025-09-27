import os, joblib, streamlit as st

MODEL_PATH = "models/tfidf_logreg.joblib"

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Paste a movie review and click Predict.")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded.")
else:
    st.warning("Model not found. Run:  python src/train_sklearn.py")

text = st.text_area("Review text:", height=200)

if st.button("Predict") and model is not None:
    proba = model.predict_proba([text])[0][1]
    label = "Positive" if proba >= 0.5 else "Negative"
    st.markdown(f"**Prediction:** {label}  \n**Confidence:** {proba:.3f}")
