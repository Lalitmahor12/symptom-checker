import streamlit as st
import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- NLTK data ----------
@st.cache_resource
def ensure_nltk():
    for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk()

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------- Preprocessing (identical to training) ----------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ---------- Emergency keywords ----------
EMERGENCY_KEYWORDS = [
    'chest pain', 'difficulty breathing', 'severe bleeding',
    'unconscious', 'stroke', 'heart attack', 'suicidal',
    'seizure', 'coughing blood', 'blood in vomit', 'paralysis'
]

# ---------- Page ----------
st.set_page_config(page_title="Symptom Checker", page_icon="🩺", layout="centered")

st.title("🩺 Symptom Checker")
st.caption("ML-powered symptom triage assistant")

st.error(
    "⚠️ **MEDICAL DISCLAIMER:** This tool is for **educational purposes only** "
    "and is **NOT** a substitute for professional medical advice, diagnosis, or "
    "treatment. Always consult a qualified healthcare provider. If you are "
    "experiencing a medical emergency, call your local emergency number immediately."
)

user_input = st.text_area(
    "Describe your symptoms in detail:",
    height=180,
    placeholder="Example: I have a high fever, severe headache, and pain behind my eyes. I also feel nauseous and have red spots on my arms."
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if analyze:
    if not user_input.strip():
        st.warning("Please describe your symptoms before analyzing.")
    elif len(user_input.split()) < 3:
        st.warning("Please provide more detail (at least a few words).")
    elif any(kw in user_input.lower() for kw in EMERGENCY_KEYWORDS):
        st.error(
            "🚨 **EMERGENCY SYMPTOMS DETECTED**\n\n"
            "Your description contains symptoms that may require **immediate "
            "emergency medical attention**. Please call your local emergency "
            "number or go to the nearest emergency room **now**.\n\n"
            "Do not rely on this tool for emergency assessment."
        )
    else:
        with st.spinner("Analyzing symptoms..."):
            clean = preprocess(user_input)
            vec = vectorizer.transform([clean])

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(vec)[0]
            else:
                scores = model.decision_function(vec)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()

            classes = model.classes_
            top3_idx = probs.argsort()[-3:][::-1]

        st.subheader("🔎 Possible Conditions")
        st.caption("These are statistical guesses, not a diagnosis.")

        for rank, idx in enumerate(top3_idx, 1):
            disease = classes[idx]
            confidence = probs[idx] * 100
            if rank == 1:
                st.success(f"**{rank}. {disease}** — {confidence:.1f}% confidence")
            else:
                st.info(f"**{rank}. {disease}** — {confidence:.1f}% confidence")

        st.divider()
        st.warning(
            "**Next steps:**\n"
            "- This is **not** a diagnosis.\n"
            "- Book an appointment with a doctor for proper evaluation.\n"
            "- If symptoms worsen suddenly, seek emergency care."
        )

with st.sidebar:
    st.header("About")
    st.write(
        "Trained on the **Symptom2Disease dataset** (1,200 symptom "
        "descriptions across 24 conditions) using TF-IDF + LinearSVC."
    )
    st.header("Model Info")
    st.write("**Dataset:** Symptom2Disease (Kaggle)")
    st.write("**Classes:** 24 diseases")
    st.write("**Features:** TF-IDF (max 5,000 n-grams)")
    st.write("**Model:** LinearSVC (with softmax)")
    st.header("⚠️ Safety")
    st.write(
        "No user input is stored. This is a portfolio project, "
        "not a medical device."
    )