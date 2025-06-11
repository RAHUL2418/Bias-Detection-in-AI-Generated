import streamlit as st
import os
import sys
import joblib
import string
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# --- NLTK Downloads ---
nltk.download("punkt")
nltk.download("stopwords")

# --- Page Configuration ---
st.set_page_config(page_title="Bias Detector | GenAI Enhanced", layout="wide")
st.title("🧠 Bias Detection in AI-Generated Content")
st.markdown("Built with Machine Learning + GPT-2")

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# --- Fallback Preprocessing if Import Fails ---
try:
    from preprocessing import clean_text
except ImportError:
    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)

# --- Load Model, Vectorizer, and GPT-2 ---
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "bias_detector.pkl"))
vectorizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl"))

@st.cache_resource
def load_model():
    return joblib.load(model_path)

@st.cache_resource
def load_vectorizer():
    return joblib.load(vectorizer_path)

@st.cache_resource
def load_genai():
    return pipeline("text-generation", model="gpt2")

model = load_model()
vectorizer = load_vectorizer()
genai = load_genai()

# --- UI Layout ---
with st.container():
    st.markdown("### ✍️ Enter your content below:")
    user_input = st.text_area("Your Text", height=150, placeholder="Type or paste AI-generated or human-written content...")

    if st.button("🔍 Detect Bias"):
        if not user_input.strip():
            st.warning("🚫 Please enter some text.")
        else:
            # Clean and transform
            cleaned = clean_text(user_input)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            prob = model.predict_proba(features)[0]

            # Show prediction
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    if prediction == 1:
                        st.error("⚠️ This content is predicted as **Biased**.")
                    else:
                        st.success("✅ This content is predicted as **Unbiased**.")

                with col2:
                    st.metric(label="Confidence Score", value=f"{np.max(prob) * 100:.2f}%")

            # --- GenAI Explanation ---
            st.markdown("### 🤖 View GenAI Explanation")
            with st.spinner("Generating explanation..."):
                prompt = f"Explain why the following content might be biased or unbiased:\n\"{user_input.strip()}\""

                try:
                    explanation_raw = genai(
                        prompt,
                        max_length=150,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        eos_token_id=50256,
                    )[0]["generated_text"]

                    # Remove repetition of prompt
                    explanation_clean = explanation_raw.replace(prompt, "").strip()

                    # Clean explanation formatting
                    lines = explanation_clean.split(". ")
                    formatted = "\n".join(
                        f"- {line.strip()}." for line in lines
                        if 10 < len(line.strip()) < 200
                    )

                    # Show fallback if GPT output is unclear
                    if not formatted.strip():
                        fallback_explanation = """
- The content does not express strong opinions, stereotypes, or assumptions.
- It uses neutral or personal language without targeting any group.
- This supports the prediction that the content is unbiased.
                        """
                        st.info(fallback_explanation)
                    else:
                        st.info(formatted)

                except Exception as e:
                    st.error(f"Failed to generate explanation: {e}")

# --- Footer ---
st.markdown("""---  
© 2025 | Built using 🐍 Python, 🤖 Scikit-learn, 🤗 Transformers, and 🧠 Streamlit
""")
