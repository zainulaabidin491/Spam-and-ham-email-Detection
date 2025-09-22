import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Load Model ----------------
MODEL_PATH = "/content/spam_detector_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer (make sure you saved it earlier as tokenizer.pkl)
with open("/content/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set max length same as training
MAX_LENGTH = 100  # change to whatever you used in training

# ---------------- Preprocessing ----------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# ---------------- Prediction Function ----------------
def predict_spam(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post", truncating="post")
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    return prediction

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam Detector")
st.write("Enter a message below to check if it's **SPAM** or **HAM**.")

user_input = st.text_area("Message:", height=150)

if st.button("Predict"):
    if user_input.strip():
        score = predict_spam(user_input)
        if score > 0.5:
            st.error(f"🚨 This looks like **SPAM** (score: {score:.4f})")
        else:
            st.success(f"✅ This looks like **HAM** (score: {score:.4f})")
    else:
        st.warning("⚠️ Please enter a message first.")
