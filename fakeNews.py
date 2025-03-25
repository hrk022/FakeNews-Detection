import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import base64
import os

# Load Model & Tokenizer
MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Error: Model file 'model.h5' not found.")
    model = None
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model Loaded Successfully!")

if not os.path.exists(TOKENIZER_PATH):
    st.error("Error: Tokenizer file 'tokenizer.pkl' not found.")
    tokenizer = None
else:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    st.success("Tokenizer Loaded Successfully!")

# Function to Set Background Image
def set_background(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{encoded_string}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Background image not found.")

# Set Background
set_background("news.jpg")

# User Input
st.title("Fake News Detection")
news_input = st.text_area("Enter the news text:")

if st.button("Predict"):
    if not news_input.strip():
        st.warning("Please enter some text before predicting.")
    elif model is None or tokenizer is None:
        st.error("Model or Tokenizer is not loaded. Check your files.")
    else:
        try:
            # Convert input to tokenized sequences
            input_seq = tokenizer.texts_to_sequences([news_input])

            if not input_seq[0]:  # Check if tokenization resulted in an empty list
                st.error("Error: Input text could not be tokenized. Try a different input.")
            else:
                input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=300, dtype='int32')

                # Make Prediction
                prediction = model.predict(input_padded)
                result = "Real News" if prediction[0][0] > 0.5 else "Fake News"
                
                st.success(f"Prediction: {result}")

        except Exception as e:
            st.error(f"Error: {e}")
