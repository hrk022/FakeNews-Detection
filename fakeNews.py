import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import base64

# Load Model & Tokenizer
try:
    model = tf.keras.models.load_model("model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    st.success("Model & Tokenizer Loaded Successfully!")
except Exception as e:
    st.error(f"Error: {e}")

# Function to Set Background Image
def set_background(image_file):
    try:
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
    except Exception as e:
        st.warning(f"Background image not found: {e}")

# Set Background
set_background("news.jpg")

# User Input
st.title("Fake News Detection")
news_input = st.text_area("Enter the news text:")

if st.button("Predict") and news_input.strip():
    try:
        # Convert input to tokenized sequences
        input_seq = np.array(tokenizer.texts_to_sequences([news_input]), dtype=object)
        input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=300, dtype='int32')

        # Make Prediction
        prediction = model.predict(input_padded)
        result = "Real News" if prediction[0][0] > 0.5 else "Fake News"
        
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {e}")
