# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

# Set the page layout
st.set_page_config(page_title='IMDB Sentiment Analysis', layout='centered')

# App Title and Header
st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: #4CAF50;">🎬 MovieMood: IMDB Sentiment Classifier </h2>
        <p style="font-size: 18px;">Enter a movie review to analyze its sentiment.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# User Input Section with Columns
st.write('')
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_input = st.text_area('Movie Review', placeholder='Type your review here...')

# Classify Button
if st.button('🎯 Classify Sentiment'):
    if user_input:
        # Preprocess the input (dummy function for now)
        preprocessed_input = preprocess_text(user_input)

        # Make Prediction (dummy model code here)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display Result with Emoji Feedback
        emoji = '😊' if sentiment == 'Positive' else '😞'
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <h2>Sentiment: {sentiment} {emoji}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.warning('Please enter a movie review.')
else:
    st.info('Awaiting your input!')

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px;">
        Made with ❤️ using Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)
