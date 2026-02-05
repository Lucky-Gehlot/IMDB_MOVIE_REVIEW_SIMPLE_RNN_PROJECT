#In this you will make your actual streamlit app of this project here
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
    #cleaned_review is my additional step because it can be possible that encoded_review can contains words which index can be out of vocabulaury (out of range of 10000)
    cleaned_review = [word if word < 10000 else 2 for word in encoded_review]
    #pad sequences excepts list of lists
    padded_review = sequence.pad_sequences([cleaned_review], maxlen=500)
    return padded_review

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis System')

st.write('Enter moview review to get sentiment prediction of that review')

user_input = st.text_area('Movie Review')

if st.button('Classify'):

    movie_review = preprocess_text(user_input)

    #make prediction 
    prediction = model.predict(movie_review)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    #Display the result
    st.write(f"sentiment : {sentiment}")
    st.write(f"prediction score : {prediction[0][0]}")

else:
    st.write("Please enter movie review")