# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data (only run once)
nltk.download('stopwords')

# Preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Load the trained TF-IDF Vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("TF-IDF Vectorizer not found. Please ensure it is saved in the current directory.")

# Load the models
try:
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    with open('nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)

except FileNotFoundError:
    st.error("One or more model files not found. Please ensure they are saved in the current directory.")

# Streamlit UI
st.title("Spam Detection App")
st.write("Enter a message to classify it as Ham or Spam")

# User input
user_input = st.text_input("Message")

if st.button("Classify"):
    if 'tfidf' not in globals():
        st.error("TF-IDF Vectorizer not loaded.")
    else:
        # Preprocess input
        processed_input = preprocess(user_input)
        input_tfidf = tfidf.transform([processed_input])

        # Predict using each model
        nb_pred = nb_model.predict(input_tfidf)
        rf_pred = rf_model.predict(input_tfidf)
        lr_pred = lr_model.predict(input_tfidf)

        # Majority Voting
        combined_pred = np.array([nb_pred[0], rf_pred[0], lr_pred[0]])
        final_prediction = np.bincount(combined_pred).argmax()

        # Output result
        result = "Spam" if final_prediction == 1 else "Ham"
        st.write(f"Prediction: {result}")
