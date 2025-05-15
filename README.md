# Email Spam Detection using Machine Learning

This project implements a spam detection system using machine learning algorithms such as Naive Bayes, Random Forest, and Logistic Regression. The system utilizes text preprocessing, TF-IDF vectorization, and a voting ensemble model to classify emails as 'Ham' or 'Spam'. The application is deployed using a Streamlit interactive interface.

## 🚀 Project Structure:
```
/Email-Spam-Detection
│── app.py                   # Streamlit app for email classification
│── tfidf_vectorizer.pkl     # Saved TF-IDF Vectorizer
│── rf_model.pkl             # Saved Random Forest Model
│── lr_model.pkl             # Saved Logistic Regression Model
│── nb_model.pkl             # Saved Naive Bayes Model
│── README.md                # Project documentation
```

## 📦 Requirements:
- Python 3.7+
- Streamlit
- NLTK
- Scikit-Learn
- Gensim
- Pandas
- Numpy

Install dependencies:
```
pip install -r requirements.txt
```

## 📂 Dataset:
- The dataset used in this project is a collection of emails labeled as 'ham' or 'spam'.
- Ensure the dataset is preprocessed and saved as a DataFrame before training the models.

## ✅ Model Training:
1. Apply text preprocessing (lowercase conversion, stopword removal, stemming).
2. Extract features using TF-IDF Vectorizer.
3. Train three classifiers:
   - Naive Bayes (TF-IDF only)
   - Random Forest (TF-IDF + Word2Vec)
   - Logistic Regression (TF-IDF + Word2Vec)
4. Save the trained models and vectorizer using pickle.

## 💻 Running the Streamlit App:
1. Navigate to the project directory:
```
cd /path/to/project
```
2. Run the app:
```
streamlit run app.py
```
3. Access the app in the browser at `http://localhost:8501`

## 🛠️ How to Use:
- Enter a sample email in the input box.
- Click **Classify** to predict whether the email is 'Ham' or 'Spam'.

## 📦 Sample Emails:
- Spam: "Congratulations! You've won a FREE vacation to the Bahamas! Click here to claim your $1000 prize now!"
- Ham: "Hi John, just wanted to check in regarding the meeting tomorrow. Let me know if you're available."

## 📝 Future Enhancements:
- Implement advanced preprocessing (lemmatization, named entity recognition).
- Incorporate Word2Vec for enhanced feature extraction.
- Deploy the app using Docker or a cloud platform.

## 🧑‍💻 Author:
Vijeeth
