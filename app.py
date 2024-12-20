import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Explicitly download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def main():
    st.title("Email/SMS Spam Classifier")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        # Preprocess the input message
        transformed_sms = transform_text(input_sms)
        # Vectorize the preprocessed message
        vector_input = tfidf.transform([transformed_sms])
        # Make prediction using the loaded model
        result = model.predict(vector_input)[0]
        # Display the prediction result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

if __name__ == '__main__':
    main()

