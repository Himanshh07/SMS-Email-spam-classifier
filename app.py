
import streamlit as st
import pickle
import string
import nltk
import os

# Set a custom nltk data path and environment variable
NLTK_PATH = "/tmp/nltk_data"
nltk.data.path.append(NLTK_PATH)
os.environ["NLTK_DATA"] = NLTK_PATH

# Download necessary NLTK data if not already present
# nltk.download('punkt', download_dir=NLTK_PATH)
nltk.download('stopwords', download_dir=NLTK_PATH)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Define the stemmer object
ps = PorterStemmer()

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)

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

# Load the vectorizer and model from the current directory
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('main.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("Enter your message here 👇")

if st.button("Predict"):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output
    if result == 1:
        st.error("🚨 Spam Message Detected!")
    else:
        st.success("✅ Not a Spam Message!")
