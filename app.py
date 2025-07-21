import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # for removing special chracter
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # for removing the punctuation and stopwords
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # for stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
"C:\Users\Manish\Desktop\dataframefile\SMS spam detector\sms_spam_detector.ipynb"

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('main.pkl', 'rb'))


st.title("Email/SMS spam Classifier")
input_sms = st.text_area("Enter the Message")

if st.button("Predict"):

    # 1. Preprocessing
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4.
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
