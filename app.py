import pickle


import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string


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
        ps.stem(i)
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Classifier")

input_msg = st.text_input("Enter the Message")

if st.button("Predict"):
    #1.preprocess
    transformed_sms = transform_text(input_msg)

    #2.vectorise
    vector_input =tfidf.transform([transformed_sms])

    #3.predic
    result = model.predict(vector_input)[0]

    #4.Display
    if result ==1:
        st.header("Spam")
    else:
        st.header("Not spam")
