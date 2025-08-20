import streamlit as st
import pickle
import nltk
import string 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps=PorterStemmer()
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model_used.pkl','rb'))

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text) ## tokenize and gives list of words
    y=[]
    for i in text:
        if i.isalnum(): ## using alnum to remove special characters
            y.append(i)

    text=y[:] ## list cannot be copied you have to clone it, done by [:] using this.
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y) ## returning as string 

st.title("Email Spam Classifier")
input_sms=st.text_input("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms=transform_text(input_sms)
    # 2. vectorize
    vector_input=tfidf.transform([transformed_sms])
    # 3. predict
    result=model.predict(vector_input)[0]
    # 4. display
    if result== 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

 

 
    



