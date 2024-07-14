import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open('prediksi_Tanggapan.pkl', 'rb'))
vectorizer = pickle.load(open('tf_idf_vectorizer.pkl', 'rb'))

st.title("Prediksi Sentiment Review Film dalam bahasa inggris")

# Input text 
komen = st.text_area("Masukkan Teks yang ingin di uji", height=400)

komen_predict = ''

if st.button('Predict'):
    if not komen:
        st.warning("Please input your Text")
    else:
        # Transform input Text using the loaded vectorizer
        komen_transformed = vectorizer.transform([komen])

        # Make prediction using the loaded model
        predict_komen = model.predict(komen_transformed)

        if predict_komen == 1:
            st.success("Teks yang anda masukkan termasuk ke dalam : Review Positive")
        else:
            st.warning("Teks yang anda masukkan termasuk ke dalam : Review Negative")
