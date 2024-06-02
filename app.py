import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO

# Fungsi untuk mengunduh resource NLTK secara senyap
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Panggil fungsi unduh NLTK resource di awal
download_nltk_resources()

# Membaca model yang sudah dilatih
logreg_model = joblib.load("model100.pkl")

# Inisialisasi objek TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()  # Case folding
    words = word_tokenize(text)  # Tokenizing
    cleaned_words = [word for word in words if word not in stop_words]  # Stopword removal
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]  # Stemming
    return " ".join(stemmed_words)

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    # Membersihkan teks input
    cleaned_text = clean_text(input_text)
    # Mengubah teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    # Melakukan prediksi menggunakan model
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# File uploader for Excel files
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    # Check if 'Text' column exists in the uploaded file
    if 'Text' in df.columns:
        # Initialize TF-IDF Vectorizer and fit_transform on the text data
        X = df['Text'].apply(clean_text)
        X_tfidf = tfidf_vectorizer.fit_transform(X)
        
        # Perform predictions
        df['Human'] = logreg_model.predict(X_tfidf)
        
        # Show the dataframe with predictions
        st.write(df)
        
        # Convert dataframe to Excel file
        @st.cache_data
        def convert_df_to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.save()
            processed_data = output.getvalue()
            return processed_data
        
        # Create a download button
        st.download_button(
            label="Unduh file dengan prediksi",
            data=convert_df_to_excel(df),
            file_name="prediksi_sentimen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("File Excel harus memiliki kolom 'Text'.")