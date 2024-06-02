import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt

# Fungsi untuk mengunduh resource NLTK secara senyap
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Panggil fungsi unduh NLTK resource di awal
download_nltk_resources()

# Membaca model yang sudah dilatih dan TF-IDF Vectorizer
logreg_model = joblib.load("model100.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

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

# Fungsi untuk menghitung akurasi, presisi, dan recall
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall

# Fungsi untuk menampilkan grafik kata yang paling sering muncul
def plot_most_common_words(texts, n_words=10):
    all_words = [word for text in texts for word in text.split()]
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(n_words)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    plt.title(f'{n_words} Kata yang Paling Sering Muncul')
    plt.xticks(rotation=45)
    st.pyplot()

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# File uploader for Excel files
uploaded_file_predict = st.file_uploader("Unggah file Excel untuk prediksi sentimen", type=["xlsx"])

if uploaded_file_predict is not None:
    # Read the Excel file
    df_predict = pd.read_excel(uploaded_file_predict)
    
    # Check if 'Text' column exists in the uploaded file
    if 'Text' in df_predict.columns:
        # Initialize TF-IDF Vectorizer and fit_transform on the text data
        X_predict = df_predict['Text'].apply(clean_text)
        X_tfidf_predict = tfidf_vectorizer.transform(X_predict)
        
        # Perform predictions
        df_predict['Predicted'] = logreg_model.predict(X_tfidf_predict)
        
        # Show the dataframe with predictions
        st.write(df_predict)
        
        # Convert dataframe to CSV file
        @st.cache
        def convert_df_to_csv(df):
            output = BytesIO()
            df.to_csv(output, index=False)
            processed_data = output.getvalue()
            return processed_data
        
        # Create a download button
        st.download_button(
            label="Unduh file dengan prediksi",
            data=convert_df_to_csv(df_predict),
            file_name="prediksi_sentimen.csv",
            mime="text/csv"
        )
    else:
        st.error("File Excel harus memiliki kolom 'Text'.")

# File uploader for CSV files
uploaded_file_analyze = st.file_uploader("Unggah file CSV untuk analisis hasil prediksi", type=["csv"])

if uploaded_file_analyze is not None:
    # Read the CSV file
    df_analyze = pd.read_csv(uploaded_file_analyze)
    
    # Check if 'Predicted' column exists in the uploaded file
    if 'Predicted' in df_analyze.columns:
        # Calculate metrics
        accuracy, precision, recall = calculate_metrics(df_analyze['Human'], df_analyze['Predicted'])
        # Show metrics
        st.write(f"Akurasi: {accuracy}")
        st.write(f"Presisi: {precision}")
        st.write(f"Recall: {recall}")
        # Plot most common words
        plot_most_common_words(df_analyze['Text'])
    else:
        st.error("File CSV harus memiliki kolom 'Predicted'.")