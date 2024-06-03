import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from wordcloud import WordCloud

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

# Fungsi untuk menampilkan Word Cloud
def generate_wordcloud(data, title):
    wordcloud = WordCloud(width=300, height=200, background_color='white').generate(' '.join(data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# File uploader for Excel and CSV files
uploaded_file = st.file_uploader("Unggah file Excel atau CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    if 'Text' in df.columns:
        X = df['Text'].apply(clean_text)
        X_tfidf = tfidf_vectorizer.transform(X)
        
        df['Human'] = logreg_model.predict(X_tfidf)
        
        # Display the dataframe with predictions
        st.write(df)
        
        # Convert dataframe to CSV file for download
        @st.cache
        def convert_df_to_csv(df):
            output = BytesIO()
            df.to_csv(output, index=False)
            processed_data = output.getvalue()
            return processed_data
        
        st.download_button(
            label="Unduh file dengan prediksi",
            data=convert_df_to_csv(df),
            file_name="prediksi_sentimen.csv",
            mime="text/csv"
        )

        # Display sentiment distribution bar chart
        st.subheader("Distribusi Sentimen")
        sentiment_counts = df['Human'].value_counts()
        st.bar_chart(sentiment_counts)

        # Display Word Clouds for each sentiment
        st.subheader("Kata-Kata yang Sering Muncul")
        for sentiment in df['Human'].unique():
            generate_wordcloud(df[df['Human'] == sentiment]['Text'], f'Word Cloud untuk Sentimen {sentiment}')

        # Display accuracy, precision, and recall
        y_true = df['Human']
        y_pred = logreg_model.predict(X_tfidf)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        
        st.subheader("Metrik Evaluasi")
        st.write(f"Akurasi: {accuracy:.2f}")
        st.write(f"Presisi: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
    else:
        st.error("File harus memiliki kolom 'Text'.")