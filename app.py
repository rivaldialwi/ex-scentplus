import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
    cleaned_text = clean_text(input_text)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk mengonversi dataframe ke file CSV
@st.cache
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

# Fungsi untuk membuat word cloud
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# File uploader for Excel files
st.header("Prediksi Sentimen dari File Excel")
uploaded_file_excel = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file_excel is not None:
    df = pd.read_excel(uploaded_file_excel)
    
    if 'Text' in df.columns:
        X = df['Text'].apply(clean_text)
        X_tfidf = tfidf_vectorizer.transform(X)
        
        df['Human'] = logreg_model.predict(X_tfidf)
        
        st.write(df)
        
        st.download_button(
            label="Unduh file dengan prediksi",
            data=convert_df_to_csv(df),
            file_name="prediksi_sentimen.csv",
            mime="text/csv"
        )
    else:
        st.error("File Excel harus memiliki kolom 'Text'.")

# File uploader for CSV files
st.header("Analisis Sentimen dari File CSV")
uploaded_file_csv = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file_csv is not None:
    df = pd.read_csv(uploaded_file_csv)
    
    if 'Text' in df.columns and 'Human' in df.columns:
        df['Cleaned_Text'] = df['Text'].apply(clean_text)
        X = df['Cleaned_Text']
        X_tfidf = tfidf_vectorizer.transform(X)
        
        y_pred = logreg_model.predict(X_tfidf)
        
        df['Predicted_Sentiment'] = y_pred
        
        st.write(df)
        
        # Menghitung metrik evaluasi
        if 'Human' in df.columns:
            accuracy = accuracy_score(df['Human'], df['Predicted_Sentiment'])
            precision = precision_score(df['Human'], df['Predicted_Sentiment'], average='weighted', zero_division=1)
            recall = recall_score(df['Human'], df['Predicted_Sentiment'], average='weighted', zero_division=1)
            
            st.write("Akurasi:", accuracy)
            st.write("Presisi:", precision)
            st.write("Recall:", recall)
        
        # Plot jumlah sentimen
        sentiment_counts = df['Predicted_Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Word Cloud untuk setiap sentimen
        for sentiment in df['Predicted_Sentiment'].unique():
            text = ' '.join(df[df['Predicted_Sentiment'] == sentiment]['Cleaned_Text'])
            create_wordcloud(text, f'Word Cloud untuk Sentimen {sentiment}')
    else:
        st.error("File CSV harus memiliki kolom 'Text' dan 'Human'.")