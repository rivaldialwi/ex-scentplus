import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

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

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# Tab untuk Upload File Excel dan CSV
tab1, tab2 = st.tabs(["Upload File Excel", "Upload File CSV"])

with tab1:
    uploaded_excel = st.file_uploader("Unggah file Excel", type=["xlsx"])

    if uploaded_excel is not None:
        df = pd.read_excel(uploaded_excel)
        
        if 'Text' in df.columns:
            X = df['Text'].apply(clean_text)
            X_tfidf = tfidf_vectorizer.transform(X)
            df['Human'] = logreg_model.predict(X_tfidf)
            st.write(df)
            
            @st.cache_data
            def convert_df_to_csv(df):
                output = BytesIO()
                df.to_csv(output, index=False)
                return output.getvalue()
            
            st.download_button(
                label="Unduh file dengan prediksi",
                data=convert_df_to_csv(df),
                file_name="prediksi_sentimen.csv",
                mime="text/csv"
            )
        else:
            st.error("File Excel harus memiliki kolom 'Text'.")

with tab2:
    uploaded_csv = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        
        if 'Text' in df.columns and 'Human' in df.columns:
            X = df['Text'].apply(clean_text)
            X_tfidf = tfidf_vectorizer.transform(X)
            y_true = df['Human']
            y_pred = logreg_model.predict(X_tfidf)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
            
            st.write(f"Akurasi: {accuracy}")
            st.write(f"Presisi: {precision}")
            st.write(f"Recall: {recall}")
            
            sentiment_counts = df['Human'].value_counts()
            st.bar_chart(sentiment_counts)
            
            st.subheader("Word Cloud untuk Setiap Sentimen")
            for sentiment in df['Human'].unique():
                text = ' '.join(df[df['Human'] == sentiment]['Text'])
                wordcloud = WordCloud(width=300, height=200, background_color='white').generate(text)
                
                st.write(f"Word Cloud untuk Sentimen {sentiment}")
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.error("File CSV harus memiliki kolom 'Text' dan 'Human'.")