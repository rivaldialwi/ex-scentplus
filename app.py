import pandas as pd
import streamlit as st
import joblib
import nltk
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
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
    # Membersihkan teks input
    cleaned_text = clean_text(input_text)
    # Mengubah teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    # Melakukan prediksi menggunakan model
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk menampilkan WordCloud
def generate_wordcloud(data, sentiment):
    text = ' '.join(data[data['Human'] == sentiment]['Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud untuk Sentimen {sentiment}')
    plt.axis('off')
    st.pyplot(plt)

# Fungsi untuk analisis sentimen dan menampilkan grafik
def analyze_sentiments(df):
    sentiment_counts = df['Human'].value_counts()
    st.bar_chart(sentiment_counts)
    
    st.write("Jumlah Sentimen Positif:", sentiment_counts.get('Positif', 0))
    st.write("Jumlah Sentimen Netral:", sentiment_counts.get('Netral', 0))
    st.write("Jumlah Sentimen Negatif:", sentiment_counts.get('Negatif', 0))

    # Menampilkan WordCloud untuk setiap sentimen
    for sentiment in df['Human'].unique():
        generate_wordcloud(df, sentiment)

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# File uploader for Excel files
uploaded_file = st.file_uploader("Unggah file Excel atau CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Check file type
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    if 'Text' in df.columns:
        # Clean and classify the text data
        df['Cleaned_Text'] = df['Text'].apply(clean_text)
        df['Human'] = df['Cleaned_Text'].apply(classify_text)

        # Show the dataframe with predictions
        st.write(df)

        # Perform sentiment analysis
        analyze_sentiments(df)

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
            data=convert_df_to_csv(df),
            file_name="prediksi_sentimen.csv",
            mime="text/csv"
        )

        # Evaluasi model
        st.header("Evaluasi Model")
        if 'Cleaned_Text' in df.columns and 'Human' in df.columns:
            X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Text'], df['Human'], test_size=0.2, random_state=42)
            X_train_tfidf = tfidf_vectorizer.transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            y_pred = logreg_model.predict(X_test_tfidf)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)

            st.write(f"Akurasi model regresi logistik multinomial: {accuracy:.2f}")
            st.write(f"Presisi model regresi logistik multinomial: {precision:.2f}")
            st.write(f"Recall model regresi logistik multinomial: {recall:.2f}")
        else:
            st.error("Kolom 'Cleaned_Text' atau 'Human' tidak ditemukan dalam DataFrame.")
    else:
        st.error("File harus memiliki kolom 'Text'.")