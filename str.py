import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
from datetime import datetime, timedelta

# Konfigurasi halaman
st.set_page_config(page_title="📊 Prediksi Popularitas TikTok", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('data/tiktok_scrapper.csv')

def preprocess_data(df):
    df.dropna(subset=['text', 'authorMeta.name', 'musicMeta.musicName'], inplace=True)

    le_name = LabelEncoder()
    df['authorMeta.name_encoded'] = le_name.fit_transform(df['authorMeta.name'])

    le_music = LabelEncoder()
    df['musicMeta.musicName_encoded'] = le_music.fit_transform(df['musicMeta.musicName'])

    df['text_length'] = df['text'].astype(str).apply(len)
    df['hashtags_str'] = df['text'].apply(lambda x: ' '.join(re.findall(r"#\w+", str(x))))

    tfidf = TfidfVectorizer(max_features=100)
    hashtag_tfidf = tfidf.fit_transform(df['hashtags_str'])

    df['createTimeISO'] = pd.to_datetime(df['createTimeISO'])
    df['createTimeISO'] = (df['createTimeISO'] - df['createTimeISO'].min()).dt.total_seconds()

    df['is_popular'] = df[['diggCount', 'commentCount', 'shareCount', 'playCount']].sum(axis=1).apply(lambda x: 1 if x > 10000 else 0)

    numerical_features = df[['videoMeta.duration', 'text_length', 'authorMeta.name_encoded', 'musicMeta.musicName_encoded', 'createTimeISO']]
    X = hstack([numerical_features, hashtag_tfidf])
    y = df['is_popular']

    return X, y, le_name, le_music, tfidf, df

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_df = pd.DataFrame(X_train.toarray())
    X_train_df['is_popular'] = y_train.values

    df_majority = X_train_df[X_train_df['is_popular'] == 0]
    df_minority = X_train_df[X_train_df['is_popular'] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

    df_train_balanced = pd.concat([df_majority, df_minority_upsampled])
    X_train_balanced = df_train_balanced.drop('is_popular', axis=1).values
    y_train_balanced = df_train_balanced['is_popular'].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_balanced, y_train_balanced)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.round(2)
    st.subheader("Classification Report")
    st.dataframe(report_df)

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor='#000000')
    sns.heatmap(cm, annot=True, fmt='d', cmap='inferno',
                xticklabels=['Tidak Populer', 'Populer'],
                yticklabels=['Tidak Populer', 'Populer'], ax=ax_cm)
    ax_cm.set_facecolor('#000000')
    ax_cm.set_title('Confusion Matrix', color='white')
    ax_cm.tick_params(colors='white')
    for spine in ax_cm.spines.values(): spine.set_color('white')
    st.pyplot(fig_cm)

    return rf

def predict_single(model, le_name, le_music, tfidf, durasi, text, nama_author, nama_musik, waktu_posting, df):
    text = str(text) if not pd.isna(text) else ""
    text_length = len(text)
    waktu = (pd.to_datetime(waktu_posting).tz_localize(None) - pd.Timestamp('2020-01-01')).total_seconds()
    encoded_author = le_name.transform([nama_author])[0] if nama_author in le_name.classes_ else 0
    encoded_music = le_music.transform([nama_musik])[0] if nama_musik in le_music.classes_ else 0
    hashtags_str = ' '.join(re.findall(r"#\w+", str(text)))
    hashtag_tfidf = tfidf.transform([hashtags_str])
    numerical_features = np.array([[durasi, text_length, encoded_author, encoded_music, waktu]])
    data = hstack([numerical_features, hashtag_tfidf])
    return model.predict(data)[0]

def predict_bulk(model, le_name, le_music, tfidf, df):
    df['predicted_popularity'] = df.apply(lambda row: predict_single(model, le_name, le_music, tfidf, row['videoMeta.duration'], row['text'], row['authorMeta.name'], row['musicMeta.musicName'], row['createTimeISO'], df), axis=1)
    return df

def main():
    st.markdown("""
        <style>
            .stApp { background-color: #000000; color: #FFFFFF; }
            .stApp, .stApp * { color: #FFFFFF !important; }
            .stMetricValue, .stMetricDelta { color: #39FF14 !important; font-weight: bold; }
            .sidebar .stButton > button {
                width: 100%; height: 48px; margin: 4px 0;
                background: linear-gradient(45deg, #FF0066, #6600FF);
                color: #FFFFFF; border: none; border-radius: 6px;
                font-size: 16px; font-weight: 600;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6);
            }
            .sidebar .stButton > button:hover { opacity: 0.85; }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("📊 Dashboard Sistem")
    if st.sidebar.button("📈 EDA dan Visualisasi Data"): st.session_state.section = 'EDA'
    if st.sidebar.button("🧠 Model Evaluasi Konten"): st.session_state.section = 'Model'
    if st.sidebar.button("📁 Informasi Data TikTok"): st.session_state.section = 'Data'
    if st.sidebar.button("🎯 Popularitas Konten TikTok"): st.session_state.section = 'Prediksi'
    st.sidebar.markdown("---")
    st.sidebar.write("🎬 TikTok Popularity Dashboard")

    if 'section' not in st.session_state:
        st.session_state.section = 'EDA'

    df = load_data()
    X, y, le_name, le_music, tfidf, df_clean = preprocess_data(df)

    if st.session_state.section == 'EDA':
        st.header("1. Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        colors = ['#39FF14', '#FF073A', '#05FFA1', '#FFD300']
        metrics = ['diggCount', 'shareCount', 'playCount', 'commentCount']
        titles = ['Like', 'Share', 'Play', 'Komentar']
        for metric, title, color, col in zip(metrics, titles, colors, [col1, col1, col2, col2]):
            fig = px.histogram(df, x=metric, nbins=20, title=title, color_discrete_sequence=[color])
            fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
            col.plotly_chart(fig, use_container_width=True)

        st.subheader("Korelasi Interaksi")
        fig_corr = plt.figure(figsize=(6, 4), facecolor='#000000')
        ax = sns.heatmap(df[metrics].corr(), annot=True, cmap='magma')
        ax.set_facecolor('#000000'); ax.tick_params(colors='white'); ax.set_title('Korelasi antar Interaksi', color='white')
        st.pyplot(fig_corr)

        st.subheader("Visualisasi Hubungan Antar Fitur")
        scatter_cols = st.columns(3)
        pairs = [('shareCount', 'diggCount'), ('playCount', 'diggCount'), ('commentCount', 'shareCount')]
        scatter_titles = ['Share vs Like', 'Play vs Like', 'Komentar vs Share']
        for (x, y_), title, color, col in zip(pairs, scatter_titles, colors[:3], scatter_cols):
            fig_sc = px.scatter(df, x=x, y=y_, title=title, color_discrete_sequence=[color])
            fig_sc.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
            col.plotly_chart(fig_sc, use_container_width=True)

    elif st.session_state.section == 'Model':
        st.header("2. Training & Evaluasi Model")
        model = train_and_evaluate(X, y)
        st.session_state.model = model
        st.session_state.le_name = le_name
        st.session_state.le_music = le_music
        st.session_state.tfidf = tfidf

    elif st.session_state.section == 'Data':
        st.header("3. Tinjau Dataset")
        st.dataframe(df_clean.head(10))
        with st.expander("📌 Statistik Deskriptif"):
            st.dataframe(df_clean.describe())

    elif st.session_state.section == 'Prediksi':
        st.header("4. Klasifikasi Popularitas Konten TikTok")

        if 'model' not in st.session_state:
            st.warning("⚠️ Model belum dilatih. Silakan jalankan terlebih dahulu bagian '🧠 Model Evaluasi Konten'.")
            return
        model = st.session_state.model
        le_name = st.session_state.le_name
        le_music = st.session_state.le_music
        tfidf = st.session_state.tfidf

        tab1, tab2 = st.tabs(["🔮 Klasifikasi Popularitas Satu Konten", "📅 Klasifikasi Popularitas Banyak Konten"])

        with tab1:
            text = st.text_input("Deskripsi Konten")
            nama_author = st.text_input("Nama Creator")

            top_music = df_clean['musicMeta.musicName'].value_counts().head(20).index.tolist()
            nama_musik = st.selectbox("Pilih Musik", top_music)

            durasi = st.slider("Durasi video (detik)", 0, 60)
            waktu_tanggal = st.date_input("Tanggal posting")
            waktu_jam = st.time_input("Jam posting", datetime.strptime("12:00", "%H:%M").time(), key="waktu_posting")
            waktu_posting = datetime.combine(waktu_tanggal, waktu_jam)

            if st.button("📈 Cek Popularitas Konten"):
                prediction = predict_single(model, le_name, le_music, tfidf, durasi, text, nama_author, nama_musik, waktu_posting, df_clean)
                status = "Populer" if prediction == 1 else "Tidak Populer"
                st.markdown(f"Konten **{status}**.")

        with tab2:
            st.subheader("📁 Prediksi Massal (Upload / Input Manual)")
            tab_upload, tab_manual = st.tabs(["🗂 Upload CSV", "✍️ Input Manual"])

            with tab_upload:
                uploaded_file = st.file_uploader("Pilih File CSV", type=["csv"])
                if uploaded_file is not None:
                    df_bulk = pd.read_csv(uploaded_file)
                    df_bulk = df_bulk[['text', 'authorMeta.name', 'musicMeta.musicName', 'videoMeta.duration', 'createTimeISO']]
                    st.dataframe(df_bulk.head())

                    if st.button("📊 Cek Popularitas dari File"):
                        predicted_df = predict_bulk(model, le_name, le_music, tfidf, df_bulk)
                        st.subheader("Hasil Popularitas konten")
                        st.dataframe(predicted_df[['text', 'authorMeta.name', 'musicMeta.musicName', 'videoMeta.duration', 'createTimeISO', 'predicted_popularity']].head())

            with tab_manual:
                st.markdown("Masukkan beberapa konten secara manual:")

                if "manual_inputs" not in st.session_state:
                    st.session_state.manual_inputs = []

                with st.form("manual_input_form", clear_on_submit=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        text = st.text_input("Deskripsi Konten")
                        author = st.text_input("Nama Creator")
                        music = st.text_input("Nama Musik")
                    with col2:
                        duration = st.number_input("Durasi Video (detik)", min_value=0, max_value=300, step=1)
                        tanggal = st.date_input("Tanggal Posting")
                        jam = st.time_input("Jam Posting", value=datetime.now().time())

                    submitted = st.form_submit_button("➕ Tambah Konten")
                    if submitted:
                        posting_time = datetime.combine(tanggal, jam).isoformat()
                        st.session_state.manual_inputs.append({
                            "text": text,
                            "authorMeta.name": author,
                            "musicMeta.musicName": music,
                            "videoMeta.duration": duration,
                            "createTimeISO": posting_time
                        })

                if st.session_state.manual_inputs:
                    df_manual = pd.DataFrame(st.session_state.manual_inputs)
                    st.write("📄 Cek Popularitas Konten:")
                    st.dataframe(df_manual)

                    if st.button("🚀 Lihat Popularitas Semua Konten Manual"):
                        predicted_df = predict_bulk(model, le_name, le_music, tfidf, df_manual)
                        st.success("✅ Prediksi selesai!")
                        st.dataframe(predicted_df[['text', 'predicted_popularity']])


if __name__ == "__main__":
    main()
