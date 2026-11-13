import pandas as pd
import streamlit as st
import seaborn as sns
import joblib
# pastikan pip install streamlit-option-menu
from streamlit_option_menu import option_menu


@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()


@st.cache_data
def load_data():
    return pd.read_csv("strawberry_v1.csv")

@st.cache_data
def generate_pairplot(data):
    grid = sns.pairplot(data, hue="grade_kualitas", palette="Set1")
    return grid.figure

st.set_page_config(
    page_icon = ":strawberry:",
    page_title = "Klasifikasi Strawberry",
    # layout = "wide"
)


with st.sidebar:
    menu = option_menu(
        menu_title = "Menu Utama",
        options = ["Beranda", "Dataset", "Prediksi", "Informasi"],
        icons = ["house", "table", "cpu", "info-circle"], # icon dari Bootstrap
        menu_icon = "menu-up",
        default_index = 2, # langsung ke halaman Prediksi
    )

if menu == "Beranda":
    st.title("Klasifikasi Strawberry")
    st.markdown("Aplikasi berbasis Machine Learning `Random Forest Classifier` sederhana untuk memprediksi kualitas buah strawberry termasuk kategori `Premium, Standar, atau Afkir` berdasarkan 9 fitur pengujian.")

    st.space()

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Akurasi Model", "~91%", "Random Forest")
        col2.metric("Target Klasifikasi", "3 Grade", "Premium, Standar, Afkir")
        col3.metric("Total Parameter", "9 Fitur", "Numerik, Kategorikal, Ordinal")
    
if menu == "Dataset":
    st.title("Dataset")
    df = load_data()
    st.subheader("Dataset Strawberry")
    st.dataframe(df.head(10))
    
    st.space()
    
    st.subheader("Statistika Deskriptif")
    st.dataframe(df.describe())

    st.space()
    
    st.subheader("Hubungan Antar Fitur (Pairplot)")
    fig = generate_pairplot(df)
    st.pyplot(fig)
    
    
    
if menu == "Informasi":
    st.title("Informasi")
    st.markdown(
            """
            ### 🎯 Tujuan Proyek
            Aplikasi ini dibangun untuk menguji performa algoritma **Random Forest** dalam 
            memisahkan tingkatan kualitas produk pertanian berdasarkan pengukuran objektif.

            ### 🧠 Informasi Algoritma
            * **Model:** Random Forest Classifier (`n_estimators: 300`, `max_depth: 8`)
            * **Cross-Validation Score:** ~91.6%
            * **Test Accuracy:** ~90.9%

            ### 👋 Pengembang
            Aplikasi ini dikembangkan oleh **Adi Setiawan** dari SMKN 1 Purbalingga.
            """
        )
    
if menu == "Prediksi":
    
    st.title("Klasifikasi Strawberry")
    st.markdown("Aplikasi machine learning untuk memprediksi kualitas strawberry termasuk `Afkir, Standar, atau Premium` dengan Random Forest Classifier.")
    
    with st.container(border=True):
        berat = st.slider("Berat", 5.0, 50.0, 20.0)
        panjang = st.slider("Panjang", 1.0, 10.0, 3.5)
        diameter = st.slider("Diameter", 1.0, 6.0, 3.5)
        kadar_gula = st.slider("Kadar Gula", 2.0, 15.0, 8.5)
        kekerasan = st.slider("Kekerasan", 0.0, 5.0, 2.4)
        varietas = st.pills("Varietas", ["Sweet Charlie", "Albion", "California", "Rosalinda"], default="Sweet Charlie")
        asal_kota = st.pills("Asal Kota", ["Lembang", "Batu", "Ciwidey", "Bedugul"], default="Lembang")
        tingkat_kematangan = st.pills("Tingkat Kematangan", ["Mentah","Mengkal","Matang","Terlalu Matang"], default="Matang")
        warna_kulit = st.pills("Warna Kulit", ["Hijau Putih","Merah Muda","Merah","Merah Tua"], default="Merah")
    
    if st.button("Prediksi", type="primary", use_container_width=True):
        data = pd.DataFrame([[berat,panjang,diameter,kadar_gula,kekerasan,varietas,asal_kota,tingkat_kematangan,warna_kulit]], columns=["berat","panjang","diameter","kadar_gula","kekerasan","varietas","asal_kota","tingkat_kematangan","warna_kulit"])
    
        prediksi = model.predict(data)[0]
        presentase = max(model.predict_proba(data)[0])
    
    
        with st.container(border=True):
            if prediksi == "Premium":
                st.success(f"Buah terklasifikasi sebagai grade **{prediksi}**")
            elif prediksi == "Standar":
                st.warning(f"Buah terklasifikasi sebagai grade **{prediksi}**")
            else:
                st.error(f"Buah terklasifikasi sebagai grade **{prediksi}**")
                
            st.metric(
                label="Tingkat keyakinan model (Confidence Score)",
                value=f"{presentase*100:.2f}%"
            )
        st.balloons()
    
    
    
    st.divider()
    st.caption("Dibuat dengan :strawberry: oleh **Zivana Oktaviani (jiaveyy@gmail.com)**")