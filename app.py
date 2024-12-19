import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import load

# Load models
rf_modelStock = load('rf_modelStock.joblib')
rf_modelPrice = load('rf_modelPrice.joblib')
with open('xgb_modelPopu.pkl', 'rb') as f:
    xgb_modelPopu = pickle.load(f)

# Define categories
categories = [
    'Elektronik', 'Lainnya', 'Perawatan Pribadi', 'Olahraga & Outdoor',
    'Peralatan Rumah Tangga', 'Pakaian & Fashion', 'Kendaraan & Aksesori',
    'Makanan & Minuman', 'Perhiasan & Aksesori', 'Mainan & Anak-Anak',
    'Gadget & Elektronik Musik'
]

# Define prediction function
def prediction(df):
    df['avg_harga_per_kategori'] = df.groupby('kategori')['harga'].transform('mean')
    df['harga_per_rating'] = df['harga'] / (df['total_rating'] + 1)
    df['harga_terjual'] = df['harga'] * df['terjual']
    df['rasio_penjualan_stok'] = df['terjual'] / (df['stock'] + 1)
    df['stok_terjual_ratio'] = df['stock'] / (df['terjual'] + 1)
    df['stok_ideal'] = np.ceil(df['stok_terjual_ratio'] * df['stock'])

    df['harga_kategori_encoding'] = rf_modelPrice.predict(
        df[['avg_harga_per_kategori', 'harga_per_rating', 'harga_terjual']]
    )
    df['restock_encoding'] = rf_modelStock.predict(
        df[['stok_ideal', 'stok_terjual_ratio', 'rasio_penjualan_stok']]
    )
    df['popularitas_encoding'] = xgb_modelPopu.predict(
        df[['harga_per_rating', 'rasio_penjualan_stok', 'total_rating']]
    )

    harga_kategori_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    restock_mapping = {1: 'Tidak Restock', 2: 'Restock', 0: 'Stok Berlebih'}
    popularitas_mapping = {0: 'Tidak Populer', 1: 'Populer', 2: 'Sangat Populer'}

    df['harga_kategori'] = df['harga_kategori_encoding'].map(harga_kategori_mapping)
    df['restock'] = df['restock_encoding'].map(restock_mapping)
    df['popularitas'] = df['popularitas_encoding'].map(popularitas_mapping)

    def rekomendasi(row):
        if row['restock'] == 'Restock':
            return "Segera lakukan restock produk ini."
        elif row['harga_kategori'] == 'Tinggi' and row['popularitas'] == 'Sangat Populer':
            return "Lakukan promosi pada produk populer ini."
        elif row['restock'] == 'Tidak Restock' and row['harga_kategori'] == 'Rendah':
            return "Evaluasi produk untuk diskon atau hapus dari katalog."
        elif row['popularitas'] == 'Tidak Populer' and row['restock'] == 'Stok Berlebih':
            return "Tunda restock produk ini dan evaluasi penjualannya."
        elif row['popularitas'] == 'Populer' and row['harga_kategori'] == 'Sedang':
            return "Pertahankan produk dengan harga dan popularitas saat ini."
        else:
            return "Pertahankan strategi saat ini."

    df['Rekomendasi'] = df.apply(rekomendasi, axis=1)
    return df[['nama_produk', 'harga_kategori', 'restock', 'popularitas', 'Rekomendasi']]

# Define Streamlit app
def main():
    st.title("Sistem Rekomendasi Produk")
    st.markdown("""
    <div style="background-color:yellow;padding:13px">
    <h1 style="color:black; text-align:center;">Aplikasi Analisis dan Rekomendasi Produk</h1>
    </div>
    """, unsafe_allow_html=True)

    if 'hasil' not in st.session_state:
        st.session_state['hasil'] = None

    with st.form(key='my_form'):
        nama_produk = st.text_input("Nama Produk", key="nama_produk_key")
        kategori = st.selectbox("Pilih Kategori", categories, key="kategori_key")
        harga = st.number_input("Harga", min_value=0, step=1000, key="harga_key")
        total_rating = st.number_input("Total Rating", min_value=0, step=1, key="rating_key")
        min_terjual = total_rating if total_rating > 0 else 0
        terjual = st.number_input("Jumlah Terjual", min_value=min_terjual, step=1, key="terjual_key")
        stock = st.number_input("Stock", min_value=0, step=1, key="stock_key")
        submit_button = st.form_submit_button(label="Proses Rekomendasi")

    if submit_button:
        kategori_index = categories.index(kategori) + 1
        data_input = {
            'nama_produk': [nama_produk],
            'kategori': [kategori_index],
            'harga': [harga],
            'total_rating': [total_rating],
            'terjual': [terjual],
            'stock': [stock]
        }
        df = pd.DataFrame(data_input)
        st.session_state['hasil'] = prediction(df)

    if st.session_state['hasil'] is not None:
        st.success("Berikut adalah hasil analisis produk:")
        st.dataframe(st.session_state['hasil'])

if __name__ == '__main__':
    main()
