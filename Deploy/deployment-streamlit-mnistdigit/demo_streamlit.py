import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

#Load Model Kembali
model = joblib.load("mymnist.pkl")
#Tampilkan header dan keterangan
st.title("Digit Recognition Demo")
st.write("Upload gambar angka 0–9, lalu model akan memprediksi angkanya.")
#Tampilkan input file uploade
file = st.file_uploader("Upload Gambar", type=["png", "jpg", "jpeg"])

#Jika file tidak kosong maka proses
if file:
    #Load file hasil upload menggunakan Library PIL, lalu convert jadi grayscale (L)
    img = Image.open(file).convert("L")
    #Data dari mnist digit tulisan putih background hitam,
    #sedangkan hasil upload tulisan hitam background putih, maka perlu diinvert
    img = ImageOps.invert(img)
    #resize jadi 8x8 supaya sama dengan data training yaitu 64
    img = img.resize((8, 8))
    #Ubah jadi numpy array
    arr = np.array(img)
    #Ubah bentuk aray 
    X = arr.reshape(1, 64)
    #Tampilkan gambar
    st.image(img, caption="Input 8x8", width=64)

    #Handle tombol predict
    if st.button("Predict"):
        #Tampilkan probabilitas setiap kelas
        prob = model.predict_proba(X)[0]
        st.bar_chart(prob)
        #Tampilkan hasil prediksi
        pred = model.predict(X)[0]
        st.success(f"Prediksi: {pred}")


