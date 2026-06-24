# 📱 Prediksi Kecanduan Media Sosial (Social Media Addiction Prediction)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble%20%26%20Stacking-success?style=for-the-badge)

Aplikasi berbasis **Artificial Intelligence (AI)** dan **Machine Learning (ML)** ini dirancang secara *end-to-end* untuk mendeteksi dan memprediksi tingkat kecanduan media sosial seseorang (**Rendah**, **Sedang**, atau **Tinggi**). Prediksi didasarkan pada berbagai indikator seperti usia, kebiasaan penggunaan gadget, pola tidur, produktivitas, dan perilaku digital secara umum.

Proyek ini menghadirkan *pipeline* kecerdasan buatan yang komprehensif, mulai dari preprocessing data otomatis, seleksi fitur (*Feature Selection*), hingga komparasi berbagai algoritma Machine Learning seperti Random Forest, Gradient Boosting, Support Vector Machine (SVM), dan Stacking Classifier untuk menemukan model yang paling optimal.

---

## 🤖 Highlight AI & Machine Learning

Proyek ini tidak hanya sebatas aplikasi biasa, tetapi mengimplementasikan alur kerja ML profesional:

- **Automated Feature Selection:** Menggunakan `SelectKBest` untuk mengidentifikasi variabel yang paling berpengaruh secara statistik terhadap kecanduan media sosial.
- **Ensemble & Stacking Modeling:** Mengevaluasi berbagai model klasik hingga modern (seperti *Random Forest*, *Gradient Boosting*, *AdaBoost*, *MLP/Neural Network*) dan menyatukannya dalam *Stacking Classifier* untuk akurasi maksimal.
- **Cross-Validation & Overfitting Penalty:** Menggunakan teknik validasi silang (Stratified K-Fold) serta sistem penalti metrik untuk memastikan model stabil dan tidak overfitting.
- **RESTful API Endpoint:** Menghidangkan model ML (melalui `joblib`) menggunakan **FastAPI** agar dapat diakses oleh berbagai jenis antarmuka (Web, Mobile, dll).

---

## 🔒 Privasi dan Keamanan Data

> [!IMPORTANT]
> Proyek ini dirancang dengan mengutamakan privasi. Kami **TIDAK** menyimpan kredensial sensitif seperti *password*, *email*, nama lengkap pengguna, atau alamat IP ke dalam repositori publik. Data yang digunakan (`data_gadget.csv`, `train.csv`, `test.csv`) murni berupa hasil kuesioner anonim yang telah melalui proses *cleaning* dan hanya berisi data numerik/kategorikal kebiasaan penggunaan gawai. Konfigurasi kredensial (jika diperlukan) disimpan secara lokal melalui `.env` yang tidak akan ter-upload ke Git.

---

## 📁 Struktur Folder Proyek

```plaintext
├── app/
│   ├── api.py                    # ⚡ Backend server FastAPI (Inference ML)
│   ├── streamlit_app.py          # 📊 Dashboard interaktif berbasis Streamlit
│   └── frontend/
│       ├── index.html            # 🌐 Antarmuka web utama (HTML)
│       ├── style.css             # 🎨 Desain gaya visual (CSS)
│       └── app.js                # ⚙️ Pengontrol interaksi frontend & fetch API
│
├── clean_data.py                 # 🧹 Skrip preprocessing dan encoding data
├── model.py                      # 🧠 Skrip pelatihan, evaluasi, & seleksi model ML
│
├── data_gadget.csv               # 📄 Data hasil pembersihan (.csv)
├── best_model.joblib             # 📦 Model Machine Learning terbaik (.joblib)
├── model_metadata.json           # 📈 Metadata model & metrik evaluasi
├── .env.example                  # ⚙️ Template konfigurasi (tanpa data sensitif)
└── requirements.txt              # 📚 Daftar dependensi library Python
```

---

## ⚡ Cara Instalasi dan Penggunaan

### 1. Clone & Masuk ke Folder Proyek
```bash
git clone <url-repository-anda>
cd social-media-addiction-prediction
```

### 2. Persiapkan File Konfigurasi (Opsional)
Salin `.env.example` menjadi `.env` jika ingin mengubah konfigurasi host/port.
*(Tidak ada informasi sensitif seperti password/IP yang akan ter-push ke GitHub karena `.env` sudah masuk dalam `.gitignore`)*.

### 3. Siapkan Virtual Environment & Aktifkan
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Instalasi Dependensi (Library)
```bash
pip install -r requirements.txt
```

### 5. Jalankan Preprocessing & Training ML Pipeline (Opsional)
Ingin melatih AI Anda sendiri dengan data baru?
```bash
# 1. Bersihkan dan encode data
python clean_data.py

# 2. Latih berbagai model ML dan cari yang terbaik secara otomatis
python model.py
```

### 6. 🚀 Jalankan Aplikasi!

Anda bisa memilih salah satu dari dua antarmuka interaktif:

#### Opsi A: Backend FastAPI + Frontend HTML Web (Rekomendasi)
1. **Jalankan API Server (Backend):**
   ```bash
   uvicorn app.api:app --reload
   ```
   *AI Backend akan siap menerima permintaan di `http://localhost:8000`.*
2. **Buka Frontend Web:**
   Buka file `app/frontend/index.html` secara langsung di browser, atau gunakan extension VS Code seperti *Live Server*.

#### Opsi B: Streamlit Dashboard Interaktif
Bagi Anda yang lebih suka visualisasi Data Science:
```bash
streamlit run app/streamlit_app.py
```
*Dashboard AI akan otomatis terbuka di browser pada alamat `http://localhost:8501`.*

---