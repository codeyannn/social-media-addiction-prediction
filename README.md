# 📱 Prediksi Kecanduan Media Sosial (Social Media Addiction Prediction)

Proyek ini adalah aplikasi Machine Learning end-to-end berbasis Python yang digunakan untuk memprediksi tingkat kecanduan media sosial seseorang (**Rendah**, **Sedang**, atau **Tinggi**) berdasarkan usia, kebiasaan penggunaan gadget, pola tidur, produktivitas, serta perilaku digital.

Proyek ini menyediakan pipeline lengkap dari preprocessing data, training model ensemble/stacking, backend API menggunakan **FastAPI**, serta dua pilihan antarmuka pengguna: **HTML Web Interface** dan **Streamlit Dashboard**.

---

## 📁 Struktur Folder Proyek

```plaintext
├── app/
│   ├── api.py                    # Backend server FastAPI
│   ├── streamlit_app.py          # Dashboard interaktif berbasis Streamlit
│   └── frontend/
│       ├── index.html            # Antarmuka web utama (HTML)
│       ├── style.css             # Desain gaya visual (CSS)
│       └── app.js                # Pengontrol interaksi frontend & fetch API
│
├── clean_data.py                 # Skrip preprocessing dan encoding data
├── model.py                      # Skrip pelatihan, evaluasi, dan pemilihan model ML
│
├── Pendataan Seberapa Sering...  # File data respon mentah (.csv)
├── data_gadget.csv               # Data hasil pembersihan (.csv)
├── encoded_data.csv              # Data hasil encoding (.csv)
├── train.csv                     # Data latih (.csv)
├── test.csv                      # Data uji (.csv)
├── best_model.joblib             # Model terbaik yang disimpan (.joblib)
├── model_metadata.json           # Metadata model dan nama fitur terpilih
├── .env.example                  # Template konfigurasi environment variables
├── requirements.txt              # Daftar dependensi library Python
└── README.md                     # Dokumentasi proyek
```

---

## 📋 File yang Perlu Disiapkan (Tidak Terbawa/Diabaikan Git)

Saat pertama kali melakukan `git clone`, ada beberapa file yang **tidak terbawa** (karena terdaftar di `.gitignore` atau merupakan berkas lingkungan lokal Anda). Anda perlu menyiapkan file-file berikut sebelum menjalankan proyek:

### 1. File Konfigurasi `.env`
File ini digunakan untuk menyimpan pengaturan port dan host aplikasi.
* **Cara menyiapkan:** Salin file `.env.example` dan ubah namanya menjadi `.env` di folder root proyek Anda.
* **Isi default `.env`:**
  ```env
  API_HOST=0.0.0.0
  API_PORT=8000
  DEBUG_MODE=True
  STREAMLIT_PORT=8501
  STREAMLIT_SERVER_ADDRESS=localhost
  ```

### 2. Virtual Environment (`venv/` atau `.venv/`)
Folder ini menyimpan semua pustaka python yang diinstal secara lokal agar tidak bentrok dengan pustaka sistem Anda.
* **Cara membuat:** Jalankan perintah `python -m venv venv` di terminal Anda.

---

## ⚡ Langkah-Langkah Menjalankan Proyek

### 1. Clone & Masuk ke Folder Proyek
```bash
git clone <url-repository-anda>
cd social-media-addiction-prediction
```

### 2. Siapkan Virtual Environment & Aktifkan
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

### 3. Instalasi Dependensi (Library)
Instal semua pustaka Python yang dibutuhkan oleh proyek menggunakan file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Preprocessing & Training Model (Opsional)
Jika Anda memiliki data kuesioner baru di file `Pendataan... - Cleaning.csv` dan ingin melatih ulang model:
```bash
# 1. Bersihkan dan encode data baru
python clean_data.py

# 2. Latih model dan simpan best_model.joblib
python model.py
```

### 5. Jalankan Antarmuka Aplikasi

Anda bisa memilih salah satu dari dua antarmuka berikut untuk melakukan prediksi:

#### Opsi A: Backend FastAPI + Frontend HTML Web (Rekomendasi)
1. **Jalankan API Server (Backend):**
   ```bash
   uvicorn app.api:app --reload
   ```
   *API backend akan berjalan pada `http://localhost:8000`.*
2. **Buka Frontend Web:**
   Buka file `app/frontend/index.html` secara langsung di browser Anda, atau gunakan extension VS Code seperti *Live Server*.

#### Opsi B: Streamlit Dashboard
Jalankan perintah berikut untuk menjalankan dashboard visual Streamlit:
```bash
streamlit run app/streamlit_app.py
```
*Aplikasi akan otomatis terbuka di browser Anda pada alamat `http://localhost:8501`.*
