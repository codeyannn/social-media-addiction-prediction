import streamlit as st
import joblib
import json
import pandas as pd
import os

st.set_page_config(page_title="Prediksi Kecanduan Medsos", page_icon="📱", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))

with open(os.path.join(BASE_DIR, "model_metadata.json"), "r") as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["class_names"]
CLASS_COLORS = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}

st.title("📱 Prediksi Tingkat Kecanduan Media Sosial")
st.markdown(f"**Model:** {metadata['model_name']} + {metadata['scaler_name']} | "
            f"**Akurasi:** {metadata['test_accuracy']*100:.1f}% | "
            f"**F1-Score:** {metadata['test_f1_macro']:.4f}")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Data Pribadi")
    usia = st.number_input("Usia", 10, 60, 20)
    status = st.selectbox("Status", ["Mahasiswa", "Pelajar", "Pekerja"])

with col2:
    st.subheader("📊 Penggunaan Gadget")
    screen_time = st.selectbox("Screen Time / Hari", [1.0, 3.0, 5.0, 7.0, 10.0],
                               format_func=lambda x: {1.0: "< 2 jam", 3.0: "2-4 jam", 5.0: "4-6 jam", 7.0: "6-8 jam", 10.0: "> 8 jam"}[x])
    hp_freq = st.selectbox("Frekuensi Buka HP / Hari", [10.0, 35.0, 75.0, 120.0],
                           format_func=lambda x: {10.0: "< 20x", 35.0: "20-50x", 75.0: "50-100x", 120.0: "> 100x"}[x])
    app_count = st.selectbox("Jumlah Aplikasi Medsos", [1.5, 3.5, 5.5, 8.5],
                             format_func=lambda x: {1.5: "1-2", 3.5: "3-4", 5.5: "5-6", 8.5: "> 7"}[x])

with col3:
    st.subheader("😴 Pola Tidur")
    sleep_dur = st.selectbox("Durasi Tidur / Hari", [2.0, 4.0, 6.0, 9.0],
                             format_func=lambda x: {2.0: "< 3 jam", 4.0: "3-5 jam", 6.0: "5-7 jam", 9.0: "> 8 jam"}[x])
    sleep_time = st.slider("Jam Tidur (desimal)", 0.0, 24.0, 23.0, 0.5)
    sleep_late = st.selectbox("Tidur Larut Karena Medsos", [0, 1, 2, 3],
                              format_func=lambda x: ["Tidak pernah", "Jarang", "Sering", "Sering sekali"][x])

st.divider()
col4, col5 = st.columns(2)

with col4:
    st.subheader("📚 Fokus & Produktivitas")
    focus_hours = st.number_input("Jam Belajar/Kerja Fokus / Hari", 0.0, 12.0, 4.0, 0.5)
    difficulty_focus = st.selectbox("Sulit Fokus Tanpa Cek HP", [0, 1, 2, 3, 4],
                                   format_func=lambda x: ["Sangat mudah", "Mudah", "Sedang", "Sulit", "Sulit sekali"][x])
    open_sosmed = st.selectbox("Sering Buka Medsos Saat Belajar/Kerja", [0, 1, 2, 3],
                               format_func=lambda x: ["Tidak pernah", "Jarang", "Sering", "Sering sekali"][x])

with col5:
    st.subheader("🧠 Perilaku Digital")
    cemas = st.selectbox("Cemas Jika Jauh dari HP", ["Tidak", "Iya"])
    no_purpose = st.selectbox("Buka Medsos Tanpa Tujuan", ["Tidak", "Iya"])
    disruptive = st.selectbox("Medsos Mengganggu Belajar/Kerja", ["Tidak", "Iya"])
    try_reduce = st.selectbox("Pernah Coba Kurangi Tapi Gagal", ["Tidak", "Iya"])

st.divider()
st.subheader("📲 Media Sosial yang Digunakan")
sosmed_cols = st.columns(7)
platforms = ["Instagram", "Tiktok", "Youtube", "X/Twitter", "Facebook", "Threads", "WhatsApp"]
sosmed_vals = []
for i, platform in enumerate(platforms):
    with sosmed_cols[i]:
        sosmed_vals.append(1 if st.checkbox(platform, value=(i < 3)) else 0)

st.divider()

status_enc = [1 if status == "Mahasiswa" else 0, 1 if status == "Pelajar" else 0, 1 if status == "Pekerja" else 0]
binary_map = {"Iya": 1, "Tidak": 0}

if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
    features = pd.DataFrame([[
        usia, screen_time, hp_freq, app_count, sleep_dur, sleep_time,
        sleep_late, focus_hours, difficulty_focus, open_sosmed,
        binary_map[cemas], binary_map[no_purpose], binary_map[disruptive], binary_map[try_reduce],
        *status_enc, *sosmed_vals
    ]], columns=metadata["feature_names"])

    pred = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    class_name = CLASS_NAMES[str(pred)]
    emoji = CLASS_COLORS[class_name]

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Hasil Prediksi", f"{emoji} {class_name}")
        st.caption(f"Confidence: {proba[pred]*100:.1f}%")
    with r2:
        st.subheader("Probabilitas per Kelas")
        chart_data = {f"{CLASS_COLORS[CLASS_NAMES[str(i)]]} {CLASS_NAMES[str(i)]}": float(p) for i, p in enumerate(proba)}
        st.bar_chart(chart_data)
