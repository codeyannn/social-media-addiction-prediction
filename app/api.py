from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))

with open(os.path.join(BASE_DIR, "model_metadata.json"), "r") as f:
    metadata = json.load(f)

app = FastAPI(title="Prediksi Kecanduan Media Sosial API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = metadata["class_names"]


class PredictionInput(BaseModel):
    Usia: float
    Rata_rata_screen_time_perhari: float
    Frekuensi_membuka_HP_perhari: float
    Jumlah_aplikasi_media_sosial_aktif_yang_digunakan: float
    Durasi_waktu_tidur_perhari: float
    Jam_tidur_hours: float
    Tidur_larut_karena_sosial_media: int
    Jam_waktu_belajar_bekerja_fokus_sehari: float
    Sulit_fokus_tanpa_mengecek_HP: int
    Sering_membuka_media_sosial_saat_belajar_bekerja: int
    Merasa_cemas_jika_jauh_dari_HP: int
    Sering_membuka_sosial_media_tanpa_tujuan_jelas: int
    Penggunaan_media_sosial_mengganggu_belajar_kerja: int
    Pernah_mencoba_mengurangi_penggunaan_media_sosial_namun_gagal: int
    Status_Mahasiswa: int
    Status_Pelajar: int
    Status_Pekerja: int
    Sosmed_Instagram: int
    Sosmed_Tiktok: int
    Sosmed_Youtube: int
    Sosmed_X_twitter: int
    Sosmed_Facebook: int
    Sosmed_Threads: int
    Sosmed_WhatsApp: int


@app.get("/")
def home():
    return {"message": "Prediksi Kecanduan Media Sosial API", "status": "active"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": metadata["model_name"],
        "scaler_name": metadata["scaler_name"],
        "feature_selection": metadata.get("feature_selection"),
        "k_features": metadata.get("k_features"),
        "selected_feature_names": metadata.get("selected_feature_names", []),
        "test_accuracy": metadata["test_accuracy"],
        "test_f1_macro": metadata["test_f1_macro"],
        "train_f1_macro": metadata.get("train_f1_macro", metadata.get("train_accuracy")),
        "cv_mean_f1": metadata["cv_mean_f1"],
        "cv_std_f1": metadata.get("cv_std_f1"),
        "gap_f1": metadata.get("gap_f1"),
        "class_names": CLASS_NAMES,
    }


@app.get("/features")
def features():
    return {
        "feature_names": metadata["feature_names"],
        "feature_info": metadata["feature_info"],
        "class_names": CLASS_NAMES,
    }


import pandas as pd

@app.post("/predict")
def predict(data: PredictionInput):
    X = pd.DataFrame([[
        data.Usia, data.Rata_rata_screen_time_perhari,
        data.Frekuensi_membuka_HP_perhari,
        data.Jumlah_aplikasi_media_sosial_aktif_yang_digunakan,
        data.Durasi_waktu_tidur_perhari, data.Jam_tidur_hours,
        data.Tidur_larut_karena_sosial_media,
        data.Jam_waktu_belajar_bekerja_fokus_sehari,
        data.Sulit_fokus_tanpa_mengecek_HP,
        data.Sering_membuka_media_sosial_saat_belajar_bekerja,
        data.Merasa_cemas_jika_jauh_dari_HP,
        data.Sering_membuka_sosial_media_tanpa_tujuan_jelas,
        data.Penggunaan_media_sosial_mengganggu_belajar_kerja,
        data.Pernah_mencoba_mengurangi_penggunaan_media_sosial_namun_gagal,
        data.Status_Mahasiswa, data.Status_Pelajar, data.Status_Pekerja,
        data.Sosmed_Instagram, data.Sosmed_Tiktok, data.Sosmed_Youtube,
        data.Sosmed_X_twitter, data.Sosmed_Facebook,
        data.Sosmed_Threads, data.Sosmed_WhatsApp,
    ]], columns=metadata["feature_names"])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()

    return {
        "prediction": pred,
        "class_name": CLASS_NAMES[str(pred)],
        "probabilities": {CLASS_NAMES[str(i)]: round(p, 4) for i, p in enumerate(proba)},
    }
