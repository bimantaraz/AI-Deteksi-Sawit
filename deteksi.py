from ultralytics import YOLO
import os

# --- Konfigurasi ---

MODEL_PATH = r'models/sawitAI.pt' # Path ke model 'sawitAI.pt'
VIDEO_PATH = 'sawit.mp4' # Path ke video yang ingin diuji
CONF_THRESHOLD = 0.3

if not os.path.exists(MODEL_PATH):
    print(f"Error: File model tidak ditemukan di {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Error: File video tidak ditemukan di {VIDEO_PATH}")
    print(f"Pastikan video '{VIDEO_PATH}' ada di folder yang sama dengan skrip ini.")
    exit()

print(f"Memuat model dari: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"Memulai deteksi pada: {VIDEO_PATH}")

try:
    results = model.predict(
        source=VIDEO_PATH,
        save=True,
        conf=CONF_THRESHOLD,
        name='hasil_deteksi_sawit'
    )

except Exception as e:
    print(f"Terjadi error saat prediksi: {e}")