
# =====================================
#  PREDIKSI ANGKA DARI KAMERA (LIVE)
# =====================================

from picamera2 import Picamera2
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Inisialisasi Kamera ---
print("[INFO] Menginisialisasi kamera...")
picam2 = Picamera2()
picam2.preview_configuration.main.size = (128, 128)  # Resolusi input model
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)  # Waktu buffer kamera

# --- Ambil Gambar ---
print("[INFO] Mengambil gambar dari kamera...")
frame = picam2.capture_array()

# --- Pra-pemrosesan ---
print("[INFO] Memproses gambar...")
image = cv2.resize(frame, (128, 128))
image = image.astype("float32") / 255.0
image_input = np.expand_dims(image, axis=0)  # Format: (1, 128, 128, 3)

# --- Load Model ---
print("[INFO] Memuat model CNN...")
model = load_model("model_ishihara.h5")

# --- Prediksi ---
print("[INFO] Melakukan prediksi...")
prediction = model.predict(image_input)
predicted_class = np.argmax(prediction)

# --- Hasil ---
print(f"Hasil prediksi angka dari kartu Ishihara: {predicted_class}")

# --- (Opsional) Tampilkan Gambar ---
cv2.imshow("Kartu Ishihara", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
