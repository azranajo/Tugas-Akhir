from picamera2 import Picamera2
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # Menggunakan TensorFlow Lite

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

# --- Memuat Model TFLite ---
print("[INFO] Memuat model TensorFlow Lite...")
interpreter = tflite.Interpreter(model_path="model_ishihara.tflite")
interpreter.allocate_tensors()

# --- Menyiapkan Input dan Output Tensors ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Menyiapkan Input untuk Inferensi ---
input_index = input_details[0]['index']
interpreter.set_tensor(input_index, image_input)

# --- Melakukan Prediksi ---
print("[INFO] Melakukan prediksi...")
interpreter.invoke()

# --- Mengambil Hasil Prediksi ---
output_index = output_details[0]['index']
prediction = interpreter.get_tensor(output_index)

# --- Hasil ---
predicted_class = np.argmax(prediction)
print(f"Hasil prediksi angka dari kartu Ishihara: {predicted_class}")

# --- (Opsional) Tampilkan Gambar ---
cv2.imshow("Kartu Ishihara", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()