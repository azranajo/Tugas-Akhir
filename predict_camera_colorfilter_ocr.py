from picamera2 import Picamera2
import time
import cv2
import numpy as np
import pytesseract

# Konfigurasi warna (HSV)
color_ranges = {
    'red': [(0, 50, 50), (10, 255, 255)],
    'green': [(35, 50, 50), (85, 255, 255)],
    'blue': [(100, 50, 50), (140, 255, 255)],
    'yellow': [(20, 50, 50), (40, 255, 255)],
    'orange': [(5, 50, 50), (15, 255, 255)],
    'purple': [(140, 50, 50), (160, 255, 255)]
}

# Inisialisasi Kamera
print("[INFO] Menginisialisasi kamera...")
picam2 = Picamera2()
picam2.preview_configuration.main.size = (128, 128)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

# Ambil gambar
print("[INFO] Mengambil gambar dari kamera...")
frame = picam2.capture_array()
image = cv2.resize(frame, (128, 128))

# Variabel untuk menyimpan hasil terbaik
best_score = 0
best_colors = []
best_thresh = None
best_digit = None

# Filter gambar berdasarkan beberapa warna
combined_mask = np.zeros_like(image[:, :, 0])  # Mask yang akan menggabungkan hasil filter warna

for color_name, (lower, upper) in color_ranges.items():
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    combined_mask = cv2.bitwise_or(combined_mask, mask)  # Gabungkan mask dari berbagai warna

# Terapkan mask gabungan pada gambar asli
filtered = cv2.bitwise_and(image, image, mask=combined_mask)

# Konversi gambar yang difilter ke grayscale
gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

# Terapkan threshold adaptif untuk mendapatkan gambar biner
thresh = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 3)

# Temukan kontur
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
score = len(contours)

# Tentukan hasil terbaik jika ditemukan kontur
if score > best_score:
    best_score = score
    best_thresh = thresh

    # Gunakan OCR untuk mengenali angka
    config = '--psm 10 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=config)
    best_digit = ''.join(filter(str.isdigit, text.strip()))

# Hasil
if best_digit:
    print(f"[HASIL] Angka '{best_digit}' terdeteksi setelah menggabungkan warna: {', '.join(best_colors)}")
else:
    print("[HASIL] Tidak berhasil membaca angka dari gambar.")

# Tampilkan hasil untuk verifikasi
if best_thresh is not None:
    cv2.imshow("Thresholded Image", best_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()