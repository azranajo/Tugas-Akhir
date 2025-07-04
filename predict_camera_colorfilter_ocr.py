
# ============================================================================
#  PREDIKSI ANGKA DARI KAMERA MENGGUNAKAN COLOR FILTERING + OCR (TESSERACT)
# ============================================================================

from picamera2 import Picamera2
import time
import cv2
import numpy as np
import pytesseract

# Konfigurasi warna (HSV)
color_ranges = {
    'red': [(0, 50, 50), (10, 255, 255)],
    'green': [(35, 50, 50), (85, 255, 255)],
    'blue': [(100, 50, 50), (140, 255, 255)]
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

best_score = 0
best_color = None
best_thresh = None
best_digit = None

for color_name, (lower, upper) in color_ranges.items():
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    filtered = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

    gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score = len(contours)

    if score > best_score:
        best_score = score
        best_color = color_name
        best_thresh = thresh

        # Gunakan OCR untuk mengenali angka
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        best_digit = ''.join(filter(str.isdigit, text.strip()))

# Hasil
if best_digit:
    print(f"[HASIL] Angka '{best_digit}' terdeteksi setelah menghilangkan warna: {best_color.upper()}")
else:
    print("[HASIL] Tidak berhasil membaca angka dari gambar.")

# Tampilkan hasil untuk verifikasi
if best_thresh is not None:
    cv2.imshow("Thresholded Image", best_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
