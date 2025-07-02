import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from picamera2 import Picamera2

# Konfigurasi Tesseract di Raspberry
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Fungsi K-Means
def kmeans(k, pixel_values, shape):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels]
    return segmented_image.reshape(shape), labels

# Pilih cluster dengan kontur terbesar
def select_cluster_by_largest_contour(segmented_image, labels, k):
    max_area = -1
    selected_cluster_image = None
    for i in range(k):
        im = np.copy(segmented_image).reshape(-1, 3)
        im[labels != i] = [255, 255, 255]
        cluster_img = im.reshape(segmented_image.shape)

        gray = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            area = cv2.contourArea(max(contours, key=cv2.contourArea))
            if area > max_area:
                max_area = area
                selected_cluster_image = cluster_img
    return selected_cluster_image

# Modifikasi warna hasil segmentasi
def modify_color(image, hex_color="#FF0000"):
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4))
    img = image.copy()
    mask = (img != [255, 255, 255]).any(axis=2)
    img[mask] = rgb
    return img

# OCR fungsi
def recognize_number(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789')
    return text.strip()

# Menangkap gambar dari kamera Raspberry Pi menggunakan picamera2
def capture_image():
    picam2 = Picamera2()  # Inisialisasi Picamera2
    picam2.configure(picam2.create_still_configuration())  # Siapkan kamera untuk menangkap gambar
    picam2.start()  # Mulai kamera
    time.sleep(2)  # Memberikan waktu untuk fokus kamera dan stabilisasi
    image_array = picam2.capture_array()  # Tangkap gambar dalam format numpy array
    picam2.stop()  # Berhenti menggunakan kamera
    return image_array

# Proses utama
results = []

# Menangkap gambar dari kamera
image = capture_image()

# Konversi gambar dari BGR ke RGB (karena OpenCV menggunakan BGR secara default)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Proses segmentasi gambar
shape = image.shape
pixels = image.reshape(-1, 3).astype(np.float32)

k = 4
segmented_image, labels = kmeans(k, pixels, shape)

# Menampilkan visualisasi untuk setiap cluster
for i in range(k):
    cluster_image = np.copy(segmented_image).reshape(-1, 3)
    cluster_image[labels != i] = [255, 255, 255]
    cluster_image = cluster_image.reshape(segmented_image.shape)

    plt.figure(figsize=(4, 4))
    plt.imshow(cluster_image)
    plt.title(f"Cluster {i + 1}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

final_image = select_cluster_by_largest_contour(segmented_image, labels, k)

if final_image is None:
    results.append(('Captured Image', ''))
else:
    colored = modify_color(final_image)
    recognized_number = recognize_number(colored)

    results.append(('Captured Image', recognized_number))

    plt.subplot(1, 1, 1)
    plt.imshow(colored)
    plt.title(f"Angka yang dikenali: {recognized_number}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Simpan hasil ke Excel
df = pd.DataFrame(results, columns=['Image', 'Recognized_Number'])
df.to_excel("hasil_segmentasi_pi_camera2.xlsx", index=False)
print("Selesai! File hasil disimpan sebagai 'hasil_segmentasi_pi_camera2.xlsx'")
