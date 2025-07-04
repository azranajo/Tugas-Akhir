import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
from tqdm import tqdm
from picamera2 import Picamera2
import time

# Konfigurasi Tesseract di Raspberry
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Direktori gambar
DATA_DIR = "data_baru_camera"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Membuat folder baru jika belum ada

# Inisialisasi Kamera Raspberry Pi
print("[INFO] Menginisialisasi kamera...")
picam2 = Picamera2()
picam2.preview_configuration.main.size = (256, 256)  # Resolusi input model
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)  # Waktu buffer kamera

# Fungsi K-Means
def kmeans(k, pixel_values, shape):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels]
    return segmented_image.reshape(shape), labels

# Preprocessing Gambar untuk meningkatkan kontras dan ketajaman
def preprocess_image(image):
    # Menambah kontras gambar dengan histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # Menggunakan filter sharpen untuk penajaman
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    return sharpened

# Pilih cluster dengan kontur terbesar
def select_cluster_by_largest_contour(segmented_image, labels, k):
    max_area = -1
    selected_cluster_image = None
    selected_cluster_index = -1
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
                selected_cluster_index = i  # Menyimpan nomor cluster yang terpilih
    return selected_cluster_image, selected_cluster_index

# Fungsi OCR yang dimodifikasi untuk meningkatkan akurasi
def recognize_number(image):
    # Preprocessing gambar sebelum OCR
    processed_image = preprocess_image(image)
    _, thresh = cv2.threshold(processed_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789')
    return text.strip()

# Ambil gambar dari kamera dan simpan ke folder
def capture_and_save_image():
    print("[INFO] Menampilkan preview kamera, tekan 'q' untuk mengambil gambar...")
    while True:
        frame = picam2.capture_array()  # Ambil gambar dari kamera
        cv2.imshow("Camera Preview", frame)  # Tampilkan gambar untuk preview
        
        # Jika pengguna menekan 'q', keluar dari preview dan simpan gambar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"image_{timestamp}.jpg"
            save_path = os.path.join(DATA_DIR, file_name)
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Simpan gambar sebagai file JPG
            print(f"[INFO] Gambar disimpan di: {save_path}")
            cv2.destroyAllWindows()  # Tutup jendela preview
            return save_path

# Proses utama
results = []

# Ambil gambar pertama untuk percakapan
image_path = capture_and_save_image()

# Baca gambar yang disimpan
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Proses K-Means dan OCR
shape = image.shape
pixels = image.reshape(-1, 3).astype(np.float32)

# Cobalah dengan nilai k yang berbeda, misalnya k=5 atau k=10 untuk eksperimen
k = 5  # Cobalah mengubah k menjadi 10
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

# Memilih cluster dengan kontur terbesar
final_image, selected_cluster_index = select_cluster_by_largest_contour(segmented_image, labels, k)

# Modifikasi warna hasil segmentasi
def modify_color(image, hex_color="#FF0000"):
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4))
    img = image.copy()
    mask = (img != [255, 255, 255]).any(axis=2)
    img[mask] = rgb
    return img

if final_image is None:
    results.append((image_path, ''))
else:
    colored = modify_color(final_image)
    recognized_number = recognize_number(colored)

    results.append((image_path, recognized_number, selected_cluster_index))

    plt.subplot(1, 1, 1)
    plt.imshow(colored)
    plt.title(f"Angka yang dikenali: {recognized_number} (Cluster {selected_cluster_index + 1})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Simpan hasil ke Excel
df = pd.DataFrame(results, columns=['Image', 'Recognized_Number', 'Best_Cluster'])
df.to_excel("hasil_segmentasi_pi.xlsx", index=False)
print("Selesai! File hasil disimpan sebagai 'hasil_segmentasi_pi.xlsx'")
