from picamera2 import Picamera2
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def select_cluster_by_digit_shape(segmented_image, labels, k):
    best_cluster = None
    best_score = 0

    for i in range(k):
        im = np.copy(segmented_image).reshape(-1, 3)
        im[labels != i] = [255, 255, 255]  # Putihkan selain cluster ke-i
        cluster_img = im.reshape(segmented_image.shape)

        # Proses ke grayscale dan threshold
        gray = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Ambil kontur terbesar
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            num_contours = len(contours)

            # Hitung skor: area terbesar dibagi jumlah kontur (untuk kurangi efek noise)
            score = largest_area / (num_contours + 1e-5)

            # Gambar kontur terbesar (untuk debugging visual)
            debug_img = cluster_img.copy()
            cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 0), 1)

            # Tampilkan visualisasi
            plt.figure()
            plt.imshow(debug_img)
            plt.title(f"Cluster {i} - Score: {score:.2f}")
            plt.axis('off')
            plt.show()

            # Simpan cluster terbaik
            if score > best_score:
                best_score = score
                best_cluster = cluster_img

    return best_cluster

# --- Inisialisasi Kamera ---
print("[INFO] Menginisialisasi kamera...")
picam2 = Picamera2()
picam2.preview_configuration.main.size = (128, 128)  # Resolusi input model
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)  # Waktu buffer kamera

# --- Ambil gambar dari kamera ---
print("[INFO] Mengambil gambar dari kamera...")
frame = picam2.capture_array()

# Misalnya, segmented_image dan labels adalah hasil dari beberapa proses klastering atau segmentasi gambar
# Gantilah kode di bawah dengan gambar yang telah diproses sebelumnya (contoh di sini menggunakan frame kamera)
segmented_image = frame  # Ganti ini dengan gambar hasil segmentasi
labels = np.zeros(segmented_image.shape[:2], dtype=int)  # Dummy labels, ganti sesuai kebutuhan

# Jumlah klaster yang ingin diproses (k)
k = 3  # Gantilah dengan jumlah klaster yang Anda inginkan

# Panggil fungsi untuk memilih cluster berdasarkan bentuk digit
best_cluster = select_cluster_by_digit_shape(segmented_image, labels, k)

# Tampilkan cluster terbaik jika ditemukan
if best_cluster is not None:
    cv2.imshow("Best Cluster", best_cluster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[INFO] Tidak ada cluster yang ditemukan.")
