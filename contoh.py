import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
import tkinter as tk
import threading
from tqdm import tqdm
from time import sleep
from PIL import Image, ImageTk
from picamera2 import Picamera2

# Konfigurasi Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Buat folder penyimpanan
os.makedirs("data_capture", exist_ok=True)

# Inisialisasi Kamera V3 (Picamera2)
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()
sleep(2)

# Variabel global
frame_np = None
captured = False
image_list = []
capture_count = 0

# Fungsi update preview kamera
def update_frame():
    global frame_np, panel, captured
    while not captured:
        frame_np = picam2.capture_array()
        rgb_image = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_image)
        image_tk = ImageTk.PhotoImage(image_pil)
        panel.config(image=image_tk)
        panel.image = image_tk
        sleep(0.05)

# Fungsi capture
def capture_image():
    global frame_np, captured, capture_count, image_list
    captured = True
    rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    filename = f"captured_{capture_count}.jpg"
    img_pil.save(f"data_capture/{filename}")
    image_list.append((filename, rgb))
    print(f"Captured: {filename}")
    root.destroy()

# Fungsi keluar
def exit_program():
    global captured
    captured = True
    root.destroy()
    print("Keluar tanpa mengambil gambar.")

# Setup GUI
root = tk.Tk()
root.attributes('-fullscreen', True)
panel = tk.Label(root, width=320, height=240)
panel.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

btn_capture = tk.Button(button_frame, text="Capture", font=("Arial", 20), bg="green", fg="white", command=capture_image)
btn_capture.pack(side="left", padx=20)

btn_exit = tk.Button(button_frame, text="Exit", font=("Arial", 20), bg="red", fg="white", command=exit_program)
btn_exit.pack(side="left", padx=20)

# Jalankan preview
threading.Thread(target=update_frame, daemon=True).start()
root.mainloop()
picam2.close()

# ---------- PROSES SEGMENTASI & OCR ----------

def resize_image(image, max_width=320, max_height=240):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def kmeans(k, pixel_values, shape):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels]
    return segmented_image.reshape(shape), labels

def select_cluster_by_digit_shape(segmented_image, labels, k):
    best_cluster = None
    best_score = 0
    for i in range(k):
        im = np.copy(segmented_image).reshape(-1, 3)
        im[labels != i] = [255, 255, 255]
        cluster_img = im.reshape(segmented_image.shape)

        gray = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            total_area = sum(cv2.contourArea(cnt) for cnt in contours)
            if total_area > best_score:
                best_score = total_area
                best_cluster = cluster_img

    return best_cluster

def modify_color(image, hex_color="#FF0000"):
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    img = image.copy()
    mask = (img != [255, 255, 255]).any(axis=2)
    img[mask] = rgb
    return img

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

def recognize_number(image):
    preprocessed = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(preprocessed, config='--psm 10 -c tessedit_char_whitelist=0123456789')
    return text.strip()

def detect_circle_and_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=140)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        x1, y1 = max(x - r, 0), max(y - r, 0)
        x2, y2 = x + r, y + r
        return masked[y1:y2, x1:x2]
    return None

# Jalankan segmentasi & OCR
results = []
for file_name, image in tqdm(image_list, desc="Processing"):
    cropped = detect_circle_and_crop(image)
    if cropped is None:
        results.append((file_name, 'Lingkaran tidak ditemukan'))
        continue
    resized = resize_image(cropped)
    shape = resized.shape
    pixels = resized.reshape(-1, 3).astype(np.float32)
    k = 4
    segmented_image, labels = kmeans(k, pixels, shape)

    # --- VISUALISASI CLUSTER 3 --- 
    cluster_yang_ditampilkan = 3  # Menentukan bahwa kita ingin fokus pada Cluster 3
    mask_cluster = (labels == cluster_yang_ditampilkan).astype("uint8").reshape(shape[:2]) * 255
    cluster_vis = cv2.bitwise_and(resized, resized, mask=mask_cluster)

    # Visualisasikan hanya Cluster 3
    plt.figure()
    plt.imshow(cv2.cvtColor(cluster_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Cluster {cluster_yang_ditampilkan}")
    plt.axis("off")
    plt.show()

    # Proses pengenalan untuk Cluster 3
    final_image = select_cluster_by_digit_shape(segmented_image, labels, k)

    if final_image is None:
        results.append((file_name, ''))
        continue

    colored = modify_color(final_image)
    recognized_number = recognize_number(colored)
    results.append((file_name, recognized_number))

    # Tampilkan hasil akhir untuk Cluster 3
    plt.imshow(colored)
    plt.title(f"Angka dikenali (Cluster 3): {recognized_number}")
    plt.axis("off")
    plt.show()

df = pd.DataFrame(results, columns=['Image', 'Recognized_Number'])
df.to_excel("hasil_segmentasi_pi.xlsx", index=False)
print("Selesai. Disimpan ke hasil_segmentasi_pi.xlsx")
