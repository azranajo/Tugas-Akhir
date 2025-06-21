import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
import tkinter as tk
import threading
from tqdm import tqdm
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
from PIL import Image,ImageTk
import io

# Konfigurasi Tesseract di Raspberry
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

os.makedirs("data_capture", exist_ok=True)
camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera, size=(640, 480))
sleep(2)  # Tunggu kamera siap

NUM_IMAGES = 2
image_list = []

# Variabel global
frame_np = None
captured = False
image_list = []
capture_count = 0

# Fungsi update live preview
def update_frame():
    global frame_np, panel, captured
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        if captured:
            break
        frame_np = frame.array
        rgb_image = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_image)
        image_tk = ImageTk.PhotoImage(image_pil)
        panel.config(image=image_tk)
        panel.image = image_tk
        raw_capture.truncate(0)

# Fungsi ketika tombol "Capture" ditekan
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

# Setup GUI
root = tk.Tk()
root.attributes('-fullscreen', True)
panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Capture", font=("Arial", 24), bg="green", fg="white", command=capture_image)
btn.pack(pady=20)

# Jalankan preview di thread
threading.Thread(target=update_frame, daemon=True).start()
root.mainloop()
camera.close()

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

# Proses utama
results = []

for idx, (file_name, image) in enumerate(tqdm(image_list, desc="Processing")):

    shape = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    k = 4
    segmented_image, labels = kmeans(k, pixels, shape)
    final_image = select_cluster_by_largest_contour(segmented_image, labels, k)

    if final_image is None:
        results.append((file_name, ''))
        continue

    colored = modify_color(final_image)
    recognized_number = recognize_number(colored)

    results.append((file_name, recognized_number))

    # Optional visualisasi di layar
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(colored)
    plt.title(f"OCR: {recognized_number}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Simpan hasil ke Excel
df = pd.DataFrame(results, columns=['Image', 'Recognized_Number'])
df.to_excel("hasil_segmentasi_pi.xlsx", index=False)
print("Selesai! File hasil disimpan sebagai 'hasil_segmentasi_pi.xlsx'")
