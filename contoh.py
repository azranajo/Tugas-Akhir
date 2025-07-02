import os
import cv2
import numpy as np
import pytesseract
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from picamera2 import Picamera2
from time import sleep
from tqdm import tqdm

# Set path Tesseract jika perlu (biasanya default sudah ok di RPi)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Folder untuk simpan gambar
os.makedirs("data_capture", exist_ok=True)

# Inisialisasi Kamera V3 (via libcamera backend)
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

# Fungsi update preview
def update_frame():
    global frame_np, panel, captured
    if captured:
        return
    frame_np = picam2.capture_array()
    rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk
    panel.after(10, update_frame)

# Fungsi capture gambar
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

update_frame()
root.mainloop()
picam2.close()

# Lanjut ke proses segmentasi dan OCR seperti sebelumnya

# Resize image
def resize_image(image, max_width=320, max_height=240):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

# K-means segmentation
def kmeans(k, pixel_values, shape):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(shape), labels

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
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=50, maxRadius=140)
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

# Proses semua hasil capture
results = []

for idx, (file_name, image) in enumerate(tqdm(image_list, desc="Processing")):
    cropped = detect_circle_and_crop(image)
    if cropped is None:
        results.append((file_name, 'Lingkaran tidak ditemukan'))
        continue

    resized = resize_image(cropped)
    shape = resized.shape
    pixels = resized.reshape(-1, 3).astype(np.float32)
    k = 5
    segmented, labels = kmeans(k, pixels, shape)
    final_image = select_cluster_by_largest_contour(segmented, labels, k)
    if final_image is None:
        results.append((file_name, ''))
        continue

    colored = modify_color(final_image)
    recognized = recognize_number(colored)
    results.append((file_name, recognized))

    plt.imshow(colored)
    plt.title(f"Angka: {recognized}")
    plt.axis("off")
    plt.show()

df = pd.DataFrame(results, columns=["Image", "Recognized_Number"])
df.to_excel("hasil_segmentasi_pi.xlsx", index=False)
print("Selesai. Hasil disimpan di hasil_segmentasi_pi.xlsx")
