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
from sklearn.metrics import silhouette_score  # Untuk menentukan nilai K terbaik

# Konfigurasi Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Buat folder penyimpanan
os.makedirs("data_capture", exist_ok=True)

# Inisialisasi Kamera V3 (Picamera2)
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (240, 240)})
picam2.configure(config)

# Mengaktifkan Autofokus
picam2.set_controls({"AfMode": 1})  # Mode autofocus, 3 berarti Continuous Autofocus

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
        rgb_image = frame_np 
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

def resize_image(image, max_width=240, max_height=240):
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
    contour_thresh = 150
    min_solidity = 0.2
    debug = True
    def circular_mask(image):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius - 2, 255, -1)
        return mask

    best_cluster = None
    best_score = 0

    for i in range(k):
        # Ambil cluster i
        im = np.copy(segmented_image).reshape(-1, 3)
        im[labels != i] = [255, 255, 255]
        cluster_img = im.reshape(segmented_image.shape)

        # Ubah ke grayscale dan threshold
        gray = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Masking hanya area dalam lingkaran Ishihara
        mask = circular_mask(gray)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        # Temukan kontur
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Skip cluster dengan terlalu banyak kontur (noise)
        if len(contours) > contour_thresh:
            if debug:
                print(f"Cluster {i} diskip karena terlalu banyak kontur: {len(contours)}")
            continue

        # Ambil kontur terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)

        # Solidity (kerapatan)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = largest_area / (hull_area + 1e-5)
        if solidity < min_solidity:
            if debug:
                print(f"Cluster {i} diskip karena solidity rendah: {solidity:.2f}")
            continue

        # Bounding box center 
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        bbox_cx = x + w_box // 2
        bbox_cy = y + h_box // 2

        # Centroid dari kontur
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = -1, -1

        h, w = gray.shape

        # Gabungan validasi posisi: centroid ATAU bbox center harus di tengah
        centroid_ok = (w * 0.15 < cx < w * 0.85 and h * 0.15 < cy < h * 0.85)
        bbox_ok = (w * 0.15 < bbox_cx < w * 0.85 and h * 0.15 < bbox_cy < h * 0.85)
        if not (centroid_ok or bbox_ok):
            if debug:
                print(f"Cluster {i} diskip karena posisi pusat di pinggir. Centroid: ({cx},{cy}), BBox: ({bbox_cx},{bbox_cy})")
            continue

        # Skor akhir (angka besar, noise kecil)
        score = largest_area / (len(contours) + 1e-5)

        # Visualisasi debugging
        if debug:
            debug_img = cluster_img.copy()
            cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 0), 1)
            cv2.circle(debug_img, (cx, cy), 3, (0, 255, 0), -1)         # centroid hijau
            cv2.circle(debug_img, (bbox_cx, bbox_cy), 3, (255, 0, 0), -1)  # bbox biru
            text = f"Area: {largest_area:.2f}"
            cv2.putText(debug_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            plt.figure()
            plt.imshow(debug_img)
            plt.title(f"Cluster {i} - Score: {score:.2f}, Solidity: {solidity:.2f}, Contours: {len(contours)}")
            plt.axis('off')
            plt.show()

        # Simpan jika lebih baik dari sebelumnya
        if score > best_score:
            best_score = score
            best_cluster = cluster_img

    return best_cluster

def remove_noise_outside_center(image, min_area=None, max_dist_ratio=0.1, min_solidity=0.2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Auto tuning: cari kontur terbesar dulu ---
    main_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            main_contour = cnt
            max_area = area

    if min_area is None:
        min_area = max_area * 0.5

    cleaned = np.zeros_like(binary)
    h, w = binary.shape
    center = np.array([w // 2, h // 2])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - center)
            if dist > max_dist_ratio * min(w, h):
                continue
        else:
            continue

        # Filter berdasarkan kerapatan (solidity)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-5)
        if solidity < min_solidity:
            continue

        # Tambahan: hanya ambil kontur dalam bounding box utama
        if main_contour is not None:
            x, y, bw, bh = cv2.boundingRect(main_contour)
            if not cv2.pointPolygonTest(main_contour, (cx, cy), False) == 1:
                if not (x <= cx <= x + bw and y <= cy <= y + bh):
                    continue

        cv2.drawContours(cleaned, [cnt], -1, 255, thickness=cv2.FILLED)
    # Tambahkan DILASI agar angka lebih solid
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=5)

    result = np.full_like(image, 255)
    result[cleaned == 255] = image[cleaned == 255]
    return result

def modify_color(image, hex_color="#FF0000"):
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    img = image.copy()
    mask = (img != [255, 255, 255]).any(axis=2)
    img[mask] = rgb
    return img

def preprocess_for_ocr(image):
    # Ubah ke HSV untuk ambil warna merah
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Ambil 2 rentang merah (karena merah wrap-around di HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Gaussian blur untuk merapikan tepi
    blurred = cv2.GaussianBlur(mask, (5, 5), sigmaX=1)

    # Optional: dilasi ringan untuk pertebal
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=5)

    # Threshold dan invert
    _, binary = cv2.threshold(dilated, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)

    # Morph closing untuk mengisi lubang kecil
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary

def show_preprocess_result(original, preprocessed):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Sebelum")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title("Setelah Preprocess")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def recognize_number(image):
    try:
        preprocessed = preprocess_for_ocr(image)
        # Visualisasi preprocess
        show_preprocess_result(image, preprocessed)
        print("Ukuran input OCR:", preprocessed.shape)

        text = pytesseract.image_to_string(preprocessed, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        print("Hasil mentah OCR:", repr(text))  # Untuk debug
        return text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"

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

    # --- VISUALISASI HASIL DETEKSI LINGKARAN & CROPPING ---
    plt.figure(figsize=(12, 4))

    # 1. Gambar asli
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Gambar Asli")
    plt.axis("off")

    # 2. Gambar hasil cropping dari lingkaran
    plt.subplot(1, 3, 2)
    plt.imshow(cropped)
    plt.title("Hasil Crop Lingkaran")
    plt.axis("off")

    # 3. Gambar setelah resize
    plt.subplot(1, 3, 3)
    plt.imshow(resize_image(cropped))
    plt.title("Setelah Resize")
    plt.axis("off")

    plt.suptitle(f"Preprocessing: {file_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

    resized = resize_image(cropped)
    shape = resized.shape
    pixels = resized.reshape(-1, 3).astype(np.float32)


    # Menentukan nilai K terbaik otomatis dengan Silhouette Score
    best_score = -1
    best_k = 2
    best_labels = None
    best_segmented = None

    for k_try in range(2, 11):  # Range nilai K dari 2 hingga 10
        try:
            segmented_k, labels_k = kmeans(k_try, pixels, shape)
            score = silhouette_score(pixels, labels_k)
            if score > best_score:
                best_score = score
                best_k = k_try
                best_labels = labels_k
                best_segmented = segmented_k
        except Exception as e:
            print(f"Silhouette error untuk K={k_try}: {e}")
            continue

    print(f"Nilai K terbaik untuk {file_name}: {best_k} (Silhouette Score: {best_score:.4f})")

    segmented_image = best_segmented
    labels = best_labels
    k = best_k

    # --- VISUALISASI CLUSTER ---
    for i in range(k):
        mask_cluster = (labels == i).astype("uint8").reshape(shape[:2]) * 255
        cluster_vis = cv2.bitwise_and(resized, resized, mask=mask_cluster)
        black_pixels = np.all(cluster_vis == [0, 0, 0], axis=-1)
        cluster_vis[black_pixels] = [255, 255, 255]
       
        plt.figure()
        plt.imshow(cv2.cvtColor(cluster_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Cluster {i}")
        plt.axis("off")
        plt.show()

    final_image = select_cluster_by_digit_shape(segmented_image, labels, k)

    if final_image is None:
        results.append((file_name, ''))
        continue

    cleaned_colored = remove_noise_outside_center(final_image)
    colored = modify_color(cleaned_colored)
    recognized_number = recognize_number(colored)
    results.append((file_name, recognized_number))

    # Tampilkan hasil akhir
    plt.imshow(colored)
    plt.title(f"Angka yang dikenali : {recognized_number}")
    plt.axis("off")
    plt.show()

df = pd.DataFrame(results, columns=['Image', 'Recognized_Number'])
df.to_excel("hasil_segmentasi_pi.xlsx", index=False)
print("Selesai. Disimpan ke hasil_segmentasi_pi.xlsx")
