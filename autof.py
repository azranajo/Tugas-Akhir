from picamera2 import Picamera2
import time

# Membuat objek Picamera2
picam2 = Picamera2()

# Menyalakan kamera
picam2.start()

# Mengatur resolusi gambar menjadi 640x480 (VGA)
picam2.set_controls({"AfMode": 3})  # Mode autofocus, 1: Single-shot
picam2.set_controls({"Resolution": (480, 320)})  # Resolusi diubah menjadi 640x480

# Menunggu autofocus untuk menyesuaikan
time.sleep(2)  # Tunggu 2 detik agar autofocus bisa menyesuaikan

# Mengambil gambar dan menyimpannya dengan resolusi yang lebih kecil
picam2.capture_file("gambar_dengan_autofocus_kecil.jpg")
print("Gambar berhasil diambil dengan resolusi kecil!")
