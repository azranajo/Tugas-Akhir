from picamera2 import Picamera2
import time

# Membuat objek Picamera2
picam2 = Picamera2()

# Membuat konfigurasi dengan resolusi yang lebih kecil (640x360)
config = picam2.create_still_configuration()
config["main"]["size"] = (640, 360)  # Mengatur resolusi menjadi 640x360

# Mengonfigurasi kamera dengan pengaturan resolusi yang baru
picam2.configure(config)

# Menyalakan kamera
picam2.start()

# Mengaktifkan autofocus
picam2.set_controls({"AfMode": 3})  # Mode autofocus, 1: Single-shot

# Menunggu autofocus untuk menyesuaikan
time.sleep(2)  # Tunggu 2 detik agar autofocus bisa menyesuaikan

# Mengambil gambar dan menyimpannya
picam2.capture_file("gambar_dengan_autofocus.jpg")
print("Gambar berhasil diambil dengan resolusi kecil!")
