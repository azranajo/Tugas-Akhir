from picamera2 import Picamera2
import time

# Membuat objek Picamera2
picam2 = Picamera2()

# Membuat konfigurasi dengan resolusi 640x480
config = picam2.create_still_configuration()
config["main"]["size"] = (640, 480)  # Resolusi gambar diatur menjadi 640x480

# Mengonfigurasi kamera dengan pengaturan resolusi yang baru
picam2.configure(config)

# Menyalakan kamera
picam2.start()

# Mengaktifkan autofocus (secara default sudah aktif, tetapi kamu bisa menyesuaikannya)
picam2.set_controls({"AfMode": 3})  # Mode autofocus, 1: Single-shot

# Menunggu autofocus untuk menyesuaikan
time.sleep(2)  # Tunggu 2 detik agar autofocus bisa menyesuaikan

# Mengambil gambar dan menyimpannya dengan resolusi yang lebih kecil
picam2.capture_file("gambar_dengan_autofocus_kecil.jpg")
print("Gambar berhasil diambil dengan resolusi kecil!")
