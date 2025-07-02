from picamera2 import Picamera2, Preview
import time

# Membuat objek Picamera2
picam2 = Picamera2()

# Mengatur konfigurasi resolusi (640x480)
config = picam2.create_still_configuration(still_width=640, still_height=480)
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
