from picamera2 import Picamera2
import time

# Membuat objek Picamera2
picam2 = Picamera2()

# Menyalakan kamera
picam2.start()

# Mengaktifkan autofocus (secara default sudah aktif, tetapi kamu bisa menyesuaikannya)
picam2.set_controls({"AfMode": 3})  # Mode autofocus, 3: Continuous (terus-menerus)

# Menunggu autofocus untuk menyesuaikan
time.sleep(2)  # Tunggu 2 detik agar autofocus bisa menyesuaikan

# Mengambil gambar
picam2.capture_file("gambar_dengan_autofocus.jpg")
print("Gambar berhasil diambil!")
