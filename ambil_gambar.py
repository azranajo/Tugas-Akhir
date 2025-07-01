import subprocess
from datetime import datetime

def ambil_gambar():
    # Buat nama file berdasarkan waktu saat ini
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    nama_file = f"gambar_{waktu}.jpg"

    # Perintah libcamera-still untuk mengambil gambar
    perintah = [
        "libcamera-still",
        "-o", nama_file,
        "--width", "1280",
        "--height", "720",
        "--nopreview",
        "-t", "1000"  # 1 detik delay sebelum ambil gambar
    ]

    try:
        subprocess.run(perintah, check=True)
        print(f"[✓] Gambar berhasil disimpan: {nama_file}")
    except subprocess.CalledProcessError as e:
        print(f"[✗] Gagal mengambil gambar: {e}")

if __name__ == "__main__":
    ambil_gambar()
