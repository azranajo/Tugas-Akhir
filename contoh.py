import cv2
import numpy as np
import subprocess
import os

def ambil_frame_dari_kamera(output_path="temp.jpg"):
    cmd = ["libcamera-jpeg", "-o", output_path, "--width", "640", "--height", "480", "--quality", "90", "--timeout", "100"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cv2.imread(output_path)

cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)

while True:
    frame = ambil_frame_dari_kamera()
    if frame is None:
        print("Gagal ambil gambar.")
        break

    cv2.imshow("Preview", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):  # Tekan C untuk capture
        cv2.imwrite("data_capture/captured_manual.jpg", frame)
        print("Captured!")
    elif key == ord("q"):  # Tekan Q untuk keluar
        break

cv2.destroyAllWindows()
