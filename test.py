import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI agar tidak error di Pi tanpa layar
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load gambar (pastikan file ada di direktori yang sama)
img = cv2.imread('captured_0.jpg')  # Ganti path jika perlu

# Cek apakah gambar berhasil dibaca
if img is None:
    print("Gambar tidak ditemukan. Pastikan nama file benar dan berada di direktori yang sama.")
    exit()

# Konversi BGR ke RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (100, 100))  # Ukuran kecil agar hemat RAM
img_reshaped = img.reshape((-1, 3))

# Simpan WCSS dan Silhouette Score
wcss = []
silhouette_scores = []
K_range = range(2, 11)

print("🔄 Sedang mengevaluasi K...")

for k in K_range:
    print(f"> Mengevaluasi K = {k}")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(img_reshaped)

    wcss.append(kmeans.inertia_)

    try:
        score = silhouette_score(img_reshaped, labels)
        silhouette_scores.append(score)
    except Exception as e:
        print(f"⚠️ Gagal menghitung silhouette score untuk K={k}: {e}")
        silhouette_scores.append(0)


# Plot hasil dan simpan sebagai gambar (karena RPi mungkin tidak punya GUI)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-', linewidth=2)
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2)
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')

plt.tight_layout()
plt.savefig('hasil_kmeans_evaluasi.png')  # Simpan hasil ke file
print("Plot evaluasi K-Means berhasil disimpan sebagai 'hasil_kmeans_evaluasi.png'")
