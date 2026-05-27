# Identitas 
Nama  : Tristan Bonardo Silalahi
NPM   : 140810240058
Kelas : B

# Dominant Color Picker

Dominant Color Picker adalah website berbasis Streamlit untuk mengekstrak
palet lima warna paling dominan dari gambar JPG atau PNG. Aplikasi ini cocok
sebagai demonstrasi tugas Artificial Intelligence karena menerapkan
K-Means Clustering pada data pixel RGB.

## Fitur

- Upload gambar JPG, JPEG, atau PNG dan tampilkan preview gambar.
- Resize otomatis maksimum `420 x 420` pixel agar proses clustering ringan.
- Analisis `KMeans(n_clusters=5)` dengan hasil konsisten (`random_state=42`).
- Palet horizontal berisi preview warna, HEX, RGB, dan persentase cluster.
- Blok HEX dengan tombol salin serta download hasil palet ke file teks.
- Tampilan modern dengan CSS custom, panduan pada halaman utama, dan footer aplikasi.

## Cara kerja K-Means

1. Gambar yang diupload dikonversi ke RGB dan diperkecil.
2. Setiap pixel dibentuk sebagai titik data tiga fitur: Red, Green, dan Blue.
3. K-Means membagi seluruh pixel menjadi lima cluster.
4. Centroid tiap cluster digunakan sebagai warna dominan.
5. Jumlah anggota cluster dibandingkan total pixel untuk menghitung
   persentase kemunculan warna.

## Teknologi

- Streamlit
- NumPy
- Pillow
- scikit-learn 
