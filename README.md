# Dominant Color Picker

**Dominant Color Picker** adalah website berbasis Streamlit untuk mengekstrak
palet lima warna paling dominan dari gambar JPG atau PNG. Aplikasi ini cocok
sebagai demonstrasi tugas Artificial Intelligence karena menerapkan
**K-Means Clustering** pada data pixel RGB.

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

## Menjalankan di komputer

Pastikan Python 3.12 atau versi kompatibel telah terinstal.

```bash
python -m venv .venv
```

Aktifkan virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependensi dan jalankan aplikasi:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, umumnya
`http://localhost:8501`.

## Struktur file

```text
WebsiteColorPicker/
|-- app.py
|-- requirements.txt
`-- README.md
```

## Deploy ke Streamlit Community Cloud

1. Push ketiga file proyek ini ke repository GitHub.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/) dan hubungkan
   akun GitHub jika belum terhubung.
3. Klik **Create app**, lalu pilih repository dan branch GitHub yang berisi
   aplikasi.
4. Isi entrypoint file dengan `app.py`.
5. Opsional: pada **Advanced settings**, pilih Python 3.12 agar sesuai dengan
   versi yang digunakan untuk dependensi proyek ini.
6. Klik **Deploy**. Streamlit akan membaca `requirements.txt`, menginstal
   dependensi, dan menyediakan URL aplikasi untuk dibagikan.

Panduan resmi:
[Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Teknologi

- Streamlit
- NumPy
- Pillow
- scikit-learn (`KMeans`)
