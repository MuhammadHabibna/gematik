# Babak Final Lomba Data Science GEMATIK VI — Soal dan Deskripsi Dataset

**Tema:** *Empowering Global Innovators for an Intelligent Future*
**Studi kasus:** Prediksi Status Pertumbuhan UMKM Digital di Provinsi Lampung

## 1. Latar Belakang Soal
Pemerintah daerah ingin menentukan UMKM mana yang perlu didampingi lebih dulu dalam program digitalisasi. Anda diminta membangun model yang memprediksi **status pertumbuhan usaha** setiap UMKM berdasarkan profil usaha, pemilik, dan jejak digitalnya.

## 2. Tugas
Bangun model klasifikasi multikelas untuk memprediksi kolom `Status_Pertumbuhan`, dengan tiga kelas:

| Kelas | Arti |
|---|---|
| Menurun | Omzet usaha cenderung turun |
| Stabil | Omzet relatif tetap |
| Tumbuh | Omzet cenderung naik |

## 3. Berkas
| Berkas | Isi |
|---|---|
| `data_mentah.csv` | 7.000 baris data mentah **berlabel** (belum dipisah train/test) |
| `sample_submission.csv` | Contoh format pengumpulan (1.400 ID holdout) |

## 3a. Protokol Pembagian Data (WAJIB, agar skor antar tim adil)
Peserta memisahkan sendiri data latih dan data uji, **dengan pembagian standar berikut**:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data_mentah.csv")          # jangan diurutkan/diacak sebelum split
train, holdout = train_test_split(df, test_size=0.2,
                                  stratify=df["Status_Pertumbuhan"], random_state=42)
```
- `holdout` (1.400 baris) **hanya boleh dipakai untuk evaluasi akhir dan pembuatan submission**, tidak boleh untuk melatih, tuning, atau memilih fitur.
- Validasi dan tuning dilakukan di dalam `train` (misalnya cross-validation).
- Peserta wajib menyertakan kode (notebook/script) yang menunjukkan protokol ini. Dewan juri dapat memeriksanya.

## 4. Deskripsi Kolom
| Kolom | Tipe | Keterangan |
|---|---|---|
| ID | teks | Identitas unik UMKM |
| Kabupaten_Kota | kategori | Lokasi usaha (15 kab/kota di Lampung) |
| Sektor_Usaha | kategori | Kuliner, Fashion, Kerajinan, Pertanian Olahan, Jasa, Perdagangan Eceran |
| Usia_Pemilik | angka | Usia pemilik (tahun) |
| Pendidikan_Pemilik | kategori | SD/SMP, SMA, Diploma, Sarjana |
| Lama_Usaha_Bulan | angka | Lama usaha berjalan (bulan) |
| Jumlah_Karyawan | angka | Jumlah karyawan |
| Modal_Awal_Juta | angka | Modal awal (juta rupiah) |
| Skor_Literasi_Digital | angka | Skor 0–100 |
| Pernah_Pelatihan | kategori | Ya/Tidak, pernah ikut pelatihan digital |
| Kualitas_Internet | kategori | Rendah, Sedang, Tinggi |
| Platform_Digital_Aktif | angka | Jumlah platform digital aktif (0–5) |
| Frekuensi_Posting_Per_Minggu | angka | Frekuensi unggahan promosi |
| Followers_Media_Sosial | angka | Total pengikut media sosial |
| Metode_Pembayaran_Digital | kategori | Tunai saja, QRIS, Multi-kanal |
| Akses_Kredit | kategori | Ya/Tidak |
| Jarak_ke_Pusat_Kota_km | angka | Jarak ke pusat kota (km) |
| Rating_Rata_Rata | angka | Rating pelanggan (1–5) |
| Jumlah_Ulasan | angka | Jumlah ulasan pelanggan |
| Omzet_Bulanan_Juta | angka | Omzet bulanan (juta rupiah) |
| **Status_Pertumbuhan** | **target** | Menurun / Stabil / Tumbuh (target prediksi) |

> Data ini **bersifat sintetis** untuk keperluan lomba dan tidak mewakili UMKM sebenarnya. Data nyata biasanya tidak rapi: perhatikan nilai kosong, nilai ekstrem, dan penulisan yang tidak konsisten.

## 5. Metrik Evaluasi
**Macro F1-score** (rata-rata F1 dari ketiga kelas).
- Dihitung pada 1.400 baris `holdout`.
- Skor *Public* (30% holdout) tampil di leaderboard sementara; skor akhir dari *Private* (70% sisanya), diumumkan setelah sesi selesai.

## 6. Format Pengumpulan
Berkas CSV dengan **tepat dua kolom**, `ID` dan `Status_Pertumbuhan`, seluruh 1.400 ID `holdout`, tanpa duplikat. Nama berkas: `<NamaTim>.csv`. Contoh:

```
ID,Status_Pertumbuhan
UMKM-03259,Stabil
UMKM-05091,Tumbuh
```
Pengumpulan yang formatnya salah tidak masuk leaderboard.

## 7. Aturan Pengerjaan
- Waktu pengerjaan **100 menit** sejak dataset dibagikan.
- Boleh memakai tools, library, dan framework apa pun.
- Dilarang berbagi pekerjaan antar tim atau meminta bantuan pihak luar tim.
- Dilarang melatih atau melakukan tuning memakai data `holdout`.
- Dokumentasikan seluruh proses (pemahaman data, pra-pemrosesan, pemodelan, evaluasi, visualisasi) dalam **PPT**.
- Presentasi daring **8 menit**, dilanjutkan tanya jawab dengan dewan juri.

## 8. Saran Struktur PPT
1. Pemahaman masalah dan data
2. Eksplorasi data (EDA) dan visualisasi
3. Pra-pemrosesan data
4. Pemodelan dan perbandingan model
5. Evaluasi (Macro F1, confusion matrix, analisis kesalahan)
6. Interpretasi: fitur yang paling berpengaruh
7. Kesimpulan dan rekomendasi bagi pemangku kepentingan
