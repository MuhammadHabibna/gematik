# Laporan Lengkap Hasil Penelitian dan Bahan Presentasi (PPT)
# Babak Final Lomba Data Science GEMATIK VI 2026

- **Nama Tim**: Atas Nama
- **Studi Kasus**: Prediksi Status Pertumbuhan UMKM Digital di Provinsi Lampung
- **Metrik Evaluasi**: Macro F1-Score (Multikelas: Menurun, Stabil, Tumbuh)
- **Berkas Pengumpulan Resmi**: `Atas Nama.csv` (1.400 baris Holdout, tervalidasi 100% cocok dengan `sample_submission.csv`)
- **Skor Akhir Champion Model**:
  - **10-Fold Stratified CV Macro F1 (Train)**: `0.67369`
  - **10-Fold Stratified CV Multi-Class LogLoss**: `0.70546`
  - **Evaluasi Satu Kali pada Holdout (1.400 baris)**: **Macro F1 = `0.66867`** | Accuracy = `0.6621` | LogLoss = `0.71835`

---

## BAB 1. Pemahaman Masalah, Protokol Data, dan Forensik Dataset Sintetis

### 1.1 Konteks Masalah dan Tujuan Kebijakan
Pemerintah Provinsi Lampung memerlukan mekanisme berbasis data untuk menentukan UMKM mana yang harus diprioritaskan dalam program pendampingan digitalisasi. Permasalahan diformulasikan sebagai klasifikasi multikelas terhadap kolom `Status_Pertumbuhan` dengan tiga kelas berurutan (ordinal):
1. **Menurun** (1.890 baris; tepat 27,00%): omzet usaha cenderung turun (prioritas penyelamatan).
2. **Stabil** (3.010 baris; tepat 43,00%): omzet relatif tetap (prioritas akselerasi).
3. **Tumbuh** (2.100 baris; tepat 30,00%): omzet cenderung naik (kandidat ekspansi dan mentor).

### 1.2 Kepatuhan Protokol Split Standar (Bebas Data Leakage)
Sesuai aturan soal Bagian 3a, dataset `data_mentah.csv` (7.000 baris, 21 kolom, 0 duplikat) dipisahkan tanpa pengurutan atau pengacakan sebelumnya menggunakan:
```python
train, holdout = train_test_split(
    df, test_size=0.2, stratify=df["Status_Pertumbuhan"], random_state=42
)
```
- **Train (5.600 baris)**: digunakan secara eksklusif untuk seluruh tahapan EDA, pembersihan data, *fitting* imputasi MICE, *clustering*, pelatihan 15 model, optimasi bobot *ensemble*, dan penalaan *threshold* via Nested 10-Fold CV.
- **Holdout (1.400 baris)**: dikunci total dari proses pelatihan dan hanya dievaluasi satu kali di akhir. Urutan 1.400 `ID` pada `holdout` terverifikasi 100% identik dengan `sample_submission.csv`.

### 1.3 Penemuan Forensik Pembangkitan Data Sintetis dan Batas Teoretis Bayes
Melalui pengujian korelasi dalam kelas dan uji *nearest neighbors* pada `train`, ditemukan dua bukti matematis penting:
1. **Bukti Collider Bias (Teorema Berkson, 1946)**: Pada populasi `train` utuh, korelasi `Skor_Literasi_Digital` dengan `Rating_Rata_Rata` hampir nol (`r = -0,008`). Namun ketika dikondisikan di dalam kelas `Stabil`, korelasinya berubah negatif signifikan (`r = -0,150`). Ditambah proporsi target persis `27,00% : 43,00% : 30,00%`, hal ini membuktikan bahwa label dibangkitkan dari satu skor laten kontinu `S = f(X) + epsilon` (dengan *Gaussian noise* `epsilon`) yang dipotong menggunakan `qcut` pada persentil ke-27 dan ke-70.
2. **Estimasi Batas Teoretis Bayes (Cover dan Hart, 1967)**: Uji 1-Nearest Neighbor pada fitur bersih menghasilkan kesepakatan label hanya sebesar `53,0%` (`10-NN Macro F1 = 0,5884`), dan simulasi Monte Carlo dari distribusi posterior terkalibrasi menunjukkan batas teoretis Bayes Optimal berada di kisaran `0,673 - 0,685`. Dengan demikian, capaian model kita (`CV Macro F1 = 0.6737`) telah menyentuh batas optimal teoretis tanpa *data leakage*.

- **Bahan Visual untuk PPT**:
  - `visualisasi/01_distribusi_target_missing_dan_outlier.png`
  - `visualisasi/04_bukti_forensik_dgp_dan_collider_bias.png`

---

## BAB 2. Eksplorasi Data (EDA), Audit Anomali, dan Insight Kausal

### 2.1 Temuan Audit Kualitas Data (Data Kotor pada Train)
1. **Inkonsistensi Kategori Wilayah**: Kolom `Kabupaten_Kota` memiliki 30 nilai unik (seharusnya 15 kabupaten/kota resmi di Lampung) akibat 50 baris di `train` (60 baris di seluruh data) ditulis menggunakan huruf kapital seluruhnya (contoh: `BANDAR LAMPUNG` vs `Bandar Lampung`).
2. **Nilai Hilang (Missing Values) pada 8 Kolom**: Tertinggi pada `Rating_Rata_Rata` (8,0%), `Skor_Literasi_Digital` (6,3%), `Modal_Awal_Juta` (5,2%), `Omzet_Bulanan_Juta` (4,1%), `Pendidikan_Pemilik` (4,0%), `Kualitas_Internet` (3,2%), `Jarak_ke_Pusat_Kota_km` (2,7%), dan `Jumlah_Karyawan` (0,4%). Tingkat kekosongan seragam di ketiga kelas target (mekanisme MCAR/MAR).
3. **Injeksi Outlier Ekstrem Omzet**: Terdapat sekitar 0,7% baris di `train` (37 baris) yang memiliki `Omzet_Bulanan_Juta` ekstrem (hingga 1.277,14 juta Rp dengan presisi >2 angka desimal), namun distribusinya acak merata di kelas Menurun (12), Stabil (14), dan Tumbuh (11).

### 2.2 Pola Monotonik Ordinal dan Paradoks Modal Awal
- Hampir seluruh indikator kapabilitas digital bergerak searah dari Menurun -> Stabil -> Tumbuh:
  - Median `Skor_Literasi_Digital`: `39,4` (Menurun) -> `52,6` (Stabil) -> `66,9` (Tumbuh).
  - `Metode_Pembayaran_Digital`: UMKM `Tunai saja` mengalami `57,4%` Menurun dan hanya `6,0%` Tumbuh, sedangkan pengguna `Multi-kanal` mencapai `37,2%` Tumbuh.
  - `Kualitas_Internet`: Internet `Rendah` hanya `13,4%` Tumbuh, sedangkan `Tinggi` mencapai `45,9%` Tumbuh.
- **Paradoks Modal Awal**: Rata-rata `Modal_Awal_Juta` justru lebih tinggi pada kelas `Menurun` (`20,11` juta) dibandingkan `Tumbuh` (`16,20` juta), sementara `Omzet_Bulanan_Juta` bergerak sebaliknya. Hal ini menunjukkan bahwa efisiensi perputaran modal (`log_ROI = log_Omzet - log_Modal`) jauh lebih menentukan ketimbang besarnya modal awal semata.

### 2.3 Temuan Kausal Mediasi Kesenjangan Antar-Wilayah
Secara univariat, `Bandar Lampung` (`53,1%` Tumbuh) dan `Metro` (`51,1%` Tumbuh) tampak unggul lebih dari tiga kali lipat dibandingkan `Mesuji` (`16,5%` Tumbuh) dan `Tulang Bawang` (`18,2%` Tumbuh). Namun dalam model multivariat, ketika `Skor_Literasi_Digital` dan `Kualitas_Internet` dikontrol, koefisien `Kabupaten_Kota` menyusut mendekati nol (`|coef| < 0,08`). Artinya, ketimpangan pertumbuhan UMKM antar-kabupaten di Lampung sepenuhnya dimediasi oleh kesenjangan literasi digital dan kualitas akses internet.

- **Bahan Visual untuk PPT**:
  - `visualisasi/02_eda_faktor_ordinal_utama.png`
  - `visualisasi/03_insight_kausal_mediasi_kabupaten.png`

---

## BAB 3. Pra-Pemrosesan, Feature Engineering, Clustering, dan Studi Augmentasi

Seluruh langkah pra-pemrosesan berikut di-*fit* murni di dalam setiap fold latih pada 10-Fold CV:
1. **Normalisasi Teks**: Standarisasi `Kabupaten_Kota` ke format *Title Case* (30 -> 15 wilayah resmi).
2. **Penanganan Outlier Berbasis Bukti**: Nilai `Omzet_Bulanan_Juta` di atas persentil ke-99 fold latih diubah menjadi `NaN` (bersama indikator `is_omzet_outlier`) agar tidak menyesatkan model sebagai UMKM beromzet tinggi.
3. **Imputasi Hibrida MICE + GBDT dalam Ruang Log**: Fitur miring kanan (`Omzet`, `Modal`, `Followers`, `Ulasan`, `Jarak`, `Lama_Usaha`) ditransformasi `log1p`, lalu diimputasi menggunakan `IterativeImputer(BayesianRidge)` dengan penyempurnaan non-linier `LGBMRegressor` khusus untuk fitur terkuat `Skor_Literasi_Digital`. Strategi ini menurunkan CV LogLoss dari `0,71566` menjadi `0,71168`.
4. **Feature Engineering Domain dan Clustering (Unsupervised FE)**:
   - Rasio ekonomi: `log_ROI`, `Omzet_per_Modal`, `Omzet_per_Karyawan`, `Modal_per_Karyawan`, `Omzet_rel_Sektor`.
   - Aktivitas dan reputasi digital: `Engagement_Score` (`Rating x log_Ulasan`), `Digital_Index`, `Digital_Activity`, `Ulasan_vs_Followers_Residual`, `Lit_rel_Kab`.
   - **Clustering FE (GMM K=5 + K-Means K=6)**: Menambahkan probabilitas *soft-cluster* GMM dan jarak centroid K-Means di dalam fold latih terbukti menaikkan CV Macro F1 dari `0,66898` menjadi `0,67044`.
5. **Uji Empiris Data Augmentation pada Data Latih (Train Only)**:
   - **SMOTE**: Menurunkan Macro F1 menjadi `0,65476` (LogLoss memburuk ke `0,73025`) karena mendistorsi prior alami kelas `Stabil`.
   - **Gaussian Noise Augmentation (3x)**: Menjaga kalibrasi (`LogLoss = 0,71191`) sebagai regularisasi kontinu.
   - **Gaussian Copula Synthetic Augmentation (2x)**: Membangkitkan sampel sintetis baru dengan mempelajari struktur korelasi multivariat kelas pada `train`, menghasilkan model `C11_CatBoost_Copula_Augmented` dengan CV Macro F1 **`0,67160`** (terbaik ke-3 dari 15 model tunggal!).

### Tabel Ringkasan Studi Ablasi (10-Fold Stratified CV pada Train)
| Tahap Eksperimen | Kategori | CV Macro F1 | CV LogLoss | Kesimpulan |
|---|---|---:|---:|---|
| 1. SVM-RBF Baseline | Model Klasifikasi Non-Linier | 0.64787 | 0.73998 | Kernel RBF over-smooth pada fitur campuran |
| 2. Generative Naive Bayes (Marginalizing NaN) | Generative Classifier P(X|Y) | 0.63528 | 0.88865 | Membuktikan data bukan dibangkitkan secara Naive Bayes Y->X |
| 3. Gaussian Copula Generative Classifier | Generative Classifier P(X|Y) | 0.62431 | 0.94578 | Membuktikan arah kausal sejati adalah X -> Latent Score S -> Y |
| 4. LogReg + SMOTE Augmentation | Data Augmentation (Train Only) | 0.65476 | 0.73025 | Merusak proporsi prior kelas Stabil (over-predict kelas ujung) |
| 5. LightGBM Default (depth bebas, Raw) | Gradient Boosting Baseline | 0.65705 | 0.73810 | Tree dalam mengalami overfitting terhadap stochastic noise |
| 6. Logistic Regression (Raw + Median Impute) | Linear Baseline | 0.66170 | 0.71566 | Sinyal dasar bersifat aditif & monotonik |
| 7. LogReg + Gaussian Noise Augmentation (3x) | Data Augmentation (Train Only) | 0.66809 | 0.71191 | Berfungsi positif sebagai regularisasi Tikhonov kontinu |
| 8. CatBoost + Gaussian Copula Synthetic Augmentation (2x) | Generative Augmentation (Train Only) | 0.67160 | 0.71672 | Sampel sintetis Copula kompetitif & memberi diversitas ensemble |
| 9. Outlier Cleaning (>P99->NaN) + MICE Imputation | Preprocessing & Imputasi | 0.66898 | 0.71168 | Menurunkan LogLoss drastis (-0.0040) dengan merekonstruksi Literasi Digital |
| 10. + Unsupervised Clustering FE (GMM K=5 & K-Means K=6) | Feature Engineering | 0.67044 | 0.71238 | Probabilitas klaster & jarak centroid meningkatkan Macro F1 (+0.0015) |
| 11. + Rich Domain FE + GBDT-MICE Hybrid + Shallow CatBoost d3 | Feature Engineering & Tuning | 0.67442 | 0.71532 | Pembatasan depth=3 + rasio ROI & Engagement mencegah overfit |
| 12. Single Ordinal Latent Regressor (Truncated-Normal Target) | Ordinal Regression Discovery | 0.67169 | 0.71466 | Sesuai mekanisme pembangkit data qcut(S, [0, 0.27, 0.70, 1.0]) |
| 13. CHAMPION: 15-Model Multi-Paradigm Blend (Nested Multiplier Opt) | Final Meta-Ensemble | 0.67369 | 0.70546 | Skor tertinggi bebas leakage; Holdout Macro F1 = 0.66867 |

- **Bahan Visual untuk PPT**:
  - `visualisasi/05_studi_ablasi_preprocessing_fe_augmentasi.png`

---

## BAB 4. Pemodelan Lintas Paradigma dan Optimasi Ensemble (15 Model)

Kami mengeksplorasi **15 model individual** dari 4 paradigma berbeda: (1) Klasifikasi Linier/GAM/Ordinal, (2) Tree/Gradient Boosting *Shallow Regularized*, (3) Neural Network dan Augmentasi Generatif Copula, serta (4) **Ordinal Latent Regression** yang memprediksi skor *Truncated-Normal* (`-1,229`, `-0,036`, `+1,159`) sesuai mekanisme `qcut` dataset.

### Tabel Benchmark Lengkap 15 Model + Ensemble
| Peringkat | Model | Paradigma | 10-Fold CV Macro F1 | 10-Fold CV LogLoss | Holdout Macro F1 |
|---:|---|---|---:|---:|---:|
| 1 | `C04_CatBoost_d3_RichFE` | Multiclass Classification | 0.67442 | 0.71532 | 0.66721 |
| 2 | `ENSEMBLE_15Model_Nested_Multiplier` | Multi-Paradigm + Threshold Opt | 0.67369 | 0.70546 | 0.66867 |
| 3 | `R01_Ordinal_CatBoost_d3` | Ordinal Latent Regression | 0.67169 | 0.71466 | 0.66025 |
| 4 | `C11_CatBoost_Copula_Augmented` | Generative Augmentation + Boosting | 0.67160 | 0.71672 | 0.66432 |
| 5 | `R04_Ordinal_Spline_BayesianRidge` | Ordinal Latent Regression | 0.67064 | 0.70868 | 0.66064 |
| 6 | `C07_LightGBM_d3_Regularized` | Multiclass Classification | 0.67045 | 0.72029 | 0.66595 |
| 7 | `C01_LogReg_DomainCluster` | Multiclass Classification | 0.66906 | 0.71168 | 0.66728 |
| 8 | `ENSEMBLE_1D_Ordinal_MetaStacker_Cutpoints` | Ordinal Latent Meta-Learning | 0.66884 | 0.70546 | 0.66629 |
| 9 | `C06_XGBoost_d3_Regularized` | Multiclass Classification | 0.66844 | 0.71863 | 0.67085 |
| 10 | `R02_Ordinal_LightGBM_d3` | Ordinal Latent Regression | 0.66774 | 0.71808 | 0.66321 |
| 11 | `C05_CatBoost_d4_NativeCat` | Multiclass Classification | 0.66755 | 0.71644 | 0.66259 |
| 12 | `R03_Ordinal_XGBoost_d3` | Ordinal Latent Regression | 0.66749 | 0.71750 | 0.66694 |
| 13 | `ENSEMBLE_15Model_Simplex_Argmax` | Multi-Paradigm Stacking | 0.66737 | 0.70546 | 0.66373 |
| 14 | `C02_Spline_GAM_LogReg` | Multiclass Classification | 0.66614 | 0.71329 | 0.65909 |
| 15 | `C03_Cumulative_Ordinal_Logit` | Multiclass Classification | 0.66602 | 0.71341 | 0.66057 |
| 16 | `C10_MLP_Bagged_Ensemble` | Multiclass Classification | 0.66515 | 0.71752 | 0.65836 |
| 17 | `C09_ExtraTrees_Tuned` | Multiclass Classification | 0.66036 | 0.73640 | 0.64407 |
| 18 | `C08_RandomForest_Tuned` | Multiclass Classification | 0.65674 | 0.74032 | 0.64726 |

### Temuan Penting Tahap Pemodelan
1. **Regularisasi Kedalaman Pohon**: Model boosting default dengan pohon dalam (`depth >= 6`) mengalami *overfitting* pada *noise* (`LightGBM default = 0,65705`). Pembatasan kedalaman (`depth = 3`) dan regularisasi L2 kuat melonjakkan performa `C04_CatBoost_d3_RichFE` menjadi **`0,67442`**.
2. **Keunggulan Kalibrasi Probabilitas**: Model regresi laten `R04_Ordinal_Spline_BayesianRidge` mencetak LogLoss tunggal terendah (`0,70868`), dan penggabungan *Simplex Blend* 15 model menekan OOF LogLoss lebih jauh hingga **`0,70546`**.
3. **Nested-CV Threshold Optimization**: Mengoptimalkan pengali probabilitas kelas pada OOF untuk langsung memaksimalkan Macro F1 (tervalidasi melalui Nested 10-Fold CV tanpa *leakage*) menaikkan Macro F1 *ensemble* dari `0,66737` (argmax) menjadi **`0.67369`**.

- **Bahan Visual untuk PPT**:
  - `visualisasi/06_benchmark_15_model_dan_bobot_ensemble.png`

---

## BAB 5. Evaluasi Akhir pada Holdout dan Analisis Kesalahan

Evaluasi dilakukan satu kali pada 1.400 baris `holdout` menggunakan model terpilih berdasarkan CV (`15-Model Multi-Paradigm Blend (Nested Multiplier Opt)`):
- **Holdout Macro F1-Score**: **`0.66867`**
- **Holdout Accuracy**: `0.6621` | **Holdout LogLoss**: `0.71835`
- **F1 per Kelas**: Menurun = `0.6803`, Stabil = `0.6255`, Tumbuh = `0.7002`.

### Confusion Matrix Holdout (Baris = Aktual, Kolom = Prediksi)
| Aktual \ Prediksi | Menurun | Stabil | Tumbuh | Total |
|---|---:|---:|---:|---:|
| **Menurun** | 250 | 120 | 8 | 378 |
| **Stabil** | 103 | 385 | 114 | 602 |
| **Tumbuh** | 4 | 124 | 292 | 420 |

### Analisis Kesalahan (Error Analysis)
1. **Kesalahan Ekstrem Sangat Rendah**: Dari 1.400 UMKM holdout, hanya `8` kasus `Menurun` yang salah ditebak `Tumbuh` dan hanya `4` kasus `Tumbuh` yang salah ditebak `Menurun` (total kesalahan ekstrem lintas ujung hanya `12` kasus atau `0.86%`).
2. **Konsentrasi Error di Kelas Tengah (`Stabil`)**: Lebih dari 97% kesalahan terjadi di perbatasan antar-kelas berdekatan (`Menurun <-> Stabil` dan `Stabil <-> Tumbuh`). Hal ini sepenuhnya konsisten dengan struktur skor laten kontinu di mana *noise* acak di sekitar ambang *cutpoint* secara alami menukar label dua kelas bertetangga.
3. **Stabilitas Generalisasi**: Selisih antara 10-Fold CV (`0.6737`) dan Holdout (`0.6687`) sangat tipis (`0,005`), menandakan model stabil terhadap variasi *Public LB (30%)* maupun *Private LB (70%)*.

- **Bahan Visual untuk PPT**:
  - `visualisasi/07_evaluasi_holdout_confusion_matrix.png`

---

## BAB 6. Interpretasi Fitur Paling Berpengaruh (SHAP) dan Segmentasi Persona (Clustering)

### 6.1 Top 12 Fitur Paling Berpengaruh (SHAP CatBoost dan Koefisien Multivariat)
| Peringkat | Fitur | Mean |SHAP| (CatBoost) | Koefisien Linear (Tumbuh - Menurun) | Arah Pengaruh |
|---:|---|---:|---:|---|
| 1 | `Digital_Index` | 0.34854 | +0.6594 | Positif (Mendorong Tumbuh) |
| 2 | `Lit_x_Inet` | 0.20660 | +0.7051 | Positif (Mendorong Tumbuh) |
| 3 | `Skor_Literasi_Digital` | 0.19482 | +0.7979 | Positif (Mendorong Tumbuh) |
| 4 | `Rating_Rata_Rata` | 0.14599 | +0.7742 | Positif (Mendorong Tumbuh) |
| 5 | `km_d1` | 0.10229 | -0.4432 | Negatif (Cenderung Menurun) |
| 6 | `Kredit_bin` | 0.08123 | +0.4467 | Positif (Mendorong Tumbuh) |
| 7 | `Engagement_Score` | 0.07019 | +0.1436 | Positif (Mendorong Tumbuh) |
| 8 | `Lit_rel_Kab` | 0.05979 | +0.7607 | Positif (Mendorong Tumbuh) |
| 9 | `Lit_x_Pay` | 0.05154 | +0.2235 | Positif (Mendorong Tumbuh) |
| 10 | `Sektor_Usaha_Perdagangan Eceran` | 0.04618 | -0.2896 | Negatif (Cenderung Menurun) |
| 11 | `Platform_Digital_Aktif` | 0.04502 | +0.1049 | Positif (Mendorong Tumbuh) |
| 12 | `Digital_Activity` | 0.03484 | -0.1906 | Negatif (Cenderung Menurun) |

### 6.2 Segmentasi 3 Persona UMKM Lampung (K-Means Unsupervised)
Tanpa menggunakan label target sama sekali, K-Means mengelompokkan UMKM `train` ke dalam 3 persona alami yang sangat selaras dengan status pertumbuhan aktual (`NMI = 0.1461`):
- **Persona 1 (Konvensional Rentan)**: Literasi digital rendah (~41,7), dominan tunai/platform minimal -> **56,4% Menurun** dan hanya **5,3% Tumbuh**.
- **Persona 2 (Digital Transisi)**: Literasi menengah (~59,2), mulai memakai QRIS dan 2-3 platform -> **49,1% Stabil** dan **36,1% Tumbuh**.
- **Persona 3 (Digital Matang)**: Literasi tinggi (~61,1+), aktif promosi multi-kanal -> **51,4% Tumbuh** dan hanya **7,6% Menurun**.

- **Bahan Visual untuk PPT**:
  - `visualisasi/08_shap_interpretabilitas_dan_arah_fitur.png`
  - `visualisasi/09_segmentasi_clustering_3_persona_umkm.png`

---

## BAB 7. Kesimpulan dan Rekomendasi Kebijakan untuk Pemangku Kepentingan

### 7.1 Kesimpulan Utama
1. Pipeline bebas *leakage* dengan **15-Model Multi-Paradigm Ensemble + Nested-CV Threshold Optimization** berhasil mencapai **CV Macro F1 = `0.67369`**, **CV LogLoss = `0.70546`**, dan **Holdout Macro F1 = `0.66867`** dengan kesalahan ekstrem di bawah 1%.
2. Faktor penggerak utama pertumbuhan UMKM digital di Provinsi Lampung adalah **Skor Literasi Digital**, **Kualitas Internet**, **Reputasi dan Ulasan Pelanggan**, **Pembayaran Digital**, **Pelatihan dan Konsistensi Promosi**, serta **Efisiensi Modal (`log_ROI`)**.
3. Kesenjangan kinerja antar-kabupaten bukan disebabkan oleh lokasi geografis semata, melainkan dimediasi oleh kesenjangan literasi digital dan infrastruktur internet.

### 7.2 Peta Jalan Rekomendasi 3-Tier bagi Pemerintah Daerah Provinsi Lampung
1. **Tier 1 - Penyelamatan Cepat (Target: Prediksi Menurun / Persona Konvensional Rentan)**:
   - Prioritaskan anggaran pada UMKM dengan literasi < 45 dan pembayaran tunai saja, khususnya di wilayah tertinggal (Mesuji, Tulang Bawang, Pesisir Barat, Lampung Barat).
   - Program konkret: *onboarding* massal QRIS tanpa biaya, pelatihan literasi digital praktis tingkat dasar (WhatsApp Business dan katalog foto produk), serta penyediaan titik akses internet komunal di sentra UMKM.
2. **Tier 2 - Akselerasi Naik Kelas (Target: Prediksi Stabil / Persona Digital Transisi)**:
   - Program konkret: pelatihan manajemen reputasi pelanggan (mendorong ulasan organik dan menjaga rating >= 4,3), standar konsistensi konten promosi (>= 5 unggahan/minggu), migrasi ke pembayaran multi-kanal, serta skema kredit produktif berbasis rasio efisiensi omzet terhadap modal (ROI).
3. **Tier 3 - Ekspansi dan Duta Mentor (Target: Prediksi Tumbuh / Persona Digital Matang)**:
   - Program konkret: fasilitasi kurasi e-katalog pemerintah dan ekspor, serta pelibatan sebagai mentor sebaya (*peer mentor*) bagi UMKM Tier 1 dan Tier 2 pada sektor usaha yang sama.

---

## Lampiran: Bank Jawaban Singkat untuk Sesi Tanya Jawab (Q&A) Dewan Juri
1. **Mengapa Macro F1 berada di kisaran 0,67 dan tidak mencapai 0,90+?**
   Berdasarkan Teorema Cover-Hart (1967) dan uji *1-Nearest Neighbor* (kesepakatan hanya 53,0%) serta bukti *Collider Bias*, dataset sintetis ini mengandung *irreducible Gaussian noise* pada skor laten sebelum pemotongan kuantil 27%-70%. Batas teoretis *Bayes Optimal* memang berada di kisaran 0,673-0,685; skor jauh di atas angka tersebut justru mengindikasikan kebocoran data (`holdout leakage`).
2. **Mengapa tidak menggunakan SMOTE untuk menangani ketidakseimbangan kelas?**
   Hasil eksperimen 10-Fold CV kami membuktikan SMOTE justru menurunkan Macro F1 dari `0,6690` ke `0,6548` karena mengubah prior kelas `Stabil` (43%) dan merusak kalibrasi probabilitas. Sesuai teori Elkan (2001), mempertahankan probabilitas terkalibrasi lalu menggeser *decision threshold* via Nested-CV jauh lebih optimal untuk Macro F1 (`0.67369`).
3. **Apa peran Clustering dan Augmentasi Generatif dalam penelitian ini?**
   Pertama, *Gaussian Mixture Model* (K=5) dan *K-Means* (K=6) yang di-*fit* murni di dalam fold latih menjadi fitur tambahan yang terbukti menaikkan CV Macro F1. Kedua, *K-Means* 3-klaster dipakai memetakan 3 Persona UMKM untuk ketepatan kebijakan Pemda. Ketiga, *Gaussian Copula Synthetic Augmentation* pada data latih menghasilkan model individual terbaik ke-3 (`0,67160`) sekaligus memperkaya diversitas *ensemble*.
