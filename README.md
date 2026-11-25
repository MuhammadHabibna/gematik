# GEMATIK VI 2026 - Tim Atas Nama

**Studi Kasus**: Prediksi Status Pertumbuhan UMKM Digital di Provinsi Lampung  
**Metrik Evaluasi**: Macro F1-Score (`Menurun`, `Stabil`, `Tumbuh`)

## Hasil Utama (Bebas Data Leakage)
- **Protokol Split Wajib**: `train_test_split(df, test_size=0.2, stratify=df["Status_Pertumbuhan"], random_state=42)` (5.600 `train`, 1.400 `holdout`).
- **Model**: 15-Model Multi-Paradigm Ensemble (Linear/Spline GAM, Cumulative Ordinal Logit, Shallow Regularized CatBoost/XGBoost/LightGBM, Gaussian Copula Synthetic Augmentation, Bagged MLP, & Ordinal Latent Regressors) + Nested-CV Macro-F1 Threshold Optimization.
- **10-Fold Stratified CV Macro F1**: `0,67369` (LogLoss: `0,70546`)
- **Holdout Macro F1 Final**: `0,66867` (Kesalahan ekstrem `Menurun <-> Tumbuh` hanya 12 dari 1.400 UMKM / `0,86%`).

## Struktur Berkas
- `web_implementasi_situmbuh/` : Aplikasi web operasional **SiTumbuh Lampung** (POV Penerapan Dinas Koperasi & UKM: Antrean Prioritas UMKM, Peta Kerentanan Wilayah, Rekomendasi Aksi Otomatis, Ekspor CSV Tugas Lapangan, dan Simulator Dampak Intervensi *What-If*).
- `Atas Nama.csv` : File submission resmi (1.400 baris Holdout).
- `Atas_Nama_Notebook_Final.ipynb` : Notebook laporan lengkap terstruktur (BAB 1-7 + seluruh kode & analisis output).
- `Atas_Nama_Pipeline_Final.py` : Pipeline reproduksibel 15 model lengkap.
- `Atas_Nama_Presentasi_Final.pptx` : Slide presentasi Babak Final (8 menit + Lampiran Q&A).
- `HASIL_PENELITIAN_ATAS_NAMA/` : Seluruh tabel CSV eksperimen, 9 visualisasi 300 DPI, dan dokumen bahan presentasi.
