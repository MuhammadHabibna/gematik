# Script Presentasi Final GEMATIK VI 2026 (Versi Ringkas: ~6 Menit 15 Detik)
**Tim**: Atas Nama (Universitas Negeri Surabaya)  
**Pembagian Pembicara**:
- **HABIB**: Slide 01 - 02, lalu Slide 07 - 14
- **MELYSA**: Slide 03 - 06

> **Catatan Tempo**: Total ~700 kata (~28% lebih ringkas). Diucapkan tenang dan tegas (~115 kata/menit + jeda ganti pembicara), selesai nyaman di **06:15 - 06:45** (jauh aman dari batas **08:00**).

---

## BAGIAN 1: HABIB (Slide 01 - 02 | 00:00 - 00:45)

### Slide 01 | 00:00 - 00:20 (20 detik) - Cover
Assalamualaikum warahmatullahi wabarakatuh, selamat sore Dewan Juri. Kami dari Tim **Atas Nama**, UNESA - saya **Muhammad Habib Nur Aiman** dan rekan saya **Melysa Dwyta Sary**.  
Kami mempresentasikan sistem prediksi pertumbuhan 7.000 UMKM Lampung yang meraih **CV Macro F1 0,6737** dan **Holdout 0,6687**, dengan kesalahan ekstrem hanya **0,86%** tanpa *data leakage*.

### Slide 02 | 00:20 - 00:45 (25 detik) - Masalah & Protokol Data
Pemda Lampung perlu memprioritaskan bantuan bagi tiga status usaha: **Menurun (27%)**, **Stabil (43%)**, dan **Tumbuh (30%)**. Dengan metrik **Macro F1**, kelas minoritas *Menurun* bernilai sama pentingnya dengan *Stabil*.  
Sesuai protokol split `random_state=42`, kami memakai **5.600 Train** untuk riset dan mengunci rapat **1.400 Holdout** (cocok 100% dengan `sample_submission.csv`) hingga evaluasi akhir.  
Selanjutnya dipaparkan oleh Melysa.

---

## BAGIAN 2: MELYSA (Slide 03 - 06 | 00:45 - 02:35)

### Slide 03 | 00:45 - 01:10 (25 detik) - Audit Data Kotor
Terima kasih Habib. Pada data Train, kami menemukan tiga data kotor: `Kabupaten_Kota` terpecah menjadi 30 kategori karena 50 baris kapital semua (dinormalkan ke 15 wilayah), 8 kolom memiliki *missing* hingga 7,9% di `Rating`, serta outlier ekstrem `Omzet` di atas 84,2 juta yang tersebar acak di tiga kelas. Nilai di atas persentil ke-99 kami ubah ke `NaN` lalu direkonstruksi dengan **Hybrid MICE**.

### Slide 04 | 01:10 - 01:35 (25 detik) - Tiga Ide Solusi
Solusi kami bertumpu pada tiga ide: pertama, bersihkan anomali, imputasi MICE plus GBDT, dan buat fitur `log ROI`. Kedua, tangkap pola ordinal lewat *Ordinal Latent Regression*, fitur klaster GMM dan K-Means, serta *Gaussian Copula*. Ketiga, gabungkan 15 model lintas 4 paradigma dengan penalaan *threshold* via *Nested-CV*.

### Slide 05 | 01:35 - 02:05 (30 detik) - EDA & Paradoks Modal
Pada grafik EDA, median `Literasi Digital` melonjak dari **39,4** (Menurun) ke **66,9** (Tumbuh), dan UMKM **Tunai saja** mengalami **57% Menurun**. Menariknya, rata-rata modal awal kelas Menurun (**20,1 juta**) justru lebih tinggi dibanding Tumbuh (**16,2 juta**). Artinya, modal besar tidak menjamin tumbuh; efisiensi perputaran modal (`log ROI`) jauh lebih menentukan.

### Slide 06 | 02:05 - 02:35 (30 detik) - Temuan Kausal Mediasi
Secara univariat, **Bandar Lampung (53,1% Tumbuh)** tampak unggul tiga kali lipat dari **Mesuji (16,5%)**. Namun antar-wilayah, literasi digital dan persentase Tumbuh berkorelasi sangat kuat (**r = 0,98**). Di model multivariat, setelah literasi dan internet dikontrol, efek lokasi menyusut mendekati nol. Jadi ketimpangan wilayah disebabkan kesenjangan literasi dan internet, bukan faktor geografis.  
Selanjutnya dilanjutkan oleh Habib.

---

## BAGIAN 3: HABIB (Slide 07 - 14 | 02:35 - 06:15)

### Slide 07 | 02:35 - 03:05 (30 detik) - Tangga Ablasi
Terima kasih Melysa. Semua teknik kami uji di **10-Fold CV**. Kami menolak **SMOTE** karena menurunkan skor ke **0,6548** dan merusak prior kelas Stabil. Sebaliknya, Hybrid MICE (**0,6690**), fitur klaster GMM dan K-Means (**0,6704**), *Copula Augmentation* (**0,6716**), hingga **Ensemble 15 Model plus Nested-CV Thresholding** menembus **0,6737** - melampaui baseline teoretis Bayes di **0,6731**.

### Slide 08 | 03:05 - 03:30 (25 detik) - Benchmark 15 Model & Ensemble
Pohon dalam seperti Random Forest (**0,6567**) mudah *overfit noise*. Membatasi kedalaman (`depth = 3`) dengan regularisasi L2 menaikkan `CatBoost_d3_RichFE` ke **0,6744**. Lewat *Simplex Blend* dari 9 model terbaik lintas paradigma, **OOF LogLoss turun hingga 0,7055** dengan **CV Macro F1 0,6737**.

### Slide 09 | 03:30 - 04:00 (30 detik) - Evaluasi Holdout & Batas Jujur
Pada **1.400 data Holdout**, model meraih **Macro F1 0,6687** - konsisten dengan CV **0,6737** tanpa *overfitting*, kuat di **Tumbuh (0,700)** dan **Menurun (0,680)**. Kesalahan fatal Menurun ke Tumbuh hanya **12 kasus (0,86%)**; sebanyak **97,5%** error wajar terjadi di perbatasan kelas tengah **Stabil** akibat *noise* kontinu data sintetis.

### Slide 10 | 04:00 - 04:30 (30 detik) - SHAP & Clustering Persona
Analisis **SHAP** mengonfirmasi penggerak utama pertumbuhan adalah `Digital_Index`, interaksi literasi dengan internet, `Rating`, akses kredit, dan `Engagement_Score`. Sejalan dengan itu, **K-Means Clustering** tanpa label membagi UMKM menjadi tiga persona yang sangat selaras dengan target aktual: **Konvensional Rentan** mengalami **56,4% Menurun**, sedangkan **Digital Matang** mencapai **51,4% Tumbuh**.

### Slide 11 | 04:30 - 05:05 (35 detik) - Rekomendasi 3-Tier Pemda
Kami merumuskan tiga paket aksi bagi Dinas Koperasi dan UKM Lampung:
- **Tier 1 Penyelamatan (357 UMKM Menurun)**: *onboarding* QRIS massal, bootcamp literasi dasar, dan internet komunal di Mesuji, Tulang Bawang, serta Pesisir Barat.
- **Tier 2 Akselerasi (629 UMKM Stabil)**: jaga rating minimal 4,3, posting minimal 5 kali seminggu, dan kredit berbasis ROI.
- **Tier 3 Ekspansi (414 UMKM Tumbuh)**: kurasi E-Katalog dan jadikan *peer mentor*.

### Slide 12 | 05:05 - 05:35 (30 detik) - Live Web App
Agar siap dipakai di lapangan, kami membangun dasbor operasional **SiTumbuh Lampung** yang sudah *live* di **`situmbuh-lampung.vercel.app`**. Petugas Dinas langsung mendapat peta kerentanan kabupaten, daftar prioritas plus rekomendasi aksi otomatis, ekspor tugas `.csv`, serta **Simulator What-If** untuk menguji dampak program sebelum dana dicairkan.

### Slide 13 & Slide 14 | 05:35 - 06:15 (40 detik) - Sintesis & Penutup
Sebagai penutup, solusi Tim Atas Nama menjamin kode bebas *leakage*, performa **Holdout 0,6687** yang jujur mendekati batas optimal Bayes, serta aplikasi web siap pakai bagi Pemda Lampung. Seluruh kode terbuka di tautan GitHub pada layar.  
Terima kasih Dewan Juri, wassalamualaikum warahmatullahi wabarakatuh. Kami siap ke sesi tanya jawab.

---

## Pintasan Q&A (Ketik Nomor Slide + Enter)
- *"Kenapa skor ~0,67 bukan 0,90+?"* -> Slide **`16`** (`L01`): *Collider Bias* & *1-NN Bayes Bound*.
- *"Detail skor 15 model?"* -> Slide **`17`** (`L02`).
- *"Kenapa SMOTE ditolak?"* -> Slide **`18`** (`L03`).
- *"Detail fitur & hyperparameter?"* -> Slide **`19`** (`L04`).
- *"Bukti bebas data leakage?"* -> Slide **`20`** (`L05`).
