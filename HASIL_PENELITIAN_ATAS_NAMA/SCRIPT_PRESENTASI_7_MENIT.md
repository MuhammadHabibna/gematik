# Script Presentasi Final GEMATIK VI 2026 (Alur Narasi Slide PPT)
**Tim**: Atas Nama (Universitas Negeri Surabaya)  
**Pembagian Pembicara**:
- **HABIB**: Slide 01 - 02, lalu Slide 07 - 14
- **MELYSA**: Slide 03 - 06

> **Panduan Tempo**: ~860 kata ucapan murni. Dibawakan tenang dan mengalir (~125-130 kata/menit + jeda ganti pembicara/slide), selesai nyaman di **06:40 - 07:05** (aman di bawah batas maksimal **08:00**). Setiap bagian langsung menuntun mata juri melihat elemen di slide (kiri -> tengah -> kanan).

---

## BAGIAN 1: HABIB (Slide 01 - 02 | 00:00 - 01:05)

### Slide 01 | 00:00 - 00:30 - Cover
Assalamualaikum warahmatullahi wabarakatuh, selamat siang Dewan Juri. Kami Tim **Atas Nama** dari Universitas Negeri Surabaya: saya **Muhammad Habib Nur Aiman**, bersama rekan saya **Melysa Dwyta Sary**.

Kami mempresentasikan sistem prediksi pertumbuhan **7.000 UMKM Provinsi Lampung**, dari profil digital hingga keputusan pendampingan Pemda. Di tiga kartu bawah layar, model kami meraih **CV Macro F1 0,6737**, **Holdout 0,6687**, dengan kesalahan fatal hanya **0,86 persen** tanpa kebocoran data.

### Slide 02 | 00:30 - 01:05 - Tantangan & Protokol Data
Mengapa ini krusial? Di kotak kiri terlihat Pemda Lampung harus membagi bantuan ke tiga kelas berurutan: **Menurun 27 persen**, **Stabil 43 persen**, dan **Tumbuh 30 persen**. Karena metriknya **Macro F1**, kelas minoritas Menurun dinilai sama pentingnya dengan Stabil agar UMKM yang hampir jatuh tidak kehilangan bantuan.

Di kotak kanan, sesuai aturan split `random_state=42`, kami memakai **5.600 Train** untuk riset dan mengunci **1.400 Holdout** (cocok `sample_submission.csv`) hingga uji akhir. Selanjutnya dipaparkan oleh Melysa.

---

## BAGIAN 2: MELYSA (Slide 03 - 06 | 01:05 - 03:10)

### Slide 03 | 01:05 - 01:40 - Audit Data Kotor
Terima kasih, Habib. Saat mengaudit 5.600 data Train, kami menemukan tiga masalah data kotor pada slide ini. Pertama, kolom kabupaten terpecah menjadi 30 kategori akibat huruf kapital semua, sehingga kami rapikan ke 15 wilayah resmi. Kedua, di grafik batang tengah ada delapan kolom kosong, tertinggi di Rating **7,9 persen**. Ketiga, di boxplot kanan ada outlier omzet di atas **84,2 juta** yang muncul acak di semua kelas. Nilai ekstrem itu kami jadikan kosong lalu direkonstruksi dengan **Hybrid MICE**.

### Slide 04 | 01:40 - 02:10 - Tiga Ide Solusi
Dari audit tersebut, solusi kami bertumpu pada tiga ide di slide keempat. Kartu pertama: bersihkan anomali dan bentuk rasio efisiensi modal, `log ROI`. Kartu kedua: tangkap urutan target lewat *Ordinal Latent Regression*, ditambah fitur klaster GMM dan K-Means serta data sintetis *Gaussian Copula*. Kartu ketiga: gabungkan 15 model lintas empat paradigma lalu setel *threshold* lewat Nested-CV agar kelas minoritas tidak kalah suara.

### Slide 05 | 02:10 - 02:40 - EDA & Paradoks Modal
Selanjutnya pada tiga grafik EDA ini, kita lihat pembeda utamanya. Di grafik kiri, median literasi digital melonjak dari **39,4** di kelas Menurun ke **66,9** di kelas Tumbuh. Di grafik tengah, UMKM **tunai saja** mengalami **57 persen** penurunan. Menariknya di grafik kanan, rata-rata modal awal kelas Menurun (**20,1 juta**) justru lebih besar dibanding Tumbuh (**16,2 juta**). Artinya, modal besar tidak menjamin tumbuh tanpa perputaran omzet yang efisien.

### Slide 06 | 02:40 - 03:10 - Temuan Kausal Mediasi
Temuan kuncinya ada pada perbandingan wilayah di slide keenam. Sekilas di grafik kiri, Bandar Lampung (**53,1 persen**) unggul tiga kali lipat atas Mesuji (**16,5 persen**). Namun di grafik kanan, pertumbuhan 15 wilayah membentuk garis lurus terhadap literasi digitalnya, **r = 0,98**. Setelah literasi dan internet dikontrol secara multivariat, efek lokasi menyusut ke nol. Jadi pemicunya bukan geografis, melainkan kesenjangan digital yang bisa diintervensi. Berikutnya dilanjutkan oleh Habib.

---

## BAGIAN 3: HABIB (Slide 07 - 14 | 03:10 - 06:50)

### Slide 07 | 03:10 - 03:40 - Tangga Ablasi
Terima kasih, Melysa. Semua ide tadi kami uji bertahap di **10-Fold CV** seperti tangga pada slide ketujuh. Dari batang merah terbawah, **SMOTE** kami tolak karena menurunkan skor ke **0,6548** dan merusak prior kelas Stabil, sementara LightGBM standar mengalami *overfit*. Sebaliknya, dari baseline **0,6617**, tahap MICE, klastering, augmentasi Copula, hingga Ensemble 15 Model naik konsisten ke **0,6737** dan melampaui garis batas teoretis Bayes.

### Slide 08 | 03:40 - 04:10 - Benchmark 15 Model & Ensemble
Slide kedelapan membedah 15 model yang diuji. Di grafik kiri, pohon yang terlalu dalam seperti Random Forest berada paling bawah (**0,6567**) karena menghafal *noise*, sedangkan pembatasan `depth=3` menaikkan CatBoost tunggal ke **0,6744**. Lalu di grafik kanan, kami padukan 9 model paling komplementer via *Simplex Blend*, didominasi regresi ordinal spline (**38,8 persen**), model linier, dan boosting ringan. Paduan ini menekan LogLoss ke **0,7055** dengan CV F1 **0,6737**.

### Slide 09 | 04:10 - 04:40 - Evaluasi Holdout & Batas Jujur
Saat diuji satu kali pada **1.400 data Holdout** di slide kesembilan, skornya konsisten di **0,6687** (berselisih 0,005 dari CV) tanpa *overfitting*. Pada *confusion matrix* kiri, garis diagonal sangat kuat dan kesalahan ekstrem Menurun-Tumbuh hanya **12 kasus (0,86 persen)**. Di grafik kanan, F1 terbaik diraih pada kelas kritis **Tumbuh (0,700)** dan **Menurun (0,680)**, sementara **97,5 persen** error wajar berada di batas kelas tengah Stabil.

### Slide 10 | 04:40 - 05:05 - SHAP & Persona
Agar transparan bagi Pemda, grafik SHAP kiri di slide kesepuluh menunjukkan penggerak utama model adalah indeks digital, interaksi literasi-internet, rating, jarak K-Means, kredit, dan *engagement*. Sejalan di grafik kanan, klastering K-Means tanpa label otomatis memisahkan tiga persona nyata: **Konvensional Rentan** mengalami **56,4 persen Menurun**, sedangkan **Digital Matang** mencapai **51,4 persen Tumbuh**.

### Slide 11 | 05:05 - 05:45 - Rekomendasi 3-Tier Pemda
Temuan ini kami ubah menjadi tiga aksi konkret bagi Dinas Koperasi dan UKM Lampung di slide kesebelas. Kolom kiri: **Tier 1 Penyelamatan (357 UMKM Menurun)**, fokus di Mesuji, Tulang Bawang, dan Pesisir Barat lewat QRIS gratis, bootcamp literasi, dan internet komunal. Kolom tengah: **Tier 2 Akselerasi (629 UMKM Stabil)** lewat target rating 4,3, posting rutin lima kali seminggu, dan kredit ROI. Kolom kanan: **Tier 3 Ekspansi (414 UMKM Tumbuh)** dikurasi ke E-Katalog dan direkrut sebagai mentor sebaya. Anggaran pun fokus menyasar **25,5 persen** UMKM paling rentan.

### Slide 12 | 05:45 - 06:15 - Live Web App
Supaya siap operasional, seluruh model kami pasang di web **SiTumbuh Lampung** (`situmbuh-lampung.vercel.app`). Di tangkapan layar kiri terlihat ringkasan tier di atas, peta kerentanan kabupaten di panel kiri, serta daftar antrean dan aksi otomatisnya di panel kanan. Sesuai fitur di sebelah kanan, petugas Dinas dapat mengekspor tugas `.csv` ke lapangan dan memakai tab **Simulator What-If** untuk menguji dampak pelatihan maupun QRIS sebelum dana cair.

### Slide 13 & Slide 14 | 06:15 - 06:50 - Sintesis & Penutup
Sebagai penutup di slide ketiga belas, alur kerja kami berjalan lewat tiga tahap teruji: rekonstruksi bebas kebocoran sesuai split `random_state=42`, pemodelan ordinal plus klaster dan Copula, serta dasbor prioritas bagi Pemda. Kami menjaga transparansi terhadap batas teoretis Bayes dengan skor Holdout **0,6687** yang jujur dan konsisten, dan seluruh kode terbuka di GitHub pada layar.

Terima kasih Dewan Juri, wassalamualaikum warahmatullahi wabarakatuh. Kami siap memasuki sesi tanya jawab.

---

## Pintasan Q&A (Ketik Nomor Slide + Enter)
- *"Kenapa skor ~0,67 bukan 0,90+?"* -> Slide **`16`** (`L01`): *Collider Bias* & *1-NN Bayes Bound*.
- *"Detail skor 15 model?"* -> Slide **`17`** (`L02`).
- *"Kenapa SMOTE ditolak?"* -> Slide **`18`** (`L03`).
- *"Detail fitur & hyperparameter?"* -> Slide **`19`** (`L04`).
- *"Bukti bebas data leakage?"* -> Slide **`20`** (`L05`).
- *"Update terbaru web SiTumbuh Lampung?"* -> Bisa demo langsung `situmbuh-lampung.vercel.app` (kini diperluas hingga 5 tab termasuk Analisis Blank-Spot, 3 Persona, Modal Lembar Kerja, dan Skrining Batch CSV).
