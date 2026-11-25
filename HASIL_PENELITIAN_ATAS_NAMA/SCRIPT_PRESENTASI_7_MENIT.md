# Script Presentasi Final GEMATIK VI 2026 (Target ~7 Menit)
**Tim**: Atas Nama (Universitas Negeri Surabaya)  
**Pembagian Pembicara**:
- **HABIB**: Slide 01 - 02, lalu Slide 07 - 14
- **MELYSA**: Slide 03 - 06

> **Catatan Tempo**: Sekitar 900 kata ucapan. Dibawakan santai tapi tegas (~130 kata/menit) selesai di **06:45 - 07:00**, masih aman dari batas **08:00**. Pola setiap slide: **apa yang kami lakukan -> apa hasilnya -> artinya apa**. Kalimat bertanda *(tunjuk layar)* diucapkan sambil menunjuk grafik/angka di slide.

---

## BAGIAN 1: HABIB (Slide 01 - 02 | 00:00 - 01:00)

### Slide 01 | 00:00 - 00:25 - Cover
Assalamualaikum warahmatullahi wabarakatuh. Selamat siang Dewan Juri. Kami dari Tim **Atas Nama**, Universitas Negeri Surabaya. Saya **Muhammad Habib Nur Aiman**, bersama rekan saya **Melysa Dwyta Sary**.

Langsung ke hasil: kami membangun model untuk memprediksi status pertumbuhan 7.000 UMKM di Lampung. *(tunjuk tiga angka)* Skor validasi silang kami **0,6737**, skor di data uji **0,6687**, dan kesalahan paling fatal, yaitu UMKM Menurun ditebak Tumbuh atau sebaliknya, hanya **0,86%**. Semua tanpa kebocoran data.

### Slide 02 | 00:25 - 01:00 - Masalah & Protokol Data
Masalahnya begini. Pemda Lampung punya anggaran pendampingan terbatas, jadi harus tahu UMKM mana yang perlu diselamatkan dan mana yang tinggal didorong. Targetnya tiga kelas berurutan: **Menurun 27%, Stabil 43%, dan Tumbuh 30%**.

Karena metriknya **Macro F1**, kelas Menurun yang lebih sedikit dinilai sama pentingnya dengan Stabil. Ini sejalan dengan kebutuhan Pemda: salah menebak UMKM yang sedang jatuh berarti bantuan tidak sampai.

Untuk menjaga kejujuran hasil, kami ikuti persis aturan split `random_state=42`: **5.600 data Train** untuk semua eksperimen, dan **1.400 data Holdout** kami kunci, tidak disentuh sama sekali sampai evaluasi akhir. Selanjutnya, temuan data akan dijelaskan oleh Melysa.

---

## BAGIAN 2: MELYSA (Slide 03 - 06 | 01:00 - 03:00)

### Slide 03 | 01:00 - 01:30 - Audit Data Kotor
Terima kasih Habib. Sebelum membuat model, kami periksa dulu kualitas datanya, dan kami menemukan tiga masalah. Pertama, nama kabupaten tercatat 30 jenis karena ada 50 baris ditulis huruf kapital semua, kami rapikan menjadi 15 wilayah resmi. Kedua, ada 8 kolom kosong, paling banyak di Rating sebesar 7,9%. Ketiga, ada omzet ekstrem di atas 84 juta yang tersebar acak di semua kelas, jadi itu noise, bukan sinyal. Nilai ekstrem ini kami anggap kosong, lalu semua data kosong kami isi ulang dengan **Hybrid MICE**.

### Slide 04 | 01:30 - 02:00 - Tiga Ide Solusi
Dari situ, solusi kami bertumpu pada tiga ide. *(tunjuk kartu 01)* Pertama, bersihkan dan rekonstruksi data, termasuk membuat fitur efisiensi modal, `log ROI`. *(kartu 02)* Kedua, tangkap struktur tersembunyi: targetnya berurutan, jadi kami pakai regresi ordinal, ditambah fitur hasil clustering dan data sintetis Gaussian Copula. *(kartu 03)* Ketiga, gabungkan 15 model, lalu atur ambang keputusan tiap kelas supaya kelas minoritas tidak kalah suara.

### Slide 05 | 02:00 - 02:30 - EDA & Paradoks Modal
Apa yang membedakan UMKM tumbuh dan menurun? *(tunjuk grafik)* Median literasi digital naik dari **39,4** di kelas Menurun menjadi **66,9** di kelas Tumbuh. UMKM yang masih **tunai saja**, **57%-nya menurun**. Yang menarik, modal awal kelas Menurun justru lebih besar, **20,1 juta**, dibanding kelas Tumbuh **16,2 juta**. Artinya, modal besar tidak menjamin tumbuh. Yang menentukan adalah seberapa efisien modal itu diputar.

### Slide 06 | 02:30 - 03:00 - Temuan Kausal Mediasi
Sekilas, Bandar Lampung jauh lebih unggul: **53,1% tumbuh**, sementara Mesuji hanya **16,5%**. Tapi *(tunjuk garis regresi)* saat kami plot 15 wilayah, persentase tumbuh hampir sejajar sempurna dengan literasi digital wilayahnya, **r = 0,98**. Setelah literasi dan internet dikontrol, pengaruh lokasi hampir hilang. Jadi masalahnya bukan letak geografis, melainkan kesenjangan literasi dan internet, dan itu bisa diintervensi. Untuk modeling dan rekomendasi, dilanjutkan oleh Habib.

---

## BAGIAN 3: HABIB (Slide 07 - 14 | 03:00 - 07:00)

### Slide 07 | 03:00 - 03:35 - Tangga Ablasi
Terima kasih Melysa. Semua ide tadi tidak kami pakai begitu saja; setiap teknik kami uji satu per satu di **10-Fold Cross Validation**, dan hanya yang terbukti menaikkan skor yang dipertahankan.

*(tunjuk tangga dari bawah)* Dari baseline **0,6617**, imputasi MICE naik ke **0,6690**, fitur clustering ke **0,6704**, data sintetis Copula ke **0,6716**, dan ensemble plus pengaturan ambang mencapai **0,6737**. Sebaliknya, **SMOTE kami tolak** karena justru menurunkan skor ke **0,6548**: SMOTE memaksa kelas jadi seimbang, padahal proporsi asli kelas Stabil adalah informasi penting.

### Slide 08 | 03:35 - 04:10 - Benchmark 15 Model & Ensemble
Di slide ini kami bandingkan 15 model dari empat pendekatan berbeda. Temuannya: model pohon yang dalam seperti Random Forest justru menghafal noise, skornya hanya **0,6567**. Ketika kedalaman pohon kami batasi menjadi 3, CatBoost naik ke **0,6744**.

Lalu kami gabungkan 9 model yang paling saling melengkapi dengan bobot optimal. Hasilnya: prediksi probabilitas makin akurat, **LogLoss turun ke 0,7055**, dengan **Macro F1 0,6737**. Intinya, model yang sederhana dan beragam lebih tahan noise dibanding satu model yang rumit.

### Slide 09 | 04:10 - 04:45 - Evaluasi Holdout & Batas Jujur
Lalu, bagaimana di data yang benar-benar baru? Pada **1.400 data Holdout**, skor kami **0,6687**, hanya selisih 0,005 dari skor validasi. Artinya model tidak overfitting.

*(tunjuk confusion matrix)* Model paling kuat di kelas yang paling penting bagi Pemda: **Tumbuh 0,700** dan **Menurun 0,680**. Kesalahan fatal Menurun-Tumbuh hanya **12 dari 1.400 UMKM**. Sebagian besar kesalahan, **97,5%**, terjadi di perbatasan dengan kelas Stabil, yang memang tipis karena data ini mengandung noise bawaan. Jadi skor sekitar 0,67 ini dekat dengan batas maksimal yang mungkin dicapai data ini.

### Slide 10 | 04:45 - 05:15 - SHAP & Persona
Model yang baik juga harus bisa dijelaskan. *(tunjuk grafik SHAP)* Analisis SHAP menunjukkan faktor penentu utama adalah **indeks digital**, **kombinasi literasi dan internet**, **rating**, **akses kredit**, dan **engagement** media sosial.

Kami juga mengelompokkan UMKM tanpa melihat label, dan muncul tiga persona. **Konvensional Rentan**, 56,4%-nya menurun. **Digital Matang**, 51,4%-nya tumbuh. Kelompok yang terbentuk sendiri ini selaras dengan target, sehingga bisa langsung dipakai sebagai dasar segmentasi program.

### Slide 11 | 05:15 - 05:55 - Rekomendasi 3-Tier Pemda
Dari prediksi di data uji, kami terjemahkan menjadi tiga keputusan untuk Dinas Koperasi dan UKM Lampung.

*(tunjuk kolom merah)* **Tier 1, Penyelamatan, 357 UMKM** yang diprediksi menurun: onboarding QRIS gratis, bootcamp literasi dasar, dan internet komunal, diprioritaskan di Mesuji, Tulang Bawang, dan Pesisir Barat.  
*(kolom kuning)* **Tier 2, Akselerasi, 629 UMKM Stabil**: perbaikan rating minimal 4,3, posting minimal lima kali seminggu, dan kredit berbasis ROI.  
*(kolom hijau)* **Tier 3, Ekspansi, 414 UMKM Tumbuh**: kurasi E-Katalog pemerintah, dan dijadikan mentor sebaya bagi Tier 1 dan 2.

Dampaknya, anggaran dasar tidak disebar rata, tetapi difokuskan ke 25,5% UMKM yang paling rentan.

### Slide 12 | 05:55 - 06:25 - Live Web App
Supaya tidak berhenti di slide, kami membangun dasbor **SiTumbuh Lampung**, sudah live di **situmbuh-lampung.vercel.app**. Petugas Dinas bisa langsung melihat antrean prioritas UMKM beserta rekomendasi aksinya, peta wilayah blank-spot, profil tiga persona, dan mengunduh daftar tugas lapangan. Ada juga **simulator What-If**: misalnya, jika UMKM diberi pelatihan dan QRIS, berapa peluang tumbuhnya naik, sehingga dampak program bisa diuji sebelum dana dicairkan. UMKM pendaftar baru pun bisa diskrining sekaligus lewat unggah CSV.

### Slide 13 & 14 | 06:25 - 07:00 - Sintesis & Penutup
Sebagai penutup, ada tiga hal yang kami bawa. Pertama, pipeline yang bersih dan bebas kebocoran data. Kedua, skor Holdout **0,6687** yang konsisten dan jujur, mendekati batas teoretis dataset. Ketiga, rekomendasi dan aplikasi web yang siap dipakai Pemda Lampung. Seluruh kode terbuka di GitHub pada layar.

Terima kasih Dewan Juri. Wassalamualaikum warahmatullahi wabarakatuh. Kami siap untuk sesi tanya jawab.

---

## Pintasan Q&A (Ketik Nomor Slide + Enter)
- *"Kenapa skor ~0,67 bukan 0,90+?"* -> Slide **`16`** (`L01`): *Collider Bias* & *1-NN Bayes Bound*.
- *"Detail skor 15 model?"* -> Slide **`17`** (`L02`).
- *"Kenapa SMOTE ditolak?"* -> Slide **`18`** (`L03`).
- *"Detail fitur & hyperparameter?"* -> Slide **`19`** (`L04`).
- *"Bukti bebas data leakage?"* -> Slide **`20`** (`L05`).
