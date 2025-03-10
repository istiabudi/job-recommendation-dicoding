# Laporan Proyek Machine Learning - Istia Budi

## Project Overview

Dalam dunia rekrutmen, menemukan pekerjaan yang sesuai dengan keterampilan dan pengalaman seseorang dapat menjadi tantangan. Dengan banyaknya lowongan pekerjaan yang tersedia, kandidat sering kali kesulitan menemukan pekerjaan yang paling relevan dengan keahlian mereka. Oleh karena itu, diperlukan sistem rekomendasi yang dapat membantu pencari kerja menemukan pekerjaan yang sesuai dengan cepat dan efisien.

## Business Understanding

Proyek ini bertujuan untuk membangun sistem rekomendasi pekerjaan berbasis Cosine Similarity, yang dapat menyarankan lowongan kerja berdasarkan kesamaan deskripsi pekerjaan dengan pekerjaan yang telah dipilih pengguna.

### Problem Statements

- Bagaimana cara merekomendasikan pekerjaan yang relevan berdasarkan judul pekerjaan dan keterampilan yang dibutuhkan?
- Pengembangan sistem rekomendasi yang kompleks sering kali memerlukan sumber daya yang besar, sehingga perlu solusi sederhana yang tetap efektif.

### Goals

- Mengembangkan sistem rekomendasi pekerjaan yang dapat mencocokkan pekerjaan berdasarkan kesamaan nama dan keterampilan.
- Merancang solusi berbasis cosine similarity yang mudah diimplementasikan dan memberikan hasil rekomendasi dengan waktu proses yang cepat dan hasil yang akurat.

### Solution statements

Menggunakan algoritma cosine similarity untuk menghitung kesamaan antara nama pekerjaan dan skill yang dibutuhkan. Sistem akan menganalisis pekerjaan dan keterampilan dari setiap pekerjaan, lalu mencocokkannya dengan data keahlian. Pendekatan ini cocok untuk pengguna yang memiliki keahlian jelas terhadap jenis pekerjaan tertentu.

## Data Understanding

### Informasi Dataset

| Jenis         | Keterangan                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Sumber        | [AI-Powered Job Recommendations - Kaggle](https://www.kaggle.com/datasets/samayashar/ai-powered-job-recommendations/data) |
| Dataset Owner | [Samay Ashar](https://www.kaggle.com/samayashar)                                                                          |
| License       | CC0: Public Domain                                                                                                        |
| Tags          | Deep Learning, Neural Networks, Recommender Systems, XGBoost, LightGBM                                                    |
| Usability     | 10.0                                                                                                                      |

Dataset ini terdiri dari 50.000 lowongan pekerjaan yang mencakup berbagai industri, lokasi, tingkat pengalaman, dan kisaran gaji. Data ini memberikan wawasan terstruktur tentang tren pasar kerja, keterampilan yang dibutuhkan, dan distribusi gaji, sehingga berguna untuk sistem rekomendasi pekerjaan yang digerakkan oleh AI, model prediksi gaji, analisis tren perekrutan, dan peramalan permintaan keterampilan.

Dataset ini dapat dimanfaatkan oleh para ilmuwan data, profesional SDM, pencari kerja, dan peneliti untuk menganalisis pola perekrutan, memprediksi gaji, dan membangun sistem pencocokan pekerjaan yang dipersonalisasi menggunakan pembelajaran mesin.

Berdasarkan data tersebut variabel-variabel pada Job Recommendation Dataset adalah sebagai berikut:

| Column Name      | Description                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| job_title        | Judul spesifik dari posisi pekerjaan (misalnya, Software Engineer, Marketing Manager). Terdapat 639 nilai unik.                                                                |
| company          | Nama perusahaan yang menawarkan posisi pekerjaan (misalnya, Google, Amazon, Microsoft). Terdapat 37.022 nilai unik.                                                            |
| location         | Kota atau wilayah tempat pekerjaan tersebut berbasis. Terdapat 7 nilai unik. Lokasi paling umum: Toronto (14%), London (14%).                                                  |
| experience_level | Tingkat senioritas yang dibutuhkan (Entry-Level: 34%, Mid-Level: 33%, Senior-Level: 33%). Terdapat 3 nilai unik.                                                               |
| salary_range     | Rentang gaji mulai dari $40.000 hingga $150.000. Rata-rata: $95,1K, Deviasi Standar: $31,8K.                                                                                   |
| industry         | Sektor industri tempat pekerjaan berada (misalnya, Software, Manufacturing). Terdapat 7 nilai unik. Industri paling umum: Software (15%).                                      |
| required_skills  | Daftar keterampilan yang diperlukan untuk pekerjaan tersebut (misalnya, Python, SQL, Sales & Merchandising). Terdapat 1.559 nilai unik. Keterampilan paling umum: Python (1%). |

### EDA

!g1

Jumlah pekerjaan yang tersedia cukup merata, pekerjaan di industri Sofware merupakan industri yang paling banyak membuka lowongan pekerjaan sebanyak 7302.

!g2

Berdasarkan barplot diatas, rata-rata gaji setiap industri hampir sama yaitu kurang lebih 95k USD. Perusahaan yang memberikan gaji tertinggi adalah perusahaan di industri Sofware dengan rata-rata 95606.272254 USD.

## Data Preparation

Data preparation bertujuan untuk mempersiapkan data agar proses pengembangan model diharapkan akurasi model akan menjadi lebih baik dan mengurangi bias pada data. Tahapan preparation data yaitu:

1. Seleksi Fitur

- Dataset memiliki berbagai fitur, tetapi untuk proyek ini hanya digunakan dua kolom, yaitu Job Title dan Required Skills. Fitur lainnya dapat dipertimbangkan untuk pengembangan model yang lebih kompleks di masa depan.

2. Mengubah Nama Kolom dengan Menghapus Spasi
   - Hapus spasi pada nama kolom untuk memudahkan proses pembuatan model
3. Mengurangi Dataset
   - Dataset dikurangi sebanyak 5000 karena keterbatasan resource.
4. Ekstraksi Fitur dengan TF-IDF

- Metode TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengukur pentingnya sebuah kata dalam dokumen tertentu relatif terhadap kumpulan dokumen lainnya.
- Proses ini dilakukan dengan fungsi TfidfVectorizer() dari library sklearn. Data difit dan ditransformasikan ke dalam matriks berukuran (5000, 40), di mana 5000 adalah jumlah data dan 40 adalah jumlah Required Skills unik yang terwakili.

## Modeling

Cosine Similarity merupakan teknik yang digunakan untuk mengukur tingkat kesamaan antara dua vektor berdasarkan sudut kosinus di antara keduanya. Nilai yang dihasilkan berkisar antara -1 hingga 1, tetapi dalam konteks sistem rekomendasi, umumnya berada dalam rentang 0 hingga 1 karena data yang digunakan jarang memiliki elemen negatif. Nilai 1 menunjukkan bahwa kedua vektor memiliki kesamaan sempurna, sedangkan nilai 0 berarti tidak ada kesamaan sama sekali.

Dalam pendekatan content-based filtering, setiap pekerjaan direpresentasikan sebagai vektor numerik berdasarkan fitur seperti industri, keterampilan yang dibutuhkan, dan tingkat pengalaman. Proses ini dapat dilakukan menggunakan metode seperti TF-IDF atau one-hot encoding. Setelah itu, vektor-vektor dinormalisasi agar perhitungan hanya bergantung pada sudut antara vektor, bukan magnitudonya. Tingkat kesamaan antar-pekerjaan kemudian dihitung menggunakan fungsi cosine_similarity dari Scikit-Learn untuk menghasilkan matriks kesamaan. Pekerjaan yang memiliki skor kesamaan tertinggi dengan pekerjaan target akan direkomendasikan kepada pengguna.

Untuk mengukur tingkat kesamaan antar-pekerjaan, sistem menerapkan Cosine Similarity menggunakan fungsi `cosine_similarity` dari library sklearn. Perhitungan ini didasarkan pada rumus berikut:

### Rumus Cosine Similarity

![images](https://github.com/user-attachments/assets/e62af498-c8d5-48c0-b023-dca2cd0a6e34)

### Keunggulan Cosine Similarity

1. Efisien dan Mudah Diterapkan
   - Teknik ini sederhana dan dapat bekerja dengan baik pada data berdimensi tinggi.
2. Tidak Terpengaruh oleh Skala Data
   - Cosine Similarity hanya mempertimbangkan arah vektor, bukan besarannya, sehingga dapat digunakan untuk data dengan skala berbeda.
3. Efektif untuk Dataset Sparse
   - Cocok untuk data yang memiliki banyak elemen nol, seperti matriks fitur pekerjaan atau dokumen teks.

### Kelemahan Cosine Similarity

1. Tidak Mempertimbangkan Nilai Absolut
   - Metode ini hanya mengukur kesamaan berdasarkan sudut, tanpa mempertimbangkan intensitas atau bobot fitur.
2. Kurang Efektif untuk Data Non-Vektor
   - Kesulitan dalam menangani data yang tidak dapat direpresentasikan sebagai vektor numerik.
3. Kurang Optimal untuk Hubungan Non-Linear
   - Tidak cocok untuk dataset yang memiliki hubungan kompleks dan memerlukan pendekatan non-linear.

Langkah selanjutnya adalah menggunakan metode argpartition untuk mengidentifikasi top-N hasil dengan skor kesamaan tertinggi. Setelah itu, sistem akan mengurutkan hasil berdasarkan bobot kesamaan dari yang terbesar hingga terkecil. Terakhir, akurasi sistem rekomendasi akan dievaluasi guna memastikan efektivitasnya dalam menemukan pekerjaan yang memiliki kemiripan dengan preferensi pengguna.

Berikut ini adalah data referensi yang digunakan untuk menentukan 5 rekomendasi pekerjaan dengan industri yang sama:

| JobTitle          | RequiredSkills                                 |
| ----------------- | ---------------------------------------------- |
| Software engineer | Social Media, SEO, Google Ads, Content Writing |
| Software engineer | Sales, Merchandising, Customer Service         |
| Software engineer | Risk Analysis                                  |
| Software engineer | Research, Teaching, Curriculum Design          |
| Software engineer | Curriculum Design, EdTech, Teaching            |
| Software engineer | Research, Teaching, Curriculum Design          |

Pengujian dilakukan dengan menggunakan pekerjaan "Software Engineer", yang termasuk dalam industri "Software" dan "Technology".

Berikut ini adalah hasil rekomendasi dari sistem yang menampilkan top-N pekerjaan dengan industri yang sama dengan "Software Engineer":

### Tabel Hasil Prediksi:

| Job Title           | RequiredSkills                                    |
| ------------------- | ------------------------------------------------- |
| Early years teacher | Pharmaceuticals                                   |
| Early years teacher | Financial Modeling, Risk Analysis                 |
| Early years teacher | Medical Research, Nursing, Patient Care, Pharm... |
| Early years teacher | Java, AWS                                         |
| Early years teacher | Sales                                             |

Sistem telah berhasil merekomendasikan top 5 job yang mirip dengan Software Engineer, yaitu job yang memiliki required skill yang sama.

Evaluasi model Content-Based Filtering dilakukan dengan menggunakan metrik Precision. Metrik ini mengukur sejauh mana model dapat memprediksi kejadian yang relevan atau positif.

Rumus Precision:

![precision_formula](https://github.com/user-attachments/assets/0943cb5f-fcf1-450c-aa48-e694e4b2ecda)

Pada contoh rekomendasi di atas: Precision = 5/5. Jadi precision = 100%.
