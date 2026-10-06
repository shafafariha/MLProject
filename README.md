# 🖐️ BISINDO Hand Sign Recognition & Classification

Proyek Machine Learning untuk klasifikasi dan pengenalan alfabet **Bahasa Isyarat Indonesia (BISINDO)** (A–Z) secara *real-time* berbasis webcam maupun unggah gambar statis menggunakan **MediaPipe Hand Landmarks** dan algoritma **Random Forest Classifier**. Dilengkapi dengan antarmuka web interaktif berbasis **Streamlit**.

---

## 📌 Daftar Isi
- [Ringkasan Proyek](#-ringkasan-proyek)
- [Fitur Utama](#-fitur-utama)
- [Alur Kerja (Pipeline)](#-alur-kerja-pipeline)
- [Hasil dan Performa Model](#-hasil-dan-performa-model)
- [Struktur Direktori](#-struktur-direktori)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Panduan Instalasi dan Menjalankan Aplikasi](#-panduan-instalasi-dan-menjalankan-aplikasi)
- [Penggunaan Dashboard](#-penggunaan-dashboard)
- [Kontributor & Lisensi](#-kontributor--lisensi)

---

## 📖 Ringkasan Proyek

Bahasa Isyarat Indonesia (BISINDO) merupakan media komunikasi alami yang digunakan oleh komunitas Tuli di Indonesia. Proyek ini bertujuan untuk menjembatani komunikasi melalui sistem penerjemah alfabet isyarat otomatis dengan pendekatan *Computer Vision* dan *Machine Learning*:

1. **Ekstraksi Fitur Geometris**: Menggunakan MediaPipe Hands untuk mendeteksi koordinat 3D dari sendi tangan ($x, y, z$).
2. **Augmentasi Data**: Memperluas variasi citra dengan augmentasi rotasi, flip horizontal, dan penyesuaian kontras/pencahayaan.
3. **Pemodelan Machine Learning**: Membandingkan 7 algoritma klasifikasi, di mana **Random Forest** menunjukkan performa terbaik dengan akurasi pengujian mencapai **~99.5%**.
4. **Antarmuka Interaktif**: Dashboard Streamlit yang mendukung deteksi *live webcam* dan analisis citra statis.

---

## ✨ Fitur Utama

- 🔴 **Real-Time Webcam Detection**: Deteksi dan prediksi alfabet isyarat langsung dari kamera dengan anotasi visual *landmark* tangan.
- 🖼️ **Image Upload Mode**: Alternatif pengujian dengan mengunggah gambar format `.jpg`, `.jpeg`, atau `.png`.
- 🎛️ **Pengaturan Kamera & Filter**: Slider langsung untuk mengatur *Brightness*, *Contrast*, dan *Saturation* guna menyesuaikan kondisi pencahayaan ruangan.
- ⚙️ **Konfigurasi Threshold Deteksi**: Pengaturan parameter *Min Detection Confidence* dan *Min Tracking Confidence* MediaPipe.
- 👐 **Dukungan Dua Tangan**: Mampu mengekstrak hingga 2 tangan secara simultan (maksimal 126 fitur koordinat), cocok untuk huruf isyarat yang membutuhkan satu maupun dua tangan.

---

## 🔄 Alur Kerja (Pipeline)

```
[ Input Citra / Webcam Frame ]
             │
             ▼
[ Ekstraksi Landmark MediaPipe ] ──► (21 titik sendi x 3 koordinat [x, y, z] per tangan)
             │
             ▼
[ Vektor Fitur (126 Dimensi) ] ──► (Zero-padding jika hanya 1 tangan terdeteksi)
             │
             ▼
[ Random Forest Classifier ]   ──► (rf_bisindo_classifier_99.pkl)
             │
             ▼
[ Hasil Prediksi Huruf (A-Z) & Visualisasi Skeleton ]
```

---

## 📊 Hasil dan Performa Model

Dalam eksperimen yang dilakukan pada notebook [`ipynb/BISINDO 1.ipynb`](ipynb/BISINDO%201.ipynb), dilakukan evaluasi terhadap beberapa algoritma *machine learning*:

| Algoritma | Validation Accuracy | Validation F1-Score | Test Accuracy | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | **99.54%** | **99.54%** | **99.45% - 99.59%** | **99.45% - 99.59%** |
| **Gradient Boosting** | 98.89% | 98.90% | 98.71% | 98.71% |
| **K-Nearest Neighbors (KNN)** | 98.30% | 98.30% | 97.79% | 97.77% |
| **Decision Tree** | 96.08% | 96.10% | 95.72% | 95.70% |
| **Support Vector Machine (SVM)** | 86.37% | 85.70% | 86.51% | 85.77% |
| **Logistic Regression** | 83.74% | 83.42% | 83.61% | 83.37% |
| **Naive Bayes** | 41.32% | 31.50% | 41.71% | 31.42% |

> Model final yang digunakan dalam sistem aplikasi adalah **Random Forest Classifier** (`model/rf_bisindo_classifier_99.pkl`).

---

## 📁 Struktur Direktori

```plaintext
MLProject/
├── .devcontainer/             # Konfigurasi development container
│   └── devcontainer.json
├── .streamlit/                # Konfigurasi tema & server Streamlit
│   └── config.toml
├── dashboard/                 # Kode sumber aplikasi antarmuka Streamlit
│   ├── dashboard.py           # Dashboard utama (Real-time Webcam + Upload Gambar)
│   ├── local.py               # Versi khusus webcam lokal
│   └── static.py              # Versi khusus klasifikasi gambar statis
├── Datasets/                  # Dataset gambar dan file ekstraksi CSV
│   ├── bisindo/               # Dataset citra asli per huruf (A - Z)
│   ├── bisindo-augmented/     # Dataset citra hasil augmentasi
│   ├── bisindo-features.csv   # Data fitur koordinat landmark lengkap
│   ├── bisindo-train.csv      # Data latih (Train set)
│   ├── bisindo-val.csv        # Data validasi (Validation set)
│   └── bisindo-test.csv       # Data uji (Test set)
├── ipynb/                     # Jupyter Notebooks eksplorasi & pelatihan model
│   ├── BISINDO 1.ipynb        # Eksperimen perbandingan 7 algoritma ML
│   └── BISINDO 2.ipynb        # Evaluasi detail & matriks konfusi Random Forest
├── model/                     # Model ML terlatih (Pickle file)
│   └── rf_bisindo_classifier_99.pkl
├── packages.txt               # Kebutuhan dependensi sistem Linux (Debian/Ubuntu)
├── requirements.txt           # Dependensi pustaka Python
└── README.md                  # Dokumentasi proyek
```

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: [Python 3.10+](https://www.python.org/)
- **Computer Vision & Hand Tracking**: [MediaPipe](https://developers.google.com/mediapipe), [OpenCV](https://opencv.org/)
- **Machine Learning & Modeling**: [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Data Manipulation & Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualisasi & Evaluasi**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Web Application & UI**: [Streamlit](https://streamlit.io/)

---

## 🚀 Panduan Instalasi dan Menjalankan Aplikasi

### 1. Klon Repositori
```bash
git clone https://github.com/shafafariha/MLProject.git
cd MLProject
```

### 2. Buat dan Aktifkan Virtual Environment (Disarankan)
- **Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Catatan untuk pengguna Linux**: Jika mengalami kendala modul grafis OpenCV (`libGL.so.1`), instal dependensi sistem:
> ```bash
> sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libxrender1 libxext6
> ```

---

## 🖥️ Penggunaan Dashboard

Jalankan dashboard Streamlit melalui terminal:

```bash
streamlit run dashboard/dashboard.py
```

Setelah perintah dijalankan, buka browser di alamat `http://localhost:8501`.

### Pilihan Mode di Dashboard:
1. **Mode Upload Gambar**:
   - Tarik atau pilih file gambar tangan yang membentuk alfabet BISINDO.
   - Hasil deteksi landmark dan prediksi huruf akan langsung ditampilkan pada layar.
2. **Mode Real-Time Webcam**:
   - Centang kotak **"Start Webcam"**.
   - Arahkan tangan Anda ke kamera. Pastikan pencahayaan cukup dan tangan terlihat jelas dalam bingkai kamera.
   - Sesuaikan slider di sidebar jika perlu mengubah kecerahan atau tingkat sensitivitas deteksi.

---

## 📄 Lisensi & Kredit

- **Dataset**: Kaggle Dataset [achmadnoer/alfabet-bisindo](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)
- Dikembangkan oleh [@shafafariha](https://github.com/shafafariha)
- Proyek ini ditujukan untuk tujuan edukasi, penelitian, dan pengembangan teknologi aksesibilitas komunikasi bahasa isyarat.
