# 🥚 OvoScan — Panduan Deploy

Aplikasi klasifikasi fertilitas telur ayam berbasis CNN Fine-Tuning (ResNet50 / EfficientNetB0 / MobileNetV2).

---

## 📁 Struktur Folder

```
ovoscan/
├── app.py                  ← Aplikasi Streamlit utama
├── requirements.txt        ← Dependensi Python
├── .streamlit/
│   └── config.toml         ← Konfigurasi tema & server
├── models/                 ← ⚠️ WAJIB DIISI — taruh file .h5 di sini
│   ├── best_ResNet50.h5
│   ├── best_EfficientNetB0.h5
│   └── best_MobileNetV2.h5
└── README.md
```

---

## 🚀 LANGKAH DEPLOY

### OPSI A — Streamlit Community Cloud (GRATIS, Direkomendasikan)

1. **Download model dari Google Colab** (sudah ada cell EV8 di notebook)

2. **Buat repository GitHub baru**, upload semua file:
   ```
   git init
   git add .
   git commit -m "OvoScan deploy"
   git remote add origin https://github.com/USERNAME/ovoscan.git
   git push -u origin main
   ```

3. **Upload model ke GitHub LFS** (karena file .h5 besar):
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   git add models/
   git commit -m "Add model files"
   git push
   ```

4. **Deploy di Streamlit Cloud:**
   - Buka https://share.streamlit.io
   - Login dengan akun GitHub
   - Klik **"New app"**
   - Pilih repo → Branch: `main` → File: `app.py`
   - Klik **Deploy**

---

### OPSI B — Jalankan Lokal

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Pastikan folder models/ berisi file .h5
ls models/

# 3. Jalankan aplikasi
streamlit run app.py
```

Buka browser: http://localhost:8501

---

### OPSI C — Deploy ke Hugging Face Spaces (Alternatif Gratis)

1. Buat akun di https://huggingface.co
2. Buat **Space baru** → pilih SDK: **Streamlit**
3. Upload semua file (termasuk model .h5 via Git LFS)
4. Space otomatis aktif di `https://huggingface.co/spaces/USERNAME/ovoscan`

---

## ⚙️ Cara Mendapatkan File Model (.h5)

1. Buka notebook `Untitled15.ipynb` di Google Colab
2. Jalankan semua cell training (Cell 1 s/d Cell 8)
3. Jalankan **Cell EV8** untuk download model:
   ```python
   from google.colab import files
   files.download("/content/drive/MyDrive/hasil_model5/best_ResNet50.h5")
   files.download("/content/drive/MyDrive/hasil_model5/best_EfficientNetB0.h5")
   files.download("/content/drive/MyDrive/hasil_model5/best_MobileNetV2.h5")
   ```
4. Pindahkan file yang terdownload ke folder `models/`

> **Minimal 1 file .h5 harus ada.** Aplikasi otomatis memilih model terbaik yang tersedia.

---

## 🔍 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError: models/best_*.h5` | Pastikan folder `models/` ada dan berisi file .h5 |
| App lambat saat pertama load | Normal — model dimuat sekali, berikutnya cepat |
| Upload gambar gagal | Cek format file (JPG/PNG) dan ukuran < 10MB |
| Error TensorFlow versi | Pastikan TF >= 2.13, sesuaikan dengan versi saat training |

---

## 📊 Akurasi Model

| Model | Akurasi Test | Ukuran File |
|-------|-------------|-------------|
| ResNet50 | 94.27% | ~100 MB |
| EfficientNetB0 | 94.27% | ~20 MB |
| MobileNetV2 | 92.99% | ~14 MB |

> Jika resource terbatas (Streamlit Cloud Free), gunakan **EfficientNetB0** atau **MobileNetV2** karena lebih ringan.


---

## 📦 Model

Download model di sini:
https://drive.google.com/drive/folders/1tGf_HMWI3f9YeSzm0M-Binx6yEe1GM4l

---

## 📊 Dataset

Download dataset di sini:
https://drive.google.com/drive/folders/1XVuoLLEGtajpMZXE2EIKMNBBg7m5zOuM
---