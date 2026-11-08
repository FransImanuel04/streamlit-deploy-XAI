# Aplikasi Prediksi Skor Kesehatan Mental (Streamlit)

Aplikasi web untuk model regresi skripsi: memprediksi `Mental_Health_Score` dari 7 variabel dan menjelaskan hasilnya dengan SHAP.

## Isi paket
| Berkas | Fungsi |
|---|---|
| `train_export.py` | Melatih model (atau memuat dari notebook) lalu menyimpan artefak ke `artifacts/` |
| `app.py` | Antarmuka Streamlit |
| `utils.py` | Fungsi bersama (pemuatan model, SHAP) |
| `requirements.txt` | Daftar pustaka |
| `artifacts/` | `model.joblib`, `metadata.json`, `background.csv` (dibuat oleh `train_export.py`) |

## 1. Menjalankan di komputer sendiri
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# letakkan AI_SocialMedia_Student_Dataset.csv di folder ini, lalu:
python train_export.py --data AI_SocialMedia_Student_Dataset.csv     # sekitar 5-10 menit (tuning MLP seperti pada skripsi)

streamlit run app.py
```
Opsi `train_export.py`:
- `--model {mlp,linear,rf,xgb}`: pilih algoritma (default `mlp`, model terbaik menurut RMSE uji pada skripsi).
- `--reuse models/model_terbaik_MLP.joblib`: memakai model hasil notebook tanpa melatih ulang. **Versi scikit-learn harus sama dengan versi di Colab** (cek dengan `!pip show scikit-learn` di Colab). Jika ragu, latih ulang saja.
- `--fast`: tanpa tuning, hanya untuk uji cepat.

## 2. Deploy gratis ke Streamlit Community Cloud
1. Buat repositori GitHub berisi seluruh berkas di atas, **termasuk folder `artifacts/` yang sudah terisi** (jangan unggah dataset jika lisensinya tidak mengizinkan).
2. Buka share.streamlit.io, masuk dengan GitHub, pilih repositori, dan isi *Main file path* dengan `app.py`.
3. Tekan Deploy. Streamlit membaca `requirements.txt` otomatis.
4. Jika memakai Random Forest, ukuran `model.joblib` bisa melebihi 100 MB (batas GitHub). Gunakan MLP, Linear Regression, atau XGBoost.

## 3. Memperbarui model
Jalankan ulang `train_export.py`, lalu commit folder `artifacts/` yang baru. Aplikasi otomatis memakai artefak terbaru.

## 4. Fitur aplikasi
- **Prediksi:** skor, posisi relatif terhadap data latih, dan kisaran galat 90%.
- **Penjelasan SHAP:** kontribusi tujuh variabel untuk masukan saat ini.
- **Simulasi:** kurva prediksi saat satu variabel diubah.
- **Prediksi massal:** unggah CSV dan unduh hasilnya.
- **Tentang model:** MAE, RMSE, R², kepentingan SHAP global, dan catatan keterbatasan.

## 5. Catatan penting
- Aplikasi adalah alat bantu riset, **bukan alat diagnosis**. Model hanya menjelaskan sebagian kecil variasi skor, sehingga galat individual besar.
- Rentang slider mengikuti rentang data latih untuk mencegah ekstrapolasi.
- Jenjang pendidikan diturunkan otomatis dari usia karena pada data latih keduanya identik secara informasi.
