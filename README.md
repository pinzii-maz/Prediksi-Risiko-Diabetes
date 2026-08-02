# Prediksi Risiko Diabetes

Proyek ini menggunakan dataset diabetes Pima untuk membangun model klasifikasi Logistic Regression dalam rangka memprediksi apakah seorang pasien menderita diabetes atau tidak.

## Struktur Proyek

- `Belajar Logistik Regresi.ipynb`: Notebook utama yang berisi alur lengkap Exploratory Data Analysis (EDA), data preparation, model training, tuning, evaluasi, dan interpretasi.
- `diabetes.csv`: Dataset yang berisi data medis wanita Pima dengan fitur numerik dan label `Outcome`.

## Tujuan

- Memahami dataset diabetes Pima
- Menyiapkan data dengan penanganan missing values pada fitur klinis
- Melatih model Logistic Regression
- Menggunakan cross-validation dan GridSearchCV untuk tuning hyperparameter
- Menetapkan threshold klasifikasi yang optimal berdasarkan metrik recall dan F1
- Mengevaluasi performa model menggunakan metrik klasifikasi, confusion matrix, ROC curve, dan koefisien model

## Langkah Utama Notebook

1. Import library dan inisialisasi.
2. Membaca dataset `diabetes.csv`.
3. Menampilkan informasi dataset:
   - Ukuran data
   - Nama kolom
   - Tipe data
   - Statistik deskriptif
   - Distribusi kelas target
4. Visualisasi data:
   - Histogram fitur
   - Boxplot
   - Korelasi
   - Boxplot fitur penting terhadap label target
   - Pairplot
5. Verifikasi kualitas data:
   - Cek missing value eksplisit
   - Cek nilai 0 pada fitur `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
   - Mengganti nilai 0 menjadi `NaN`
6. Persiapan data:
   - Membagi fitur (`X`) dan target (`y`)
   - Split data menjadi train dan test dengan stratifikasi
7. Pipeline model:
   - `SimpleImputer(strategy='median')`
   - `StandardScaler()`
   - `LogisticRegression(random_state=42, max_iter=1000)`
8. Evaluasi cross-validation:
   - StratifiedKFold dengan 5 fold
   - Metrik: accuracy, precision, recall, f1, ROC-AUC
9. Hyperparameter tuning:
   - GridSearchCV pada nilai `C` logistic regression
   - Scoring menggunakan ROC-AUC
10. Threshold tuning:
    - Menentukan threshold kelas berdasarkan probabilitas prediksi
    - Memilih threshold dengan recall >= 0.75 dan F1 tertinggi
11. Evaluasi final pada data test:
    - Confusion matrix
    - Accuracy, precision, recall, specificity, F1, ROC-AUC
    - ROC curve
    - Perbandingan threshold 0.50 vs threshold optimal
12. Interpretasi model:
    - Koefisien final logistic regression
    - Odds ratio
    - Visualisasi koefisien fitur

## Temuan dan Insight

- Dataset diolah dengan mengganti nilai 0 untuk fitur medis tertentu menjadi `NaN` sebelum imputasi.
- Model Logistic Regression dievaluasi dengan cross-validation dan tuned menggunakan GridSearchCV.
- Threshold klasifikasi disesuaikan untuk menjaga recall tinggi, yang penting dalam konteks prediksi risiko diabetes.
- Evaluasi final mencakup interpretasi koefisien model untuk melihat fitur mana yang paling berpengaruh.

## Dependensi

Beberapa library yang digunakan dalam notebook:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Cara Menjalankan

1. Pastikan Python dan dependensi terpasang, misalnya via `pip install -r requirements.txt` jika menggunakan file dependensi.
2. Buka `Belajar Logistik Regresi.ipynb` di Jupyter Notebook / JupyterLab / VS Code.
3. Jalankan sel notebook secara berurutan.

## Catatan

- Model menggunakan `random_state=42` agar hasil reproducible.
- Stratified split digunakan agar proporsi kelas target tetap terjaga antara train dan test.
- Threshold akhir dipilih berdasarkan analisis metrik pada training out-of-fold predictions.
