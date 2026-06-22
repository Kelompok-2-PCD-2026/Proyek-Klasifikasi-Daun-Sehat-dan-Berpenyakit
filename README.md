# Klasifikasi Daun Sehat dan Berpenyakit Menggunakan KNN, SVM, dan Random Forest

## Nama Anggota
- JAUDA AVARO ZUFARU :F1D02410059
- KADIVA ALIFIA NURHIDAYAH : F1D02410060
- NURHAYATI NINGSIH : F2D02410085
- SITI ANANDA RAHMA : F1D02410095

---

## Project Overview

Project ini merupakan implementasi Pengolahan Citra Digital (PCD) untuk melakukan klasifikasi citra daun sehat dan daun berpenyakit berdasarkan dataset gambar. Dataset yang digunakan terdiri dari dua kelas utama, yaitu:

- `healthy`
- `diseased`

Tujuan utama dari project ini adalah membandingkan beberapa tahapan preprocessing citra untuk melihat pengaruhnya terhadap hasil ekstraksi fitur dan performa model klasifikasi. Proses klasifikasi dilakukan dengan memanfaatkan fitur tekstur dari citra menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)**, kemudian hasil fiturnya digunakan untuk melatih beberapa model machine learning.

Model klasifikasi yang digunakan dalam project ini adalah:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Project ini tidak hanya berfokus pada nilai akurasi akhir, tetapi juga pada pemilihan preprocessing yang sesuai, proses ekstraksi fitur, serta analisis hasil evaluasi model.

---

## Struktur Repository

```text
Kelompok-2-PCD-2026/
│
├── dataset/
│   ├── healthy/
│   └── diseased/
│
├── percobaan/
│   ├── Percobaan1.ipynb
│   ├── Percobaan2.ipynb
│   └── Percobaan3.ipynb
│   └── Percobaan4.ipynb
│
└── README.md
```

Keterangan:

- Folder `dataset/` berisi gambar asli yang dikelompokkan berdasarkan label kelas.
- Folder `percobaan/` berisi notebook utama untuk preprocessing, ekstraksi fitur, dan klasifikasi.
- File `README.md` berisi dokumentasi project.

---

## Import Library

Library yang digunakan pada project ini menyesuaikan kebutuhan setiap tahap, mulai dari pembacaan gambar, pengolahan citra, ekstraksi fitur, hingga klasifikasi.

Beberapa library utama yang digunakan:

```python
import os
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
```

---

## Load Data

Tahap pertama adalah membaca dataset gambar dari folder `dataset/`. Setiap subfolder pada folder dataset dianggap sebagai label kelas.

Label yang digunakan:

```text
healthy
diseased
```

Contoh alur load data:

```python
data = []
labels = []
file_name = []
for sub_folder in os.listdir("/content/drive/MyDrive/PCD_Project/dataset/"):
    sub_folder_files = os.listdir(os.path.join("/content/drive/MyDrive/PCD_Project/dataset/", sub_folder))
    for i, filename in enumerate(sub_folder_files):
        img_path = os.path.join("/content/drive/MyDrive/PCD_Project/dataset/", sub_folder, filename)
        img = cv.imread(img_path)
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        data.append(img)
        labels.append(sub_folder)
        name = os.path.splitext(filename)[0]
        file_name.append(filename)

data = np.array(data)
labels = np.array(labels)
```

Pada project ini, gambar kemudian diseragamkan ukurannya menjadi:

```python
img = cv.resize(img, (128, 128))
```

Resize dilakukan agar seluruh gambar memiliki ukuran yang sama sebelum masuk ke tahap preprocessing dan ekstraksi fitur.

---

## Data Understanding

Dataset terdiri dari dua kelas citra kondisi daun. Setiap kelas disimpan dalam folder yang berbeda sehingga proses labeling dapat dilakukan secara otomatis berdasarkan nama folder.

Distribusi dataset:

| Label | Jumlah Data |
|---|---:|
| `healthy` | 80 gambar |
| `diseased` | 80 gambar |
| **Total** | **160 gambar** |

Karakteristik umum dataset:

- Gambar berasal dari dua jenis kondisi daun yang berbeda.
- Ukuran dan orientasi gambar dapat bervariasi.
- Background gambar tidak selalu seragam.
- Kondisi pencahayaan dapat berbeda pada setiap gambar.
- Beberapa objek memiliki tekstur yang cukup mirip dengan background, sehingga preprocessing diperlukan untuk memperjelas informasi citra.

Karena dataset memiliki jumlah data yang seimbang untuk setiap kelas, model tidak terlalu terdampak oleh masalah imbalance antar kelas.

---

## Data Preparation

### Data Augmentation

Pada project ini, data augmentation tidak menjadi tahap utama karena jumlah data pada tiap kelas sudah mencapai 80 gambar. Jumlah tersebut masih berada pada batas minimal yang dapat digunakan untuk percobaan klasifikasi sederhana.

Namun, augmentation tetap dapat diterapkan apabila ingin menambah variasi data, misalnya dengan:

- Rotasi gambar
- Flip horizontal
- Perubahan brightness
- Zoom atau cropping ringan

Augmentation dapat membantu model mengenali objek dalam berbagai posisi dan kondisi pencahayaan, tetapi pada project ini fokus utama diarahkan pada perbandingan preprocessing.

---

## Preprocessing

Preprocessing dilakukan untuk menyeragamkan citra, mengurangi noise, memperbaiki kualitas visual, dan menonjolkan karakteristik tertentu sebelum dilakukan ekstraksi fitur GLCM.

Terdapat 4 percobaan untuk melakukan preprocessing

| Preprocessing | Tahapan |
|---|---|
| `Percobaan1.ipynb` | Resize → Grayscale |
| `Percobaan2.ipynb` | Resize → Grayscale → Histogram Equalization → Median Filter |
| `Percobaan3.ipynb` | Resize → Grayscale → Histogram Equalization → Median Filter → Sobel → Thresholding |
| `Percobaan4.ipynb` | Resize → Grayscale → Histogram Equalization → Median Filter → Sobel → Thresholding → Opening → Closing |


### Penjelasan Preprocessing

#### 1. Resize

Proses resize digunakan untuk mengubah dimensi piksel atau ukuran fisik suatu gambar menjadi standar tertentu, seperti 128x128. Hal ini sangat diperlukan agar proses pengolahan citra dan ekstraksi fitur selanjutnya dapat berjalan secara konsisten. Selain itu, penyesuaian ukuran ini juga efektif untuk membuat ukuran file gambar menjadi lebih ringan.

#### 2. Grayscale

Proses grayscale digunakan untuk mengubah citra RGB/BGR menjadi citra derajat keabuan. Metode ini bekerja dengan mereduksi tiga saluran warna menjadi hanya satu saluran intensitas tanpa kehilangan karakteristik penting seperti bentuk, tepi, dan tekstur objek. Dengan menyederhanakan informasi piksel tersebut, proses komputasi pada tahapan analisis citra selanjutnya akan menjadi jauh lebih cepat dan efisien.

#### 3. Histogram Equalization

Histogram Equalization adalah metode dalam Pengolahan Citra Digital yang digunakan untuk meningkatkan kontras gambar secara otomatis. Cara kerjanya adalah dengan meratakan distribusi nilai intensitas piksel yang menumpuk agar tersebar merata ke seluruh rentang derajat keabuan. Proses ini sangat berguna untuk memunculkan detail objek yang tersembunyi pada citra yang terlalu gelap atau terlalu terang. 

#### 4. Median Filter

Median Filter digunakan dalam Pengolahan Citra Digital untuk menghilangkan derau atau noise, khususnya jenis salt-and-pepper, tanpa mengaburkan tepi objek. Metode ini bekerja dengan cara mengganti nilai suatu piksel dengan nilai median dari piksel-piksel di area sekitarnya. Hasilnya, kualitas citra meningkat secara signifikan karena gangguan bintik hitam-putih tereduksi dengan tetap mempertahankan detail struktural gambar.

#### 5. Operator Sobel

Operator Sobel digunakan dalam pengolahan citra digital untuk mendeteksi tepi dari suatu objek secara efektif. Tahap ini bekerja dengan menonjolkan bentuk, garis kontur, serta batas-batas fisik pada objek hama yang sedang diamati. Melalui proses tersebut, informasi struktur visual citra menjadi lebih jelas sehingga mempermudah algoritma dalam mengenali pola objek.

#### 6. Thresholding

Thresholding digunakan untuk mengubah citra hasil deteksi tepi menjadi citra biner yang hanya terdiri dari warna hitam dan putih. Tahap ini sangat membantu dalam memisahkan area objek utama secara tegas dari latar belakangnya (background). Namun, proses pengambangan ini juga berpotensi menghilangkan detail tekstur penting jika nilai ambang yang ditentukan kurang sesuai.

#### 7. Operasi Opening

Operasi Opening dalam pengolahan citra digital digunakan untuk menghilangkan derau (noise) berukuran kecil sekaligus menghaluskan batas objek. Proses morfologi ini dilakukan dengan menjalankan operasi erosi terlebih dahulu, kemudian dilanjutkan dengan operasi dilasi menggunakan structuring element yang sama. Melalui tahapan tersebut, objek-objek kecil yang mengganggu akan terhapus tanpa mengubah ukuran asli dari objek utama secara signifikan.

#### 8. Operasi Closing

Operasi Closing dalam pengolahan citra digital digunakan untuk menutup lubang kecil di dalam objek sekaligus menyambungkan retakan atau batas yang terputus. Proses morfologi ini bekerja dengan cara menjalankan operasi dilasi terlebih dahulu untuk memperluas area, kemudian diikuti dengan operasi erosi menggunakan structuring element yang sama. Melalui tahapan tersebut, struktur utama objek dapat dipertahankan dan disatukan menjadi lebih solid tanpa mengubah ukuran aslinya secara signifikan.

---

## Skenario Percobaan

Project ini menggunakan ... jenis preprocessing. Untuk melihat pengaruh penambahan preprocessing terhadap performa model, percobaan dapat dibagi menjadi beberapa skenario:

| Percobaan | Preprocessing yang Digunakan |
|---|---|
| Percobaan 1 | `percobaan1` |
| Percobaan 2 | `percobaan1`, `percobaan2` |
| Percobaan 3 | `percobaan1`, `percobaan2`, `percobaan3` |
| Percobaan 4 | `percobaan1`, `percobaan2`, `percobaan3`, `percobaan4` |

Dengan skenario ini, setiap hasil klasifikasi dapat dibandingkan untuk mengetahui preprocessing mana yang paling sesuai terhadap dataset hama tanaman.

---

## Feature Extraction

Tahap ekstraksi fitur dilakukan menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)**.

GLCM digunakan untuk mengambil informasi tekstur dari citra berdasarkan hubungan spasial antar piksel. Pada project ini, fitur GLCM dihitung pada empat arah sudut:

- 0°
- 45°
- 90°
- 135°

Implementasi pada notebook menggunakan `graycomatrix` dan `graycoprops` dari `skimage.feature`.

Fitur yang diekstraksi:

| Fitur | Keterangan |
|---|---|
| Contrast | Mengukur perbedaan intensitas antara piksel bertetangga |
| Homogeneity | Mengukur keseragaman tekstur citra |
| Dissimilarity | Mengukur tingkat perbedaan antar piksel |
| Entropy | Mengukur kompleksitas atau ketidakteraturan tekstur |
| ASM | Mengukur keseragaman distribusi nilai GLCM |
| Energy | Mengukur energi atau kekuatan pola tekstur |
| Correlation | Mengukur hubungan linear antar piksel |

Contoh struktur fungsi ekstraksi fitur:

```python
def extract_glcm_features(image):
    angles = [0, 45, 90, 135]
    features = {}

    for angle in angles:
        matriks = glcm(image, angle)

        features[f"Contrast{angle}"] = contrast(matriks)
        features[f"Homogeneity{angle}"] = homogenity(matriks)
        features[f"Dissimilarity{angle}"] = dissimilarity(matriks)
        features[f"Entropy{angle}"] = entropyGlcm(matriks)
        features[f"ASM{angle}"] = ASM(matriks)
        features[f"Energy{angle}"] = energy(matriks)
        features[f"Correlation{angle}"] = correlation(matriks)

    return features
```

Hasil ekstraksi fitur disimpan ke dalam folder:

```text
ekstraksi/
```

File CSV yang dihasilkan:

```text
hasil_ekstraksi_percobaan1.csv
hasil_ekstraksi_percobaan2.csv
hasil_ekstraksi_percobaan3.csv
hasil_ekstraksi_percobaan4.csv
```

---

## Feature Selection

Feature selection dilakukan untuk memilih fitur yang paling relevan dan mengurangi fitur yang terlalu berkorelasi satu sama lain.

Pada template project, feature selection dapat dilakukan menggunakan correlation. Tahap ini membantu mengurangi redundansi fitur sehingga proses klasifikasi menjadi lebih efisien.

Contoh pendekatan correlation:

```python
correlation_matrix = hasilEkstrak.drop(columns=['Label','Filename']).corr()
```

Fitur yang memiliki korelasi terlalu tinggi dapat dipertimbangkan untuk dihapus agar model tidak mempelajari informasi yang berulang.

---

## Splitting Data

Dataset hasil ekstraksi fitur dibagi menjadi data training dan data testing.

Perbandingan data yang dapat digunakan:

```text
80% data training
20% data testing
```

Contoh kode:

```python
X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

---

## Normalization

Normalisasi dilakukan agar seluruh fitur memiliki skala nilai yang lebih seragam. Hal ini penting terutama untuk model seperti KNN dan SVM yang sensitif terhadap jarak atau skala fitur.

Metode normalisasi yang dapat digunakan:

```python
X_test = (X_test - X_train.mean()) / X_train.std()
X_train = (X_train - X_train.mean()) / X_train.std()
```

---

## Modeling

Model klasifikasi yang digunakan pada project ini terdiri dari tiga algoritma:

### 1. K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) melakukan klasifikasi dengan cara mengukur tingkat kemiripan atau kedekatan jarak antara data baru dan data latihan. Meskipun konsep algoritmanya tergolong sederhana, performa model ini sangat sensitif terhadap perbedaan skala pada setiap fitur data. Oleh karena itu, tahapan normalisasi fitur sangat penting dilakukan terlebih dahulu agar variabel dengan nilai besar tidak mendominasi perhitungan jarak.

### 2. Support Vector Machine (SVM)

Support Vector Machine (SVM) berfungsi dengan menentukan hyperplane pembatas paling optimal guna mengelompokkan kelas-kelas data secara akurat. Metode ini terbukti sangat andal ketika menangani data yang memiliki dimensi fitur tinggi dan kompleks. Karena keunggulan tersebut, SVM menjadi pilihan yang sangat tepat untuk mengklasifikasikan data hasil ekstraksi fitur Gray-Level Co-occurrence Matrix (GLCM).

### 3. Random Forest

Random Forest merupakan algoritma ensemble yang bekerja dengan cara menggabungkan hasil prediksi dari banyak pohon keputusan (decision tree). Struktur ini membuat model menjadi sangat stabil dan tidak mudah terpengaruh oleh adanya variasi atau perubahan kecil pada fitur data. Selain itu, algoritma ini juga memiliki kemampuan yang andal dalam memetakan serta menangani hubungan non-linear yang kompleks antar variabel di dalamnya.

Contoh training model:

```python
knn = KNeighborsClassifier()
svm = SVC()
rf = RandomForestClassifier(random_state=35)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
```

---

## Evaluation

Evaluasi model dilakukan untuk mengetahui performa klasifikasi pada setiap preprocessing.

Metrik evaluasi yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Contoh kode evaluasi:

```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Format tabel hasil evaluasi:

| Preprocessing | Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Resize + Grayscale | Random Forest | 87.50%% | 90.00% | 87.50% | 87.30% |
| Resize + Grayscale | SVM | 90.62% | 92.10% | 90.62% | 90.54% |
| Resize + Grayscale | KNN | 87.50%% | 90.00% | 87.50% | 87.30% |
| Resize + Grayscale + Median Filter + Histogram Equalization | Random Forest | 90.62% | 91% | 91% | 91% |
| Resize + Grayscale + Median Filter + Histogram Equalization | SVM | 90.62% | 91% | 91% | 91% |
| Resize + Grayscale + Median Filter + Histogram Equalization | KNN | 87.50% | 88% | 88% | 87% |
| Resize + Grayscale + Median Filter + Sobel + Thresholding | Random Forest | 81.25% | 83.33% | 81.25% | 80.95% |
| Resize + Grayscale + Median Filter + Sobel + Thresholding | SVM | 84.37% | 88.09% | 84.37% | 83.98% |
| Resize + Grayscale + Median Filter + Sobel + Thresholding | KNN | 84.37% | 85.62% | 84.37% | 84.23% |
| Resize + Grayscale + Median Filter + Sobel + Thresholding + Opening + Closing| Random Forest | 98.44% | 87.50% | 87.50% | 87.50% |
| Resize + Grayscale + Median Filter + Sobel + Thresholding + Opening + Closing| SVM | 89.00% | 90.62% | 90.62% | 90.62% |
| Resize + Grayscale + Median Filter + Sobel + Thresholding + Opening + Closing| KNN | 88.00% | 84.00% | 88.00% | 84.00% |


### Analisis Preprocessing
Analisis:

- Pada percobaan pertama dilakukan resize dan grayscale, dilakukan resize 128x128 yang membuat ukuran citra seragam dan grayscale untuk mengubah citra RGB/BGR menjadi citra derajat keabuan. Grayscale mengubah citra tiga saluran warna menjadi hanya satu saluran warna tanpa menghilangkan karakteristik penting seperti bentuk, tepi, dan tekstur objek. Dengan menyederhanakan informasi piksel tersebut, proses pengolahan citra akan menjadi jauh lebih cepat dan efisien.
- Pada percobaan kedua dilakukan preprocessing Resize → Grayscale → Histogram Equalization → Median Filter sebelum melakukan ekstraksi fitur GLCM. Berdasarkan hasil pengujian yang telah dilakukan, didapatkan hasil SVM dan Random Forest memiliki performa terbaik yakni dengan akurasi sebesar 90,62% dengan  hasil paling stabil dan akurat, sedangkan KNN memperoleh akurasi 87,50%. Peningkatan kontras karena proses histogram equalization dan pengurangan noise-noise dengan median filter membantu menghasilkan fitur tekstur yang lebih jelas. Hal ini menunjukkan bahwa kombinasi preprocessing pada percobaan kedua cukup efektif dalam mendukung proses klasifikasi pada daun sehat dan daun berpenyakit.
- Histogram Equalization → Median Filter → Sobel → Thresholding sebelum melakukan ekstraksi fitur GLCM. Berdasarkan hasil pengujian yang telah dilakukan, didapatkan hasil SVM dan KNN memiliki performa terbaik yakni dengan akurasi sebesar 84,37% dengan hasil paling stabil dan akurat, sedangkan Random Forest memperoleh akurasi 81,25%. Penerapan deteksi tepi Sobel yang dilanjutkan dengan thresholding membantu menonjolkan struktur garis tepi pada citra biner daun, sehingga fitur tekstur GLCM yang diekstraksi menjadi lebih representatif terhadap perbedaan pola antara daun sehat dan daun berpenyakit. Namun demikian, hasil ini menunjukkan bahwa penambahan preprocessing Sobel dan thresholding pada percobaan ketiga tidak memberikan peningkatan performa klasifikasi dibandingkan percobaan kedua, karena proses thresholding menjadi citra biner hitam-putih justru menghilangkan sebagian informasi tekstur yang sebelumnya masih tersimpan dalam variasi nilai keabuan piksel, sehingga fitur GLCM yang diekstraksi menjadi kurang kaya dibandingkan percobaan kedua.


---

## Cara Menjalankan Project

1. Clone repository:

```bash
git clone https://github.com/Kelompok-2-PCD-2026/Proyek-Klasifikasi-Daun-Sehat-dan-Berpenyakit.git
```

2. Install library yang dibutuhkan:

```bash
pip install numpy pandas matplotlib opencv-python scikit-image scipy scikit-learn
```

3. Jalankan notebook secara berurutan:

```text
Percobaan1.ipynb
Percobaan2.ipynb
Percobaan3.ipynb
Percobaan4.ipynb
```

---

## Output Project

Output utama dari project ini adalah:

```text
percobaan/
ekstraksi/
classification report
confusion matrix
tabel perbandingan akurasi model
```

Folder `percobaan/` berisi hasil citra dari setiap tahapan preprocessing. Folder `ekstraksi/` berisi file CSV fitur GLCM yang digunakan pada tahap klasifikasi.

## Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan, dapat disimpulkan bahwa melalui tahapan resize, grayscale, histogram equalization, median filtering, deteksi tepi Sobel, thresholding, serta operasi morfologi opening dan closing, diperoleh citra daun yang memiliki kualitas lebih baik dibandingkan citra awal. Noise berhasil dikurangi, kontras meningkat, tepi daun menjadi lebih jelas, dan objek daun dapat dipisahkan dari latar belakang dengan baik. Hasil ini menghasilkan citra yang lebih representatif untuk proses ekstraksi fitur dan klasifikasi pada tahap selanjutnya.
