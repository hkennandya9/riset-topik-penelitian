# Deteksi Bahasa Pada Kalimat Teks Menggunakan Klasifikasi Naives Bayes

Nama : Hafizh Kennandya Maulana <br>
NPM : 20081010077 <br>
Kelas : Riset Informatika D081

## Problem Statement
Teks adalah salah satu bentuk komunikasi yang paling umum digunakan di dunia digital. Teks dapat ditulis dalam berbagai bahasa, tergantung pada preferensi dan latar belakang penulis. Namun, tidak semua orang dapat mengenali bahasa yang digunakan dalam teks secara otomatis. Hal ini dapat menyulitkan proses pemrosesan teks lebih lanjut, seperti penerjemahan, klasifikasi, atau analisis sentimen.

Salah satu cara untuk mengenali bahasa yang digunakan dalam teks adalah dengan menggunakan metode klasifikasi. Metode klasifikasi adalah teknik yang dapat mengelompokkan data berdasarkan karakteristik atau fitur tertentu. Salah satu metode klasifikasi yang populer dan sederhana adalah algoritma Naives Bayes. Algoritma Naives Bayes adalah algoritma yang berdasarkan pada teorema Bayes, yang menghitung probabilitas suatu kelas atau label berdasarkan frekuensi kemunculan fitur atau atribut dalam data.

## Research Questions
- Bagaimana cara mengimplementasikan algoritma Naives Bayes untuk mendeteksi bahasa pada teks?
- Seberapa akurat algoritma Naives Bayes dalam mendeteksi bahasa pada teks?

## Dataset
[Dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst/data) yang digunakan terdiri dari 22 ribu sampel dengan berbagai nilai unik. Target yang digunakan diambil dari kolom "language" yang memuat 22 varian bahasa, yaitu English, Arabic, French, Hindi, Urdu, Portuguese, Persian, Pushto, Spanish, Korean, Tamil, Turkish, Estonian, Russian, Romanian, Chinese, Swedish, Latin, Indonesian, Dutch, Japanese dan Thai.

Tiap nilai dalam kolom 'language' merepresentasikan bahasa yang bersesuaian dengan sampel tersebut. Gambaran lebih detail mengenai dataset, dapat dilihat pada tabel berikut

| Text | language |
| ------------- | ------------- |
| klement gottwaldi surnukeha palsameeriti ning paigutati mausoleumi surnukeha oli aga liiga hilja ja ... | Estonian |
| sebes joseph pereira thomas på eng the jesuits and the sino-russian treaty of nerchinsk the diary ... | Swedish |
| ถนนเจริญกรุง อักษรโรมัน thanon charoen krung เริ่มตั้งแต่ถนนสนามไชยถึงแม่น้ำเจ้าพระยาที่ถนนตก กรุงเท... | Thai |
| விசாகப்பட்டினம் தமிழ்ச்சங்கத்தை இந்துப் பத்திரிகை-விசாகப்பட்டின ஆசிரியர் சம்பத்துடன் இணைந்து விரிவுப... | Tamil |
| de spons behoort tot het geslacht haliclona en behoort tot de familie chalinidae de wetenschappelijk... | Dutch |
| エノが行きがかりでバスに乗ってしまい、気分が悪くなった際に助けるが、今すぐバスを降りたいと運転手に頼む際、本当のことを言ってしまうと彼女が恥ずかしい思いをすると察して「僕ウンコしたいんです」と言ってバ... | Japanese |
| ... | ... |

Pada tabel tersebut terdapat 2 kolom berupa "Text" dan "language". 
- <b>Text</b> adalah setiap baris yang berisi beberapa kalimat dalam bahasa tertentu yang dipilih
- <b>language</b> adalah nama bahasa di mana teks ditulis pada kolom "Text"

### Distribusi Data Pada Setiap Bahasa
![figure 1](https://github.com/hkennandya9/riset-topik-penelitian/assets/127032854/7f43e458-dd4e-4a9e-b613-92fab573645f)

Distribusi menunjukkan bahwa dataset memiliki tingkat keseimbangan yang lengkap, karena jumlah instan data setiap kategori bahasa berada pada tingkat kesetaraan yang sama yaitu 1000.

### Distribusi Panjang Kalimat Pada Data
![figure 2](https://github.com/hkennandya9/riset-topik-penelitian/assets/127032854/f2517afe-c5c8-4170-89b1-1d2e79cef365)

Distribusi menunjukkan sebagian besar kalimat teks pada dataset memiliki jumlah kata yang kurang dari 50 kata.

## Preprocessing
Preprocessing merupakan serangkaian langkah yang diterapkan pada data sebelum digunakan dalam analisis atau pelatihan model, bertujuan untuk membersihkan, mentransformasi, atau menyesuaikan data agar sesuai dengan algoritma yang akan dijalankan.

### Memuat Dataset
Mengimpor atau memuat dataset yang akan digunakan untuk analisis atau pelatihan model.
```python
import pandas

data = pandas.read_csv(f'dataset.csv')
data.columns = ('text', 'language')
data
```

### Membersihkan Kalimat Teks Pada Data
Proses membersihkan teks pada data melibatkan langkah-langkah detil yang bertujuan untuk menghasilkan kalimat yang bersih, termasuk penghapusan karakter tidak diinginkan, normalisasi kata, dan pembersihan umum untuk memastikan kualitas teks yang optimal sebelum dilibatkan dalam tahap analisis atau pemodelan lebih lanjut.
```python
import re

def clean_txt(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]',' ',text)
    text=re.sub(r'[_0-9]',' ',text)
    text=re.sub(r'\s\s+',' ',text)
    return text

# Penerapan Function clean_txt
txt = 'Saya (&*(()))mencuci tangan $agar #terhindar dari$ kuman'
print(clean_txt(txt))
```

### Pemisahan Data Latih dan Uji (Train Test Split)
Prosedur validasi model yang mengindikasikan seberapa baik kinerja model pada data yang baru.
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.text.values, data.language.values, test_size=0.1, random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
```
#### Output
```
((19800,), (19800,), (2200,), (2200,))
```

### Vektorisasi (Vectorization)
Metode pendekatan untuk mengoptimalkan algoritma agar lebih efisien.
 ```python
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

x_train = [clean_txt(text) for text in tqdm(x_train)]
x_test = [clean_txt(text) for text in tqdm(x_test)]

# Tfidf Vectorizer
tfidf = TfidfVectorizer()
tfidf.fit(x_train)

x_train_ready = tfidf.transform(x_train)
x_test_ready = tfidf.transform(x_test)
```
#### Output
```
100%|██████████| 19800/19800 [00:01<00:00, 16246.68it/s]
100%|██████████| 2200/2200 [00:00<00:00, 16688.22it/s]
```

### Label Encoding
Metode untuk melakukan konversi label menjadi bentuk numerik pada model machine learning.
```python
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
enc.fit(y_train)
y_train_ready = enc.transform(y_train)
y_test_ready = enc.transform(y_test)
labels = enc.classes_

# Pengujian Encoder
preds = enc.inverse_transform([0,3,6])
preds
```
#### Output
```
array(['Arabic', 'English', 'Hindi'], dtype=object)
```

## Implementasi Klasifikasi Naives Baiyes
Setelah melalui proses preprocessing, data yang telah diolah disajikan ke dalam model machine learning untuk dilatih. Data yang telah terlatih dapat digunakan untuk melakukan prediksi bahasa dari suatu teks kalimat.
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(x_train_ready, y_train_ready)
```
### Membuat Model Pipeline
Membuat model pipeline untuk menggabungkan vektorizer dan model terlatih menjadi satu objek

```python
from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer', tfidf),('nb', nb)])
```
### Uji Model Prediksi Bahasa Melalui Kalimat Teks 
Melakukan prediksi berdasarkan model dalam menentukan bahasa pada input berupa suatu kalimat teks 

```python
# Function untuk melakukan prediksi bahasa dari kalimat teks
def predict(text):
    pred = model.predict([clean_txt(text)])
    ans = enc.inverse_transform(pred)
    return ans[0]

example_text = ['Saya mencuci tangan agar terhindar dari kuman', 
                'I wash my hands to avoid germs', 
                'أغسل يدي لتجنب الجراثيم',
                'Ik was mijn handen om ziektekiemen te voorkomen',
                'कीटाणुओं से बचने के लिए मैं अपने हाथ धोता हूं',
                'Me lavo las manos para evitar gérmenes']
result_language = []

for text in example_text:
  result_language.append(predict(text))

for language in result_language:
  print(language)
```
#### Output
```
Indonesian
English
Arabic
Dutch
Hindi
Spanish
```

## Analisis Pengujian
Pada tahapan ini dilakukan analisis untuk menilai sejauh mana model deteksi bahasa dapat secara akurat mengenali dan mengelompokkan bahasa dalam kalimat teks. Sejumlah metode digunakan dalam uji coba, yaitu 
- Mean Squared Error (MSE)
- Matriks Pengujian (Confusion Matrix)
- Metrik Kinerja (akurasi, presisi, recall, dan F1-score)

Dalam analisis pengujian yang lebih mendalam, diperlukan persiapan data uji  yang akan menjadi subjek prediksi oleh model. Proses ini melibatkan serangkaian langkah termasuk pengumpulan, penyaringan, dan pengaturan data agar sesuai dengan kebutuhan pengujian.

#### Data Uji
| Language     | Jumlah |
|--------------|--------|
| Swedish      |   7    |
| Indonesian   |   2    |
| English      |   2    |
| Latin        |   2    |
| Romanian     |   2    |
| Estonian     |   2    |
| Tamil        |   2    |
| Korean       |   2    |
| Pushto       |   2    |
| Persian      |   2    |
| Urdu         |   2    |
| Hindi        |   2    |
| Russian      |   2    |
| Spanish      |   2    |
| Portugese    |   2    |
| French       |   2    |
| Arabic       |   2    |
| Dutch        |   2    |
| Thai         |   2    |
| Turkish      |   1    |

Hasil analisis pengujian dapat membantu menemukan bagian-bagian yang perlu diperbaiki atau dioptimalkan pada model. Hal ini membantu menilai seberapa baik model dalam mendeteksi bahasa yang mnngkin melibatkan penyesuaian parameter model, penambahan lebih banyak data latih, atau menggunakan teknik pemrosesan teks tambahan.

### MSE (Mean Square Error)
Analisis Mean Square Error bertujuan untuk mengukur sejauh mana perbedaan antara hasil prediksi model Naive Bayes dan nilai sebenarnya dalam melakukan deteksi bahasa. Semakin rendah nilai MSE, semakin kecil deviasi antara hasil prediksi dan nilai sebenarnya, menunjukkan kinerja model yang lebih baik.
#### Rumus MSE
![image](https://github.com/hkennandya9/riset-topik-penelitian/assets/127032854/865e2602-d390-4232-8d98-b1e122fc938e)
#### Hasil Perhitungan MSE
```
0.11363636363636363
```
- Hasil perhitungan menunjukkan tingkat kesalahan rata-rata kuadrat antara prediksi model Naive Bayes dengan nilai sebenarnya dalam deteksi bahasa yaitu 0.11363636363636363.
- Deviasi antara prediksi model dan nilai memiliki hasil yang cukup rendah, hal ini menandakan bahwa model secara umum sudah dapat memahami variasi dalam bahasa yang ada dalam dataset.

### Matriks Pengujian (Confusion Matrix)
Matriks Pengujian memberikan pemahaman yang lebih mendalam tentang hasil prediksi model serta membantu mengidentifikasi area di mana model cenderung membuat kesalahan.

#### Hasil Matriks Pengujian
![image](https://github.com/hkennandya9/riset-topik-penelitian/assets/127032854/545d426e-cbdf-4694-81f1-020a42390301)

- Model memiliki kinerja sempurna dalam mengenali kasus negatif (True Negative tinggi)
- Model tidak membuat kesalahan positif palsu (False Positive = 0)
- Terdapat beberapa kesalahan dalam mengenali kasus positif (False Negative = 5)
- Model memiliki kinerja baik dalam mengenali kasus positif (True Positive tinggi)

| Label               | Hasil |
|---------------------|-------|
| True Negative (TN)  | 0     |
| False Positive (FP) | 0     |
| False Negative (FN) | 5     |
| True Positive (TP)  | 39    |

Meskipun hasil yang ditunjukkan baik dalam mengenali kasus positif, perhatian tetap diperlukan untuk memahami dampak False Negative dan memastikan bahwa model dapat diandalkan dalam situasi di mana mendeteksi kasus positif sangat penting.

### Metrik Kinerja
