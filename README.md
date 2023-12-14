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
import pandas as pd

data = pd.read_csv(f'dataset.csv')
data.columns = ('text','language')
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
txt = 'Adik (&*(()))mencuci tangan $agar #terhindar dari$ kuman'
print(clean_txt(txt))
```

### Pemisahan Data Latih dan Uji (Train Test Split)
Prosedur validasi model yang mengindikasikan seberapa baik kinerja model pada data yang baru.
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.text.values, data.language.values, test_size=0.1, random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
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
