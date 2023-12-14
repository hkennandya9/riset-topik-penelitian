# Riset Informatika

Nama : Hafizh Kennandya Maulana <br>
NPM : 20081010077 <br>
Kelas : Riset Informatika D081

## Deteksi Bahasa Pada Teks Menggunakan Klasifikasi Naives Bayes

### Problem Statement
Teks adalah salah satu bentuk komunikasi yang paling umum digunakan di dunia digital. Teks dapat ditulis dalam berbagai bahasa, tergantung pada preferensi dan latar belakang penulis. Namun, tidak semua orang dapat mengenali bahasa yang digunakan dalam teks secara otomatis. Hal ini dapat menyulitkan proses pemrosesan teks lebih lanjut, seperti penerjemahan, klasifikasi, atau analisis sentimen.

Salah satu cara untuk mengenali bahasa yang digunakan dalam teks adalah dengan menggunakan metode klasifikasi. Metode klasifikasi adalah teknik yang dapat mengelompokkan data berdasarkan karakteristik atau fitur tertentu. Salah satu metode klasifikasi yang populer dan sederhana adalah algoritma Naives Bayes. Algoritma Naives Bayes adalah algoritma yang berdasarkan pada teorema Bayes, yang menghitung probabilitas suatu kelas atau label berdasarkan frekuensi kemunculan fitur atau atribut dalam data.

### Research Questions
- Bagaimana cara mengimplementasikan algoritma Naives Bayes untuk mendeteksi bahasa pada teks?
- Seberapa akurat algoritma Naives Bayes dalam mendeteksi bahasa pada teks?
- Bagaimana performa algoritma Naives Bayes dalam mendeteksi bahasa pada teks, dibandingkan dengan metode klasifikasi lainnya?

### Dataset : Language Identification dataset
[Dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst/data) yang digunakan terdiri dari 22 ribu sampel dengan berbagai nilai unik. Target yang digunakan diambil dari kolom 'language' yang memuat 22 varian bahasa, yaitu English, Arabic, French, Hindi, Urdu, Portuguese, Persian, Pushto, Spanish, Korean, Tamil, Turkish, Estonian, Russian, Romanian, Chinese, Swedish, Latin, Indonesian, Dutch, Japanese dan Thai.

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
- <b>Text</b> adalah setiap baris yang berisi beberapa kalimat dalam bahasa tertentu yang dipilih.
- <b>language</b> adalah Nama bahasa di mana teks ditulis pada kolom "Text"
