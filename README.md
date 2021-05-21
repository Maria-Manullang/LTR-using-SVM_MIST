# Learning To Rank with Pairwise Approach using Support Vector Machine
Learning to rank mengacu pada melakukan peringkat dalam sistem pengambilan informasi menggunakan machine learning. Dengan adanya sekumpulan dokumen dan user query, fungsi ini dapat secara tepat memprediksi skor untuk masing-masing dokumen tersebut pada saatnya untuk menentukan rank secara efektif [2]. Pada percobaan ini akan dibahas mengenai konsep dari learning to rank dengan pendekatan pairwise menggunakan support vector machine dimana algoritma ini akan menggunakan pasangan kueri dan dokumen untuk dapat melakukan pemeringkatan. Pemeringkatan dilakukan pada dataset LETOR 4.0 yang memiliki 46 fitur yang berbeda dengan 5 fold yang terdiri dari kombinasi 4 dataset untuk membangun data training, testing, dan validasi. Pada metode algoritma SVM, digunakan kernel linear untuk membangun model agar memberikan kemungkinan terkecil terjadinya kesalahan pada model apabila terdapat kesalahan pada data. Evaluasi performa perankingan Learning to Rank menggunakan daftar peringkat Normalized Discounted Cumulative Gain (NDCG).


### Dataset
Million queries dataset from TREC 2008 :
[MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0)

| Dataset name |       | rows   | columns | num samples in queries (min, median, max) | 
|--------------|-------|--------|---------|-------------------------------------------| 
| mq2008       | train | 9630   | 49      | (5, 8, 121)                               | 
|              | test  | 2874   | 49      | (6, 14, 119)                              | 

### Hasil Evaluasi
Hasil eksperimen dari proyek ini berupa skor NDCG yang dihasilkan dari kelima fold. Hasil fold yang didapatkan berasal dari data testing yang telah dilakukan pemeringkatan menggunakan model dengan algoritma SVM.
| Fold ke-  | NDCG |
|-----------|------|
| 1         | 0.37 |
| 2         | 0.39 |
| 3         | 0.37 |
| 4         | 0.44 |
| 5         | 0.42 |

Pada percobaan kali ini dilakukan pemeringkatan terhadap dataset LETOR 4.0, yaitu dataset MQ2008 yang terdiri dari 5 fold dengan kombinasi dari data S1, S2, S3, dan S4 dengan menerapkan metode algoritma Support Vector Machine (SVM). Hasil percobaan dievaluasi dengan menggunakan metric NDCG untuk dapat menentukan apakah pemeringkatan telah dilakukan dengan baik. Dari percobaan yang dilakukan, didapatkan nilai NDCG untuk fold 1 sebesar 0.37, untuk fold 2 sebesar 0.39, untuk fold 3 sebesar 0.37, untuk fold 4 sebesar 0.44, dan untuk fold 5 sebesar 0.42 dengan nilai rata-rata NDCG keseluruhan dataset adalah sebesar 0.4. Sehingga dapat disimpulkan bahwa pemeringkatan dengan menggunakan algoritma SVM pada dataset LETOR MQ2008 berjalan lebih baik pada dataset fold 4 dibandingkan pada fold lainnya. 


A project by:
```
12S17027 Stella Sitinjak
12S17029 Silvany Lumban Gaol
12S17056 Maria Manullang
12S17063 Meilysa Tarigan
```

