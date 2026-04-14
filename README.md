🏠 Ames House Price Prediction & Web API
Bu proje, Ames Housing veri setini kullanarak ev fiyatlarını tahmin etmek amacıyla geliştirilmiş uçtan uca bir makine öğrenmesi projesidir. Proje, veri analizinden (EDA) başlayarak karmaşık özellik mühendisliği (Feature Engineering), model seçimi ve bir Web API (FastAPI) üzerinden modelin sunulmasını kapsamaktadır.

🚀 Proje Özellikleri
Gelişmiş Veri Analizi (EDA): Korelasyon matrisleri, outlier tespiti ve hedef değişken (SalePrice) dağılım analizi.

Özellik Mühendisliği: Toplam metrekare (TotalSF), banyo sayısı (TotalBathrooms) ve evin yaşı gibi türetilmiş özelliklerin oluşturulması.

Veri Ön İşleme: Logaritmik dönüşüm (Log Transformation), Winsorize (Outlier baskılama), Ordinal ve One-Hot Encoding işlemleri.

Model Karşılaştırma: Linear Regression, Lasso, Ridge, Random Forest ve XGBoost gibi 10 farklı modelin R² ve RMSE metriklerine göre değerlendirilmesi.

Web Arayüzü: FastAPI altyapısı ve modern bir HTML/JS arayüzü ile kullanıcı dostu fiyat tahmini.

🛠 Kullanılan Teknolojiler
Dil: Python 3.12

Kütüphaneler: Scikit-learn, XGBoost, Pandas, NumPy, Joblib

Web: FastAPI, Uvicorn, Jinja2, HTML5/CSS3

Görselleştirme: Matplotlib, Seaborn

📊 Model Performansı
Proje kapsamında yapılan testlerde en iyi sonucu AdaBoost Regressor vermiştir. Modellerin karşılaştırmalı test sonuçları ve görselleştirilmiş başarı oranları proje dizinindeki model_performans.png dosyasında mevcuttur.

<img width="1000" height="600" alt="model_performans" src="https://github.com/user-attachments/assets/d6c272b3-90d6-484e-afe8-6cadf767c655" />


💻 Kurulum ve Çalıştırma
Gerekli Kütüphaneleri Yükleyin:

Bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib jinja2
Modeli Eğitin ve Paketleyin:

Bash
python Tubitak.py
Bu işlem sonunda house_price_package.pkl dosyası oluşturulacaktır.

Web API'yi Başlatın:

Bash
python app.py
Erişim:
Tarayıcınızdan http://127.0.0.1:8001 adresine giderek ev fiyatlarını tahmin etmeye başlayabilirsiniz.
