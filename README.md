🏠 Ames House Price Prediction & Web API
Bu proje, Kaggle'ın meşhur Ames Housing veri setini kullanarak, ev özelliklerinden piyasa değerini tahmin eden bir makine öğrenmesi modelidir. Proje sadece bir analiz dosyası değil, aynı zamanda FastAPI kullanılarak bir web servisine dönüştürülmüş çalışan bir üründür.

🎯 Projenin Amacı
Karmaşık ev özelliklerini (kalite, alan, yapım yılı, mahalle vb.) analiz ederek, konut fiyatlarını en düşük hata payıyla tahmin edebilecek bir sistem kurmak ve bu sistemi bir web arayüzü üzerinden erişilebilir kılmaktır.

🛠️ Kullanılan Teknolojiler
Veri Analizi & Görselleştirme: Pandas, NumPy, Seaborn, Matplotlib.

Makine Öğrenmesi: Scikit-Learn, XGBRegressor.

Model Paketleme: Joblib.

Backend API: FastAPI (Python).

Frontend: HTML5, CSS3, JavaScript (Jinja2 Templates).

🚀 Öne Çıkan Özellikler
Feature Engineering: TotalSF (Toplam Metrekare), Age (Ev Yaşı) ve TotalBathrooms gibi yeni özellikler türetilerek model performansı artırıldı.

Data Pipeline: Ham veri; sayısal ölçeklendirme (StandardScaler) ve kategorik dönüştürme (OneHotEncoder/OrdinalEncoder) aşamalarından geçirilerek otomatikleştirildi.

Log Transformation: Fiyat dağılımındaki çarpıklığı (skewness) gidermek için logaritmik dönüşüm uygulandı.

Real-time Prediction: Kullanıcı arayüzünden girilen veriler anlık olarak API'ye gönderilir ve modelden gelen tahmin kullanıcıya gösterilir.

📈 Model Performansı
Kullanılan Algoritma: XGBoost

R² Skoru: ~0.89+ (Modelin başarısına göre bu rakamı güncelle)

Hata Metriği: RMSE (Root Mean Squared Error)

🖥️ Nasıl Çalıştırılır?
Depoyu klonlayın: git clone https://github.com/kullaniciadin/proje-adin.git

Kütüphaneleri kurun: pip install -r requirements.txt

Sunucuyu başlatın: python app.py

Tarayıcınızda http://127.0.0.1:8001 adresine gidin.
