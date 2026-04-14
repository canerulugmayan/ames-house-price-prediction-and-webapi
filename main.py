import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib # Modeli disari aktarmak icin
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

df = pd.read_csv("../train.csv")
pd.set_option("display.max_column",None)

def show_info(df):
    print(
        f"Info:\n{df.info()}\n\n"
        f"First rows:\n{df.head()}\n\n"
        f"Describe:\n{df.describe()}\n\n"
        f"Columns:\n{df.columns}\n\n"
        f"Null-data:\n{df.isnull().sum()}\n\n"
        )
show_info(df)

#EDA
plt.figure(figsize=(8,5))
sns.histplot(df["SalePrice"],kde=True)
plt.title("Sağa Çarpık SalePrice Dağılımı")
plt.show()
"""
sns.histplot(np.log1p(df["SalePrice"]),kde=True)
Neden log dönüşüm yapıyoruz?
SalePrice değişkeni (ve bazı alanlar mesela LotArea, GrLivArea) genelde sağa çarpık (right-skewed) oluyor.
Yani veri setinde çok sayıda orta fiyatlı ev var ama birkaç tane çok pahalı ev var. Bu uç değerler dağılımı bozuyor.

Lineer regresyon gibi birçok algoritma, hedef değişkenin ve hataların normal dağılıma yakın olmasını varsayar.
Skewed dağılım → model hatalarını büyütür, tahmin performansını düşürür.
Ayrıca MSE (mean squared error) gibi metrikler, uç fiyatlardan dolayı orantısız derecede cezalandırıcı hale gelir.
"""
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df["SalePrice"]),kde=True)
plt.title("SalePrice Dağılımı")
plt.show()
"""
1)Hedef değişkenin yapısını anlamak
Ortalama fiyat, medyan fiyat, min–max, piyasadaki fiyat aralığını görüyoruz.
Eğer veri çok geniş aralıklıysa, modellerin bazı evleri yanlış tahmin etme ihtimali artar.

2)Skewness (çarpıklık) ve Kurtosis (basıklık)
Çarpıklık → dağılım sağa mı sola mı kayık? (log dönüşüm burada işe yarıyor)
Basıklık → uç değerlerin fazlalığı.
Bunları bilmezsek, residual (hata) dağılımın sorunlu olur, regresyon varsayımları bozulur.

3) Outlier etkisini görme
Çok pahalı birkaç ev varsa histogramın kuyruğu uzar , model onları öğrenmeye çalışırken geneli kötü tahmin eder.
Yani outlier temizlemenin gerekliliğini anlıyoruz.

4)  Hedef değişkenin “transformasyonu”na karar vermek
Lineer modeller, normal dağılım ister log dönüşüm uygularsın.
Tree-based modeller (Random Forest, XGBoost), normal dağılıma ihtiyaç duymaz, ama log dönüşüm yine de RMSLE gibi metriklerde performansı iyileştirebilir.

5) Tahmin performansını nasıl ölçeceğini belirlemek
Eğer SalePrice çok çarpıksa → RMSE (Root Mean Squared Error) çok etkilenir.
Bu yüzden Kaggle bu yarışmada RMSLE (Root Mean Squared Log Error) kullanıyor → yani log dönüşümle çalışmak zaten zorunlu gibi.

 Özet: SalePrice EDA’sı sana sadece “normal dağılım var mı” bilgisini vermez. Aynı zamanda:
Outlier var mı?
Hangi dönüşüm (log, box-cox, yeo-johnson) lazım?
Hangi metriği kullanmak daha mantıklı?
Hangi modeller daha uygun?
Bunları netleştiriyor.
"""
#numerik değişkenlerle ilişki
plt.figure(figsize=(20,16))
corr=df.corr(numeric_only=True)
sns.heatmap(corr,cmap="coolwarm",center=0)
plt.title("Korelasyon Matrisi")
plt.show()

def yuksek_corr_sutunlar(df, sutun_sayısı, hedef) :
    df_numeric = df.select_dtypes(include=["number"])
    corr = df_numeric.corr()
    corr_abs = corr.abs()
    print (corr_abs.nlargest(sutun_sayısı, hedef)[hedef])
    return corr_abs.nlargest(sutun_sayısı, hedef)[hedef].index.tolist()
yuksek_corr_sutunlar(df,15,"SalePrice")

print("########################################################################")

def plot_corr_matrix(df,sutun_sayısı,hedef):
    #sadece sayısal sütunları al
    df_numeric=df.select_dtypes(include=["number"])

    if hedef not in df_numeric.columns:
        raise ValueError(f"Hedef sütun '{hedef}' sayısal değil veya bulunamadı.")

    corr = df_numeric.corr()
    corr_abs = corr.abs()

    #En yüksek korelasyonlu sütunlar
    cols = corr_abs.nlargest(sutun_sayısı,hedef)[hedef].index
    corr_matrix = np.corrcoef(df_numeric[cols].values.T)

    plt.figure(figsize=(sutun_sayısı/1.5,sutun_sayısı/1.5))
    sns.set(font_scale=1.25)
    sns.heatmap(corr_matrix,linewidths=1.5,annot=True,square=True,
                fmt=".2f",annot_kws={"size":15},
                yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
plot_corr_matrix(df,15,"SalePrice")

"""
Korelasyon ne işe yarar?
1)Özelliklerin hedef değişkenle ilişkisini anlamak
Korelasyon, her özelliğin hedef değişkenle ne kadar ilişkili olduğunu hızlıca gösterir.
Örnek: Ev fiyatı tahmini yapıyorsak:
OverallQual ile SalePrice korelasyonu 0.8 → güçlü ilişki → bu özelliği modelde mutlaka kullan.
GarageYrBlt ile SalePrice korelasyonu 0.05 → neredeyse yok → modelde çok fayda sağlamaz, çıkarılabilir.
Bu, feature selection yani önemli özellikleri seçmek için pratik bir yöntemdir.

2) Çoklu doğrusal bağlantıyı (Multicollinearity) tespit etmek
İki veya daha fazla özellik çok yüksek korelasyona sahipse, bazı modellerde (özellikle lineer modellerde) multicollinearity sorunu oluşur.
Bu da modelin katsayılarını dengesizleştirir ve yorumlamayı zorlaştırır.
Örnek: GrLivArea ile TotalBsmtSF korelasyonu 0.9 → iki sütunu birden modele eklemek mantıklı olmayabilir.

3) Özellik mühendisliği için ipuçları
Negatif korelasyon → bir özellik arttığında hedef azalıyor → bazı algoritmalarda ters etkili olabilir.
Bu bilgiyi kullanarak:
Özellikleri dönüştürebilir
Yeni özellikler oluşturabilir
Fazla bağımlı olanları çıkarabilirsin

4) Modelin performansını artırmak
Yüksek korelasyonlu ve anlamlı özellikler → model daha iyi öğrenir → tahmin doğruluğu artar.
Düşük korelasyonlu veya gereksiz özellikler → model karmaşıklaşır ama çok katkı sağlamaz.

Özet
Korelasyon, ML’de EDA ve feature selection aşamasında bize:
Hangi özellikler hedefi güçlü şekilde etkiliyor,
Hangi özellikler birbirine çok bağlı,
Hangi özellikler gereksiz veya düşük katkılı
konularında hızlı ve görselleştirilebilir bilgi verir.
"""
#Kategorik değişkenler ile ilişkiler
plt.figure(figsize=(8,5))
sns.scatterplot(x="GrLivArea",y="SalePrice",data=df)
plt.title("GrLivArea vs SalePrice")
plt.xlabel("GrLivArea (Yaşanabilir Alan, sqrt)")
plt.ylabel("SalePrice")
plt.show()
#Yaşam alanı arttıkça fiyat genellikle artıyor, güçlü doğrusal ilişki var.
#Bazı uç noktalar (outlier) var: 4000 sqft üstü ama ucuz evler, bunları temizlemek modeli iyileştirir.

plt.figure(figsize=(10,6))
sns.boxplot(x="OverallQual",y="SalePrice",data=df)
plt.title("OverallQual vs SalePrice")
plt.xlabel("OverallQuall (Genel Kalite, 1-10)")
plt.ylabel("SalePrice")
plt.show()
#Kalite arttıkça fiyat da artıyor, çoğunlukla logaritmik bir trend var.
#Çok net bir pattern gözlemleniyor, OverallQual tahmin için güçlü bir feature.

plt.figure(figsize=(14,6))
sns.boxplot(x="Neighborhood",y="SalePrice",data=df)
plt.xticks(rotation=45)
plt.title("Neighborhood vs SalePrice")
plt.xlabel("Mahalle")
plt.ylabel("SalePrice")
plt.show()
#Bazı mahalleler (StoneBr, NoRidge, NridgHt) çok pahalı, lokasyon çok belirleyici.

plt.figure(figsize=(10,6))
sns.boxplot(x="HouseStyle", y="SalePrice", data=df)
plt.title("HouseStyle vs SalePrice")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x="MSZoning", y="SalePrice", data=df)
plt.title("MSZoning vs SalePrice")
plt.show()
#Ev stili ve zoning fiyat farklılıklarını açıklıyor.

plt.figure(figsize=(14,6))
sns.countplot(x="Neighborhood", data=df)
plt.xticks(rotation=45)
plt.title("Neighborhood Frekansları")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x="HouseStyle", data=df)
plt.title("HouseStyle Frekansları")
plt.show()
#Bazı kategoriler çok az, dengesizlikleri fark edip “rare label encoding” yapılabilir.

#Eksik değer analizi
def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title("Eksik Değer Sayıları")
    plt.show()
plot_missing_values(df)

#outlier tespiti
import pandas as pd
import numpy as np
from scipy import stats
def outlier_tespit(df, columns,treshold):
    outliers_dict = {}  # Her kolonun outlier değerlerini saklayacağız
    #NaN oranı yapacağız ve eğer çok fazla değil ise geçici olarak nan olan satırları kaldıracağız
    for col in columns:
        percenage = df[col].isnull().sum()/len(df)*100
        if percenage >= 50 :
            print(f"{col} Sütunu %50 den büyük olduğu için outlier tespiti yapılamaz!")
            continue
        print(f"{col} Kolonu toplam {df[col].isnull().sum()} adet NaN barındırıyor, Yüzdesi: %{percenage:.2f}\n")

        """
        Verilen kolon için Z-score ve IQR yöntemine göre outlier'ları tespit eder ve raporlar.
        """
        # Geçici NaN temizleme
        data = df[col].dropna()

        # Z-score yöntemi
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]

        # IQR yöntemi
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - treshold * IQR
        upper_bound = Q3 + treshold * IQR
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

        # Raporlama
        print(f"--- {col} için Outlier Raporu ---")
        print(f"Toplam veri: {len(data)}")
        print(f"Z-score yöntemi ile outlier sayısı: {len(z_outliers)}")
        print(f"Z-score outlier_percentage: {100 * z_outliers.shape[0] / df.shape[0]}")
        print(f"IQR yöntemi ile outlier sayısı: {len(iqr_outliers)}")
        print(f"IQ outlier_percentage: {100 * iqr_outliers.shape[0] / df.shape[0]}")
        print(f"lower bound: {lower_bound}")
        print(f"upper bound: {upper_bound}")
        print("---------------------------------------------")

        # Outlier değerlerini dict içine ekle
        outliers_dict[col] = {
            "z_score_outliers": z_outliers.values,
            "iqr_outliers": iqr_outliers.values
        }
    return outliers_dict

outlier_tespit(df,df.dtypes[df.dtypes != "object"].index,1.5)

print("########################################################################")

#GrLivArea ve LotArea ve GarageArea, SalePrice ile yüksek korelasyona sahipse, bu kolonlarda uç değerler modeli ciddi etkiler
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)  # 2 satır, 2 sütun, 1. grafik
sns.boxplot(x=df["LotArea"])
plt.title("LotArea Outlier Kontrolü")

plt.subplot(2,2,2)
sns.boxplot(x=df["GrLivArea"])
plt.title("GrLivArea Outlier Kontrolü")

plt.subplot(2,2,3)
sns.boxplot(x=df["GarageArea"])
plt.title("GarageArea Outlier Kontrolü")

plt.subplot(2,2,4)
sns.boxplot(x=df["SalePrice"])
plt.title("SalePrice Outlier Kontrolü")
plt.tight_layout()
plt.show()
#Çok büyük arsa alanı veya çok büyük yaşam alanı veya çok büyük garaj alanı ama ucuz evler, modelde hata yaratabilir, ayıklanmalı

#Zaman analizi
plt.figure(figsize=(8,5))
sns.boxplot(x="YrSold", y="SalePrice", data=df)
plt.title("Satış Yılına Göre Fiyatlar")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="MoSold", y="SalePrice", data=df)
plt.title("Satış Ayına Göre Fiyatlar")
plt.show()
#Satış yılı ve ayının fiyata etkisi genelde düşük.
#Ama bazı aylar (özellikle yaz ayları) fiyatlar biraz yüksek olabilir.
"""
Özetle EDA’dan çıkarmamız gereken ana içgörüler:
SalePrice sağa çarpık → log dönüşüm faydalı.
OverallQual, GrLivArea, GarageCars, TotalBsmtSF fiyatla en çok ilişkili.
Mahalle (Neighborhood) çok güçlü bir kategorik belirleyici.
Eksik değerler aslında çoğunlukla “özellik yok” → doğru şekilde doldurulmalı.
Outlier’lar (çok büyük alanlı ama ucuz evler) temizlenmeli.
Zaman değişkenlerinin etkisi düşük.
"""

#Feature Engineering
"""
Feature engineering önceliği
İlk aşamada, en yüksek etkiyi veren ve mantıklı kolonlar ile çalışmak genelde daha pratiktir.
Geride kalan kolonlar:
Ya etkisi düşük
Ya eksik değer çok
Ya da direkt kullanılmadan önce transformasyon/encoding gerek
"""
numerical_feats = df.dtypes[df.dtypes != "object"].index
print(f"Numerik özellikler:{numerical_feats},ve sayıları:{len(numerical_feats)}")

categorical_feats = df.dtypes[df.dtypes == "object"].index
print(f"Kategorik özellikler:{categorical_feats},ve sayıları:{len(categorical_feats)}\n")

#Seçeceğimiz değişkenler hedef değişkenle (SalePrice) anlamlı ilişkisi olan ve modelin öğrenmesine katkı sağlayabilecek
#özellikler olacakları için seçildi.
"""
Neighborhood(Mahalle):	Lokasyon fiyat üzerinde çok belirleyici; bazı mahalleler çok pahalı.
MSZoning(Arazi tipi): (Residential, Commercial, vb.)	Fiyat farklılıklarını açıklıyor.
HouseStyle(Ev tipi): (1-story, 2-story, vb.)	Ev stili fiyatı etkiliyor.
ExterQual(Dış kalite): (Ex, Gd, TA, Fa)	Sıralı (ordinal) değişken, kalite → fiyat ilişkisi güçlü.
ExterCond(Dış koşul durumu):	Ordinal, evin dış durumuna göre fiyat değişiyor.
BsmtQual(Bodrum kalite):	Bodrum varsa, kalitesi fiyatı etkiliyor.
GarageType(Garaj tipi):	Garajın varlığı ve tipi fiyatı etkiler.
FireplaceQu(Şömine kalitesi):	Özelliğin varlığı ve kalitesi fiyatı etkiler.
"""
print("Ordinal Değişkenler ve Tipleri")
print(f"Dış Kaliteler: {df["ExterQual"].unique()}\n"
    f"Dış Koşul Durumları: {df["ExterCond"].unique()}\n"
    f"Bodrum Kaliteleri: {df["BsmtQual"].unique()}\n"
    f"Şömine Kalitesi: {df["FireplaceQu"].unique()}\n\n")
print("Nominal Değişkenler ve Tipleri")
print(f"Mahalle Tipleri: {df["Neighborhood"].unique()}\n"
    f"Arazi Tipleri: {df["MSZoning"].unique()}\n"
    f"Ev Tipleri: {df["HouseStyle"].unique()}\n"
    f"Garaj Tipleri: {df["GarageType"].unique()}\n\n")

print("########################################################################")

"""
FireplaceQu: 770 non-null → 690 missing ≈ 47.3%
BsmtQual: 1423 non-null → 37 missing ≈ 2.53%
GarageType: 1379 non-null → 81 missing ≈ 5.55%
Bu oranlara bakınca: BsmtQual ve GarageType için eksik çok az, FireplaceQu için eksik çok fazla.
Fakat Kaggle House Prices datasında bu NA’ların çoğu “özellik yok” anlamına geliyor (örn. NA in mostcases = no basement / no fireplace / no garage).
Bu yüzden satır silme yerine anlamlı doldurma daha mantıklı.
"""
ordinal_feats=["ExterQual","ExterCond","BsmtQual","FireplaceQu"]
nominal_feats=["Neighborhood","MSZoning","HouseStyle","GarageType"]

#1)Türevlenen Kolonlar
#Bu işlemler, farklı ama birbiriyle ilişkili kolonları tek bir mantıklı feature haline getiriyor.
df["TotalSF"]=df[["TotalBsmtSF","1stFlrSF","2ndFlrSF"]].sum(axis=1)
#Amaç:Evin tüm kullanılabilir alanını tek bir feature ile göstermek.
#Böylece model, “evin toplam büyüklüğü” ile fiyat ilişkisini daha rahat öğrenir.

df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
#İkinci katı olan evler genelde daha pahalı.

df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
#Fiyat üzerinde toplam veranda alanının etkisini modelin öğrenmesini sağlamak.
#Tek tek kolonlar yerine tek bir feature ile temsil edince, model daha hızlı ve verimli öğrenir.

df["TotalBathrooms"] = df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
#Amaç:Model, toplam “fonksiyonel banyo sayısını” öğrenebilsin.
#Çünkü tek tek kolonlar yerine toplam banyo sayısı, fiyat üzerinde daha net bir etki gösterir.

df["Age"] = df["YrSold"] - df["YearBuilt"]
#Amaç: Evin yaşı evin satış yılı ile yapım yılı arasındaki fark. Önemi ise Genellikle yeni evler daha pahalıdır, yaşlı evler ise değer kaybına uğramış olabilir.

df["YearsSinceRemod"] = df["YearRemodAdd"] - df["YearBuilt"]
#Amaç: Evin orijinal yapımından sonra yapılan remodeling (yenileme/ tadilat) süresi.Önemi Daha yeni yenilenmiş evler genellikle daha yüksek fiyatlıdır.

df["GarageYrBlt"]=df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#Garajı olan evler genellikle daha değerli.
"""
df['Neighborhood_avgPrice'] = df.groupby('Neighborhood')['SalePrice'].transform('mean')
df['HouseStyle_avgPrice'] = df.groupby('HouseStyle')['SalePrice'].transform('mean')
DATA LEAKAGE !!!
Burada bütün dataset (hem train, hem test) kullanılarak SalePrice ortalaması hesaplanıyor.
Yani, test verisinin hedef bilgisini train tarafına sızdırmış oluyoruz o yüzden bunu train-test splitten sonra yapacağız.
"""
#Artık bizim için anlamsız olan bazı sütunları atabiliriz
df.drop(columns=[
    "TotalBsmtSF","1stFlrSF","2ndFlrSF",        # TotalSF için
    "OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch",  # TotalPorchSF için
    "FullBath","HalfBath","BsmtFullBath","BsmtHalfBath",      # TotalBathrooms için
    "GarageYrBlt",   # GarageAge için
    "Id"    # buda bi işimize yaramayacak
        ],axis=1,inplace=True)
print(f"{df.head()}\n\n")
#Yapılan bu amele işi şeylerin hepsi modelin tahmin gücünü artıracak anlamlı özellikler üretmeye yöneliktir.


#2) Eksik değerleri doldurma
df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')
df['BsmtQual']   = df['BsmtQual'].fillna('NoBasement')
df['GarageType'] = df['GarageType'].fillna('NoGarage')

#3) Outlier sınırlama
def winsorize_outliers(df, columns, threshold=1.5):
    #Avantajımız Veri kaybetmeyiz ve dağılım dengelenir.
    df_copy = df.copy()
    for col in columns:
        if df[col].dtype != "object":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound, df_copy[col])
            df_copy[col] = np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
    return df_copy
"""
“Outlier değerler IQR yöntemine göre tespit edilmiş ve üst-alt sınırlar içinde Winsorize edilerek veri kaybı önlenmiştir.
”"""
num_cols = df.select_dtypes(include=[np.number]).columns
df = winsorize_outliers(df, num_cols)

#4) Indicator sütunları ( var ise 1 yok ise 0 bilgisi)
df["HasFireplaceQu"]=df["FireplaceQu"].notna().astype(int)
df["HasBsmtQual"]=df["BsmtQual"].notna().astype(int)
df["HasGarageType"]=df["GarageType"].notna().astype(int)

X=df.drop("SalePrice",axis=1)
y=df["SalePrice"]
# KRİTİK DÜZELTME: Log dönüşümünü yap ve DEĞİŞKENİ GÜNCELLE
y = np.log1p(y)
#Bu modeller, hata terimlerinin normal dağılıma yakın olmasını ister.Sağa çarpık dağılım yani model, çok pahalı evleri iyi tahmin edemeyebilir.
#bu yüzden log dönüşümü uyguladık ve eda kısmında da zaten bunu anlattım tahminlerimizi bu değişken üzerinden yapacağız

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=45)

"FEATURE ENGİNEERİNGE DEVAM EDİYORUZ EĞER BU AŞAMADAN ÖNCE ENCODİNG YAPSAYDIK DATA LEAKAGE YAŞANIRDI !!!"
#6)Kendi ordinal encoder fonksiyonum
from sklearn.base import BaseEstimator
from custom_transformers import OrdinalEncoderr
mappings = {
    "ExterQual": {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
    "ExterCond": {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
    "BsmtQual": {"NoBasement":0, "Fa":2, "TA":3, "Gd":4, "Ex":5},
    "FireplaceQu": {"NoFireplace":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
}
fill_map = {"BsmtQual":"NoBasement", "FireplaceQu":"NoFireplace"}
encoder = OrdinalEncoderr(mappings,fillna_map=fill_map)#sadece dönüşüm mantığını tutar, veri içermez.
df = encoder.fit_transform(df)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Sadece sayısal sütunlarda median
num_imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),          # NaN'leri medyan ile doldur
    ('scaler', StandardScaler())                            # Standardizasyon
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # NaN'leri en sık değer ile doldur
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encoding
])

# ColumnTransformer ile birleştir
preprocessor = ColumnTransformer(transformers=
                                [
                                    ('FEYZA WANNA BE ',StandardScaler(),numerical_cols),
                                    ('A ROCKSTAR', OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
                                ], remainder= "passthrough"
                                )

from sklearn.model_selection import KFold

def kfold_target_encoding(X, y, col, n_splits=5):
    X = X.copy()
    X[col + '_te'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(X):
        means = y.iloc[tr_idx].groupby(X.iloc[tr_idx][col]).mean()
        X.iloc[val_idx, X.columns.get_loc(col + '_te')] = X.iloc[val_idx][col].map(means)
    # training için kalan boşlar (nadiren) fill
    X[col + '_te'].fillna(y.mean(), inplace=True)
    return X[col + '_te']
# kullanım: X_train['Neighborhood_te'] = kfold_target_encoding(X_train, y_train, 'Neighborhood', 5)
# test için mapping train'den hesaplanmış genel means kullan:
mapping = y_train.groupby(X_train['Neighborhood']).mean()
X_test['Neighborhood_te'] = X_test['Neighborhood'].map(mapping).fillna(mapping.mean())
"""
K-Fold Cross Validation (Katlamalı Doğrulama) Nedir?
K-Fold, veriyi tek bir defada “train-test” diye bölmek yerine, veriyi k parçaya (fold’a) ayırır.
Sonra bu parçaları dönüşümlü olarak hem eğitim hem test için kullanır.
Örneğin k = 5 olsun:
Veriyi 5 parçaya bölersin.
1. seferde: 4 parça train, 1 parça test olur
2. seferde: başka bir parça test olur, kalan 4 train
Bu işlem 5 kere tekrarlanır → böylece model 5 farklı test seti üzerinde denenmiş olur.

Neden Kullanılır?
Modelin genelleme kabiliyetini ölçmek için:
Tek bir train-test ayrımı şansa bağlı olabilir, ama K-Fold birçok farklı ayrımı test ettiği için daha güvenilir sonuç verir.
Overfitting riskini azaltır:
Çünkü model, her iterasyonda farklı verilerle eğitilip test edilir.
Verinin tamamını değerlendirmiş olursun:
Her örnek en az bir defa test setinde yer alır.

K-Fold Target Encoding ile Ne İlgisi Var?
Target encoding (örneğin Neighborhood → ortalama SalePrice) yaptığında data leakage riski doğar.
Yani model, “tahmin etmesi gereken hedef bilgiyi” encoding sırasında öğrenmiş olur.
Bu da modeli gerçekte olduğundan daha iyi gösterir.
K-Fold target encoding bu sorunu çözer:
Veriyi 5 fold’a ayırırsın.
Her fold için encoding işlemini, sadece o fold’un dışında kalan verinin ortalamasına göre yaparsın.
Yani test fold’u “hedef bilgiyi” encoding aşamasında görmez.
"""

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print(type(X_train))       # <class 'numpy.ndarray'> olmalı
print(X_train.shape)       # (1168, 258)
print(X_train[0])          # ilk satırı göster, liste mi array mi

#Modellerimiz
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from sklearn.base import RegressorMixin
from scipy.sparse import hstack, csr_matrix
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import timeout_decorator

class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, adding_constraint=True, stochastic_constraint=False, coef_nonnegative=True):
        self.adding_constraint = adding_constraint
        self.stochastic_constraint = stochastic_constraint
        self.coef_nonnegative = coef_nonnegative
        self.coef_ = None
        self.intercept_ = None
        # Kullanıcının hangi kısıtları uygulamak istediğini kontrol eder.
        # coef_ ve intercept_ eğitim sonrası tahmin edilen katsayıları tutar

    def fit(self, X, y):
        # Sparse veya dense matrisleri destekle
        if isinstance(X, csr_matrix):
            # intercept eklemek için bir sütun ekliyoruz
            X_aug = hstack([X, csr_matrix(np.ones((X.shape[0], 1)))])
        else:
            X = np.array(X)
            X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

        y = np.array(y)
        n_features = X.shape[1]

        # Modelin minimize edeceği fonksiyon: MSE
        def objective(beta):
            return np.mean((y - (X_aug @ beta)) ** 2)  # beta[:-1] katsayılar, beta[-1] intercept

        cons = []
        # Katsayıların negatif olmaması (intercept hariç)
        if self.coef_nonnegative:
            cons.append({'type': 'ineq', 'fun': lambda beta: beta[:-1]})  # β >= 0

        # Katsayıların toplamı ≈ 1 (±%5 tolerans)
        if self.adding_constraint:
            cons.append({'type': 'ineq', 'fun': lambda beta: 1.05 - np.sum(beta[:-1])})
            cons.append({'type': 'ineq', 'fun': lambda beta: np.sum(beta[:-1]) - 0.95})

        # Stokastik çeşitlilik kısıtı (opsiyonel)
        if self.stochastic_constraint:
            cons.append({'type': 'ineq', 'fun': lambda beta: np.std(beta[:-1]) - 0.01})

        # Başlangıç değerleri: katsayılar eşit, intercept = 0
        x0 = np.ones(n_features + 1) / n_features
        x0[-1] = 0  # intercept başlangıçta 0

        # Optimize etme
        result = minimize(objective, x0, constraints=cons, method='SLSQP')

        # Sonuçları kaydet
        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]
        return self

    def predict(self, X):
        if isinstance(X, csr_matrix):
            X_aug = hstack([X, csr_matrix(np.ones((X.shape[0], 1)))])
            return X_aug @ np.append(self.coef_, self.intercept_)
        else:
            X = np.array(X)
            return np.dot(X, self.coef_) + self.intercept_

models_params = {
    "Linear Regression": {
        "model":LinearRegression(),
        "params":{
        }
    },
    "LinearRegression_Constrained": {
        "model":ConstrainedLinearRegression(
            adding_constraint=True,         #Normalize edilmiş ağırlık
            stochastic_constraint=True,     #Aşırı ağırlık farkını engeller
            coef_nonnegative=True),         #Ekonomik modellerde anlamlı
        "params":{
        }
    },
    "Lasso": {
        "model": Lasso(max_iter=5000),
        "params": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
    },
    "Ridge": {
        "model": Ridge(max_iter=5000),
        "params": {"model__alpha": [0.01, 0.1, 1, 10, 100, 200, 500]
        }
    },
    "K Neighbors Regressor": {
        "model": KNeighborsRegressor(),
        "params": {"model__n_neighbors": [3,5,7,9,11,15],
                   "model__weights": ["uniform", "distance"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(),
        "params": {"model__criterion" : ["squared_error", "absolute_error", "friedman_mse", "poisson"],"model__splitter" : ["best", "random"],
                   "model__max_depth": [5, 8, 10 ,15, None],"model__max_features" : ["sqrt", "log2", None],
                   "model__min_samples_split": [2,5,10],"model__min_samples_leaf": [1,2,4,10]
        }
    },
    "Random Forest Regressor": {
        "model": RandomForestRegressor(),
        "params": {"model__max_depth": [5, 8, 10 ,15, None],"model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
                   "model__min_samples_split": [5, 8, 15, 20],"model__n_estimators": [500, 650, 800, 1000],
                   "model__min_samples_leaf": [1,2,4,10]
        }
    },
    "Adaboost Regressor": {
        "model": AdaBoostRegressor(estimator=DecisionTreeRegressor()),
        "params": {"model__estimator__max_depth" : [2,3,4,5],"model__n_estimators" : [200, 350, 500],
                   "model__learning_rate" : [0.01,0.05,0.1,0.2,0.5],"model__loss" : ["linear", "squared_error", "exponential"]
        }
    },
    "Gradient Boost Regressor": {
        "model": GradientBoostingRegressor(),
        "params": {"model__n_estimators" : [500, 1000, 1500, 2000],"model__max_depth" : [3,4,5],
                   "model__loss" : ["squared_error", "absolute_error", "huber", "quantile"],"model__learning_rate": [0.001,0.01,0.02,0.05]
        }
    },
    "XGBoost Regressor": {
        "model": XGBRegressor(),
        "params": {"model__n_estimators" : [500, 1000, 1500, 2000],"model__learning_rate" : [0.001,0.01,0.02,0.05],
                   "model__max_depth": [3,4,5,6],"model__colsample_bytree": [0.7,0.8,0.9],
                   "model__subsample": [0.7,0.8,0.9]
        }
    },
}
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

@timeout_decorator.timeout(120)
def run_gridsearch(grid, X_train, y_train):
    grid.fit(X_train, y_train)
    return grid

results = []  # tüm sonuçları buraya ekleyeceğiz

# --- MODEL EĞİTİM DÖNGÜSÜ ---
best_model = None
best_r2 = -float('inf')

for model_key, model_info in models_params.items():
    print(f"{model_key} için eğitim başlıyor...")
    current_trained_model = None

    # 1. Eğitim ve Hiperparametre Optimizasyonu
    if model_info['params']:
        pipe = Pipeline([('model', model_info['model'])])
        try:
            grid = GridSearchCV(pipe, model_info['params'], cv=5, scoring='r2', n_jobs=-1)
            grid_result = run_gridsearch(grid, X_train, y_train)
            current_trained_model = grid_result.best_estimator_
        except:
            print(f"{model_key} -> RandomizedSearch'e geçiliyor...")
            rand = RandomizedSearchCV(pipe, model_info['params'], cv=5, scoring='r2', n_jobs=-1, n_iter=30, random_state=42)
            rand.fit(X_train, y_train)
            current_trained_model = rand.best_estimator_
    else:
        current_trained_model = model_info['model']
        current_trained_model.fit(X_train, y_train)

    # 2. Tahminler (Tüm modeller için ortak alan)
    y_train_pred = current_trained_model.predict(X_train)
    y_test_pred = current_trained_model.predict(X_test)

    # 3. Skorlar (Tüm modeller için ortak alan)
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    # 4. Sonuçları listeye ekle
    results.append({
        "Model": model_key,
        "Train_RMSE": model_train_rmse,
        "Test_RMSE": model_test_rmse,
        "Train_MAE": model_train_mae,
        "Test_MAE": model_test_mae,
        "Train_R2": model_train_r2,
        "Test_R2": model_test_r2
    })

    # 5. En iyi Modeli Belirleme (Test R2'ye göre)
    if model_test_r2 > best_r2:
        best_r2 = model_test_r2
        best_model = current_trained_model


# --- SONUÇLARIN TABLOSU ---
df_results = pd.DataFrame(results)
# Test R2'ye göre sırala
df_results_sorted = df_results.sort_values(by="Test_R2", ascending=False)

print("\n" + "="*100)
print("MODELLERİN KARŞILAŞTIRMASI")
print("="*100)
# Tablonun tamamını (tüm sütunları) terminalde gösterir
print(df_results_sorted.to_string(index=False))
print("="*100)

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(10, 6))
sns.barplot(x="Test_R2", y="Model", data=df_results_sorted, palette="viridis")
plt.title("Modellerin Test R2 Skorları")
plt.xlabel("Test R2")
plt.ylabel("Model")
plt.savefig("model_performans.png")
plt.show()

# --- KAYIT İŞLEMİ ---
import joblib
# all_cols_list için preprocessor'ı kullanıyoruz
if hasattr(preprocessor, "feature_names_in_"):
    all_cols_list = preprocessor.feature_names_in_.tolist()
else:
    all_cols_list = numerical_cols.tolist() + categorical_cols.tolist() # Yedek plan

artifacts = {
    "model": best_model,
    "preprocessor": preprocessor,
    "numerical_cols": numerical_cols.tolist(),
    "categorical_cols": categorical_cols.tolist(),
    "all_cols": all_cols_list
}

joblib.dump(artifacts, 'house_price_package.pkl')
print("işlem tamamlandı")


























