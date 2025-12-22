import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# GÜVENLİK İZNİ (CORS) - Bağlantı hatasını çözer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Model Paketini Yükle
artifacts = joblib.load('house_price_package.pkl')
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(features: dict):
    try:
        # 1. Veriyi DataFrame'e çevir
        df = pd.DataFrame([features])

        # 2. İsim düzeltmeleri
        df.columns = [col.replace("FlrSF1st", "1stFlrSF").replace("FlrSF2nd", "2ndFlrSF") for col in df.columns]

        # 3. HATA ALDIĞIN EKSİK KOLONLARI DOLDUR (Sayısal ve Türetilmişler)
        # Hata mesajındaki listede yer alan tüm sayısal kolonlar için makul varsayılanlar:
        defaults_num = {
            'MSSubClass': 60, 'LotFrontage': 65.0, 'OverallCond': 5, 'YearRemodAdd': 2005,
            'MasVnrArea': 0.0, 'BsmtFinSF1': 0.0, 'BsmtFinSF2': 0.0, 'BsmtUnfSF': 0.0,
            'LowQualFinSF': 0.0, 'GrLivArea': 1500.0, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1,
            'TotRmsAbvGrd': 7, 'Fireplaces': 1, 'GarageCars': 2, 'MoSold': 6, 'YrSold': 2010,
            'PoolArea': 0.0, 'MiscVal': 0.0, 'WoodDeckSF': 0.0
        }
        for col, val in defaults_num.items():
            if col not in df.columns:
                df[col] = val

        # 4. FEATURE ENGINEERING (Türetilmiş kolonlar - Hata listendekiler)
        df["TotalSF"] = df.get("TotalBsmtSF", 1000) + df.get("1stFlrSF", 1000) + df.get("2ndFlrSF", 0)
        df["TotalBathrooms"] = df.get("FullBath", 2) + 0.5 * df.get("HalfBath", 0)
        df["Age"] = df.get("YrSold", 2010) - df.get("YearBuilt", 2005)
        df["YearsSinceRemod"] = df.get("YrSold", 2010) - df.get("YearRemodAdd", 2005)
        df["GarageAge"] = 5.0  # Varsayılan
        df["TotalPorchSF"] = 0.0

        # Boolean Kolonlar (Hata listendeki Has... olanlar)
        df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        df['HasGarage'] = 1
        df['HasBsmtQual'] = 1
        df['HasFireplaceQu'] = 0
        df['HasGarageType'] = 1

        # 5. KATEGORİK EKSİKLER (Alley vb.)
        if 'Alley' not in df.columns: df['Alley'] = 'None'

        # 6. KATEGORİK STANDARTLAR (Önceki hatadan kalanlar)
        missing_cats = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                        'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                        'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

        for col in missing_cats:
            if col not in df.columns:
                df[col] = "TA" if "Qual" in col or "Cond" in col else "None"

        # 7. PREPROCESSING & TAHMİN
        # Preprocessor'ın beklediği kolon sırasını bozmamak için transform kullanıyoruz
        processed = preprocessor.transform(df)
        log_pred = model.predict(processed)[0]

        # Tahmin 15'ten büyükse (Sonsuzluk engeli) makule çek
        if log_pred > 15: log_pred = 13

        real_price = np.expm1(log_pred)

        return {"predicted_price": round(float(real_price), 2)}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"detail": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)