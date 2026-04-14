import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import traceback
# Önemli: custom_transformers.py dosyanın aynı dizinde olduğundan emin ol
from custom_transformers import OrdinalEncoderr

app = FastAPI()

# CORS Ayarları: Tarayıcıdan gelen isteklerin engellenmemesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# 1. Model ve Preprocessor Yükleme
try:
    artifacts = joblib.load('house_price_package.pkl')
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    # Modelin eğitimde gördüğü tüm kolon listesini alıyoruz
    all_training_cols = preprocessor.feature_names_in_
    print("Model ve Preprocessor başarıyla yüklendi.")
except Exception as e:
    print(f"Yükleme Hatası: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(features: dict):
    try:
        # 1. Ham veriyi al ve isimleri standartlaştır
        raw_data = features.copy()
        if "FlrSF1st" in raw_data: raw_data["1stFlrSF"] = raw_data.pop("FlrSF1st")
        if "FlrSF2nd" in raw_data: raw_data["2ndFlrSF"] = raw_data.pop("FlrSF2nd")

        # 2. Sinyal Onarımı (Feature Engineering)
        gr_area = float(raw_data.get("GrLivArea", 1500))
        f1st = float(raw_data.get("1stFlrSF", gr_area))
        total_bsmt = float(raw_data.get("TotalBsmtSF", 1000))
        f2nd = float(raw_data.get("2ndFlrSF", 0))

        raw_data["TotalSF"] = total_bsmt + f1st + f2nd
        raw_data["TotalBathrooms"] = float(raw_data.get("FullBath", 2)) + 0.5 * float(raw_data.get("HalfBath", 0))
        raw_data["Age"] = 2026 - int(raw_data.get("YearBuilt", 2005))
        raw_data["Has2ndFloor"] = 1 if f2nd > 0 else 0

        # 3. Modelin Beklediği TAM SIRAYLA Veri İnşası
        input_row = []
        # artifacts["all_cols"] artık Tubitak.py'den gelen tam listedir
        target_cols = artifacts.get("all_cols", [])
        num_cols = artifacts.get("numerical_cols", [])

        for col in target_cols:
            if col in raw_data:
                input_row.append(raw_data[col])
            elif col in num_cols:
                input_row.append(0.0)
            else:
                input_row.append("None")

        df_final = pd.DataFrame([input_row], columns=target_cols)

        print(f"\n[DEBUG] Modele giden son satır:\n{df_final.iloc[0].to_dict()}\n")

        # 4. Tahmin ve Sonsuzluk Kontrolü
        processed = preprocessor.transform(df_final)
        log_pred = model.predict(processed)[0]

        # --- GÜVENLİK SINIRI ---
        # Logaritmik düzlemde 15'in üstü (yaklaşık 3.2 milyon $) hatadır
        if log_pred > 15:
            log_pred = 13.5  # Yaklaşık 730 bin $ (Ames veri seti için makul üst limit)

        real_price = np.expm1(log_pred)

        # JSON'un 'inf' hatası vermemesi için son kontrol
        if np.isinf(real_price) or np.isnan(real_price):
            return {"predicted_price": "Hesaplanamadı (Veri Uyumsuzluğu)"}

        return {"predicted_price": round(float(real_price), 2)}

    except Exception as e:
        print(traceback.format_exc())
        return {"detail": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
