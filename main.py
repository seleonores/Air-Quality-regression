from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class Features(BaseModel):
    PT08_S1_CO: float = Field(alias="PT08.S1(CO)")
    NMHC_GT: float = Field(alias="NMHC(GT)")
    C6H6_GT: float = Field(alias="C6H6(GT)")
    PT08_S2_NMHC: float = Field(alias="PT08.S2(NMHC)")
    NOx_GT: float = Field(alias="NOx(GT)")
    PT08_S3_NOx: float = Field(alias="PT08.S3(NOx)")
    NO2_GT: float = Field(alias="NO2(GT)")
    PT08_S4_NO2: float = Field(alias="PT08.S4(NO2)")
    PT08_S5_O3: float = Field(alias="PT08.S5(O3)")
    T: float
    RH: float
    AH: float
    hour: int
    weekday: int
    month: int

@app.post("/predict")
def predict(features: Features):
    input_data = np.array([[
        features.PT08_S1_CO,
        features.NMHC_GT,
        features.C6H6_GT,
        features.PT08_S2_NMHC,
        features.NOx_GT,
        features.PT08_S3_NOx,
        features.NO2_GT,
        features.PT08_S4_NO2,
        features.PT08_S5_O3,
        features.T,
        features.RH,
        features.AH,
        features.hour,
        features.weekday,
        features.month,
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return {"prediction": prediction[0]}
