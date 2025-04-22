# Air Quality Prediction API

This is a FastAPI-based web service that predicts air quality based on various environmental sensor readings and temporal features.

## Features

- Predicts air quality using a pre-trained machine learning model
- Accepts 15 input features including chemical concentrations, temperature, humidity, and time-related features
- Scales input features using a pre-fitted scaler for model compatibility
- Returns predictions in JSON format

## Requirements

- Python 3.7+
- FastAPI
- Pydantic
- NumPy
- scikit-learn (for model and scaler compatibility)
- joblib (for model loading)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install fastapi pydantic numpy scikit-learn joblib
   ```
3. Ensure you have the following files in your working directory:
   - `model.pkl` (your pre-trained model)
   - `scaler.pkl` (your pre-fitted scaler)

## Usage

### Running the API

Start the server with:
```
uvicorn main:app --reload
```

### Making Predictions
- http://127.0.0.1:8000/docs
- Send a POST request to `/predict` with a JSON body containing the required features. Example:

```json
{
  "PT08.S1(CO)": 1000.5,
  "NMHC(GT)": 150.2,
  "C6H6(GT)": 12.3,
  "PT08.S2(NMHC)": 950.1,
  "NOx(GT)": 200.7,
  "PT08.S3(NOx)": 1300.4,
  "NO2(GT)": 150.9,
  "PT08.S4(NO2)": 1200.3,
  "PT08.S5(O3)": 1100.6,
  "T": 22.5,
  "RH": 45.2,
  "AH": 0.8,
  "hour": 14,
  "weekday": 3,
  "month": 7
}
```

### Response

The API will return a JSON response with the prediction:
```json
{
  "prediction": 2.3790000000000004
}
```

