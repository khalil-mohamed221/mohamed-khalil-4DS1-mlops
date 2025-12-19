from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

# from model_pipeline import load_model
import pandas as pd
import mlflow.pyfunc


app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained CatBoost model.",
    version="1.0.0",
)

# Template engine
templates = Jinja2Templates(directory="templates")

# Load trained model
# ============================================================
# Load Production model from MLflow Model Registry
# ============================================================

print("🔎 Loading Production model from MLflow Registry...")

model = mlflow.pyfunc.load_model("models:/house_price_model/Production")

print("✅ Production model loaded.")


# ================================
#  JSON INPUT SCHEMA (API)
# ================================
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    street: str
    city: str
    statezip: str
    sale_year: int
    sale_month: int


# ================================
#  HOME PAGE (HTML FORM)
# ================================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


# ================================
#  JSON PREDICTION ENDPOINT
# ================================
@app.post("/predict")
def predict(features: HouseFeatures):

    # Convert BaseModel → dictionary
    data = features.dict()

    # Convert dict → DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)[0]

    return {"predicted_price": float(prediction)}


# ================================
#  HTML FORM PREDICTION ENDPOINT
# ================================
@app.post("/predict-form")
def predict_form(
    request: Request,
    bedrooms: int = Form(...),
    bathrooms: float = Form(...),
    sqft_living: float = Form(...),
    sqft_lot: float = Form(...),
    floors: float = Form(...),
    waterfront: int = Form(...),
    view: int = Form(...),
    condition: int = Form(...),
    sqft_above: float = Form(...),
    sqft_basement: float = Form(...),
    yr_built: int = Form(...),
    yr_renovated: int = Form(...),
    street: str = Form(...),
    city: str = Form(...),
    statezip: str = Form(...),
    sale_year: int = Form(...),
    sale_month: int = Form(...),
):

    # Convert form inputs to a dictionary
    data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "street": street,
        "city": city,
        "statezip": statezip,
        "sale_year": sale_year,
        "sale_month": sale_month,
    }

    # Convert dict → DataFrame
    df = pd.DataFrame([data])

    # Predict
    pred = model.predict(df)[0]

    # Return the same page with the result
    return templates.TemplateResponse(
        "form.html",
        {"request": request, "result": round(float(pred), 2)},
    )
