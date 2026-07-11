import os
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model import PossumSexClassifier, PossumAgeRegressor, PossumHeadLengthRegressor



app = FastAPI(
    title = "Possum Morphometric Predictor API",
    description = "Production endpoint delivering inference scores for possum population metrics.",
    version = "1.1.0"
)

class PossumFeatures(BaseModel):
    """
        Strictly type-checks incoming web requests to ensure schema validation safety.
    """
    site:    int = Field(..., description="Site code where possum was captured (1-7)", ge=1, le=7)
    skull_w: float = Field(..., description="Skull width in millimeters", ge=30.0, le=80.0)
    total_l: float = Field(..., description="Total length in centimeters", ge=60.0, le=110.0)
    tail_l:  float = Field(..., description="Tail length in centimeters", ge=20.0, le=55.0)



predict_sex = None
predict_age = None
predict_headln = None

sex_path = "models/possum_sex.pth"
age_path = "models/possum_age.pth"
headln_path = "models/possum_headln.pth"

def load_predictor():
    """
        Lazy loads the weights from the optimized models.
    """

    global predict_sex, predict_age, predict_headln

    if predict_sex is None or predict_age is None or predict_headln is None:
        predict_sex = PossumSexClassifier(input_dim=4, hidden_dim=16)
        predict_age = PossumAgeRegressor(input_dim=4, hidden_dim=16)
        predict_headln = PossumHeadLengthRegressor(input_dim=4, hidden_dim=16)
    
        if os.path.exists(sex_path): predict_sex.load_state_dict(torch.load(sex_path, weights_only=True))
        if os.path.exists(age_path): predict_age.load_state_dict(torch.load(age_path, weights_only=True))
        if os.path.exists(headln_path): predict_headln.load_state_dict(torch.load(headln_path, weights_only=True))

        predict_sex.eval()
        predict_age.eval()
        predict_headln.eval()
    
    return predict_sex, predict_age, predict_headln



@app.get("/")
def get_status():
    return {
        "status": "online", 
        "models_found": {
            "sex_classifier": os.path.exists(sex_path),
            "age_regressor": os.path.exists(age_path),
            "head_length_regressor": os.path.exists(headln_path)
        }
    }


@app.post("/predict")
def predict_possum_metrics(payload: PossumFeatures):
    """
        Accepts JSON payloads, parses feature strings, and runs matrix evaluations on sex, age, and head length.
    """
    
    sex_net, age_net, head_net = load_predictor()

    input_tensor = torch.tensor([[
        payload.site, 
        payload.skull_w, 
        payload.total_l, 
        payload.tail_l
    ]], dtype=torch.float32)

    with torch.no_grad():
        raw_sex = sex_net(input_tensor)
        raw_age = age_net(input_tensor)
        raw_head = head_net(input_tensor)

    sex_logit = float(raw_sex.item())
    sex_prediction = "Male" if sex_logit > 0.0 else "Female"
    if sex_logit > 0.0:
        sex_confidence = (1.0 / (1/0 + np.exp(-sex_logit))) * 100
    else: 
        sex_confidence = (1.0 - (1.0 / (1.0 + np.exp(-sex_logit)))) * 100.0

    predicted_age = max(0.0, float(raw_age.item()))

    predicted_headln = float(raw_head.item())

    return {
        "predictions": {
            "sex": {
                "result": sex_prediction,
                "confidence_probability": f"{sex_confidence:.2f}%"
            },
            "estimated_age_years": round(predicted_age, 1),
            "estimated_head_length_mm": round(predicted_headln, 2)
        },
        "submitted_measurements": payload.dict()
    }