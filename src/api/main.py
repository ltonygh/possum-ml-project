import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



app = FastAPI(
    title = "Possum Morphometric Predictor API",
    description = "Production endpoint delivering inference scores for possum population metrics.",
    version = "1.0.0"
)

class PossumInferenceRequest(BaseModel):
    """Strictly type-checks incoming web requests to ensure schema validation safety."""
    site: int = Field(..., description="Site code where possum was captured (1-7)", ge=1, le=7)
    age: float = Field(..., description="Age of the possum in years", ge=0.0, le=15.0)
    head_ln: float = Field(..., description="Head length in millimeters", ge=50.0, le=120.0)
    skull_w: float = Field(..., description="Skull width in millimeters", ge=30.0, le=80.0)
    total_l: float = Field(..., description="Total length in centimeters", ge=60.0, le=110.0)
    tail_l: float = Field(..., description="Tail length in centimeters", ge=20.0, le=55.0)



predict = None
model = "models/possum_classifier.pkl"

def load_cached_predictor():
    """
        Lazy loads the weights from the optimized model.
    """

    global predict
    if predict is None:
        if not os.path.exists(model):
            raise RuntimeError(f"Weights asset file missing from disk path: {model}")
        with open(model, "rb") as f:
            predict = pickle.load(f)
    return predict



@app.get("/", tags=["System Status"])
def read_root():
    return {"status": "online", "model_configured": os.path.exists(model)}


@app.post("/predict", tags=["Machine Learning Inference"])
def predict_possum_metrics(payload: PossumInferenceRequest):
    """
        Accepts JSON payloads, parses feature strings, and runs matrix evaluations.
    """
    
    try:
        model = load_cached_predictor()
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Inference engine unavailable: {str(err)}")

    features_array = np.array([[
        payload.site,
        payload.age,
        payload.head_ln,
        payload.skull_w,
        payload.total_l,
        payload.tail_l
    ]], dtype=np.float32)

    prediction_index = int(model.predict(features_array)[0])

    label_mapping = {0: "Vic (Victoria Population)", 1: "other (New South Wales/Queensland)"}
    result_population = label_mapping.get(prediction_index, "Unknown")

    return {
        "prediction_index": prediction_index,
        "predicted_population": result_population,
        "input_features_validated": payload.dict()
    }