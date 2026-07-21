import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model import PossumNetwork



app = FastAPI(
    title = "Possum Morphometric Predictor API",
    description = "Production endpoint delivering inference scores for possum population metrics.",
    version = "1.2.0"
)

class PossumJSONRequest(BaseModel):
    """
        Strictly type-checks incoming requests to ensure comprehensive feature validation.
    """
    totlngth: float = Field(..., description="Total length in centimeters", ge=60.0, le=110.0)
    footlgth: float = Field(..., description="Foot length in millimeters", ge=50.0, le=80.0)
    chest:    float = Field(..., description="Chest girth in centimeters", ge=20.0, le=45.0)
    earconch: float = Field(..., description="Ear conch length in millimeters", ge=35.0, le=60.0)
    belly:    float = Field(..., description="Belly circumference in centimeters", ge=20.0, le=45.0)
    skullw:   float = Field(..., description="Skull width in millimeters", ge=30.0, le=80.0)
    eye:      float = Field(..., description="Eye size dimension index", ge=10.0, le=25.0)



model_cache = {}
model_path = {
    "sex": "models/possum_sex_clf.pth",
    "age": "models/possum_age_reg.pth",
    "hdlngth": "models/possum_hdlngth_reg.pth"
}

def load_pytorch_model(target_key: str):
    """
        Lazy-loads models, re-instantiates optimized topologies, and maps state weights.
    """
    if target_key not in model_cache:
        path = model_path[target_key]
        if not os.path.exists(path):
            raise RuntimeError(f"Model asset binary missing from disk: {path}")
            
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        hparams = checkpoint["hyperparameters"]
        
        model = PossumNetwork(
            input_dim=len(checkpoint["features_ordered"]),
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            dropout_rate=hparams["dropout_rate"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        model_cache[target_key] = {
            "model": model,
            "features_ordered": checkpoint["features_ordered"],
            "scale_mean": checkpoint["scale_mean"],
            "scale_std": checkpoint["scale_std"]
        }
    return model_cache[target_key]



@app.get("/", tags=["System Status"])
def read_root():
    all_configured = all(os.path.exists(p) for p in model_path.values())
    return {"status": "online", "models_configured": all_configured}


@app.post("/predict")
def predict_possum_metrics(payload: PossumJSONRequest):
    """
        Accepts JSON payloads, parses feature strings, and runs matrix evaluations on sex, age, and head length.
    """
    
    raw_input = payload.model_dump()
    predictions = {}

    for target, _ in model_path.items():
        try:
            meta = load_pytorch_model(target)
            model = meta["model"]
            ordered_features = meta["features_ordered"]
            means = meta["scale_mean"]
            stdevs = meta["scale_std"]

            ordered_raw_vector = [raw_input[feat] for feat in ordered_features]
            
            scaled_vector = []
            for i, feat in enumerate(ordered_features):
                scaled_val = (ordered_raw_vector[i] - means[feat]) / stdevs[feat]
                scaled_vector.append(scaled_val)

            input_tensor = torch.tensor([scaled_vector], dtype=torch.float32)
            
            with torch.no_grad():
                raw_output = model(input_tensor).item()
                
            if target == "sex":
                prob = 1 / (1 + torch.exp(torch.tensor(-raw_output)).item())
                pred_class = 1 if prob >= 0.5 else 0
                predictions["sex_prediction"] = {
                    "index": pred_class,
                    "label": "Female" if pred_class == 1 else "Male",
                    "confidence_probability": round(prob if pred_class == 1 else 1 - prob, 4)
                }
            elif target == "age":
                predictions["predicted_age_years"] = max(0.0, round(raw_output, 2))
            elif target == "hdlngth":
                predictions["predicted_head_length_mm"] = max(0.0, round(raw_output, 2))
                
        except Exception as err:
            raise HTTPException(status_code=503, detail=f"Inference failure on [{target}]: {str(err)}")
            
    return {
        "predictions": predictions,
        "input_features_validated": raw_input
    }