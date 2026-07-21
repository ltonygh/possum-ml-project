import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient
from src.api import main



client = TestClient(main.app)

@pytest.fixture(autouse=True)
def reset_and_mock_api_globals(tmp_path, monkeypatch):
    """
        Reset cached global predictor from leaking.
    """
    monkeypatch.setattr(main, "model_cache", {})
    return None



class DummyPossumModel:
    """
        A minimal dummy class mirroring scikit-learn's interface for pipeline test injections.
    """
    def __init__(self):
        super().__init__()
    def __call__(self, x):
        return torch.tensor([[1.5]], dtype=torch.float32)
    def eval(self):
        pass



def test_api_health_check_endpoint():
    """
        Test if root GET endpoint returns 200 - OK status.
    """
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "online"
    assert "models_configured" in data

def test_api_predict_all_successful_inference(monkeypatch):
    """
        Verifies multi-task inference mapping formats when weights are active.
    """
    mock_meta = {
        "model": DummyPossumModel(),
        "features_ordered": ["totlngth", "footlgth", "chest", "earconch", "belly"],
        "scale_mean": {"totlngth": 89.0, "footlgth": 68.0, "chest": 27.5, "earconch": 46.5, "belly": 32.5},
        "scale_std": {"totlngth": 3.0, "footlgth": 2.0, "chest": 1.5, "earconch": 2.0, "belly": 2.1}
    }
    
    monkeypatch.setattr(main, "model_cache", {
        "sex": mock_meta, "age": mock_meta, "hdlngth": mock_meta
    })
    
    sample_payload = {
        "totlngth": 89.0, "footlgth": 68.0, "chest": 27.5, 
        "earconch": 46.5, "belly": 32.5, "skullw": 57.5, "eye": 15.0
    }
    
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "input_features_validated" in data
    assert data["input_features_validated"]["totlngth"] == 89.0

def test_api_predict_validation_gate_failures():
    """
        Ensures Pydantic rejects out-of-bound values for total length = 150.0.
    """
    invalid_payload = {
        "totlngth": 150.0,
        "footlgth": 68.0, "chest": 27.5, "earconch": 46.5, "belly": 32.5, "skullw": 57.5, "eye": 15.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422