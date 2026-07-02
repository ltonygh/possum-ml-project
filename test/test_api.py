import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api import main



client = TestClient(main.app)

@pytest.fixture(autouse=True)
def reset_and_mock_api_globals(tmp_path, monkeypatch):
    """
        Reset cached global predictor from leaking.
    """
    monkeypatch.setattr(main, "predict", None)
    
    dummy_path = tmp_path / "mock_possum_model.pkl"
    monkeypatch.setattr(main, "model", str(dummy_path))
    return dummy_path



class DummyPossumModel:
    """
        A minimal dummy class mirroring scikit-learn's interface for pipeline test injections.
    """
    def predict(self, features_array):
        return np.array([0])



def test_api_health_check_endpoint():
    """
        Test if root GET endpoint returns 200 - OK status.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "model_configured": False}

def test_api_predict_endpoint_missing_weights():
    """
        Test if endpoint returns 503 - Service Unavailable status code if the weights file is missing.
    """

    payload = {
        "site": 1, "age": 4.0, "head_ln": 91.5,
        "skull_w": 56.2, "total_l": 84.0, "tail_l": 36.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Inference engine unavailable" in response.json()["detail"]

def test_api_predict_endpoint_successful_inference(monkeypatch, reset_and_mock_api_globals):
    """
        Test if successful request returns a proper label formatting when weights are active.
    """
    dummy_path = reset_and_mock_api_globals
    
    dummy_path.touch()
    
    monkeypatch.setattr(main, "predict", DummyPossumModel())
    
    sample = {
        "site": 3, "age": 5.5, "head_ln": 94.0,
        "skull_w": 60.0, "total_l": 89.5, "tail_l": 38.0
    }
    
    response = client.post("/predict", json = sample)
    assert response.status_code == 200
    
    data = response.json()
    assert data["prediction_index"] == 0
    assert "Vic (Victoria Population)" in data["predicted_population"]
    assert data["input_features_validated"]["site"] == 3

def test_api_predict_endpoint_validation_gate_failure():
    """
        Test if endpoint returns 422 - Unprocessable Entity when receiving invalid parameter bounds with a 422 Unprocessable Entity.
    """

    sample = {
        "site": 9, 
        "age": -2.0,
        "head_ln": 91.5, "skull_w": 56.2, "total_l": 84.0, "tail_l": 36.0
    }
    response = client.post("/predict", json = sample)
    assert response.status_code == 422