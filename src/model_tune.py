import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset
import logging

from src.data_prepare import load_dataset
from src.data_split import split_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)



class PossumNetwork(nn.Module):
    """
        A unified neural network architecture. Dynamically constructs hidden layers
        and adapts outputs for either classification or regression tasks.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout_rate: float):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)



def train_and_evaluate(X_train_np, y_train_np, X_val_np, y_val_np, trial, is_regression: bool) -> float:
    """
        Wraps isolated training steps inside a trial scope to calculate validation 
        accuracy scores across candidate hyperparameter vectors.
    """ 

    hidden_dim = trial.suggest_int("hidden_dim", 16, 64, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_np), torch.FloatTensor(y_train_np).unsqueeze(1)), 
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_np), torch.FloatTensor(y_val_np).unsqueeze(1)), 
        batch_size=batch_size, shuffle=False
    )

    model = PossumNetwork(
        input_dim=X_train_np.shape[1], 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        dropout_rate=dropout_rate
    )
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    criterion = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(20):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss_accumulator = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            
            if is_regression:
                loss = criterion(outputs, batch_y)
                val_loss_accumulator += loss.item() * batch_x.size(0)
                total_samples += batch_y.size(0)
            else:
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                correct_predictions += (predictions == batch_y).sum().item()
                total_samples += batch_y.size(0)

    return -(val_loss_accumulator / total_samples) if is_regression else correct_predictions / total_samples



def run_pipeline_tuning(csv_path: str, output_dir: str = "models"):
    """
        Loads data, isolates cross-validation matrices, calculates 
        Bayesian parameters, and exports the final trained state dict array.
    """

    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(csv_path)
    
    df["sex_encoded"] = df["sex"].astype(str).str.strip().str.lower().map({"f": 1, "m": 0})
        
    relation_features = ["age", "hdlngth", "skullw", "totlngth", "taill"]
    
    target_tasks = {
        "sex_encoded": {"is_reg": False, "file": "possum_sex_clf.pth"},
        "age":         {"is_reg": True, "file": "possum_age_reg.pth"},
        "hdlngth":     {"is_reg": True, "file": "possum_hdlngth_reg.pth"}
    }

    for target_col, meta in target_tasks.items():
        logging.info(f"Hypertuning for {target_col.upper()}")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df=df,
            feature_list=[f for f in relation_features if f != target_col], # Prevent data leakage
            target_col=target_col,
            stratify_target=not meta["is_reg"],
            train_validate_test_ratio=(0.7, 0.1, 0.2),
            random_state=42
        )

        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.float32)
        X_val_np = X_val.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.float32)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(
            lambda trial: train_and_evaluate(X_train_np, y_train_np, X_val_np, y_val_np, trial, meta["is_reg"]), 
            n_trials=15
        )

        logging.info(f"{target_col} Best Trial Optimization Score: {np.abs(study.best_value):.4f}")
        logging.info(f"{target_col} Best Configuration Found: {study.best_params}")

        best_params = study.best_params
        final_model = PossumNetwork(X_train_np.shape[1], best_params["hidden_dim"], best_params["num_layers"], best_params["dropout_rate"])
        export_path = os.path.join(output_dir, meta["file"])
        torch.save({
            "state_dict": final_model.state_dict(),
            "hyperparameters": best_params,
            "features_ordered": [f for f in relation_features if f != target_col],
            "is_classification": not meta["is_reg"]
        }, export_path)
        
        logging.info(f"Production weights successfully serialized to: {export_path}\n")




if __name__ == "__main__":
    run_pipeline_tuning("data/possum.csv")