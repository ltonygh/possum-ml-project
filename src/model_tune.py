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
from src.model import PossumNetwork

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)



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
    for _ in range(30):
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
        Loads data, isolates cross-validation matrices, calculates Bayesian parameters, and exports the final trained state dict array.
    """

    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(csv_path)
    
    df["sex_encoded"] = df["sex"].astype(str).str.strip().str.lower().map({"f": 1, "m": 0})
        
    target_tasks = {
        "sex_encoded": {"is_reg": False, "file": "possum_sex_clf.pth", "features": ["totlngth", "footlgth", "chest", "earconch", "belly"]},
        "age":         {"is_reg": True,  "file": "possum_age_reg.pth",  "features": ["belly", "chest", "skullw", "totlngth", "eye"]},
        "hdlngth":     {"is_reg": True,  "file": "possum_hdlngth_reg.pth", "features": ["skullw", "totlngth", "chest", "belly", "footlgth"]}
    }

    for target_col, meta in target_tasks.items():
        logging.info(f"Hypertuning for {target_col.upper()}")
        features = meta["features"]

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df=df,
            feature_list=features, 
            target_col=target_col,
            stratify_target=not meta["is_reg"],
            train_validate_test_ratio=(0.7, 0.1, 0.2),
            random_state=42
        )

        mean_scale = X_train.mean()
        std_scale = X_train.std().replace(0, 1)

        X_train_scaled = (X_train - mean_scale) / std_scale
        X_val_scaled = (X_val - mean_scale) / std_scale

        X_train_np = X_train_scaled.values.astype(np.float32)
        X_val_np = X_val_scaled.values.astype(np.float32)

        if target_col == "age":
            y_train_np = np.log1p(y_train.values).astype(np.float32)
            y_val_np = np.log1p(y_val.values).astype(np.float32)
        else:
            y_train_np = y_train.values.astype(np.float32)
            y_val_np = y_val.values.astype(np.float32)



        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(
            lambda trial: train_and_evaluate(X_train_np, y_train_np, X_val_np, y_val_np, trial, meta["is_reg"]), 
            n_trials=20
        )

        logging.info(f"{target_col} Best Trial Optimization Score: {np.abs(study.best_value):.4f}")
        logging.info(f"{target_col} Best Configuration Found: {study.best_params}")

        best_params = study.best_params
        final_model = PossumNetwork(X_train_np.shape[1], best_params["hidden_dim"], best_params["num_layers"], best_params["dropout_rate"])
        


        if best_params["optimizer"] == "Adam":
            optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
        elif best_params["optimizer"] == "RMSprop":
            optimizer = optim.RMSprop(final_model.parameters(), lr=best_params["lr"])
        else:
            optimizer = optim.SGD(final_model.parameters(), lr=best_params["lr"], momentum=0.9)

        criterion = nn.MSELoss() if meta["is_reg"] else nn.BCEWithLogitsLoss()

        X_full_scaled = pd.concat([X_train_scaled, X_val_scaled]).values.astype(np.float32)
        if target_col == "age":
            y_full_labels = np.log1p(pd.concat([y_train, y_val]).values).astype(np.float32)
        else:
            y_full_labels = pd.concat([y_train, y_val]).values.astype(np.float32)

        full_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_full_scaled), torch.FloatTensor(y_full_labels).unsqueeze(1)),
            batch_size=best_params["batch_size"], shuffle=True
        )

        final_model.train()
        for _ in range(50):
            for bx, by in full_loader:
                optimizer.zero_grad()
                loss = criterion(final_model(bx), by)
                loss.backward()
                optimizer.step()
        
        final_model.eval()

        export_path = os.path.join(output_dir, meta["file"])
        torch.save({
            "state_dict": final_model.state_dict(),
            "hyperparameters": best_params,
            "features_ordered": features,
            "is_classification": not meta["is_reg"],
            "scale_mean": mean_scale.to_dict(),
            "scale_std": std_scale.to_dict()
        }, export_path)
        
        logging.info(f"Production weights successfully serialized to: {export_path}\n")




if __name__ == "__main__":
    run_pipeline_tuning("data/possum.csv")