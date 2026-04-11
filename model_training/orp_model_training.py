from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"
MODEL_FILE = MODEL_DIR / "orp_model.pkl"
CSV_FILE = BASE_DIR / "KrishiLink_MultiOutput_Training_Data_v4.csv"


def train_orp_model(orp_df: pd.DataFrame):
    """Train and evaluate the Phase 2 ORP recommendation model."""
    x = orp_df[["ORP_mV"]]
    y = orp_df["Phase2_ORP_Flood_Water_Liters"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("ORP Model Evaluation")
    print("-" * 30)
    print("Phase2_ORP_Flood_Water_Liters:")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2:  {r2:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved trained model to: {MODEL_FILE}")

    return model


if __name__ == "__main__":
    data = pd.read_csv(CSV_FILE)
    data = data.drop(columns=["Health_Status"], errors="ignore")
    orp_df = data[["ORP_mV", "Phase2_ORP_Flood_Water_Liters"]]
    train_orp_model(orp_df)