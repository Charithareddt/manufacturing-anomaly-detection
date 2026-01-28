import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from src.config import PATHS
from src.utils import load_csv, ensure_dir
from src.feature_engineering import build_features

NUM_COLS = ["sales_units", "equipment_temp", "line_speed", "dow", "month", "claim_rate_7d_mean", "claim_rate_7d_std"]
CAT_COLS = ["product", "plant", "region"]

def train_model(df: pd.DataFrame) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    model = IsolationForest(
        n_estimators=200,
        contamination=0.04,
        random_state=42
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(df[CAT_COLS + NUM_COLS])
    return pipe

if __name__ == "__main__":
    df = load_csv(PATHS.data_raw / "manufacturing_claims_raw.csv")
    df = build_features(df)

    ensure_dir(PATHS.models)
    pipe = train_model(df)

    joblib.dump(pipe, PATHS.models / "model.pkl")
    # Save baseline feature distribution for drift checks
    baseline = df[NUM_COLS].describe().to_dict()
    joblib.dump(baseline, PATHS.models / "baseline_stats.pkl")

    print("Saved model to:", PATHS.models / "model.pkl")
