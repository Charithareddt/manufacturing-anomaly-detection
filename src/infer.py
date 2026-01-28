import joblib
import pandas as pd
from src.config import PATHS
from src.utils import load_csv, save_csv, ensure_dir
from src.feature_engineering import build_features
from src.train import NUM_COLS, CAT_COLS

def score(df: pd.DataFrame) -> pd.DataFrame:
    pipe = joblib.load(PATHS.models / "model.pkl")
    X = df[CAT_COLS + NUM_COLS]
    # IsolationForest: -1 anomaly, 1 normal
    pred = pipe.predict(X)
    score = pipe.decision_function(X)  # higher = more normal
    out = df.copy()
    out["anomaly_flag"] = (pred == -1).astype(int)
    out["anomaly_score"] = score
    return out

if __name__ == "__main__":
    ensure_dir(PATHS.data_outputs)
    df = load_csv(PATHS.data_raw / "manufacturing_claims_raw.csv")
    df = build_features(df)
    scored = score(df)
    save_csv(scored, PATHS.data_outputs / "scored_claims.csv")
    print("Saved:", PATHS.data_outputs / "scored_claims.csv")
