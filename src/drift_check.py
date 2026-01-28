import joblib
import numpy as np
from src.config import PATHS
from src.utils import load_csv
from src.feature_engineering import build_features
from src.train import NUM_COLS

def drift_flag(baseline_stats: dict, current_df, z_thresh: float = 2.5):
    """
    baseline_stats is saved as: baseline_stats[col][stat] from df.describe().to_dict()
    Example: baseline_stats["equipment_temp"]["mean"]
    """
    flags = {}
    for col in NUM_COLS:
        base_mean = float(baseline_stats[col]["mean"])
        base_std = float(baseline_stats[col]["std"]) if float(baseline_stats[col]["std"]) != 0 else 1e-6

        cur_mean = float(current_df[col].mean())
        z = abs(cur_mean - base_mean) / (base_std + 1e-6)

        flags[col] = {"z_score": float(z), "drift": bool(z > z_thresh)}
    return flags


if __name__ == "__main__":
    baseline = joblib.load(PATHS.models / "baseline_stats.pkl")
    df = load_csv(PATHS.data_raw / "manufacturing_claims_raw.csv")
    df = build_features(df)

    flags = drift_flag(baseline, df)
    drifted = [k for k, v in flags.items() if v["drift"]]
    print("Drifted features:", drifted if drifted else "None")
    # Print top 3 largest z-scores
    top = sorted(flags.items(), key=lambda kv: kv[1]["z_score"], reverse=True)[:3]
    print("Top shifts:", top)
