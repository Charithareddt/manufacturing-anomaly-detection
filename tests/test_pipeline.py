from src.generate_data import generate
from src.feature_engineering import build_features

def test_feature_engineering_outputs():
    df = generate(n_days=20, seed=1)
    fe = build_features(df)
    assert "claim_rate" in fe.columns
    assert "claim_rate_7d_mean" in fe.columns
    assert fe.isnull().sum().sum() == 0
