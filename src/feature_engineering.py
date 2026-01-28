import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Target: claim_rate (claims per unit)
    df["claim_rate"] = df["claims"] / df["sales_units"].clip(lower=1)

    # Rolling signals (per plant+product)
    df = df.sort_values(["plant", "product", "date"])
    df["claim_rate_7d_mean"] = (
        df.groupby(["plant", "product"])["claim_rate"]
          .transform(lambda s: s.rolling(7, min_periods=3).mean())
    )
    df["claim_rate_7d_std"] = (
        df.groupby(["plant", "product"])["claim_rate"]
          .transform(lambda s: s.rolling(7, min_periods=3).std())
    )

    df["claim_rate_7d_std"] = df["claim_rate_7d_std"].fillna(0.0)
    df["claim_rate_7d_mean"] = df["claim_rate_7d_mean"].fillna(df["claim_rate"].mean())

    return df
