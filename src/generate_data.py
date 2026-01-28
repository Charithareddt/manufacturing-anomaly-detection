import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import PATHS
from src.utils import save_csv, ensure_dir

def generate(n_days: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=n_days)

    products = ["Motorboat", "Motorhome", "Towable"]
    plants = ["Plant_A", "Plant_B"]
    regions = ["West", "South", "Midwest", "Northeast"]

    rows = []
    for d in range(n_days):
        date = start + timedelta(days=d)
        for _ in range(rng.integers(80, 140)):  # daily volume
            product = rng.choice(products, p=[0.35, 0.35, 0.30])
            plant = rng.choice(plants, p=[0.55, 0.45])
            region = rng.choice(regions)

            sales_units = int(max(1, rng.normal(1.5, 0.7)))
            equipment_temp = float(rng.normal(75, 6))  # equipment metric
            line_speed = float(rng.normal(1.0, 0.08))

            # Base claim rate by product
            base_claim_rate = {"Motorboat": 0.05, "Motorhome": 0.07, "Towable": 0.06}[product]

            # Plant_B slightly riskier; higher temp/speed increases risk
            risk = base_claim_rate
            if plant == "Plant_B":
                risk += 0.01
            risk += max(0, (equipment_temp - 80) * 0.002)
            risk += max(0, (line_speed - 1.05) * 0.10)

            # Inject a realistic incident window (anomaly period)
            if 60 <= d <= 75 and plant == "Plant_B" and product == "Motorhome":
                risk += 0.08  # spike

            claims = rng.binomial(n=sales_units, p=min(0.95, risk))
            rows.append({
                "date": str(date),
                "product": product,
                "plant": plant,
                "region": region,
                "sales_units": sales_units,
                "claims": claims,
                "equipment_temp": equipment_temp,
                "line_speed": line_speed
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    ensure_dir(PATHS.data_raw)
    df = generate()
    save_csv(df, PATHS.data_raw / "manufacturing_claims_raw.csv")
    print("Saved:", PATHS.data_raw / "manufacturing_claims_raw.csv")
