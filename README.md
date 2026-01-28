# Manufacturing Claims Anomaly Detection (Batch + Drift)

This project demonstrates a production-style batch ML workflow for detecting anomalies in manufacturing-related data such as sales, claims, and equipment signals.  
It includes:
- Feature engineering
- Anomaly detection model training (Isolation Forest)
- Batch inference (scheduled scoring)
- Simple drift detection using statistical distribution checks

## Why this project
In manufacturing environments (e.g., motorboats, motorhomes, towables), abnormal patterns in claims or equipment signals can indicate quality issues or operational risks.  
This repo simulates a realistic workflow used in enterprise settings where batch inference is preferred for cost and stability.

## Data
The repo generates a synthetic dataset that mimics:
- Sales units
- Warranty claims
- Equipment temperature and line speed
- Product, plant, and region attributes  
An incident window is injected to simulate a real anomaly spike.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

python src/generate_data.py
python src/train.py
python src/infer.py
python src/drift_check.py
