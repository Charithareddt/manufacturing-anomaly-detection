from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    data_outputs: Path = root / "data" / "outputs"
    models: Path = root / "models"

PATHS = Paths()
