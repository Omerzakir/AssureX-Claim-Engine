"""Import bridge for the existing model_integration package."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "model_integration" / "backend"
if str(P) not in sys.path:
    sys.path.insert(0, str(P))
