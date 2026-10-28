"""Generate all findings figures into results/figures/findings/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ymlab.figures import generate_static  # noqa: E402

if __name__ == "__main__":
    out = generate_static(ROOT / "results" / "figures" / "findings")
    for k, v in out.items():
        print(k, *[p.name for p in v])
