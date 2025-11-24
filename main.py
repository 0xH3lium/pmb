import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcmc_pmb.run_pipeline import main as run_pipeline


def main():
    run_pipeline(sampler="nuts")


if __name__ == "__main__":
    main()
