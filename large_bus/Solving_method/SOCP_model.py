"""12 时段统一建模的 DistFlow SOCP 松弛/Gurobi 基线。"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Solving_method.Distflow_model_ipopt import run_validation


def main():
    run_validation(relaxation="socp")


if __name__ == "__main__":
    main()
