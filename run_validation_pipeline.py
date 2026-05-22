from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from hvac_v3_engine import run_validation_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROM vs DesignBuilder validation and residual correction.")
    parser.add_argument("--input", required=True, help="CSV/TXT/XLSX file with date, DesignBuilder/reference and ROM/model columns.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--no_holdout", action="store_true", help="Use all data for correction calibration. Not recommended for manuscript validation.")
    args = parser.parse_args()

    outputs = run_validation_workflow(args.input, args.output, holdout_last_year=not args.no_holdout)
    print("Saved validation outputs to:", Path(args.output).resolve())
    print("\nMetrics:")
    print(outputs["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
