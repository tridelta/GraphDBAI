"""Stage 2 analysis helper.

Usage:
  python tools/analyze_stage2.py --run-dir runs --prefix stage2_pilot_real_ --output-dir runs/stage2_pilot_real_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experience_graph.scripts import analyze_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a filtered set of ExperienceGraph runs.")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--prefix", required=True, help="Only analyze runs whose run_id starts with this prefix.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    sys.argv = [
        "eg-analyze-results",
        "--run-dir",
        args.run_dir,
        "--output-dir",
        args.output_dir,
        "--window",
        str(args.window),
        "--run-id-prefix",
        args.prefix,
    ]
    analyze_results.main()


if __name__ == "__main__":
    main()
