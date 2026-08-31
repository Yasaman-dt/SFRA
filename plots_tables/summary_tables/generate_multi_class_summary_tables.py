"""Generate the multi-class summary tables from one public command.

The 2-class and 5-vs-10-class tables use different input layouts and table
formats. This launcher gives them one public interface while leaving those
specialized implementations independent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent / "multi_class_summary"
GENERATORS = {
    "2": SCRIPT_DIR / "_two_class.py",
    "5-10": SCRIPT_DIR / "_five_vs_ten_class.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the 2-class table, the 5-vs-10-class table, or both. "
            "Each table is written to its existing results directory."
        )
    )
    parser.add_argument(
        "--setting",
        choices=("2", "5-10", "all"),
        default="all",
        help="Table setting to generate (default: all).",
    )
    return parser.parse_args()


def run_generator(setting: str) -> None:
    script = GENERATORS[setting]
    if not script.is_file():
        raise FileNotFoundError(f"Generator not found: {script}")
    print(f"[INFO] Running {script.name}")
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    args = parse_args()
    settings = tuple(GENERATORS) if args.setting == "all" else (args.setting,)
    for setting in settings:
        run_generator(setting)


if __name__ == "__main__":
    main()
