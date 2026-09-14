"""
Script to generate high-fidelity synthetic ICU cohort for demo and testing.
"""

import argparse
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.data.synthetic_generator import SyntheticDataGenerator
from medguard.utils.logging import get_logger
from medguard.utils.config import get_project_root, load_yaml

logger = get_logger("medguard.scripts.generate_synthetic")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ICU cohort for MEDGUARD AI.")
    parser.add_argument("--num_patients", type=int, default=600, help="Number of synthetic patients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    args = parser.parse_args()

    root = get_project_root()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDataGenerator(seed=args.seed)
    df_patients, df_timeseries, df_notes = generator.generate_cohort(
        num_patients=args.num_patients,
        base_mortality_rate=0.18,
        obs_window_hours=24,
    )

    df_patients.to_parquet(out_dir / "patients.parquet", index=False)
    df_timeseries.to_parquet(out_dir / "timeseries.parquet", index=False)
    df_notes.to_parquet(out_dir / "notes.parquet", index=False)

    # Also save small CSV copies for easy inspection
    df_patients.head(100).to_csv(out_dir / "patients_sample.csv", index=False)
    df_timeseries.head(500).to_csv(out_dir / "timeseries_sample.csv", index=False)
    df_notes.head(100).to_csv(out_dir / "notes_sample.csv", index=False)

    logger.info(f"Synthetic dataset saved successfully to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
