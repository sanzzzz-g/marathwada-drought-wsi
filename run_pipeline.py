"""
Entry point to execute the Marathwada WSI Data Engineering Pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    print("Running Marathwada Water Stress Index (WSI) data pipeline...")
    try:
        results = run_pipeline(config_path="config/pipeline.yaml", output_dir=".")
        print("Pipeline execution completed successfully.")
        print("Outputs created:")
        print("  - outputs/Marathwada_MASTER_2003_2024.parquet")
        print("  - outputs/Marathwada_MASTER_AUDIT_DATASET_2003_2024.csv")
        print("  - outputs/Marathwada_MODEL_DATASET_2003_2024.csv")
        print("  - artifacts/train_statistics.json")
        print("  - artifacts/validation_report.json")
        print("  - artifacts/preprocessing_metadata.json")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
