"""
run_pipeline.py
===============
Root-level entry point to execute the Marathwada WSI Rebuild Pipeline.

Usage:
  python run_pipeline.py
"""

import sys
import os

# Add workspace directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    print("=" * 70)
    print("MARATHWADA WATER STRESS INDEX (WSI) DATA ENGINEERING PIPELINE")
    print("Rebuilding from raw sources with zero leakage & strict temporal order")
    print("=" * 70)
    
    try:
        results = run_pipeline(config_path="config/pipeline.yaml", output_dir=".")
        print("\nPipeline execution finished successfully!")
        print("Authoritative outputs created:")
        print("  - outputs/Marathwada_MASTER_2003_2024.parquet")
        print("  - outputs/Marathwada_MASTER_AUDIT_DATASET_2003_2024.csv")
        print("  - outputs/Marathwada_MODEL_DATASET_2003_2024.csv")
        print("  - artifacts/train_statistics.json")
        print("  - artifacts/validation_report.json")
        print("  - artifacts/preprocessing_metadata.json")
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
