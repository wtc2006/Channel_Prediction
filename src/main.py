"""
Main Orchestrator for the Channel Prediction Pipeline.
Integrates data generation, model training (ML & DL), and warning analysis.
"""

import subprocess
import sys

import config


PIPELINE_STEPS = [
    ("Generating simulated channel data", config.PROJECT_ROOT / "src" / "data_generator.py"),
    ("Training Basic ML Model (Random Forest)", config.PROJECT_ROOT / "src" / "model_training.py"),
    ("Training Deep Learning Model (LSTM)", config.PROJECT_ROOT / "src" / "dl_model_training.py"),
    ("Running Smart Warning Engine", config.PROJECT_ROOT / "src" / "early_warning.py"),
]


def run_script(script_path):
    """Run a pipeline script and fail fast if it exits unsuccessfully."""
    subprocess.run([sys.executable, str(script_path)], cwd=config.PROJECT_ROOT, check=True)


def run_pipeline():
    """Execute the full pipeline end-to-end."""
    print("--- Channel Prediction & Warning System Started ---", flush=True)

    for index, (title, script_path) in enumerate(PIPELINE_STEPS, start=1):
        print(f"\n[Step {index}/{len(PIPELINE_STEPS)}] {title}...", flush=True)
        run_script(script_path)

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("Check the generated plots for analysis.")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
