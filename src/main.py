"""
Main Orchestrator for the Channel Prediction Pipeline.
Integrates data generation, model training (ML & DL), and warning analysis.
"""

import os
import config

def run_pipeline():
    """Executes the full pipeline end-to-end."""
    print("--- Channel Prediction & Warning System Started ---")
    
    # 1. Data Generation
    print("\n[Step 1/4] Generating simulated channel data...")
    os.system("python src/data_generator.py")
    
    # 2. Model Training (ML + DL)
    print("\n[Step 2/4] Training Basic ML Model (Random Forest)...")
    os.system("python src/model_training.py")
    
    print("\n[Step 3/4] Training Deep Learning Model (LSTM)...")
    os.system("python src/dl_model_training.py")
    
    # 3. Warning Analysis
    print("\n[Step 4/4] Running Smart Warning Engine...")
    os.system("python src/early_warning.py")

    print("\n" + "="*50)
    print("✅ Pipeline Completed Successfully!")
    print("Check the generated plots for analysis.")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
