"""
Main entry point for ML Pathogenicity Classification pipeline.

Runs:
1. Data ingestion
2. Data processing
3. Modeling
4. Visualization outputs
"""

from src.data_ingestion import load_data
from src.data_processing import preprocess_data
from src.modeling import run_models
from src.visualization import generate_visuals


def main():
    print("Loading data...")
    raw_data = load_data()

    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    print("Running models...")
    model_results = run_models(processed_data)

    print("Generating visualizations...")
    generate_visuals(model_results)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
