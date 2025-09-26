#!/bin/bash
python -m src.data --output data/sample_churn.csv --n 5000
python -m src.model --input data/sample_churn.csv --output_dir artifacts
