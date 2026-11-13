# Restaurant Sentiment Analysis - Experiment Index

## Dataset Preparation
- prepare_dataset.py: Prepares the train, validation, and test datasets.

## Model Experiments
- compare_models.py: Compares baseline models.
- improve_neutral_model.py: Tests neutral-class weighting options.
- refine_neutral_model.py: Refines neutral-class weighting.
- test_neutral_thresholds.py: Tests neutral probability thresholds using training and validation data.

## Final Evaluation
- evaluate_final_test.py: Evaluates the selected model on held-out test data.

## Frozen Final Test Results
- Accuracy: 82.49%
- Macro Precision: 70.72%
- Macro Recall: 73.51%
- Macro F1: 71.96%

## Main Limitation
Neutral sentiment is the most challenging class.

IMPORTANT: Do not use final test data for model tuning.
