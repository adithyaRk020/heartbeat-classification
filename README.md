# Heartbeat Classification

Deep learning pipeline for ECG heartbeat classification using the Kaggle MIT-BIH heartbeat dataset.

## Models

1. **Residual MLP** baseline with skip connections
2. **TCN + CNN + Attention** model inspired by MDPI Sensors 2024 architecture patterns

## Project Structure

```
├── data_loading.py
├── models.py
├── train.py
├── evaluate.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Dataset

Download from Kaggle:
- https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data

Place files in `data/`:
- `data/mitbih_train.csv`
- `data/mitbih_test.csv`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and Evaluate

```bash
python train.py --data-dir data --output-dir outputs --epochs 50 --batch-size 256
```

## Outputs

For each model, `train.py` saves:
- Training vs validation loss and accuracy curves
- Confusion matrix (`.npy` and `.png`)
- Classification report (`.txt`, `.json`, and `.png` heatmap)
- Best model checkpoint (`best_model.keras`)
- Training history (`history.json`)

Output folders:
- `outputs/residual_mlp/`
- `outputs/tcn_cnn_attention/`

## Reference

- MDPI Sensors 2024: https://www.mdpi.com/1424-8220/24/8/2484
