# AI-Generated Text Detection Using Transformers

A comprehensive project for detecting AI-generated text using transformer-based models. This repository includes data loading pipelines, feature engineering, and model training infrastructure for binary text classification.

## Features

- **Data Loading**: Efficient dataset management and preprocessing
- **Feature Engineering**: Advanced text feature extraction and vectorization
- **Transformer Models**: Pretrained transformer encoders for text classification
- **Modular Architecture**: Clean separation of concerns with organized module structure

## Project Structure

```
├── data_loader/
│   ├── dataset.py      - PyTorch dataset implementation
│   ├── loader.py       - Data loading utilities
│   └── utils.py        - Data processing helpers
├── feature_engineering/
│   ├── config.py       - Configuration management
│   ├── feature_engineering.py  - Feature extraction pipeline
│   ├── preprocessing.py        - Text preprocessing
│   ├── vectorizer.py           - Text vectorization
│   ├── utils.py                - Utility functions
│   └── feature_report.md       - Feature analysis documentation
└── README.md           - Project overview
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- NumPy
- Pandas (optional, for data analysis)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Dharsan5/AI-Generated-Text-Detection-Using-Transformers.git
cd AI-Generated-Text-Detection-Using-Transformers
pip install -r requirements.txt
```

## Usage

### Data Loading

```python
from data_loader.loader import load_data
from data_loader.dataset import TextDataset

# Load and prepare data
data = load_data('path/to/data.csv')
dataset = TextDataset(data)
```

### Feature Engineering

```python
from feature_engineering.feature_engineering import FeatureEngineer

# Extract features
engineer = FeatureEngineer()
features = engineer.transform(texts)
```

## Getting Started

1. Prepare your dataset with text samples and labels
2. Configure settings in `feature_engineering/config.py`
3. Run the data loading and feature engineering pipeline
4. Train and evaluate transformer models

## Project Goals

- Detect AI-generated text with high accuracy
- Provide interpretable feature analysis
- Enable easy model training and evaluation
- Support multiple transformer architectures

## Notes

Refer to `feature_engineering/feature_report.md` for detailed feature analysis and engineering decisions.
