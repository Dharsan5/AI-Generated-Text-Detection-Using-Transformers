# AI-Generated Text Detection Using Transformers

This repository contains a minimal transformer-based text classification example for AI-generated text detection.

The current implementation in `sample.py`:
- loads a pretrained Hugging Face transformer encoder
- defines a PyTorch dataset for tokenized text inputs
- adds a classification head for binary prediction
- runs inference on sample sentences and prints class probabilities

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- NumPy

## Install

```bash
pip install torch transformers numpy
```

If you are using a GPU-enabled PyTorch build, install the version recommended by the official PyTorch instructions for your platform.

## Run

```bash
python sample.py
```

## What It Does

`sample.py` creates a `TransformerClassifier` using `distilbert-base-uncased` by default, tokenizes input text, and produces probability scores for two classes.

## Project Structure

- `sample.py` - minimal example model and inference script
- `README.md` - project overview and usage

## Notes

This is a starter example, not a trained detector. To make it useful for AI-generated text detection, you would need to train the classifier on a labeled dataset and evaluate it on held-out data.
