# Stars vs Galaxies Classification Project

## Overview

This project implements a deep learning pipeline to classify astronomical images into two categories: **Stars** and **Galaxies**.

**Key goals:**
- Reliable training pipeline
- Reproducible experiments
- Deployable model serving
- End‑to‑end CNN classification (PyTorch Lightning)

## Key Features

- **CNN classifier** in PyTorch Lightning
- **LightningDataModule** for data splits (train/val/test)
- **MLflow** logging (loss, accuracy, F1, ROC‑AUC)
- **Artifact storage** (plots, checkpoints, ONNX)
- **Inference script** for new datasets

## Setup

### Prerequisites

- Python 3.10
- Git
- Virtual environment (`venv`, `pyenv`, or `conda`)
- Optional: GPU with CUDA (for accelerated training)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/makemebright/stars-vs-galaxies.git
   cd stars-vs-galaxie
   ```
git clone https://github.com/makemebright/stars-vs-galaxies.git
cd stars-vs-galaxies
Create a virtual environment (using venv):

python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
Install Python dependencies:
   ```bash
pip install --upgrade pip
pip install -r requirements.txt
   ```
Optional: start Docker (if using containerized environment):

Prepare data (if not already downloaded):

The dataset directory structure should look like this:
   ```bash
data/
├─ raw/
│  ├─ train/
│  │  ├─ GALAXY/
│  │  └─ STAR/
│  ├─ validation/
│  │  ├─ GALAXY/
│  │  └─ STAR/
│  └─ test/
│     ├─ GALAXY/
│     └─ STAR/
Train
   ```
To train the CNN classifier, run the training pipeline using Hydra configuration.

Steps

Start the FastAPI server (required for some commands):
   ```bash
python -m stars_galaxies.server
   ```
Run training:
   ```bash
python -m stars_galaxies.commands train
   ```
This will:

Load and preprocess the dataset.

Initialize the CNN model.

Train the model with PyTorch Lightning.

Log metrics and artifacts to MLflow (plots/ and checkpoints).

Notes

Configurations are located in stars_galaxies/configs/.

Training parameters like lr, batch_size, max_epochs are controlled via Hydra.

Production Preparation

After training, prepare your model for deployment.

Steps

Save checkpoint: automatically saved in stars_galaxies/checkpoints/galaxy_star_model.ckpt.

Export ONNX model: automatically saved in stars_galaxies/checkpoints/galaxy_star_model.onnx.

Include necessary modules for inference:
   ```bash
stars_galaxies/infer/
stars_galaxies/models/model.py
stars_galaxies/data/datamodule.py
   ```
Optional: convert ONNX to TensorRT or other optimized formats for production.

Infer

Once the model is trained, you can run inference on new images.

Example Usage
   ```bash
python -m stars_galaxies.commands infer --data_dir /path/to/new/imagesInput Format
   ```
Images should be structured in subfolders (similar to training data), e.g.:
   ```bash
new_data/
├─ GALAXY/
├─ STAR/
   ```
The infer script will read images, preprocess them (resize/crop), and produce predictions.

Output

Predictions are returned as a CSV or logged directly via MLflow.

Optionally, you can visualize results using the provided plotting utilities.

Notes for Developers

All code is formatted with black and isort.

Linting is performed with flake8.

MLflow is used for tracking experiments.

Hydra manages configuration, allowing easy overrides via CLI.

To add new datasets or models, extend GalaxyStarDataModule and GalaxyStarClassifier.
