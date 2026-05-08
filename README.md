# Lung Cancer Detection using Vision Transformer and Radiomics

## Overview

This project implements a two-stage lung cancer detection system using:

- **Vision Transformer (ViT)**
- **Radiomics feature extraction**
- **Multimodal feature fusion**

The system classifies lung CT scan images into:

- **Normal**
- **Benign**
- **Malignant**

The project also includes:

- **Explainable AI (XAI)**
- **Attention map analysis**
- **Feature caching for faster processing**
- **Dataset**

## Dataset used in this project:

**Lung Cancer 14K Dataset (Kaggle)**

## Project Architecture

### Two-Stage Classification Pipeline

#### Stage 1
- **Normal vs Abnormal classification**

#### Stage 2
- **Benign vs Malignant classification**

## Technologies Used

- **Python**
- **PyTorch**
- **Vision Transformer (ViT)**
- **OpenCV**
- **NumPy**
- **Scikit-learn**
- **Radiomics**
- **Matplotlib**

## Features

- **Vision Transformer based feature extraction**
- **Radiomics feature extraction**
- **Multimodal fusion**
- **Explainable AI (XAI)**
- **Attention map visualization**
- **Feature caching**
- **Two-stage classification pipeline**

## Performance

| Metric | Value |
|----------|--------|
| Overall Accuracy | 93.78% |
| Stage 1 Accuracy | 95.22% |
| Stage 2 Accuracy | 97.04% |
| Malignant Recall | 99.65% |

## Folder Structure

```
project/
│
├── data/
├── models/
├── outputs/
├── utils/
├── cancer_detection_test.py
├── run_two_stage_pipeline.py
└── README.md
```

## Installation

### Clone the repository:

```bash
git clone <repository-link>
cd project-folder
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

### Train the model

```bash
python run_two_stage_pipeline.py
```

### Test the model

```bash
python cancer_detection_test.py
```
