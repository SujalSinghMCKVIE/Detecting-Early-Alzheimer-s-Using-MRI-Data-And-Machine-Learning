# Detecting Early Alzheimer's Using MRI Data and Machine Learning

## Overview
This project applies machine learning to MRI data for early Alzheimer's diagnosis. By leveraging multiple classification models, the system aims to detect subtle patterns in brain scans that indicate early-stage Alzheimer’s Disease (AD).

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction
Alzheimer’s disease is a progressive neurodegenerative disorder that affects memory and cognitive functions. Early detection allows for timely intervention and potential treatment planning. In this project, we use MRI scans combined with machine learning models to detect early-stage Alzheimer's.

## Dataset
- **Source**: MRI scans from a diverse cohort including healthy individuals, mild cognitive impairment (MCI) cases, and AD patients.
- **Preprocessing**: Includes standardization, skull stripping, alignment, and feature extraction (e.g., hippocampal volume, cortical thickness).

## Project Workflow
1. **Data Collection & Preprocessing** - Cleaning MRI images, normalizing, and extracting features.
2. **Exploratory Data Analysis (EDA)** - Understanding the dataset and feature distributions.
3. **Model Training** - Implementing various machine learning models.
4. **Evaluation** - Comparing performance using key metrics.
5. **Interpretability & Insights** - Utilizing SHAP values for model explainability.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/alzheimers-mri-detection.git
cd alzheimers-mri-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the main script to train and evaluate the model:
```bash
python alzheimers_detection.py
```

## Models Used
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest
- AdaBoost

## Evaluation Metrics
- **Accuracy** - Measures overall correctness.
- **Recall (Sensitivity)** - Measures the ability to correctly identify AD cases.
- **AUC (Area Under the ROC Curve)** - Measures discrimination power.
- **Confusion Matrix** - Visualizes false positives and false negatives.

## Results
- **Best Model:** Random Forest
- **Accuracy:** 87%
- **Sensitivity for early-stage AD:** 78%
- **Key Features:** Hippocampal volume, cortical thickness

## Future Improvements
- Integration of deep learning models (CNNs, LSTMs)
- Using multimodal imaging data (MRI + PET scans)
- Incorporating genetic and clinical test data

## License
This project is licensed under the MIT License. See `LICENSE` for details.
