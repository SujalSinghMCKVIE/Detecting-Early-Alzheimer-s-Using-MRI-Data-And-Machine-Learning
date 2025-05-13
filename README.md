
# 🧠 Detecting Early Alzheimer's Using MRI Data and Machine Learning

## 📝 Overview
This project combines neuroimaging data and machine learning to detect early-stage Alzheimer's Disease (AD) from MRI-derived metrics. A suite of classifiers is trained and evaluated to support early diagnosis by identifying structural brain changes and cognitive scores.

---

## 📊 Live Streamlit Demo

Experience the Alzheimer's detection app interactively:

👉 **[Click here to launch the app](https://alzheimer-detection-ml.streamlit.app/)**

This interactive web application enables users to enter patient data (e.g., MMSE, brain volume metrics) and instantly see predictions from five ML models. Featuring dark mode, SHAP explanations, and CSV exports — it's built for clinicians, researchers, and learners alike.

---

## 🧠 Dataset
- **Source**: The OASIS longitudinal MRI dataset.
- **Samples**: Healthy controls, MCI (mild cognitive impairment), and Alzheimer's patients.
- **Features**:
  - Demographic: Age, Gender, Education, SES
  - Cognitive: MMSE Score
  - MRI-derived: eTIV, nWBV, ASF

---

## 🔁 Project Workflow
1. **Data Collection & Cleaning**: Filtering, imputation, standardization.
2. **EDA**: Visualizing MMSE, brain volumes, age, and SES distributions.
3. **Feature Engineering**: Normalized brain volume ratios, gender encoding, and SES handling.
4. **Model Training**: Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost.
5. **Evaluation**: Accuracy, Recall, AUC, Confusion Matrices, SHAP.
6. **Deployment**: Hosted Streamlit app with live prediction and download capability.

---

## ⚙️ Technologies Used

- **Frontend**: Streamlit (with dark mode CSS)
- **Backend/ML**: 
  - Python (Pandas, NumPy, Scikit-learn)
  - Plotly, Seaborn, Matplotlib
  - Joblib, SHAP, Pickle

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/alzheimers-mri-detection.git
cd alzheimers-mri-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage

To run the backend training pipeline:
```bash
python alzheimers_detection.py
```

To launch the Streamlit app:
```bash
streamlit run app.py
```

Test sample Excel file is available from the app UI for demo inputs.

---

## 🧮 Models Used

| Model               | Type         | Strengths                        |
|--------------------|--------------|----------------------------------|
| Logistic Regression| Linear       | Simple, interpretable            |
| SVM                | Non-linear   | Good with margins & kernel trick |
| Decision Tree      | Tree-based   | Fast, explainable                |
| Random Forest      | Ensemble     | Robust, high accuracy            |
| AdaBoost           | Ensemble     | Adaptive learning, boosting weak learners |

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall correctness.
- **Recall (Sensitivity)**: AD detection rate.
- **AUC**: True vs. false positive trade-off.
- **F1 Score**: Balance between precision and recall.
- **Confusion Matrix**: True Positive / False Positive insights.

---

## 📊 Results Summary

| Model            | Accuracy | Recall | AUC   |
|------------------|----------|--------|-------|
| Logistic Regression | 80.5%   | 75%    | 0.75  |
| SVM                 | 81.5%   | 70%    | 0.82  |
| Decision Tree       | 81.5%   | 65%    | 0.825 |
| **Random Forest**   | **86.8%** | **80%**  | **0.872** |
| AdaBoost            | 86.8%   | 65%    | 0.825 |

✅ **Best Performer**: Random Forest – strong balance of accuracy and sensitivity.

---

## 🔮 Future Improvements

- Deep learning integration (e.g., CNNs on raw MRI)
- Multi-modal data: PET scans, cognitive assessments
- Genetic and clinical data inclusion
- Continuous learning via feedback loop

---

## 📄 Sample Input File

The application includes a downloadable Excel file (`alzheimers_test_samples.xlsx`) to test the prediction system with multiple cases in bulk.

---

## 🛡 License

This project is licensed under the **MIT License**. See `LICENSE` for more details.

---

## 🙏 Acknowledgments

Special thanks to the OASIS MRI dataset and contributors, Streamlit community, and the inspiration to apply machine learning for healthcare impact.

---

*Crafted with ❤️ by Sujal Singh*
