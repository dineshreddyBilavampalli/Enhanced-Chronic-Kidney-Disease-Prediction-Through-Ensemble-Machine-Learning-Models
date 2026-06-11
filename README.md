# 🩺 Enhanced Chronic Kidney Disease Prediction Through Ensemble Machine Learning Models

> A mini project submitted to **SASTRA Deemed to be University** for the course **CSE300 / INT300 / ICT300**  
> Academic Year: 2024–25 | School of Computing, Thanjavur

---

## 👥 Team Members

| Name | Reg. No. | Branch |
|------|----------|--------|
| Kumar Sambhavam | 126156074 | B.Tech CSE (AI & DS) |
| Bilavampalli Dinesh Reddy | 126014011 | B.Tech ICT |


---

## 📌 Project Overview

Chronic Kidney Disease (CKD) is a serious global health challenge that often goes undetected until advanced stages. This project builds and compares multiple **Machine Learning models** to predict CKD early using clinical data from the UCI Machine Learning Repository.

The key novelty of this project is the use of **Ensemble Learning techniques** (Bagging, Boosting, and Stacking) on top of traditional classifiers, achieving up to **100% accuracy** with the Stacking ensemble.

- **Dataset:** [UCI CKD Dataset](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease) — 400 instances, 24 features
- **Task:** Binary Classification (CKD / Not CKD)
- **Best Result:** Stacking Ensemble → **100% accuracy**

---

## 📁 Repository Structure

```
├── final_code_research.ipynb       # Main Jupyter Notebook with all code
├── kidney_disease.csv              # Dataset (UCI CKD Dataset)
└── README.md                       # Project documentation
```

---

## 🔬 Methodology

The project follows a structured ML pipeline:

```
Raw Dataset
    ↓
Data Preprocessing (Missing value handling, Encoding)
    ↓
Feature Selection (CFS / LASSO / Wrapper)
    ↓
Model Training (ANN, DT, KNN, SVM)
    ↓
Ensemble Techniques (Bagging, Boosting, Stacking)
    ↓
Model Evaluation & Comparison
```

---

## 📊 What's Inside the Notebook

### 1. 📦 Data Loading & Preprocessing
- Loaded `kidney_disease.csv` using pandas
- Filled missing values: **mean** for numerical columns, **mode** for categorical columns
- Applied **Label Encoding** to convert categorical features to numeric
- Visualized feature correlations using a **Heatmap** and class distribution using a **Count Plot**

---

### 2. 🔍 Feature Selection Techniques

Three different feature selection methods were applied and compared:

#### A. Correlation-Based Feature Selection (CFS)
- Dropped features with pairwise correlation above a threshold of **0.5**
- Selected features: `age, bp, sg, al, su, rbc, ba, bu, pot, wc, dm, cad, appet, pe`

#### B. LASSO Feature Selection
- Used `Lasso(alpha=0.1)` regression to identify features with non-zero coefficients
- Reduced the feature set to **8 most relevant features**

#### C. Wrapper Method (Forward Sequential Feature Selection)
- Used `SequentialFeatureSelector` with an MLP classifier
- Iteratively added features that improved model performance

---

### 3. 🤖 Classification Models

The following classifiers were trained and evaluated under each feature selection method:

| Model | Details |
|-------|---------|
| **ANN** (Artificial Neural Network) | 3-layer Sequential model: Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid) |
| **Decision Tree** | `DecisionTreeClassifier(random_state=42)` |
| **KNN** | `KNeighborsClassifier`, K tuned from 1–20; K vs Accuracy curve plotted |
| **SVM** | Tested with `linear`, `poly`, `rbf`, `sigmoid` kernels; also `LinearSVC` with L1 penalty |

**Evaluation plots generated for each model:**
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve (for KNN)
- K vs Accuracy Curve (for KNN)
- Loss & Accuracy Curves (for ANN)
- Decision Boundary Plot (for SVM)

---

### 4. ⚖️ Class Imbalance Handling with SMOTE

- The dataset had more CKD-positive cases than negative
- **SMOTE** (Synthetic Minority Oversampling Technique) was used to balance the classes
- Before/After class distribution visualized (Fig 3.6.1)
- All classifiers (DT, KNN, ANN, SVM) were re-evaluated after SMOTE

---

### 5. 🎯 Ensemble Learning (Core Novelty)

#### Bagging
- **Random Forest:** `RandomForestClassifier(n_estimators=30)` → **99% accuracy**
- **Bagging KNN:** `BaggingClassifier` with `KNeighborsClassifier(n_neighbors=3)` → **97% accuracy**
- Learning curves plotted for both

#### Stacking
- Base learners: **SVC + Random Forest + Decision Tree**
- Meta-model: **Logistic Regression** (5-fold cross-validation)
- Achieved → **100% accuracy**
- Base Models vs Stacking accuracy bar chart plotted

#### Boosting
- **Gradient Boosting** → **98.75% accuracy**
- **XGBoost** → **99% accuracy** + Feature Importance plot generated

---

## 📈 Results Summary

### Table 1.1 — LASSO Feature Selection

| Model | Base Paper Accuracy (%) | Our Implementation Accuracy (%) |
|---|---|---|
| Decision Tree | 88.29 | 95.00 |
| ANN | 90.24 | 95.00 |
| KNN (k=1) | 56.59 | 96.25 |
| SVM with Linear Kernel | — | 96.25 |
| SVM with L1 (Lambda=0.5) | 97.05 | **97.50** |

---

### Table 1.2 — Wrapper Feature Selection

| Model | Base Paper Accuracy (%) | Our Implementation Accuracy (%) |
|---|---|---|
| Decision Tree | 96.10 | **98.75** |
| ANN | 94.63 | 97.00 |
| KNN (k=3) | 76.10 | **98.75** |
| SVM with Linear Kernel | 93.66 | 97.50 |
| SVM with L1 | 94.15 | 95.00 |

---

### Table 1.3 — Correlation-Based Feature Selection (CFS)

| Model | Base Paper Accuracy (%) | Our Implementation Accuracy (%) |
|---|---|---|
| Decision Tree | 92.68 | 95.00 |
| ANN | 91.71 | 94.00 |
| KNN (k=3) | 53.17 | 96.25 |
| SVM with Linear Kernel | 93.66 | 93.75 |
| SVM with L1 | 95.12 | 93.75 |

---

### Table 1.4 — Bagging (Ensemble)

| Model | Implementation Accuracy (%) |
|---|---|
| Random Forest | 97.50 |
| Decision Tree | **98.75** |
| KNN (k=3) | 97.50 |
| SVM with Linear Kernel | **98.75** |

---

### Table 1.5 — Stacking (Ensemble)

| Base Model | Base Model Accuracy (%) | Meta Model Accuracy — Logistic Regression (%) |
|---|---|---|
| Decision Tree | 95.80 | |
| Random Forest | 98.50 | **100** |
| SVC | 99.50 | |

---

### Table 1.6 — Boosting (Ensemble)

| Model | Implementation Accuracy (%) |
|---|---|
| Gradient Boosting | 98.75 |
| XG Boosting | 98.75 |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Notebook:** Jupyter Notebook
- **Libraries:**

```
pandas, numpy, matplotlib, seaborn
scikit-learn
tensorflow / keras
imbalanced-learn (SMOTE)
xgboost
```

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn xgboost
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook final_code_research.ipynb
   ```

4. Make sure `kidney_disease.csv` is in the same directory as the notebook, then run all cells top to bottom.

---

## 📚 Key References

- Pankaj Chittora et al. (2021), *"Prediction of Chronic Kidney Disease — A Machine Learning Perspective"*, IEEE Access — [Link](https://ieeexplore.ieee.org/document/9333572)
- Kommuri Venkatrao et al. (2023), *"HDLNET: A Hybrid Deep Learning Network Model With Intelligent IoT for CKD"*, IEEE Access
- Qiong Bai et al. (2022), *"Machine learning to predict end-stage kidney disease"*, Scientific Reports
- UCI Machine Learning Repository — CKD Dataset

---

## 🔮 Future Scope

- Test on larger, more recent real-world CKD datasets
- Deploy as a web/mobile app for healthcare professionals
- Explore deep learning architectures (CNN, LSTM) for further improvement
- Integrate explainability tools (SHAP, LIME) for model interpretability

---

## ❓ Frequently Asked Questions

**Q1. What is Chronic Kidney Disease (CKD) and why does early prediction matter?**  
CKD is a condition where the kidneys gradually lose their ability to filter waste from the blood. It often shows no symptoms in early stages, making early detection critical. ML-based prediction helps identify at-risk patients before the disease progresses to kidney failure.

---

**Q2. Where does the dataset come from and how large is it?**  
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease). It contains **400 patient records** with **24 clinical features** such as blood pressure, albumin, sugar, red blood cells, hemoglobin, serum creatinine, and more.

---

**Q3. Why are there three different feature selection methods used?**  
Different feature selection methods can capture different aspects of the data. CFS removes highly correlated redundant features, LASSO penalizes less important features to zero, and the Wrapper method iteratively selects features that actually improve model performance. Comparing all three helps identify which approach gives the best predictive accuracy.

---

**Q4. Why was SMOTE applied? Is the dataset imbalanced?**  
Yes, the original dataset has more CKD-positive cases (~250) than CKD-negative cases (~150). Without handling this imbalance, models tend to be biased toward the majority class. SMOTE synthetically generates new minority-class samples to balance the distribution, leading to fairer and more reliable predictions.

---

**Q5. Which model gave the best accuracy and why?**  
The **Stacking Ensemble** (base models: SVC + Random Forest + Decision Tree; meta-model: Logistic Regression) achieved **100% accuracy** on the test set. Stacking works well here because it combines the strengths of multiple diverse classifiers, with the meta-model learning how to best combine their predictions.

---

**Q6. Is 100% accuracy realistic or a sign of overfitting?**  
On a small dataset of 400 records, very high accuracy is achievable but should be interpreted carefully. The model was tested on an 80/20 train-test split (80 test samples). While the results are strong, testing on a larger and more diverse real-world dataset would be needed to fully validate generalization. This is acknowledged in the Future Scope section.

---

**Q7. What is the difference between Bagging, Boosting, and Stacking?**  
- **Bagging** trains multiple models independently on random subsets of data and averages their predictions (e.g., Random Forest).  
- **Boosting** trains models sequentially, where each model tries to correct the errors of the previous one (e.g., XGBoost, Gradient Boosting).  
- **Stacking** trains multiple different base models and feeds their predictions as input to a meta-model that makes the final decision.

---

**Q8. Can I use this project on a different medical dataset?**  
Yes! The pipeline — preprocessing, feature selection, model training, and ensemble methods — is generic enough to be adapted for other binary classification problems in healthcare. You would need to update the feature names, preprocessing steps, and tune hyperparameters accordingly.

---

**Q9. What Python version and environment is recommended to run this?**  
Python **3.8 or above** is recommended. You can run it in a standard Jupyter Notebook environment or Google Colab. All required libraries are installable via `pip` as described in the How to Run section.

---

**Q10. Can this model be deployed as a real application?**  
Not directly in its current form — it is a research/academic prototype. For deployment, the trained model would need to be serialized (e.g., using `joblib` or `pickle`), wrapped in a backend API (e.g., Flask or FastAPI), and connected to a user-friendly front-end. This is one of the planned future enhancements.

---

## 📄 License

This project was developed as an academic mini project at SASTRA Deemed to be University. Feel free to use it for learning and educational purposes.
