# Ensemble Models: Random Forest vs Gradient Boosting

This project compares two powerful ensemble techniques — **Random Forest** and **Gradient Boosting** — for binary classification using a credit risk dataset. The focus is to demonstrate how **Bagging (Random Forest)** and **Boosting (Gradient Boosting)** differ in performance, model behavior.

---

## Objective

- To compare Random Forest and Gradient Boosting models on a real-world classification task.
- Analyze the difference in accuracy, precision-recall trade-offs, and feature importance.
- Visualize the impact of each model using test accuracy and feature importance plots.

---

## Dataset

- **Dataset:** Credit Risk Dataset  
- **Target Variable:** `Loan_Status`  
- **Features:** Gender, Married, Dependents, Education, ApplicantIncome, etc.  
- **Shape:** 614 samples × 13 features

---

## Preprocessing Steps

1. **Categorical Encoding:** Label Encoding for object columns.
2. **Missing Values:** Imputed with `SimpleImputer(strategy='mean')`.
3. **Scaling:** StandardScaler applied post-imputation.
4. **Train/Test Split:** 80% train, 20% test.

---

## Models Compared

| Model            | Type     | Ensemble Strategy |
|------------------|----------|-------------------|
| Random Forest    | Bagging  | Parallel Trees    |
| Gradient Boosting| Boosting | Sequential Trees  |

---

## Performance Metrics

| Metric        | Random Forest | Gradient Boosting |
|---------------|---------------|-------------------|
| Accuracy      | ✅ 75.6%       | ✅ 73.1%           |
| Precision     | 0.76 (class 1) | 0.74 (class 1)    |
| Recall        | 0.93 (class 1) | 0.91 (class 1)    |
| F1-score      | 0.83 (class 1) | 0.80 (class 1)    |

---

## Outputs

### Test Accuracy Comparison
Random Forest vs Gradient Boosting

![E4](https://github.com/user-attachments/assets/76179e48-55bd-40ac-8386-b6908e6eec67)

---

## Conclusion

- **Random Forest** performs slightly better in overall accuracy and recall.
- **Gradient Boosting** can be more sensitive to class imbalances but is tunable.
- Feature importance helps interpret model focus — both agree on key contributors like `Credit_History` and `ApplicantIncome`.

---

## Skills Applied

- Ensemble Learning
- Tree-Based Models
- Model Evaluation Metrics
- Data Preprocessing (Encoding, Imputation, Scaling)
- Visualization using Matplotlib & Seaborn

---

## How to Run

```bash
# Clone repository
git clone https://github.com/yourusername/Ensemble_Models_RandomForest_vs_GradientBoosting

# Open and run the Jupyter Notebook
