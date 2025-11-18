# Project Narrative Summary: Interpretable Customer Churn Prediction

This project presents a full-cycle machine learning pipeline to predict customer churn using the Telco dataset. Beyond predictive accuracy, the core emphasis lies in **model interpretability**—leveraging SHAP (SHapley Additive exPlanations) to understand not just what the model predicts, but why.

---

##  Business Context

Customer churn poses a significant financial risk for subscription-based services. Identifying at-risk customers early enables targeted retention strategies. However, black-box models often lack transparency, making it difficult to justify decisions to stakeholders.

This project bridges that gap by combining:
- A tuned XGBoost classifier for robust prediction
- SHAP for global and local interpretability
- Narrative reporting to translate model insights into business actions

---

##  Data Preparation

The raw dataset included 7,000+ customer records with 21 features spanning demographics, service usage, billing, and contract details.

Key preprocessing steps:
- Cleaned missing values in `TotalCharges` and categorical fields
- Encoded binary and multi-class categorical features
- Dropped non-predictive identifiers (`customerID`)
- Final dataset prepared with one-hot encoded features and binary target (`Churn`)

---

##  Model Development

An XGBoost classifier was selected for its balance of performance and interpretability. Hyperparameters were tuned using `GridSearchCV` with 3-fold cross-validation.

**Best Parameters**:
- `n_estimators = 100`
- `max_depth = 3`
- `learning_rate = 0.1`

**Final Metrics**:
- **AUC Score**: `0.8246` → strong ability to distinguish churners
- **F1 Score**: `0.4875` → moderate precision-recall balance

---

##  Global SHAP Interpretation

SHAP summary plots revealed key drivers of churn across the dataset:

- **Top churn indicators**:
  - Low `tenure`
  - Missing services: `OnlineSecurity_No`, `OnlineBackup_No`
  - Month-to-month contracts
  - `InternetService_Fiber optic` (context-dependent)

- **Top retention signals**:
  - Long `tenure`
  - High `TotalCharges`
  - Two-year contracts

These insights align with business intuition: customers with longer tenure and bundled services are less likely to churn.

---

##  Local SHAP Interpretation: 5 Customer Profiles

Five individual customers were analyzed using SHAP force plots:

- **Churn Cases**:
  - Short tenure, missing security/backup, no long-term contracts
  - High SHAP values (+1.37, +2.64) indicate strong churn confidence

- **No Churn Cases**:
  - Long tenure, high total charges, bundled services
  - Negative SHAP values (-0.85 to -1.12) reflect retention stability

Each force plot visualizes how specific features push the prediction toward or away from churn.

---

##  Counter-Intuitive Interactions 

SHAP dependence plots uncovered three surprising patterns:

1. **Tenure × MonthlyCharges**  
   - New customers with high bills churn faster than expected  
   - Contradicts assumption that high spenders are loyal

2. **Fiber optic × No Security**  
   - Fiber users without bundled services show elevated churn risk  
   - Challenges belief that tech-savvy users need fewer add-ons

3. **SeniorCitizen × Contract Type**  
   - Month-to-month contracts increase churn risk for older customers  
   - Unexpected, as seniors are often seen as loyal

These findings highlight the value of SHAP in surfacing nuanced, non-obvious relationships.

---

##  Conclusion

This project demonstrates how interpretable machine learning can support churn mitigation by:

- Predicting churn with reasonable accuracy
- Explaining predictions at both global and individual levels
- Revealing unexpected behavioral patterns through SHAP

> 🧠 *Future work includes improving recall, testing ensemble models, and integrating SHAP insights into customer retention workflows.*



