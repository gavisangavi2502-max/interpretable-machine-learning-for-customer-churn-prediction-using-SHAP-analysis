#  Interpretable Churn Prediction: Model Report & Interpretability Analysis

This report presents a complete walkthrough of the churn prediction pipeline, from data preprocessing to model evaluation and SHAP-based interpretability. The goal is to build a transparent, high-performing model that not only predicts churn but explains why each prediction was made.

---

##  Data Preprocessing Summary

The raw dataset contained customer demographics, service usage, billing information, and churn labels. To prepare the data for modeling:

- **Missing Values**:  
  - `TotalCharges` was cleaned and converted to numeric.  
  - Categorical nulls were imputed using mode values.

- **Feature Engineering**:  
  - Binary features were mapped to `0/1`.  
  - Multi-class categorical features were one-hot encoded using `pd.get_dummies()`.  
  - Non-predictive identifiers like `customerID` were dropped.

- **Target Encoding**:  
  - `Churn` was encoded as `1` for churn and `0` for retention.

- **Final Shape**:  
  - After encoding, the dataset contained **X features** and **1 target column**.

---

##  Model Training & Evaluation

An XGBoost classifier was trained using 80/20 stratified split and tuned via `GridSearchCV`:

- **Best Parameters**:  
  - `n_estimators = 100`, `max_depth = 3`, `learning_rate = 0.1`

- **Performance Metrics**:  
  - **AUC Score**: `0.8246`  
  - **F1 Score**: `0.4875`  
  - These metrics reflect moderate predictive power, with room for improvement in recall.

---

##  SHAP Global Interpretation

SHAP summary plots revealed the most influential features across all predictions:

- **Top Positive Contributors to Churn**:
  - Low `tenure`
  - `Contract_One year` and `Contract_Two year = 0`
  - `OnlineSecurity_No`, `OnlineBackup_No`
  - `InternetService_Fiber optic`

- **Top Retention Signals**:
  - Long `tenure`
  - High `TotalCharges`
  - Two-year contracts

These insights align with business intuition: customers with longer tenure and bundled services are less likely to churn.

---

##  SHAP Local Interpretation: 5 Customer Profiles

Five customers were selected for force plot analysis to understand individual predictions:

- **Customer 1 (Index: 0)**  
  - Prediction: Churn  
  - SHAP Value: +1.37  
  - Drivers: Very low tenure, no security, fiber optic internet, no contract commitment  
  - Interpretation: Classic early-stage churn profile

- **Customer 2 (Index: 6)**  
  - Prediction: Churn  
  - SHAP Value: +2.64  
  - Drivers: Missing tech support, backup, short tenure, moderate charges  
  - Interpretation: Amplified churn risk due to multiple missing services

- **Customer 3 (Index: 4316)**  
  - Prediction: No Churn  
  - SHAP Value: +0.70  
  - Drivers: High monthly charges, fiber optic internet, senior citizen  
  - Interpretation: Mixed signals, but tenure and spending stabilize prediction

- **Customer 4 (Index: 4000)**  
  - Prediction: No Churn  
  - SHAP Value: -1.12  
  - Drivers: Long tenure, bundled services, traditional billing  
  - Interpretation: Strong retention profile

- **Customer 5 (Index: 4976)**  
  - Prediction: No Churn  
  - SHAP Value: -0.97  
  - Drivers: High total charges, two-year contract, security features  
  - Interpretation: Loyal and engaged customer

---

##  Counter-Intuitive Interaction Discovery

Using SHAP dependence plots, we uncovered unexpected feature interactions:

- **Tenure vs. MonthlyCharges**:  
  - Customers with low tenure and high monthly charges showed elevated churn risk, contradicting the assumption that high spenders are retained.

- **Fiber Optic Internet vs. Add-on Services**:  
  - Fiber optic users without security or backup were more likely to churn, suggesting that speed alone doesn’t guarantee retention.

- **SeniorCitizen vs. Contract Type**:  
  - Senior citizens on month-to-month contracts had higher churn risk than expected, indicating a need for targeted retention strategies.

---

##  Conclusion

This project demonstrates the power of interpretable machine learning in customer churn prediction. By combining XGBoost with SHAP, we not only achieved reasonable predictive performance but also gained actionable insights into why customers leave — and how to keep them.

>  *Next steps include improving recall, exploring ensemble models, and integrating SHAP explanations into customer outreach workflows.*
