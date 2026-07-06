Cardiovascular Disease Prediction Using Machine Learning

Comparison of three supervised machine learning algorithms — Logistic Regression, Random Forest, and Support Vector Machine (SVM) — applied to cardiovascular disease prediction, using dimensionality reduction (PCA), stratified cross-validation, and hyperparameter optimization with GridSearchCV.

Project Scope
 Dataset: Cardiovascular Disease Dataset (Kaggle, ~70,000 records)
 At least three supervised algorithms: Logistic Regression, Random Forest, and SVM
 Cross-validation + hyperparameter tuning: StratifiedKFold (k=5) + GridSearchCV
 Dimensionality reduction: PCA (automatic component selection to retain ≥95% explained variance)
 Model comparison: dedicated notebook consolidating the results of all three algorithms
 Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and ROC Curve
Repository Structure
├── 1 - Logistic Regression.ipynb     # Complete pipeline + Logistic Regression model
├── 2 - Random Forest.ipynb           # Complete pipeline + Random Forest model
├── 3 - SVM.ipynb                     # Complete pipeline + SVM model
├── 4 - Final Comparison.ipynb        # Consolidated comparison of the three models
└── README.md

Each of the three algorithm notebooks is self-contained and follows exactly the same data processing pipeline, ensuring that all models are evaluated under identical conditions.

Dataset
Source: Cardiovascular Disease Dataset (sulianova, Kaggle)
Size: ~70,000 records, 11 input features + 1 target variable (cardio)
Features: age, gender, height, weight, systolic/diastolic blood pressure, cholesterol, glucose, smoking status, alcohol consumption, and physical activity
Data Preprocessing Pipeline

The following preprocessing steps were applied identically across all three algorithm notebooks:

Converted age from days to years
Feature engineering:
Body Mass Index (BMI)
Pulse Pressure
Mean Arterial Pressure (MAP)
Systolic/Diastolic Ratio
Obesity Indicator
Composite Risk Score
Removed physiologically invalid outliers (e.g., systolic pressure lower than diastolic pressure, implausible height and weight values)
Standardized numerical features using StandardScaler
Applied dimensionality reduction with PCA (≥95% explained variance)
Performed a fixed train/test split (random_state=42), identical across all notebooks
Models and Optimization
Step	Method
Cross-validation	StratifiedKFold(n_splits=5)
Hyperparameter tuning	GridSearchCV
Algorithms	LogisticRegression, RandomForestClassifier, SVC

Note on the SVM model: Due to the computational cost of SVM in the Google Colab environment, GridSearchCV was performed on a stratified subsample of approximately 6,000 instances. The final model was then retrained on the full PCA-transformed training set, ensuring a fair comparison with the other algorithms.

Results
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Random Forest	0.7326	0.7502	0.6891	0.7183	0.8001
Logistic Regression	0.7281	0.7532	0.6699	0.7091	0.7918
SVM	0.7296	0.7750	0.6388	0.7004	0.7916

Metrics were computed on the same test set (13,723 samples) for all three models.

The Random Forest model achieved the best overall balance across the evaluation metrics, obtaining the highest Recall and ROC-AUC. The SVM achieved the highest Precision, at the cost of a lower Recall. Logistic Regression delivered competitive intermediate performance across nearly all metrics, suggesting that the decision boundary between patients with and without cardiovascular disease is largely close to linear for this dataset.

A complete comparison—including summary tables, bar charts, and a heatmap—is available in the notebook 4 - Final Comparison.ipynb.

How to Run
Open any notebook in Google Colab.
Run the cells sequentially. The dataset is downloaded automatically using kagglehub (or the CSV file can be uploaded manually).
To reproduce the final comparison, first execute the three algorithm notebooks and then run 4 - Final Comparison.ipynb.
Technologies
Python
scikit-learn (GridSearchCV, StratifiedKFold, PCA, StandardScaler)
pandas
NumPy
Matplotlib
Seaborn
kagglehub
Author

WesdleyR
