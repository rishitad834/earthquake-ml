These four Python scripts implement different machine learning models for earthquake magnitude prediction using the same dataset (earthquake_1995-2023.csv). Each script follows a similar structure but employs a different algorithm:

1. The Random Forest Regressor script (earthquake_rf.py) uses ensemble learning with multiple decision trees to predict earthquake magnitudes, optimizing parameters like n_estimators, max_depth, and min_samples_split.

2. The Decision Tree Regressor script (eg_dt.py) implements a single decision tree model with hyperparameter tuning for max_depth, min_samples_split, min_samples_leaf, and max_features.

3. The LightGBM script (eg_lbgm.py) utilizes gradient boosting with the LightGBM library, including special handling for column names to ensure compatibility with the algorithm.

4. The Support Vector Machine script (eg_svm.py) applies SVR with different kernels (linear, rbf, poly) and regression parameters.

All scripts share common elements: they load and preprocess the earthquake data, perform one-hot encoding for categorical features, handle missing values, split data into training/testing sets, apply feature scaling, and conduct hyperparameter tuning via GridSearchCV. Each model's performance is evaluated using metrics like MSE, MAE, and R² score.

The scripts generate similar visualizations: actual vs. predicted plots, residual analysis, feature importance rankings (except for SVM), and binned confusion matrices to assess prediction accuracy across different magnitude ranges. Additionally, they include ROC curve analysis at various magnitude thresholds (4.0, 5.0, 6.0, 7.0) to evaluate each model's ability to classify earthquakes above specific magnitudes.

