import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgbm
import re

# ----------------------------
# 1. Load the dataset
# ----------------------------
# Replace with the actual path to your dataset
data = pd.read_csv("/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv")

data = data.drop(columns=['title', 'date_time'])

string_cols = data.select_dtypes(include=['object']).columns.tolist()

# Ensure the target column is not one-hot encoded (assuming target is 'magnitude')
if 'magnitude' in string_cols:
    string_cols.remove('magnitude')

# One-hot encode the identified string columns
data = pd.get_dummies(data, columns=string_cols)

# ----------------------------
# 2. Clean Column Names - FIX FOR THE ERROR
# ----------------------------
# Clean column names to remove special characters that LightGBM can't handle
def clean_column_name(name):
    # Replace special characters with underscores
    return re.sub(r'[^\w]+', '_', name)

# Apply cleaning to all column names
data.columns = [clean_column_name(col) for col in data.columns]

# ----------------------------
# 3. Data Preprocessing
# ----------------------------
# Assuming 'magnitude' is the target variable
# and all other columns are features.
X = data.drop(columns=['magnitude'])
y = data['magnitude']

# ----------------------------
# 4. Split the Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# ----------------------------
# 5. Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrames with cleaned column names for LightGBM
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ----------------------------
# 6. Train the LightGBM Model
# ----------------------------
# Start with a simplified parameter grid to ensure it works
lgbm_model = lgbm.LGBMRegressor(
    random_state=42,
    verbose=-1  # Reduce verbosity
)

# Simplified parameter grid for initial testing
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.05, 0.1]
}

# ----------------------------
# 7. Perform Grid Search with Cross-Validation
# ----------------------------
# Use try-except to catch and display any errors more clearly
try:
    grid_search = GridSearchCV(
        lgbm_model, 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        verbose=2,
        error_score='raise'  # To get detailed error information
    )
    
    # Use the pandas DataFrame with proper column names
    grid_search.fit(X_train_df, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score (MSE):", -grid_search.best_score_)
    
    # ----------------------------
    # 8. Evaluate the Best Model
    # ----------------------------
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score: {r2}")
    
    residuals = y_test - y_pred
    
    # ----------------------------
    # 9. Generate Informative Plots
    # ----------------------------
    
    # Function to ensure each plot is displayed on a separate page
    def show_plot():
        plt.show(block=True)
    
    # Plot 1: Actual vs. Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Actual vs. Predicted Earthquake Magnitudes')
    show_plot()
    
    # Plot 2: Residual Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='g')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Magnitude')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals vs. Predicted Values')
    show_plot()
    
    # Plot 3: Histogram of Residuals
    plt.figure(figsize=(7, 6))
    plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    show_plot()
    
    # Plot 4: Feature Importance
    plt.figure(figsize=(10, 8))
    importances = best_model.feature_importances_
    features = X_train_df.columns
    indices = np.argsort(importances)[::-1]
    
    # Display only the top 20 features if there are many
    n_features_to_show = min(20, len(indices))
    plt.barh(range(n_features_to_show), importances[indices[:n_features_to_show]], align='center', color='skyblue')
    plt.yticks(range(n_features_to_show), features[indices[:n_features_to_show]])
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.xlabel('Relative Importance')
    plt.title('Top Feature Importances (LightGBM)')
    show_plot()
    
    # Define magnitude bins
    bins = np.arange(min(y_test), max(y_test), 0.5)  # Adjust bin size as needed
    y_test_binned = np.digitize(y_test, bins)
    y_pred_binned = np.digitize(y_pred, bins)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned)
    
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=bins, yticklabels=bins)
    plt.xlabel('Predicted Magnitude Bin')
    plt.ylabel('Actual Magnitude Bin')
    plt.title('Binned Confusion Matrix')
    plt.show()

except Exception as e:
    print(f"Error during model training: {e}")
    
    # Print a sample of feature names to help diagnose issues
    print("\nSample of feature names (first 10):")
    print(list(X_train_df.columns)[:10])