import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # for saving the model if desired
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# ----------------------------
# 1. Load the dataset
# ----------------------------
# Replace 'earthquake_data.csv' with the actual path to your dataset
data = pd.read_csv("/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv")

data = data.drop(columns=['title','date_time'])

string_cols = data.select_dtypes(include=['object']).columns.tolist()

# Ensure the target column is not one-hot encoded (assuming target is 'magnitude')
if 'magnitude' in string_cols:
    string_cols.remove('magnitude')

# One-hot encode the identified string columns
data = pd.get_dummies(data, columns=string_cols)
# ----------------------------
# 2. Data Preprocessing
# ----------------------------
# Example: Drop any rows with missing values (customize as needed)


# Assuming 'magnitude' is the target variable
# and all other columns are features.
# Adjust the list of feature columns as per your dataset.
X = data.drop(columns=['magnitude'])
y = data['magnitude']

# Optional: If you need to convert certain columns to numeric or handle categorical features,
# add your conversion or encoding steps here.

# ----------------------------
# 3. Split the Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# ----------------------------
# 4. Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 5. Train the Random Forest Model
# ----------------------------
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# ----------------------------
# 7. Perform Grid Search with Cross-Validation
# ----------------------------
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error',verbose=2)
grid_search.fit(X_train, y_train.values.ravel())  # Flatten y_train if needed

print("Best parameters:", grid_search.best_params_)
print("Best score (MSE):", -grid_search.best_score_)  # Negative MSE converted back to positive

# ----------------------------
# 8. Evaluate the Best Model
# ----------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

residuals = y_test - y_pred

# ----------------------------
# 9. Generate Informative Plots on Separate Pages
# ----------------------------

# Function to ensure each plot is displayed on a separate page
def show_plot():
    plt.show(block=True)  # Ensures each figure waits until closed before proceeding

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

# Plot 4: Improved Feature Importance Plot
if hasattr(best_model, 'feature_importances_'):
    # Get feature importances and names
    importances = best_model.feature_importances_
    features = X.columns
    
    # Sort importances and get indices of top 15 features
    indices = np.argsort(importances)[::-1]
    top_n = 15  # Adjust this number based on your needs
    
    # Create a larger figure with more height
    plt.figure(figsize=(10, 12))
    
    # Plot only the top N features
    plt.barh(range(top_n), importances[indices[:top_n]], align='center', color='skyblue')
    plt.yticks(range(top_n), [features[i] for i in indices[:top_n]])
    
    # Format the plot
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.xlabel('Relative Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()  # Ensures labels fit within the figure
    
    # Add a note about remaining features
    remaining_importance = sum(importances[indices[top_n:]])
    plt.figtext(0.5, 0.01, f"Sum of importance for {len(indices) - top_n} remaining features: {remaining_importance:.4f}", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    show_plot()
else:
    print("Feature importances not available.")

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
