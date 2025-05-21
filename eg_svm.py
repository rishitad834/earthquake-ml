import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ----------------------------
# 1. Load the dataset
# ----------------------------
data = pd.read_csv("/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv")

data = data.drop(columns=['title', 'date_time'])

string_cols = data.select_dtypes(include=['object']).columns.tolist()
if 'magnitude' in string_cols:
    string_cols.remove('magnitude')

data = pd.get_dummies(data, columns=string_cols)

# ----------------------------
# 2. Data Preprocessing
# ----------------------------
X = data.drop(columns=['magnitude'])
y = data['magnitude']

# ----------------------------
# 3. Split the Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
# 5. Train the SVM Model
# ----------------------------
svr_model = SVR()
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# ----------------------------
# 6. Perform Grid Search with Cross-Validation
# ----------------------------
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train_scaled, y_train.values.ravel())

print("Best parameters:", grid_search.best_params_)
print("Best score (MSE):", -grid_search.best_score_)

# ----------------------------
# 7. Evaluate the Best Model
# ----------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

residuals = y_test - y_pred

# ----------------------------
# 8. Generate Informative Plots as Individual Figures
# ----------------------------

# Plot 1: Actual vs. Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs. Predicted Earthquake Magnitudes')
plt.tight_layout()
plt.show()

# Plot 2: Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='g')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Magnitude')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs. Predicted Values')
plt.tight_layout()
plt.show()

# Plot 3: Histogram of Residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Create a Binned Confusion Matrix
# ----------------------------

# Define magnitude bins
bins = [0, 2, 4, 6, 8, 10]  # Adjust as needed
bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10']

# Discretize actual and predicted values
y_test_binned = pd.cut(y_test, bins=bins, labels=bin_labels)
y_pred_binned = pd.cut(y_pred, bins=bins, labels=bin_labels)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_binned, y_pred_binned, labels=bin_labels)

# Plot confusion matrix as a separate figure
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=bin_labels, yticklabels=bin_labels)
plt.xlabel("Predicted Magnitude Range")
plt.ylabel("Actual Magnitude Range")
plt.title("Confusion Matrix for Binned Magnitudes")
plt.tight_layout()
plt.show()

# ----------------------------
# 10. Add AUC-ROC Analysis
# ----------------------------

# Function to create binary labels for ROC analysis
def create_binary_labels(y_true, y_pred, thresholds):
    """
    Create binary classification scenarios at different magnitude thresholds
    """
    results = {}
    for threshold in thresholds:
        # Create binary labels: 1 if magnitude >= threshold, 0 otherwise
        y_true_binary = (y_true >= threshold).astype(int)
        # Use predicted values as scores
        y_scores = y_pred
        
        results[threshold] = {
            'y_true_binary': y_true_binary,
            'y_scores': y_scores
        }
    return results

# Define magnitude thresholds of interest (e.g., major earthquake thresholds)
magnitude_thresholds = [4.0, 5.0, 6.0, 7.0]

# Create binary classification scenarios
binary_scenarios = create_binary_labels(y_test, y_pred, magnitude_thresholds)

# Plot ROC curves for each threshold
plt.figure(figsize=(10, 8))

for threshold in magnitude_thresholds:
    scenario = binary_scenarios[threshold]
    y_true_binary = scenario['y_true_binary']
    y_scores = scenario['y_scores']
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, 
             label=f'Magnitude ≥ {threshold} (AUC = {roc_auc:.3f})')

# Add diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Format plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Magnitude Thresholds')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate and print AUC scores for each threshold
print("\nAUC-ROC Scores for Different Magnitude Thresholds:")
for threshold in magnitude_thresholds:
    scenario = binary_scenarios[threshold]
    y_true_binary = scenario['y_true_binary']
    y_scores = scenario['y_scores']
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_true_binary, y_scores)
    
    # Calculate percentage of earthquakes above threshold
    percent_above = (y_true_binary.sum() / len(y_true_binary)) * 100
    
    print(f"Magnitude ≥ {threshold}: AUC = {auc_score:.3f} ({percent_above:.1f}% of data above threshold)")
