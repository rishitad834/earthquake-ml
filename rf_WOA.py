import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class WhaleOptimizationAlgorithm:
    def __init__(self, n_whales=20, max_iter=100, dim=None):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.dim = dim
        self.best_whale = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
    def initialize_population(self):
        """Initialize whale population with random binary solutions"""
        population = np.random.randint(0, 2, (self.n_whales, self.dim))
        # Ensure at least one feature is selected for each whale
        for i in range(self.n_whales):
            if np.sum(population[i]) == 0:
                population[i][np.random.randint(0, self.dim)] = 1
        return population
    
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        """Fitness function based on model performance with selected features"""
        selected_features = np.where(solution == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            # Select features
            X_train_selected = X_train.iloc[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features]
            
            # Train model
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_selected, y_train)
            
            # Predict and calculate fitness (MSE + penalty for too many features)
            y_pred = rf.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            
            # Add penalty for using too many features (feature selection pressure)
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except:
            return float('inf')
    
    def optimize(self, X_train, X_test, y_train, y_test):
        """Main WOA optimization loop"""
        self.dim = X_train.shape[1]
        
        # Initialize population
        population = self.initialize_population()
        fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test) 
                                 for whale in population])
        
        # Find best whale
        best_idx = np.argmin(fitness_values)
        self.best_whale = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.convergence_curve.append(self.best_fitness)
        
        print(f"Initial best fitness: {self.best_fitness:.4f}")
        print(f"Initial features selected: {np.sum(self.best_whale)}/{self.dim}")
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            a = 2 - 2 * iteration / self.max_iter  # Decreases from 2 to 0
            
            for i in range(self.n_whales):
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                
                if np.random.random() < 0.5:
                    if abs(A) < 1:
                        # Shrinking encircling mechanism
                        D = abs(C * self.best_whale - population[i])
                        population[i] = self.best_whale - A * D
                    else:
                        # Search for prey (exploration)
                        random_whale = population[np.random.randint(0, self.n_whales)]
                        D = abs(C * random_whale - population[i])
                        population[i] = random_whale - A * D
                else:
                    # Spiral updating position (bubble-net attacking)
                    distance = abs(self.best_whale - population[i])
                    b = 1  # defines shape of logarithmic spiral
                    l = np.random.uniform(-1, 1)  # random number in [-1,1]
                    population[i] = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_whale
                
                # Convert to binary using sigmoid function
                population[i] = np.where(1 / (1 + np.exp(-population[i])) > 0.5, 1, 0)
                
                # Ensure at least one feature is selected
                if np.sum(population[i]) == 0:
                    population[i][np.random.randint(0, self.dim)] = 1
            
            # Evaluate new population
            fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test) 
                                     for whale in population])
            
            # Update best whale if improvement found
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < self.best_fitness:
                self.best_whale = population[current_best_idx].copy()
                self.best_fitness = fitness_values[current_best_idx]
            
            self.convergence_curve.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.best_fitness:.4f}, "
                      f"Features = {np.sum(self.best_whale)}/{self.dim}")
        
        return self.best_whale, self.best_fitness

def woa_feature_selection(X_train, X_test, y_train, y_test, n_whales=20, max_iter=50):
    """Wrapper function for WOA feature selection"""
    print("Starting Whale Optimization Algorithm for Feature Selection...")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Population size: {n_whales}, Max iterations: {max_iter}")
    print("-" * 60)
    
    woa = WhaleOptimizationAlgorithm(n_whales=n_whales, max_iter=max_iter)
    best_features, best_fitness = woa.optimize(X_train, X_test, y_train, y_test)
    
    selected_feature_indices = np.where(best_features == 1)[0]
    selected_feature_names = X_train.columns[selected_feature_indices].tolist()
    
    print("-" * 60)
    print(f"WOA Feature Selection Complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, selected_feature_names, woa.convergence_curve

# ----------------------------
# 1. Load and Preprocess Data (Your existing code)
# ----------------------------
data = pd.read_csv("/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv")
data = data.drop(columns=['title','date_time'])

string_cols = data.select_dtypes(include=['object']).columns.tolist()
if 'magnitude' in string_cols:
    string_cols.remove('magnitude')

data = pd.get_dummies(data, columns=string_cols)

X = data.drop(columns=['magnitude'])
y = data['magnitude']

# ----------------------------
# 2. Split and Preprocess Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# ----------------------------
# 3. Apply WOA Feature Selection
# ----------------------------
selected_indices, selected_features, convergence = woa_feature_selection(
    X_train, X_test, y_train, y_test, 
    n_whales=30, max_iter=50
)

print(f"\nSelected features: {selected_features}")

# ----------------------------
# 4. Train Model with Selected Features
# ----------------------------
X_train_selected = X_train.iloc[:, selected_indices]
X_test_selected = X_test.iloc[:, selected_indices]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Grid Search with selected features
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_selected, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (MSE): {-grid_search.best_score_:.4f}")

# ----------------------------
# 5. Model Evaluation and Comparison
# ----------------------------
best_model_selected = grid_search.best_estimator_
y_pred_selected = best_model_selected.predict(X_test_selected)

# Metrics for WOA-selected features
mse_selected = mean_squared_error(y_test, y_pred_selected)
mae_selected = mean_absolute_error(y_test, y_pred_selected)
r2_selected = r2_score(y_test, y_pred_selected)

print(f"\n=== WOA Feature Selection Results ===")
print(f"Features used: {len(selected_features)}/{X.shape[1]}")
print(f"MSE: {mse_selected:.4f}")
print(f"MAE: {mae_selected:.4f}")
print(f"R² Score: {r2_selected:.4f}")

# Train model with all features for comparison
rf_all = RandomForestRegressor(**grid_search.best_params_, random_state=42)
rf_all.fit(X_train, y_train)
y_pred_all = rf_all.predict(X_test)

mse_all = mean_squared_error(y_test, y_pred_all)
mae_all = mean_absolute_error(y_test, y_pred_all)
r2_all = r2_score(y_test, y_pred_all)

print(f"\n=== All Features Results (Baseline) ===")
print(f"Features used: {X.shape[1]}/{X.shape[1]}")
print(f"MSE: {mse_all:.4f}")
print(f"MAE: {mae_all:.4f}")
print(f"R² Score: {r2_all:.4f}")

# Performance comparison
print(f"\n=== Performance Comparison ===")
print(f"MSE Improvement: {((mse_all - mse_selected)/mse_all)*100:.2f}%")
print(f"Feature Reduction: {((X.shape[1] - len(selected_features))/X.shape[1])*100:.1f}%")

# ----------------------------
# 6. Visualization
# ----------------------------
def show_plot():
    plt.show(block=True)

# Plot 1: WOA Convergence Curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(convergence)), convergence, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Fitness (MSE + Feature Penalty)')
plt.title('WOA Convergence Curve')
plt.grid(True, alpha=0.3)
show_plot()

# Plot 2: Performance Comparison
metrics = ['MSE', 'MAE', 'R²']
woa_values = [mse_selected, mae_selected, r2_selected]
all_features_values = [mse_all, mae_all, r2_all]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(metrics):
    axes[i].bar(['WOA Selected', 'All Features'], 
                [woa_values[i], all_features_values[i]], 
                color=['skyblue', 'lightcoral'])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
plt.tight_layout()
show_plot()

# Plot 3: Feature Importance for Selected Features
if hasattr(best_model_selected, 'feature_importances_'):
    importances = best_model_selected.feature_importances_
    feature_names = [selected_features[i] for i in range(len(selected_features))]
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importances)), importances[indices], align='center', color='lightgreen')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (WOA Selected Features)')
    plt.tight_layout()
    show_plot()

# Plot 4: Actual vs Predicted (WOA Selected Features)
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.scatter(y_test, y_pred_selected, alpha=0.6, color='b', label='WOA Selected')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted (WOA Selected Features)')
plt.legend()

plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred_all, alpha=0.6, color='orange', label='All Features')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted (All Features)')
plt.legend()

plt.tight_layout()
show_plot()

# Plot 5: Feature Selection Summary
plt.figure(figsize=(10, 6))
categories = ['Selected Features', 'Removed Features']
values = [len(selected_features), X.shape[1] - len(selected_features)]
colors = ['lightgreen', 'lightcoral']

plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title(f'Feature Selection Summary\nTotal Features: {X.shape[1]}')
show_plot()

# Save the selected features for future use
selected_features_df = pd.DataFrame({
    'Feature_Index': selected_indices,
    'Feature_Name': selected_features
})
selected_features_df.to_csv('woa_selected_features.csv', index=False)
print(f"\nSelected features saved to 'woa_selected_features.csv'")

# Save the trained model with selected features
joblib.dump(best_model_selected, 'earthquake_rf_woa_model.pkl')
joblib.dump(scaler, 'feature_scaler_woa.pkl')
print("WOA-optimized model saved as 'earthquake_rf_woa_model.pkl'")
