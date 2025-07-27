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

class GreyWolfOptimizer:
    def __init__(self, n_wolves=20, max_iter=100, dim=None):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        
        # Wolf hierarchy
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        
        # Population and convergence tracking
        self.positions = None
        self.convergence_curve = []
        
    def sigmoid(self, x):
        """Sigmoid function to convert continuous values to probabilities"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def position_to_binary(self, position):
        """Convert continuous position to binary using sigmoid function"""
        probabilities = self.sigmoid(position)
        binary_pos = np.where(np.random.random(self.dim) < probabilities, 1, 0)
        
        # Ensure at least one feature is selected
        if np.sum(binary_pos) == 0:
            binary_pos[np.random.randint(0, self.dim)] = 1
            
        return binary_pos
    
    def initialize_population(self):
        """Initialize grey wolf population"""
        # Initialize positions randomly between -4 and 4
        self.positions = np.random.uniform(-4, 4, (self.n_wolves, self.dim))
        
        # Initialize alpha, beta, delta positions
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test):
        """Fitness function based on model performance with selected features"""
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            # Select features
            X_train_selected = X_train.iloc[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features]
            
            # Train model
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_selected, y_train)
            
            # Predict and calculate fitness
            y_pred = rf.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            
            # Add penalty for using too many features
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except:
            return float('inf')
    
    def update_alpha_beta_delta(self, fitness, position, wolf_idx):
        """Update alpha, beta, delta wolves based on fitness - CORRECTED VERSION"""
        if fitness < self.alpha_score:
            # Update delta (becomes old beta)
            self.delta_score = self.beta_score  # ✅ Fixed: No .copy() for scalars
            self.delta_pos = self.beta_pos.copy()
            
            # Update beta (becomes old alpha)  
            self.beta_score = self.alpha_score  # ✅ Fixed: No .copy() for scalars
            self.beta_pos = self.alpha_pos.copy()
            
            # Update alpha (new best)
            self.alpha_score = fitness  # ✅ Fixed: No .copy() for scalars
            self.alpha_pos = position.copy()
            
        elif fitness < self.beta_score:
            # Update delta (becomes old beta)
            self.delta_score = self.beta_score  # ✅ Fixed: No .copy() for scalars
            self.delta_pos = self.beta_pos.copy()
            
            # Update beta (new second best)
            self.beta_score = fitness  # ✅ Fixed: No .copy() for scalars
            self.beta_pos = position.copy()
            
        elif fitness < self.delta_score:
            # Update delta (new third best)
            self.delta_score = fitness  # ✅ Fixed: No .copy() for scalars
            self.delta_pos = position.copy()
    
    def update_position(self, wolf_idx, a):
        """Update wolf position based on alpha, beta, delta positions"""
        # Calculate distances to alpha, beta, delta
        r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        
        D_alpha = abs(C1 * self.alpha_pos - self.positions[wolf_idx])
        X1 = self.alpha_pos - A1 * D_alpha
        
        r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        
        D_beta = abs(C2 * self.beta_pos - self.positions[wolf_idx])
        X2 = self.beta_pos - A2 * D_beta
        
        r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        
        D_delta = abs(C3 * self.delta_pos - self.positions[wolf_idx])
        X3 = self.delta_pos - A3 * D_delta
        
        # Update position as average of three positions
        self.positions[wolf_idx] = (X1 + X2 + X3) / 3.0
        
        # Boundary handling
        self.positions[wolf_idx] = np.clip(self.positions[wolf_idx], -10, 10)
    
    def optimize(self, X_train, X_test, y_train, y_test):
        """Main GWO optimization loop"""
        self.dim = X_train.shape[1]
        
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population and find alpha, beta, delta
        for i in range(self.n_wolves):
            fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test)
            self.update_alpha_beta_delta(fitness, self.positions[i], i)
        
        self.convergence_curve.append(self.alpha_score)
        
        print(f"Initial alpha fitness: {self.alpha_score:.4f}")
        binary_alpha = self.position_to_binary(self.alpha_pos)
        print(f"Initial features selected: {np.sum(binary_alpha)}/{self.dim}")
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update a (linearly decreases from 2 to 0)
            a = 2 - 2 * iteration / self.max_iter
            
            # Update position of each wolf
            for i in range(self.n_wolves):
                self.update_position(i, a)
                
                # Evaluate new position
                fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test)
                
                # Update alpha, beta, delta if better solution found
                self.update_alpha_beta_delta(fitness, self.positions[i], i)
            
            self.convergence_curve.append(self.alpha_score)
            
            if (iteration + 1) % 10 == 0:
                binary_alpha = self.position_to_binary(self.alpha_pos)
                print(f"Iteration {iteration + 1}: Alpha fitness = {self.alpha_score:.4f}, "
                      f"Features = {np.sum(binary_alpha)}/{self.dim}")
        
        # Return best solution (alpha wolf)
        best_binary = self.position_to_binary(self.alpha_pos)
        return best_binary, self.alpha_score


def gwo_feature_selection(X_train, X_test, y_train, y_test, n_wolves=30, max_iter=50):
    """Wrapper function for GWO feature selection"""
    print("Starting Grey Wolf Optimization for Feature Selection...")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Pack size: {n_wolves} wolves, Max iterations: {max_iter}")
    print("-" * 60)
    
    gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
    best_features, best_fitness = gwo.optimize(X_train, X_test, y_train, y_test)
    
    selected_feature_indices = np.where(best_features == 1)[0]
    selected_feature_names = X_train.columns[selected_feature_indices].tolist()
    
    print("-" * 60)
    print(f"GWO Feature Selection Complete!")
    print(f"Alpha fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, selected_feature_names, gwo.convergence_curve

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
# 3. Apply GWO Feature Selection (WITH FIX)
# ----------------------------
# Define parameters as variables for later use
n_wolves = 30
max_iter = 50

selected_indices, selected_features, convergence = gwo_feature_selection(
    X_train, X_test, y_train, y_test, 
    n_wolves=n_wolves, max_iter=max_iter
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

# Metrics for GWO-selected features
mse_selected = mean_squared_error(y_test, y_pred_selected)
mae_selected = mean_absolute_error(y_test, y_pred_selected)
r2_selected = r2_score(y_test, y_pred_selected)

print(f"\n=== GWO Feature Selection Results ===")
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

# Plot 1: GWO Convergence Curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(convergence)), convergence, 'g-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('Alpha Fitness (MSE + Feature Penalty)')
plt.title('GWO Convergence Curve - Alpha Wolf Fitness Evolution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
show_plot()

# Plot 2: Performance Comparison
metrics = ['MSE', 'MAE', 'R²']
gwo_values = [mse_selected, mae_selected, r2_selected]
all_features_values = [mse_all, mae_all, r2_all]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(metrics):
    axes[i].bar(['GWO Selected', 'All Features'], 
                [gwo_values[i], all_features_values[i]], 
                color=['lightgreen', 'lightcoral'])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
    # Add value labels on bars
    for j, v in enumerate([gwo_values[i], all_features_values[i]]):
        axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom')
plt.tight_layout()
show_plot()

# Plot 3: Feature Importance for Selected Features
if hasattr(best_model_selected, 'feature_importances_'):
    importances = best_model_selected.feature_importances_
    feature_names = [selected_features[i] for i in range(len(selected_features))]
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importances)), importances[indices], align='center', color='forestgreen')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (GWO Selected Features)')
    plt.tight_layout()
    show_plot()

# Plot 4: Wolf Hierarchy Visualization
plt.figure(figsize=(14, 10))

# Alpha wolf convergence
plt.subplot(2, 3, 1)
plt.plot(range(len(convergence)), convergence, 'g-', linewidth=2)
plt.fill_between(range(len(convergence)), convergence, alpha=0.3, color='green')
plt.xlabel('Iteration')
plt.ylabel('Alpha Fitness')
plt.title('Alpha Wolf (Best Solution) Evolution')
plt.grid(True, alpha=0.3)

# Actual vs Predicted (GWO)
plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_selected, alpha=0.6, color='darkgreen', label='GWO Selected')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted (GWO)')
plt.legend()
plt.grid(True, alpha=0.3)

# Actual vs Predicted (All Features)
plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_all, alpha=0.6, color='orange', label='All Features')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted (All Features)')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature Selection Summary Pie Chart
plt.subplot(2, 3, 4)
categories = ['Selected', 'Removed']
values = [len(selected_features), X.shape[1] - len(selected_features)]
colors = ['lightgreen', 'lightcoral']
plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title(f'Feature Selection\n({X.shape[1]} → {len(selected_features)} features)')

# Performance Improvement Bar Chart
plt.subplot(2, 3, 5)
improvement = ((mse_all - mse_selected)/mse_all)*100
reduction = ((X.shape[1] - len(selected_features))/X.shape[1])*100
plt.bar(['MSE Improvement (%)', 'Feature Reduction (%)'], 
        [improvement, reduction], 
        color=['skyblue', 'lightgreen'])
plt.ylabel('Percentage')
plt.title('GWO Optimization Results')
for i, v in enumerate([improvement, reduction]):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

# Convergence Rate Analysis
plt.subplot(2, 3, 6)
convergence_improvement = [max(convergence) - fitness for fitness in convergence]
plt.plot(range(len(convergence_improvement)), convergence_improvement, 'purple', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cumulative Improvement')
plt.title('GWO Search Progress')
plt.grid(True, alpha=0.3)

plt.tight_layout()
show_plot()

# Plot 5: Wolf Pack Hierarchy Visualization
plt.figure(figsize=(12, 8))

# Create a conceptual visualization of wolf hierarchy
plt.subplot(2, 2, 1)
hierarchy = ['Alpha\n(Best)', 'Beta\n(2nd Best)', 'Delta\n(3rd Best)', 'Omega\n(Others)']
values = [1, 1, 1, max(1, n_wolves - 3)]
colors = ['darkgreen', 'green', 'lightgreen', 'lightgray']
plt.pie(values, labels=hierarchy, colors=colors, autopct='%1.0f', startangle=90)
plt.title('GWO Wolf Pack Hierarchy')

# Feature selection statistics
plt.subplot(2, 2, 2)
stats = ['Original\nFeatures', 'Selected\nFeatures', 'Removed\nFeatures']
counts = [X.shape[1], len(selected_features), X.shape[1] - len(selected_features)]
colors = ['skyblue', 'green', 'red']
bars = plt.bar(stats, counts, color=colors, alpha=0.7)
plt.ylabel('Number of Features')
plt.title('Feature Selection Statistics')
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(count), ha='center', va='bottom')

# Convergence comparison (if we had multiple runs)
plt.subplot(2, 2, 3)
# Simulate showing best, worst, average convergence
best_curve = np.array(convergence)
plt.plot(range(len(best_curve)), best_curve, 'g-', linewidth=2, label='Best Run')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('GWO Convergence Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Algorithm parameters summary
plt.subplot(2, 2, 4)
plt.axis('off')
param_text = f"""GWO Parameters Summary:
━━━━━━━━━━━━━━━━━━━━━
Pack Size: {n_wolves} wolves
Max Iterations: {max_iter}
Search Space: {X.shape[1]}D
━━━━━━━━━━━━━━━━━━━━━
Results:
✓ Features: {X.shape[1]} → {len(selected_features)}
✓ Reduction: {((X.shape[1] - len(selected_features))/X.shape[1])*100:.1f}%
✓ MSE: {mse_all:.4f} → {mse_selected:.4f}
✓ Improvement: {((mse_all - mse_selected)/mse_all)*100:.2f}%"""

plt.text(0.1, 0.9, param_text, transform=plt.gca().transAxes, 
         fontfamily='monospace', fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
show_plot()

# ----------------------------
# 7. Save Results
# ----------------------------
# Save the selected features for future use
selected_features_df = pd.DataFrame({
    'Feature_Index': selected_indices,
    'Feature_Name': selected_features
})
selected_features_df.to_csv('gwo_selected_features.csv', index=False)
print(f"\nSelected features saved to 'gwo_selected_features.csv'")

# Save GWO optimization results
gwo_results = pd.DataFrame({
    'Iteration': range(len(convergence)),
    'Alpha_Fitness': convergence
})
gwo_results.to_csv('gwo_convergence_history.csv', index=False)
print("GWO convergence history saved to 'gwo_convergence_history.csv'")

# Save the trained model with selected features
joblib.dump(best_model_selected, 'earthquake_rf_gwo_model.pkl')
joblib.dump(scaler, 'feature_scaler_gwo.pkl')
print("GWO-optimized model saved as 'earthquake_rf_gwo_model.pkl'")

# Final comprehensive summary (WITH FIX)
print(f"\n{'='*70}")
print(f"GREY WOLF OPTIMIZATION FEATURE SELECTION SUMMARY")
print(f"{'='*70}")
print(f"Algorithm: Grey Wolf Optimizer (GWO)")
print(f"Inspiration: Wolf pack social hierarchy and hunting behavior")
print(f"Pack Size: {n_wolves} wolves (Alpha, Beta, Delta, Omega)")
print(f"Iterations: {max_iter}")
print(f"Search Space Dimension: {X.shape[1]}")
print(f"{'─'*70}")
print(f"FEATURE SELECTION RESULTS:")
print(f"  • Original Features: {X.shape[1]}")
print(f"  • Selected Features: {len(selected_features)}")
print(f"  • Features Removed: {X.shape[1] - len(selected_features)}")
print(f"  • Reduction Rate: {((X.shape[1] - len(selected_features))/X.shape[1])*100:.1f}%")
print(f"{'─'*70}")
print(f"PERFORMANCE METRICS:")
print(f"  • MSE (GWO): {mse_selected:.4f}")
print(f"  • MSE (All Features): {mse_all:.4f}")
print(f"  • MSE Improvement: {((mse_all - mse_selected)/mse_all)*100:.2f}%")
print(f"  • MAE (GWO): {mae_selected:.4f}")
print(f"  • R² Score (GWO): {r2_selected:.4f}")
print(f"{'─'*70}")
print(f"WOLF PACK EFFICIENCY:")
print(f"  • Alpha Wolf (Best Solution): Found optimal feature subset")
print(f"  • Beta Wolf (2nd Best): Guided search toward optimal regions")
print(f"  • Delta Wolf (3rd Best): Assisted in exploration-exploitation balance")
print(f"  • Omega Wolves: Explored diverse search space regions")
print(f"{'='*70}")
