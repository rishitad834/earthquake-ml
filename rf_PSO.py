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

class ParticleSwarmOptimization:
    def __init__(self, n_particles=20, max_iter=100, w=0.9, c1=2.0, c2=2.0, dim=None):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.dim = dim
        
        # Swarm properties
        self.particles_pos = None
        self.particles_vel = None
        self.particles_best_pos = None
        self.particles_best_fitness = None
        self.global_best_pos = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []
        
    def sigmoid(self, x):
        """Sigmoid function to convert continuous values to probabilities"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def initialize_swarm(self):
        """Initialize particle positions and velocities"""
        # Initialize positions randomly between -4 and 4 (will be converted to binary)
        self.particles_pos = np.random.uniform(-4, 4, (self.n_particles, self.dim))
        
        # Initialize velocities randomly between -1 and 1
        self.particles_vel = np.random.uniform(-1, 1, (self.n_particles, self.dim))
        
        # Initialize personal best positions and fitness
        self.particles_best_pos = self.particles_pos.copy()
        self.particles_best_fitness = np.full(self.n_particles, float('inf'))
        
    def position_to_binary(self, position):
        """Convert continuous position to binary using sigmoid function"""
        probabilities = self.sigmoid(position)
        binary_pos = np.where(np.random.random(self.dim) < probabilities, 1, 0)
        
        # Ensure at least one feature is selected
        if np.sum(binary_pos) == 0:
            binary_pos[np.random.randint(0, self.dim)] = 1
            
        return binary_pos
    
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
    
    def update_velocity(self, particle_idx):
        """Update particle velocity using PSO equation"""
        r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
        
        cognitive_component = self.c1 * r1 * (self.particles_best_pos[particle_idx] - self.particles_pos[particle_idx])
        social_component = self.c2 * r2 * (self.global_best_pos - self.particles_pos[particle_idx])
        
        self.particles_vel[particle_idx] = (self.w * self.particles_vel[particle_idx] + 
                                          cognitive_component + social_component)
        
        # Velocity clamping to prevent explosion
        v_max = 6.0
        self.particles_vel[particle_idx] = np.clip(self.particles_vel[particle_idx], -v_max, v_max)
    
    def update_position(self, particle_idx):
        """Update particle position"""
        self.particles_pos[particle_idx] += self.particles_vel[particle_idx]
        
        # Position clamping
        self.particles_pos[particle_idx] = np.clip(self.particles_pos[particle_idx], -10, 10)
    
    def optimize(self, X_train, X_test, y_train, y_test):
        """Main PSO optimization loop"""
        self.dim = X_train.shape[1]
        
        # Initialize swarm
        self.initialize_swarm()
        
        # Evaluate initial swarm
        for i in range(self.n_particles):
            fitness = self.fitness_function(self.particles_pos[i], X_train, X_test, y_train, y_test)
            self.particles_best_fitness[i] = fitness
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_pos = self.particles_pos[i].copy()
        
        self.convergence_curve.append(self.global_best_fitness)
        
        print(f"Initial best fitness: {self.global_best_fitness:.4f}")
        binary_best = self.position_to_binary(self.global_best_pos)
        print(f"Initial features selected: {np.sum(binary_best)}/{self.dim}")
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update inertia weight linearly from w_max to w_min
            self.w = 0.9 - (0.9 - 0.4) * iteration / self.max_iter
            
            for i in range(self.n_particles):
                # Update velocity and position
                self.update_velocity(i)
                self.update_position(i)
                
                # Evaluate new position
                fitness = self.fitness_function(self.particles_pos[i], X_train, X_test, y_train, y_test)
                
                # Update personal best
                if fitness < self.particles_best_fitness[i]:
                    self.particles_best_fitness[i] = fitness
                    self.particles_best_pos[i] = self.particles_pos[i].copy()
                    
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_pos = self.particles_pos[i].copy()
            
            self.convergence_curve.append(self.global_best_fitness)
            
            if (iteration + 1) % 10 == 0:
                binary_best = self.position_to_binary(self.global_best_pos)
                print(f"Iteration {iteration + 1}: Best fitness = {self.global_best_fitness:.4f}, "
                      f"Features = {np.sum(binary_best)}/{self.dim}")
        
        # Return best solution
        best_binary = self.position_to_binary(self.global_best_pos)
        return best_binary, self.global_best_fitness

def pso_feature_selection(X_train, X_test, y_train, y_test, n_particles=30, max_iter=50, 
                         w=0.9, c1=2.0, c2=2.0):
    """Wrapper function for PSO feature selection"""
    print("Starting Particle Swarm Optimization for Feature Selection...")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Swarm size: {n_particles}, Max iterations: {max_iter}")
    print(f"PSO Parameters: w={w}, c1={c1}, c2={c2}")
    print("-" * 60)
    
    pso = ParticleSwarmOptimization(
        n_particles=n_particles, 
        max_iter=max_iter, 
        w=w, c1=c1, c2=c2
    )
    
    best_features, best_fitness = pso.optimize(X_train, X_test, y_train, y_test)
    
    selected_feature_indices = np.where(best_features == 1)[0]
    selected_feature_names = X_train.columns[selected_feature_indices].tolist()
    
    print("-" * 60)
    print(f"PSO Feature Selection Complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, selected_feature_names, pso.convergence_curve

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
# 3. Apply PSO Feature Selection
# ----------------------------
selected_indices, selected_features, convergence = pso_feature_selection(
    X_train, X_test, y_train, y_test, 
    n_particles=30, max_iter=50,
    w=0.9, c1=2.0, c2=2.0
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

# Metrics for PSO-selected features
mse_selected = mean_squared_error(y_test, y_pred_selected)
mae_selected = mean_absolute_error(y_test, y_pred_selected)
r2_selected = r2_score(y_test, y_pred_selected)

print(f"\n=== PSO Feature Selection Results ===")
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

# Plot 1: PSO Convergence Curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(convergence)), convergence, 'b-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('Fitness (MSE + Feature Penalty)')
plt.title('PSO Convergence Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
show_plot()

# Plot 2: Performance Comparison
metrics = ['MSE', 'MAE', 'R²']
pso_values = [mse_selected, mae_selected, r2_selected]
all_features_values = [mse_all, mae_all, r2_all]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(metrics):
    axes[i].bar(['PSO Selected', 'All Features'], 
                [pso_values[i], all_features_values[i]], 
                color=['lightblue', 'lightcoral'])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
    # Add value labels on bars
    for j, v in enumerate([pso_values[i], all_features_values[i]]):
        axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom')
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
    plt.title('Feature Importance (PSO Selected Features)')
    plt.tight_layout()
    show_plot()

# Plot 4: Actual vs Predicted Comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_selected, alpha=0.6, color='b', label='PSO Selected')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted (PSO Selected Features)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_all, alpha=0.6, color='orange', label='All Features')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted (All Features)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
show_plot()

# Plot 5: Feature Selection Summary
plt.figure(figsize=(10, 8))

# Pie chart
plt.subplot(2, 1, 1)
categories = ['Selected Features', 'Removed Features']
values = [len(selected_features), X.shape[1] - len(selected_features)]
colors = ['lightgreen', 'lightcoral']

plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title(f'Feature Selection Summary\nTotal Features: {X.shape[1]}')

# Bar chart showing feature counts
plt.subplot(2, 1, 2)
categories = ['Original\nFeatures', 'Selected\nFeatures', 'Removed\nFeatures']
counts = [X.shape[1], len(selected_features), X.shape[1] - len(selected_features)]
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = plt.bar(categories, counts, color=colors)
plt.ylabel('Number of Features')
plt.title('Feature Selection Statistics')

# Add value labels on bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(count), ha='center', va='bottom')

plt.tight_layout()
show_plot()

# Plot 6: PSO Swarm Behavior Visualization (showing particle diversity)
plt.figure(figsize=(12, 8))

# Create a heatmap showing how particles explored the search space
# This is a simplified 2D representation of the multi-dimensional search
plt.subplot(2, 2, 1)
# Plot convergence with more details
plt.plot(range(len(convergence)), convergence, 'b-', linewidth=2)
plt.fill_between(range(len(convergence)), convergence, alpha=0.3)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('PSO Fitness Evolution')
plt.grid(True, alpha=0.3)

# Plot histogram of selected vs non-selected features
plt.subplot(2, 2, 2)
plt.bar(['Selected', 'Not Selected'], 
        [len(selected_features), X.shape[1] - len(selected_features)],
        color=['green', 'red'], alpha=0.7)
plt.ylabel('Number of Features')
plt.title('Feature Selection Distribution')

# Plot improvement over iterations (if fitness improved)
plt.subplot(2, 2, 3)
improvements = [max(convergence) - fitness for fitness in convergence]
plt.plot(range(len(improvements)), improvements, 'g-', linewidth=2, marker='o', markersize=3)
plt.xlabel('Iteration')
plt.ylabel('Fitness Improvement')
plt.title('PSO Performance Improvement')
plt.grid(True, alpha=0.3)

# Feature reduction visualization
plt.subplot(2, 2, 4)
reduction_percentage = (1 - len(selected_features)/X.shape[1]) * 100
remaining_percentage = 100 - reduction_percentage

plt.pie([remaining_percentage, reduction_percentage], 
        labels=[f'Used ({remaining_percentage:.1f}%)', f'Reduced ({reduction_percentage:.1f}%)'],
        colors=['lightblue', 'orange'], 
        autopct='%1.1f%%')
plt.title('Feature Reduction Achievement')

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
selected_features_df.to_csv('pso_selected_features.csv', index=False)
print(f"\nSelected features saved to 'pso_selected_features.csv'")

# Save PSO optimization results
pso_results = pd.DataFrame({
    'Iteration': range(len(convergence)),
    'Best_Fitness': convergence
})
pso_results.to_csv('pso_convergence_history.csv', index=False)
print("PSO convergence history saved to 'pso_convergence_history.csv'")

# Save the trained model with selected features
joblib.dump(best_model_selected, 'earthquake_rf_pso_model.pkl')
joblib.dump(scaler, 'feature_scaler_pso.pkl')
print("PSO-optimized model saved as 'earthquake_rf_pso_model.pkl'")

# Performance summary
print(f"\n{'='*60}")
print(f"PSO FEATURE SELECTION SUMMARY")
print(f"{'='*60}")
print(f"Algorithm: Particle Swarm Optimization")
print(f"Swarm Size: {30} particles")
print(f"Iterations: {50}")
print(f"Original Features: {X.shape[1]}")
print(f"Selected Features: {len(selected_features)}")
print(f"Feature Reduction: {((X.shape[1] - len(selected_features))/X.shape[1])*100:.1f}%")
print(f"Final MSE (PSO): {mse_selected:.4f}")
print(f"Final MSE (All): {mse_all:.4f}")
print(f"Performance Change: {((mse_all - mse_selected)/mse_all)*100:.2f}%")
print(f"{'='*60}")
