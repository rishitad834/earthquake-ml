import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ========================================
# PARTICLE SWARM OPTIMIZATION CLASS
# ========================================

class ParticleSwarmOptimization:
    def __init__(self, n_particles=30, max_iter=50, w=0.9, c1=2.0, c2=2.0, dim=None):
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
        """Fitness function based on SVM model performance with selected features"""
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            # Select features
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            
            # Train SVM model
            svm_model = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
            svm_model.fit(X_train_selected, y_train)
            
            # Predict and calculate fitness
            y_pred = svm_model.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            
            # Add penalty for using too many features
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except Exception as e:
            print(f"Error in fitness function: {e}")
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
        
        print(f"Starting PSO optimization with {self.dim} features...")
        
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
    print("=" * 60)
    print("PARTICLE SWARM OPTIMIZATION - FEATURE SELECTION")
    print("=" * 60)
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
    
    print("-" * 60)
    print(f"PSO Feature Selection Complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, pso.convergence_curve

# ========================================
# MAIN ANALYSIS PIPELINE
# ========================================

def main():
    # Load and preprocess data
    print("Loading and preprocessing earthquake data...")
    
    # Load data
    data = pd.read_csv("/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv")
    data = data.drop(columns=['title', 'date_time'])
    
    # Handle categorical variables
    string_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'magnitude' in string_cols:
        string_cols.remove('magnitude')
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=string_cols)
    
    # Separate features and target
    X = data.drop(columns=['magnitude'])
    y = data['magnitude']
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===================================
    # BASELINE: SVM with All Features
    # ===================================
    
    print("\n" + "="*60)
    print("BASELINE: SVM WITH ALL FEATURES")
    print("="*60)
    
    start_time = time.time()
    
    # Train baseline model
    svm_baseline = SVR()
    
    # Parameter grid for baseline
    param_grid_baseline = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    
    grid_search_baseline = GridSearchCV(
        svm_baseline,
        param_grid_baseline,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    grid_search_baseline.fit(X_train_scaled, y_train)
    
    # Evaluate baseline
    best_baseline = grid_search_baseline.best_estimator_
    y_pred_baseline = best_baseline.predict(X_test_scaled)
    
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    r2_baseline = r2_score(y_test, y_pred_baseline)
    time_baseline = time.time() - start_time
    
    print(f"Baseline Results:")
    print(f"  Best parameters: {grid_search_baseline.best_params_}")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  MAE: {mae_baseline:.4f}")
    print(f"  R² Score: {r2_baseline:.4f}")
    print(f"  Features used: {X_train_scaled.shape[1]}")
    print(f"  Training time: {time_baseline:.2f} seconds")
    
    # ===================================
    # PSO FEATURE SELECTION
    # ===================================
    
    # Apply PSO feature selection
    selected_indices, convergence_curve = pso_feature_selection(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        n_particles=30, max_iter=50,
        w=0.9, c1=2.0, c2=2.0
    )
    
    print(f"\nSelected feature indices: {selected_indices[:10]}...")  # Show first 10
    
    # ===================================
    # TRAIN MODEL WITH SELECTED FEATURES
    # ===================================
    
    print("\n" + "="*60)
    print("SVM WITH PSO-SELECTED FEATURES")
    print("="*60)
    
    start_time = time.time()
    
    # Get selected features
    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    # Parameter grid for PSO-selected features
    param_grid_pso = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    
    svm_pso = SVR()
    grid_search_pso = GridSearchCV(
        svm_pso,
        param_grid_pso,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search_pso.fit(X_train_selected, y_train)
    
    # Evaluate PSO model
    best_pso_model = grid_search_pso.best_estimator_
    y_pred_pso = best_pso_model.predict(X_test_selected)
    
    mse_pso = mean_squared_error(y_test, y_pred_pso)
    mae_pso = mean_absolute_error(y_test, y_pred_pso)
    r2_pso = r2_score(y_test, y_pred_pso)
    time_pso = time.time() - start_time
    
    print(f"\nPSO-Optimized Results:")
    print(f"  Best parameters: {grid_search_pso.best_params_}")
    print(f"  MSE: {mse_pso:.4f}")
    print(f"  MAE: {mae_pso:.4f}")
    print(f"  R² Score: {r2_pso:.4f}")
    print(f"  Features used: {len(selected_indices)}")
    print(f"  Training time: {time_pso:.2f} seconds")
    
    # ===================================
    # COMPARISON AND RESULTS
    # ===================================
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Baseline (All Features):")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  Features: {X_train_scaled.shape[1]}")
    
    print(f"\nPSO Feature Selection:")
    print(f"  MSE: {mse_pso:.4f}")
    print(f"  Features: {len(selected_indices)}")
    print(f"  Feature reduction: {(1 - len(selected_indices)/X_train_scaled.shape[1])*100:.1f}%")
    print(f"  MSE improvement: {((mse_baseline - mse_pso)/mse_baseline)*100:.2f}%")
    
    # ===================================
    # VISUALIZATIONS
    # ===================================
    
    def show_plot():
        plt.show(block=True)
    
    # Plot 1: PSO Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(convergence_curve)), convergence_curve, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Fitness (MSE + Feature Penalty)')
    plt.title('PSO Convergence Curve - Swarm Intelligence Evolution')
    plt.grid(True, alpha=0.3)
    show_plot()
    
    # Plot 2: Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['Baseline\n(All Features)', 'PSO\n(Selected Features)']
    mse_values = [mse_baseline, mse_pso]
    mae_values = [mae_baseline, mae_pso]
    r2_values = [r2_baseline, r2_pso]
    
    # MSE comparison
    bars1 = axes[0].bar(methods, mse_values, color=['lightcoral', 'lightblue'])
    axes[0].set_title('MSE Comparison')
    axes[0].set_ylabel('MSE')
    for i, v in enumerate(mse_values):
        axes[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # MAE comparison
    bars2 = axes[1].bar(methods, mae_values, color=['lightcoral', 'lightblue'])
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('MAE')
    for i, v in enumerate(mae_values):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # R² comparison
    bars3 = axes[2].bar(methods, r2_values, color=['lightcoral', 'lightblue'])
    axes[2].set_title('R² Score Comparison')
    axes[2].set_ylabel('R² Score')
    for i, v in enumerate(r2_values):
        axes[2].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    show_plot()
    
    # Plot 3: Actual vs Predicted Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline
    axes[0].scatter(y_test, y_pred_baseline, alpha=0.6, color='coral')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Magnitude')
    axes[0].set_ylabel('Predicted Magnitude')
    axes[0].set_title(f'Baseline (All Features)\nR² = {r2_baseline:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # PSO
    axes[1].scatter(y_test, y_pred_pso, alpha=0.6, color='lightblue')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Magnitude')
    axes[1].set_ylabel('Predicted Magnitude')
    axes[1].set_title(f'PSO Selected Features\nR² = {r2_pso:.4f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    show_plot()
    
    # Plot 4: Residual Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Baseline residuals
    residuals_baseline = y_test - y_pred_baseline
    axes[0, 0].scatter(y_pred_baseline, residuals_baseline, alpha=0.6, color='coral')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Magnitude')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Baseline: Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSO residuals
    residuals_pso = y_test - y_pred_pso
    axes[0, 1].scatter(y_pred_pso, residuals_pso, alpha=0.6, color='lightblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Magnitude')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('PSO: Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Baseline residual histogram
    axes[1, 0].hist(residuals_baseline, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Baseline: Distribution of Residuals')
    
    # PSO residual histogram
    axes[1, 1].hist(residuals_pso, bins=30, color='lightblue', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('PSO: Distribution of Residuals')
    
    plt.tight_layout()
    show_plot()
    
    # Plot 5: Confusion Matrix for Binned Magnitudes
    # Define magnitude bins
    bins = [0, 2, 4, 6, 8, 10]
    bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline confusion matrix
    y_test_binned = pd.cut(y_test, bins=bins, labels=bin_labels)
    y_pred_baseline_binned = pd.cut(y_pred_baseline, bins=bins, labels=bin_labels)
    conf_matrix_baseline = confusion_matrix(y_test_binned, y_pred_baseline_binned, labels=bin_labels)
    
    sns.heatmap(conf_matrix_baseline, annot=True, fmt='d', cmap='Reds', 
                xticklabels=bin_labels, yticklabels=bin_labels, ax=axes[0])
    axes[0].set_xlabel("Predicted Magnitude Range")
    axes[0].set_ylabel("Actual Magnitude Range")
    axes[0].set_title("Baseline: Confusion Matrix (Binned)")
    
    # PSO confusion matrix
    y_pred_pso_binned = pd.cut(y_pred_pso, bins=bins, labels=bin_labels)
    conf_matrix_pso = confusion_matrix(y_test_binned, y_pred_pso_binned, labels=bin_labels)
    
    sns.heatmap(conf_matrix_pso, annot=True, fmt='d', cmap='Blues', 
                xticklabels=bin_labels, yticklabels=bin_labels, ax=axes[1])
    axes[1].set_xlabel("Predicted Magnitude Range")
    axes[1].set_ylabel("Actual Magnitude Range")
    axes[1].set_title("PSO: Confusion Matrix (Binned)")
    
    plt.tight_layout()
    show_plot()
    
    # Plot 6: PSO Swarm Behavior Visualization
    plt.figure(figsize=(12, 8))
    
    # Create a comprehensive PSO analysis
    plt.subplot(2, 2, 1)
    plt.plot(range(len(convergence_curve)), convergence_curve, 'b-', linewidth=2)
    plt.fill_between(range(len(convergence_curve)), convergence_curve, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('PSO Fitness Evolution')
    plt.grid(True, alpha=0.3)
    
    # Feature reduction visualization
    plt.subplot(2, 2, 2)
    categories = ['Selected Features', 'Removed Features']
    values = [len(selected_indices), X_train_scaled.shape[1] - len(selected_indices)]
    colors = ['lightblue', 'lightcoral']
    
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'PSO Feature Selection\n({X_train_scaled.shape[1]} → {len(selected_indices)} features)')
    
    # Performance improvement
    plt.subplot(2, 2, 3)
    improvement = ((mse_baseline - mse_pso)/mse_baseline)*100
    reduction = ((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100
    plt.bar(['MSE Improvement (%)', 'Feature Reduction (%)'], 
            [improvement, reduction], 
            color=['skyblue', 'lightgreen'])
    plt.ylabel('Percentage')
    plt.title('PSO Optimization Results')
    for i, v in enumerate([improvement, reduction]):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # PSO parameters summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    param_text = f"""PSO Parameters Summary:
━━━━━━━━━━━━━━━━━━━━━
Swarm Size: 30 particles
Max Iterations: 50
Inertia Weight: 0.9 → 0.4
Cognitive (c1): 2.0
Social (c2): 2.0
━━━━━━━━━━━━━━━━━━━━━
Results:
✓ Features: {X_train_scaled.shape[1]} → {len(selected_indices)}
✓ Reduction: {((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100:.1f}%
✓ MSE: {mse_baseline:.4f} → {mse_pso:.4f}
✓ Improvement: {((mse_baseline - mse_pso)/mse_baseline)*100:.2f}%"""

    plt.text(0.1, 0.9, param_text, transform=plt.gca().transAxes, 
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    show_plot()
    
    # ===================================
    # SAVE RESULTS
    # ===================================
    
    # Save selected features
    selected_features_df = pd.DataFrame({
        'Feature_Index': selected_indices,
        'Feature_Name': [f'Feature_{i}' for i in selected_indices]  # Generic names since we don't have original feature names
    })
    selected_features_df.to_csv('pso_svm_selected_features.csv', index=False)
    
    # Save convergence history
    convergence_df = pd.DataFrame({
        'Iteration': range(len(convergence_curve)),
        'Global_Best_Fitness': convergence_curve
    })
    convergence_df.to_csv('pso_svm_convergence.csv', index=False)
    
    # Save models
    joblib.dump(best_baseline, 'svm_baseline_model.pkl')
    joblib.dump(best_pso_model, 'svm_pso_model.pkl')
    joblib.dump(scaler, 'feature_scaler_pso_svm.pkl')
    
    print(f"\nFiles saved:")
    print(f"  • pso_svm_selected_features.csv")
    print(f"  • pso_svm_convergence.csv")
    print(f"  • svm_baseline_model.pkl")
    print(f"  • svm_pso_model.pkl")
    print(f"  • feature_scaler_pso_svm.pkl")
    
    print("\n" + "="*60)
    print("PSO FEATURE SELECTION SUMMARY")
    print("="*60)
    print(f"Algorithm: Particle Swarm Optimization")
    print(f"Inspiration: Bird flocking and fish schooling behavior")
    print(f"Swarm Size: 30 particles")
    print(f"Iterations: 50")
    print(f"Search Space Dimension: {X_train_scaled.shape[1]}")
    print(f"Final MSE: {mse_pso:.4f}")
    print(f"Feature Reduction: {((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100:.1f}%")
    print(f"Performance Improvement: {((mse_baseline - mse_pso)/mse_baseline)*100:.2f}%")
    print("PSO Swarm Intelligence:")
    print("  • Particles: Explored search space using social learning")
    print("  • Cognitive Component: Learned from personal best solutions")
    print("  • Social Component: Followed global best particle guidance")
    print("  • Inertia Weight: Balanced exploration and exploitation")
    print("="*60)

if __name__ == "__main__":
    main()
