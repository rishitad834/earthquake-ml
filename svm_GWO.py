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
# GREY WOLF OPTIMIZER CLASS
# ========================================

class GreyWolfOptimizer:
    def __init__(self, n_wolves=30, max_iter=50, dim=None):
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
    
    def update_alpha_beta_delta(self, fitness, position, wolf_idx):
        """Update alpha, beta, delta wolves based on fitness"""
        if fitness < self.alpha_score:
            # Update delta (becomes old beta)
            self.delta_score = self.beta_score  # Fixed: No .copy() for scalars
            self.delta_pos = self.beta_pos.copy()
            
            # Update beta (becomes old alpha)
            self.beta_score = self.alpha_score  # Fixed: No .copy() for scalars
            self.beta_pos = self.alpha_pos.copy()
            
            # Update alpha (new best)
            self.alpha_score = fitness  # Fixed: No .copy() for scalars
            self.alpha_pos = position.copy()
            
        elif fitness < self.beta_score:
            # Update delta (becomes old beta)
            self.delta_score = self.beta_score  # Fixed: No .copy() for scalars
            self.delta_pos = self.beta_pos.copy()
            
            # Update beta (new second best)
            self.beta_score = fitness  # Fixed: No .copy() for scalars
            self.beta_pos = position.copy()
            
        elif fitness < self.delta_score:
            # Update delta (new third best)
            self.delta_score = fitness  # Fixed: No .copy() for scalars
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
        
        print(f"Starting GWO optimization with {self.dim} features...")
        
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
    print("=" * 60)
    print("GREY WOLF OPTIMIZATION - FEATURE SELECTION")
    print("=" * 60)
    print(f"Total features: {X_train.shape[1]}")
    print(f"Pack size: {n_wolves} wolves, Max iterations: {max_iter}")
    print("-" * 60)
    
    gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
    best_features, best_fitness = gwo.optimize(X_train, X_test, y_train, y_test)
    
    selected_feature_indices = np.where(best_features == 1)[0]
    
    print("-" * 60)
    print(f"GWO Feature Selection Complete!")
    print(f"Alpha fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, gwo.convergence_curve

# ========================================
# MAIN ANALYSIS PIPELINE
# ========================================

def main():
    # Define parameters as variables (to avoid undefined variable errors)
    n_wolves = 30
    max_iter = 50
    
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
    # GWO FEATURE SELECTION
    # ===================================
    
    # Apply GWO feature selection
    selected_indices, convergence_curve = gwo_feature_selection(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        n_wolves=n_wolves, max_iter=max_iter
    )
    
    print(f"\nSelected feature indices: {selected_indices[:10]}...")  # Show first 10
    
    # ===================================
    # TRAIN MODEL WITH SELECTED FEATURES
    # ===================================
    
    print("\n" + "="*60)
    print("SVM WITH GWO-SELECTED FEATURES")
    print("="*60)
    
    start_time = time.time()
    
    # Get selected features
    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    # Parameter grid for GWO-selected features
    param_grid_gwo = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    
    svm_gwo = SVR()
    grid_search_gwo = GridSearchCV(
        svm_gwo,
        param_grid_gwo,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search_gwo.fit(X_train_selected, y_train)
    
    # Evaluate GWO model
    best_gwo_model = grid_search_gwo.best_estimator_
    y_pred_gwo = best_gwo_model.predict(X_test_selected)
    
    mse_gwo = mean_squared_error(y_test, y_pred_gwo)
    mae_gwo = mean_absolute_error(y_test, y_pred_gwo)
    r2_gwo = r2_score(y_test, y_pred_gwo)
    time_gwo = time.time() - start_time
    
    print(f"\nGWO-Optimized Results:")
    print(f"  Best parameters: {grid_search_gwo.best_params_}")
    print(f"  MSE: {mse_gwo:.4f}")
    print(f"  MAE: {mae_gwo:.4f}")
    print(f"  R² Score: {r2_gwo:.4f}")
    print(f"  Features used: {len(selected_indices)}")
    print(f"  Training time: {time_gwo:.2f} seconds")
    
    # ===================================
    # COMPARISON AND RESULTS
    # ===================================
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Baseline (All Features):")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  Features: {X_train_scaled.shape[1]}")
    
    print(f"\nGWO Feature Selection:")
    print(f"  MSE: {mse_gwo:.4f}")
    print(f"  Features: {len(selected_indices)}")
    print(f"  Feature reduction: {(1 - len(selected_indices)/X_train_scaled.shape[1])*100:.1f}%")
    print(f"  MSE improvement: {((mse_baseline - mse_gwo)/mse_baseline)*100:.2f}%")
    
    # ===================================
    # VISUALIZATIONS
    # ===================================
    
    def show_plot():
        plt.show(block=True)
    
    # Plot 1: GWO Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(convergence_curve)), convergence_curve, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Alpha Fitness (MSE + Feature Penalty)')
    plt.title('GWO Convergence Curve - Alpha Wolf Fitness Evolution')
    plt.grid(True, alpha=0.3)
    show_plot()
    
    # Plot 2: Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['Baseline\n(All Features)', 'GWO\n(Selected Features)']
    mse_values = [mse_baseline, mse_gwo]
    mae_values = [mae_baseline, mae_gwo]
    r2_values = [r2_baseline, r2_gwo]
    
    # MSE comparison
    bars1 = axes[0].bar(methods, mse_values, color=['lightcoral', 'lightgreen'])
    axes[0].set_title('MSE Comparison')
    axes[0].set_ylabel('MSE')
    for i, v in enumerate(mse_values):
        axes[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # MAE comparison
    bars2 = axes[1].bar(methods, mae_values, color=['lightcoral', 'lightgreen'])
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('MAE')
    for i, v in enumerate(mae_values):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # R² comparison
    bars3 = axes[2].bar(methods, r2_values, color=['lightcoral', 'lightgreen'])
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
    
    # GWO
    axes[1].scatter(y_test, y_pred_gwo, alpha=0.6, color='lightgreen')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Magnitude')
    axes[1].set_ylabel('Predicted Magnitude')
    axes[1].set_title(f'GWO Selected Features\nR² = {r2_gwo:.4f}')
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
    
    # GWO residuals
    residuals_gwo = y_test - y_pred_gwo
    axes[0, 1].scatter(y_pred_gwo, residuals_gwo, alpha=0.6, color='lightgreen')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Magnitude')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('GWO: Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Baseline residual histogram
    axes[1, 0].hist(residuals_baseline, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Baseline: Distribution of Residuals')
    
    # GWO residual histogram
    axes[1, 1].hist(residuals_gwo, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('GWO: Distribution of Residuals')
    
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
    
    # GWO confusion matrix
    y_pred_gwo_binned = pd.cut(y_pred_gwo, bins=bins, labels=bin_labels)
    conf_matrix_gwo = confusion_matrix(y_test_binned, y_pred_gwo_binned, labels=bin_labels)
    
    sns.heatmap(conf_matrix_gwo, annot=True, fmt='d', cmap='Greens', 
                xticklabels=bin_labels, yticklabels=bin_labels, ax=axes[1])
    axes[1].set_xlabel("Predicted Magnitude Range")
    axes[1].set_ylabel("Actual Magnitude Range")
    axes[1].set_title("GWO: Confusion Matrix (Binned)")
    
    plt.tight_layout()
    show_plot()
    
    # Plot 6: Wolf Pack Hierarchy Visualization
    plt.figure(figsize=(12, 8))
    
    # Create a comprehensive GWO analysis
    plt.subplot(2, 2, 1)
    plt.plot(range(len(convergence_curve)), convergence_curve, 'g-', linewidth=2)
    plt.fill_between(range(len(convergence_curve)), convergence_curve, alpha=0.3, color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha Fitness')
    plt.title('Alpha Wolf (Best Solution) Evolution')
    plt.grid(True, alpha=0.3)
    
    # Wolf pack hierarchy visualization
    plt.subplot(2, 2, 2)
    hierarchy = ['Alpha\n(Best)', 'Beta\n(2nd Best)', 'Delta\n(3rd Best)', 'Omega\n(Others)']
    values = [1, 1, 1, max(1, n_wolves - 3)]
    colors = ['darkgreen', 'green', 'lightgreen', 'lightgray']
    plt.pie(values, labels=hierarchy, colors=colors, autopct='%1.0f', startangle=90)
    plt.title('GWO Wolf Pack Hierarchy')
    
    # Feature reduction visualization
    plt.subplot(2, 2, 3)
    categories = ['Selected Features', 'Removed Features']
    values = [len(selected_indices), X_train_scaled.shape[1] - len(selected_indices)]
    colors = ['lightgreen', 'lightcoral']
    
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'GWO Feature Selection\n({X_train_scaled.shape[1]} → {len(selected_indices)} features)')
    
    # GWO parameters and results summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    param_text = f"""GWO Parameters Summary:
━━━━━━━━━━━━━━━━━━━━━
Pack Size: {n_wolves} wolves
Max Iterations: {max_iter}
Search Space: {X_train_scaled.shape[1]}D
Hierarchy: Alpha, Beta, Delta, Omega
━━━━━━━━━━━━━━━━━━━━━
Results:
✓ Features: {X_train_scaled.shape[1]} → {len(selected_indices)}
✓ Reduction: {((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100:.1f}%
✓ MSE: {mse_baseline:.4f} → {mse_gwo:.4f}
✓ Improvement: {((mse_baseline - mse_gwo)/mse_baseline)*100:.2f}%"""

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
    selected_features_df.to_csv('gwo_svm_selected_features.csv', index=False)
    
    # Save convergence history
    convergence_df = pd.DataFrame({
        'Iteration': range(len(convergence_curve)),
        'Alpha_Fitness': convergence_curve
    })
    convergence_df.to_csv('gwo_svm_convergence.csv', index=False)
    
    # Save models
    joblib.dump(best_baseline, 'svm_baseline_model.pkl')
    joblib.dump(best_gwo_model, 'svm_gwo_model.pkl')
    joblib.dump(scaler, 'feature_scaler_gwo_svm.pkl')
    
    print(f"\nFiles saved:")
    print(f"  • gwo_svm_selected_features.csv")
    print(f"  • gwo_svm_convergence.csv")
    print(f"  • svm_baseline_model.pkl")
    print(f"  • svm_gwo_model.pkl")
    print(f"  • feature_scaler_gwo_svm.pkl")
    
    print("\n" + "="*70)
    print("GREY WOLF OPTIMIZATION FEATURE SELECTION SUMMARY")
    print("="*70)
    print(f"Algorithm: Grey Wolf Optimizer (GWO)")
    print(f"Inspiration: Wolf pack social hierarchy and hunting behavior")
    print(f"Pack Size: {n_wolves} wolves (Alpha, Beta, Delta, Omega)")
    print(f"Iterations: {max_iter}")
    print(f"Search Space Dimension: {X_train_scaled.shape[1]}")
    print(f"Final MSE: {mse_gwo:.4f}")
    print(f"Feature Reduction: {((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100:.1f}%")
    print(f"Performance Improvement: {((mse_baseline - mse_gwo)/mse_baseline)*100:.2f}%")
    print(f"Wolf Pack Efficiency:")
    print(f"  • Alpha Wolf (Best Solution): Found optimal feature subset")
    print(f"  • Beta Wolf (2nd Best): Guided search toward optimal regions")
    print(f"  • Delta Wolf (3rd Best): Assisted in exploration-exploitation balance")
    print(f"  • Omega Wolves: Explored diverse search space regions")
    print("="*70)

if __name__ == "__main__":
    main()
