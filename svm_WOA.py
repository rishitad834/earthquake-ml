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
# WHALE OPTIMIZATION ALGORITHM CLASS
# ========================================

class WhaleOptimizationAlgorithm:
    def __init__(self, n_whales=20, max_iter=50, dim=None):
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
        """Fitness function based on SVM model performance with selected features"""
        selected_features = np.where(solution == 1)[0]
        
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
            
            # Add penalty for using too many features (feature selection pressure)
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except Exception as e:
            print(f"Error in fitness function: {e}")
            return float('inf')
    
    def optimize(self, X_train, X_test, y_train, y_test):
        """Main WOA optimization loop"""
        self.dim = X_train.shape[1]
        
        print(f"Starting WOA optimization with {self.dim} features...")
        
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

def woa_feature_selection(X_train, X_test, y_train, y_test, n_whales=30, max_iter=50):
    """Wrapper function for WOA feature selection"""
    print("=" * 60)
    print("WHALE OPTIMIZATION ALGORITHM - FEATURE SELECTION")
    print("=" * 60)
    print(f"Total features: {X_train.shape[1]}")
    print(f"Population size: {n_whales}, Max iterations: {max_iter}")
    print("-" * 60)
    
    woa = WhaleOptimizationAlgorithm(n_whales=n_whales, max_iter=max_iter)
    best_features, best_fitness = woa.optimize(X_train, X_test, y_train, y_test)
    
    selected_feature_indices = np.where(best_features == 1)[0]
    
    print("-" * 60)
    print(f"WOA Feature Selection Complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, woa.convergence_curve

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
    # WOA FEATURE SELECTION
    # ===================================
    
    # Apply WOA feature selection
    selected_indices, convergence_curve = woa_feature_selection(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        n_whales=30, max_iter=50
    )
    
    print(f"\nSelected feature indices: {selected_indices[:10]}...")  # Show first 10
    
    # ===================================
    # TRAIN MODEL WITH SELECTED FEATURES
    # ===================================
    
    print("\n" + "="*60)
    print("SVM WITH WOA-SELECTED FEATURES")
    print("="*60)
    
    start_time = time.time()
    
    # Get selected features
    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    # Parameter grid for WOA-selected features
    param_grid_woa = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    
    svm_woa = SVR()
    grid_search_woa = GridSearchCV(
        svm_woa,
        param_grid_woa,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search_woa.fit(X_train_selected, y_train)
    
    # Evaluate WOA model
    best_woa_model = grid_search_woa.best_estimator_
    y_pred_woa = best_woa_model.predict(X_test_selected)
    
    mse_woa = mean_squared_error(y_test, y_pred_woa)
    mae_woa = mean_absolute_error(y_test, y_pred_woa)
    r2_woa = r2_score(y_test, y_pred_woa)
    time_woa = time.time() - start_time
    
    print(f"\nWOA-Optimized Results:")
    print(f"  Best parameters: {grid_search_woa.best_params_}")
    print(f"  MSE: {mse_woa:.4f}")
    print(f"  MAE: {mae_woa:.4f}")
    print(f"  R² Score: {r2_woa:.4f}")
    print(f"  Features used: {len(selected_indices)}")
    print(f"  Training time: {time_woa:.2f} seconds")
    
    # ===================================
    # COMPARISON AND RESULTS
    # ===================================
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Baseline (All Features):")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  Features: {X_train_scaled.shape[1]}")
    
    print(f"\nWOA Feature Selection:")
    print(f"  MSE: {mse_woa:.4f}")
    print(f"  Features: {len(selected_indices)}")
    print(f"  Feature reduction: {(1 - len(selected_indices)/X_train_scaled.shape[1])*100:.1f}%")
    print(f"  MSE improvement: {((mse_baseline - mse_woa)/mse_baseline)*100:.2f}%")
    
    # ===================================
    # VISUALIZATIONS
    # ===================================
    
    def show_plot():
        plt.show(block=True)
    
    # Plot 1: WOA Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(convergence_curve)), convergence_curve, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (MSE + Feature Penalty)')
    plt.title('WOA Convergence Curve - Whale Population Evolution')
    plt.grid(True, alpha=0.3)
    show_plot()
    
    # Plot 2: Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['Baseline\n(All Features)', 'WOA\n(Selected Features)']
    mse_values = [mse_baseline, mse_woa]
    mae_values = [mae_baseline, mae_woa]
    r2_values = [r2_baseline, r2_woa]
    
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
    
    # WOA
    axes[1].scatter(y_test, y_pred_woa, alpha=0.6, color='lightblue')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Magnitude')
    axes[1].set_ylabel('Predicted Magnitude')
    axes[1].set_title(f'WOA Selected Features\nR² = {r2_woa:.4f}')
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
    
    # WOA residuals
    residuals_woa = y_test - y_pred_woa
    axes[0, 1].scatter(y_pred_woa, residuals_woa, alpha=0.6, color='lightblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Magnitude')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('WOA: Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Baseline residual histogram
    axes[1, 0].hist(residuals_baseline, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Baseline: Distribution of Residuals')
    
    # WOA residual histogram
    axes[1, 1].hist(residuals_woa, bins=30, color='lightblue', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('WOA: Distribution of Residuals')
    
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
    
    # WOA confusion matrix
    y_pred_woa_binned = pd.cut(y_pred_woa, bins=bins, labels=bin_labels)
    conf_matrix_woa = confusion_matrix(y_test_binned, y_pred_woa_binned, labels=bin_labels)
    
    sns.heatmap(conf_matrix_woa, annot=True, fmt='d', cmap='Blues', 
                xticklabels=bin_labels, yticklabels=bin_labels, ax=axes[1])
    axes[1].set_xlabel("Predicted Magnitude Range")
    axes[1].set_ylabel("Actual Magnitude Range")
    axes[1].set_title("WOA: Confusion Matrix (Binned)")
    
    plt.tight_layout()
    show_plot()
    
    # Plot 6: Feature Selection Summary
    plt.figure(figsize=(10, 8))
    
    # Feature reduction pie chart
    plt.subplot(2, 1, 1)
    categories = ['Selected Features', 'Removed Features']
    values = [len(selected_indices), X_train_scaled.shape[1] - len(selected_indices)]
    colors = ['lightblue', 'lightcoral']
    
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'WOA Feature Selection Summary\nTotal Features: {X_train_scaled.shape[1]}')
    
    # Performance improvement bar chart
    plt.subplot(2, 1, 2)
    improvement_metrics = ['MSE Improvement (%)', 'Feature Reduction (%)']
    improvement_values = [
        ((mse_baseline - mse_woa)/mse_baseline)*100,
        ((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100
    ]
    
    bars = plt.bar(improvement_metrics, improvement_values, color=['skyblue', 'lightgreen'])
    plt.ylabel('Percentage')
    plt.title('WOA Optimization Results')
    for i, v in enumerate(improvement_values):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
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
    selected_features_df.to_csv('woa_svm_selected_features.csv', index=False)
    
    # Save convergence history
    convergence_df = pd.DataFrame({
        'Iteration': range(len(convergence_curve)),
        'Fitness': convergence_curve
    })
    convergence_df.to_csv('woa_svm_convergence.csv', index=False)
    
    # Save models
    joblib.dump(best_baseline, 'svm_baseline_model.pkl')
    joblib.dump(best_woa_model, 'svm_woa_model.pkl')
    joblib.dump(scaler, 'feature_scaler_svm.pkl')
    
    print(f"\nFiles saved:")
    print(f"  • woa_svm_selected_features.csv")
    print(f"  • woa_svm_convergence.csv")
    print(f"  • svm_baseline_model.pkl")
    print(f"  • svm_woa_model.pkl")
    print(f"  • feature_scaler_svm.pkl")
    
    print("\n" + "="*60)
    print("WOA FEATURE SELECTION SUMMARY")
    print("="*60)
    print(f"Algorithm: Whale Optimization Algorithm")
    print(f"Inspiration: Humpback whale hunting behavior")
    print(f"Population Size: 30 whales")
    print(f"Iterations: 50")
    print(f"Search Space Dimension: {X_train_scaled.shape[1]}")
    print(f"Final MSE: {mse_woa:.4f}")
    print(f"Feature Reduction: {((X_train_scaled.shape[1] - len(selected_indices))/X_train_scaled.shape[1])*100:.1f}%")
    print(f"Performance Improvement: {((mse_baseline - mse_woa)/mse_baseline)*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
