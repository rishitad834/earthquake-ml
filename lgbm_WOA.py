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
        """Fitness function based on LightGBM model performance with selected features"""
        selected_features = np.where(solution == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            # Select features
            X_train_selected = X_train.iloc[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features]
            
            # Train LightGBM model
            lgb_model = lgbm.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            lgb_model.fit(X_train_selected, y_train)
            
            # Predict and calculate fitness
            y_pred = lgb_model.predict(X_test_selected)
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
    selected_feature_names = X_train.columns[selected_feature_indices].tolist()
    
    print("-" * 60)
    print(f"WOA Feature Selection Complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Selected {len(selected_feature_indices)} features out of {X_train.shape[1]}")
    print(f"Feature reduction: {(1 - len(selected_feature_indices)/X_train.shape[1])*100:.1f}%")
    
    return selected_feature_indices, selected_feature_names, woa.convergence_curve

# ========================================
# DATA LOADING AND PREPROCESSING
# ========================================

def clean_column_name(name):
    """Clean column names to remove special characters that LightGBM can't handle"""
    return re.sub(r'[^\w]+', '_', name)

def load_and_preprocess_data(file_path):
    """Load and preprocess earthquake data"""
    print("Loading and preprocessing earthquake data...")
    
    # Load data
    data = pd.read_csv(file_path)
    data = data.drop(columns=['title', 'date_time'])
    
    # Handle categorical variables
    string_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'magnitude' in string_cols:
        string_cols.remove('magnitude')
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=string_cols)
    
    # Clean column names for LightGBM compatibility
    data.columns = [clean_column_name(col) for col in data.columns]
    
    # Separate features and target
    X = data.drop(columns=['magnitude'])
    y = data['magnitude']
    
    return X, y

# ========================================
# MAIN ANALYSIS PIPELINE
# ========================================

def main():
    # Load and preprocess data
    file_path = "/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv"
    X, y = load_and_preprocess_data(file_path)
    
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
    
    # Create DataFrames with cleaned column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # ===================================
    # BASELINE: LightGBM with All Features
    # ===================================
    
    print("\n" + "="*60)
    print("BASELINE: LIGHTGBM WITH ALL FEATURES")
    print("="*60)
    
    start_time = time.time()
    
    # Train baseline model
    lgbm_baseline = lgbm.LGBMRegressor(random_state=42, verbose=-1)
    
    # Parameter grid for baseline
    param_grid_baseline = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1]
    }
    
    grid_search_baseline = GridSearchCV(
        lgbm_baseline,
        param_grid_baseline,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    grid_search_baseline.fit(X_train_df, y_train)
    
    # Evaluate baseline
    best_baseline = grid_search_baseline.best_estimator_
    y_pred_baseline = best_baseline.predict(X_test_df)
    
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    r2_baseline = r2_score(y_test, y_pred_baseline)
    time_baseline = time.time() - start_time
    
    print(f"Baseline Results:")
    print(f"  Best parameters: {grid_search_baseline.best_params_}")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  MAE: {mae_baseline:.4f}")
    print(f"  R² Score: {r2_baseline:.4f}")
    print(f"  Features used: {X_train_df.shape[1]}")
    print(f"  Training time: {time_baseline:.2f} seconds")
    
    # ===================================
    # WOA FEATURE SELECTION
    # ===================================
    
    # Apply WOA feature selection
    selected_indices, selected_features, convergence_curve = woa_feature_selection(
        X_train_df, X_test_df, y_train, y_test, 
        n_whales=30, max_iter=50
    )
    
    print(f"\nSelected features: {selected_features[:10]}...")  # Show first 10
    
    # ===================================
    # TRAIN MODEL WITH SELECTED FEATURES
    # ===================================
    
    print("\n" + "="*60)
    print("LIGHTGBM WITH WOA-SELECTED FEATURES")
    print("="*60)
    
    start_time = time.time()
    
    # Get selected features
    X_train_selected = X_train_df.iloc[:, selected_indices]
    X_test_selected = X_test_df.iloc[:, selected_indices]
    
    # Parameter grid for WOA-selected features
    param_grid_woa = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.05, 0.1, 0.15]
    }
    
    lgbm_woa = lgbm.LGBMRegressor(random_state=42, verbose=-1)
    grid_search_woa = GridSearchCV(
        lgbm_woa,
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
    print(f"  Features used: {len(selected_features)}")
    print(f"  Training time: {time_woa:.2f} seconds")
    
    # ===================================
    # COMPARISON AND RESULTS
    # ===================================
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Baseline (All Features):")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  Features: {X_train_df.shape[1]}")
    
    print(f"\nWOA Feature Selection:")
    print(f"  MSE: {mse_woa:.4f}")
    print(f"  Features: {len(selected_features)}")
    print(f"  Feature reduction: {(1 - len(selected_features)/X_train_df.shape[1])*100:.1f}%")
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
    plt.title('WOA Convergence Curve')
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
    
    # Plot 4: Feature Importance (Top 20 selected features)
    if hasattr(best_woa_model, 'feature_importances_'):
        importances = best_woa_model.feature_importances_
        feature_names = [selected_features[i] for i in range(len(selected_features))]
        
        indices = np.argsort(importances)[::-1]
        n_features_to_show = min(20, len(importances))
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(n_features_to_show), importances[indices[:n_features_to_show]], 
                align='center', color='lightgreen')
        plt.yticks(range(n_features_to_show), [feature_names[i] for i in indices[:n_features_to_show]])
        plt.gca().invert_yaxis()
        plt.xlabel('Feature Importance')
        plt.title('Top 20 WOA-Selected Feature Importances (LightGBM)')
        show_plot()
    
    # Plot 5: Feature Selection Summary
    plt.figure(figsize=(10, 6))
    categories = ['Selected Features', 'Removed Features']
    values = [len(selected_features), X_train_df.shape[1] - len(selected_features)]
    colors = ['lightgreen', 'lightcoral']
    
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'WOA Feature Selection Summary\nTotal Features: {X_train_df.shape[1]}')
    show_plot()
    
    # ===================================
    # SAVE RESULTS
    # ===================================
    
    # Save selected features
    selected_features_df = pd.DataFrame({
        'Feature_Index': selected_indices,
        'Feature_Name': selected_features
    })
    selected_features_df.to_csv('woa_lightgbm_selected_features.csv', index=False)
    
    # Save convergence history
    convergence_df = pd.DataFrame({
        'Iteration': range(len(convergence_curve)),
        'Fitness': convergence_curve
    })
    convergence_df.to_csv('woa_lightgbm_convergence.csv', index=False)
    
    # Save models
    joblib.dump(best_baseline, 'lightgbm_baseline_model.pkl')
    joblib.dump(best_woa_model, 'lightgbm_woa_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print(f"\nFiles saved:")
    print(f"  • woa_lightgbm_selected_features.csv")
    print(f"  • woa_lightgbm_convergence.csv")
    print(f"  • lightgbm_baseline_model.pkl")
    print(f"  • lightgbm_woa_model.pkl")
    print(f"  • feature_scaler.pkl")

if __name__ == "__main__":
    main()
