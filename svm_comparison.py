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
# METAHEURISTIC ALGORITHM CLASSES
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
        population = np.random.randint(0, 2, (self.n_whales, self.dim))
        for i in range(self.n_whales):
            if np.sum(population[i]) == 0:
                population[i][np.random.randint(0, self.dim)] = 1
        return population
    
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution == 1)[0]
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            
            svm = SVR(kernel='rbf', C=1.0, gamma='scale')
            svm.fit(X_train_selected, y_train)
            
            y_pred = svm.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except:
            return float('inf')
    
    def optimize(self, X_train, X_test, y_train, y_test):
        self.dim = X_train.shape[1]
        population = self.initialize_population()
        fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test) 
                                 for whale in population])
        
        best_idx = np.argmin(fitness_values)
        self.best_whale = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.convergence_curve.append(self.best_fitness)
        
        for iteration in range(self.max_iter):
            a = 2 - 2 * iteration / self.max_iter
            
            for i in range(self.n_whales):
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                
                if np.random.random() < 0.5:
                    if abs(A) < 1:
                        D = abs(C * self.best_whale - population[i])
                        population[i] = self.best_whale - A * D
                    else:
                        random_whale = population[np.random.randint(0, self.n_whales)]
                        D = abs(C * random_whale - population[i])
                        population[i] = random_whale - A * D
                else:
                    distance = abs(self.best_whale - population[i])
                    b = 1
                    l = np.random.uniform(-1, 1)
                    population[i] = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_whale
                
                population[i] = np.where(1 / (1 + np.exp(-population[i])) > 0.5, 1, 0)
                
                if np.sum(population[i]) == 0:
                    population[i][np.random.randint(0, self.dim)] = 1
            
            fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test) 
                                     for whale in population])
            
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < self.best_fitness:
                self.best_whale = population[current_best_idx].copy()
                self.best_fitness = fitness_values[current_best_idx]
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_whale, self.best_fitness

class ParticleSwarmOptimization:
    def __init__(self, n_particles=20, max_iter=50, w=0.9, c1=2.0, c2=2.0, dim=None):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = dim
        self.particles_pos = None
        self.particles_vel = None
        self.particles_best_pos = None
        self.particles_best_fitness = None
        self.global_best_pos = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def initialize_swarm(self):
        self.particles_pos = np.random.uniform(-4, 4, (self.n_particles, self.dim))
        self.particles_vel = np.random.uniform(-1, 1, (self.n_particles, self.dim))
        self.particles_best_pos = self.particles_pos.copy()
        self.particles_best_fitness = np.full(self.n_particles, float('inf'))
        
    def position_to_binary(self, position):
        probabilities = self.sigmoid(position)
        binary_pos = np.where(np.random.random(self.dim) < probabilities, 1, 0)
        if np.sum(binary_pos) == 0:
            binary_pos[np.random.randint(0, self.dim)] = 1
        return binary_pos
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test):
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            
            svm = SVR(kernel='rbf', C=1.0, gamma='scale')
            svm.fit(X_train_selected, y_train)
            
            y_pred = svm.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except:
            return float('inf')
    
    def update_velocity(self, particle_idx):
        r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
        cognitive_component = self.c1 * r1 * (self.particles_best_pos[particle_idx] - self.particles_pos[particle_idx])
        social_component = self.c2 * r2 * (self.global_best_pos - self.particles_pos[particle_idx])
        
        self.particles_vel[particle_idx] = (self.w * self.particles_vel[particle_idx] + 
                                          cognitive_component + social_component)
        
        v_max = 6.0
        self.particles_vel[particle_idx] = np.clip(self.particles_vel[particle_idx], -v_max, v_max)
    
    def update_position(self, particle_idx):
        self.particles_pos[particle_idx] += self.particles_vel[particle_idx]
        self.particles_pos[particle_idx] = np.clip(self.particles_pos[particle_idx], -10, 10)
    
    def optimize(self, X_train, X_test, y_train, y_test):
        self.dim = X_train.shape[1]
        self.initialize_swarm()
        
        for i in range(self.n_particles):
            fitness = self.fitness_function(self.particles_pos[i], X_train, X_test, y_train, y_test)
            self.particles_best_fitness[i] = fitness
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_pos = self.particles_pos[i].copy()
        
        self.convergence_curve.append(self.global_best_fitness)
        
        for iteration in range(self.max_iter):
            self.w = 0.9 - (0.9 - 0.4) * iteration / self.max_iter
            
            for i in range(self.n_particles):
                self.update_velocity(i)
                self.update_position(i)
                
                fitness = self.fitness_function(self.particles_pos[i], X_train, X_test, y_train, y_test)
                
                if fitness < self.particles_best_fitness[i]:
                    self.particles_best_fitness[i] = fitness
                    self.particles_best_pos[i] = self.particles_pos[i].copy()
                    
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_pos = self.particles_pos[i].copy()
            
            self.convergence_curve.append(self.global_best_fitness)
        
        best_binary = self.position_to_binary(self.global_best_pos)
        return best_binary, self.global_best_fitness

class GreyWolfOptimizer:
    def __init__(self, n_wolves=20, max_iter=50, dim=None):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = None
        self.convergence_curve = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def position_to_binary(self, position):
        probabilities = self.sigmoid(position)
        binary_pos = np.where(np.random.random(self.dim) < probabilities, 1, 0)
        if np.sum(binary_pos) == 0:
            binary_pos[np.random.randint(0, self.dim)] = 1
        return binary_pos
    
    def initialize_population(self):
        self.positions = np.random.uniform(-4, 4, (self.n_wolves, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test):
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            
            svm = SVR(kernel='rbf', C=1.0, gamma='scale')
            svm.fit(X_train_selected, y_train)
            
            y_pred = svm.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except:
            return float('inf')
    
    def update_alpha_beta_delta(self, fitness, position, wolf_idx):
        if fitness < self.alpha_score:
            self.delta_score = self.beta_score
            self.delta_pos = self.beta_pos.copy()
            self.beta_score = self.alpha_score
            self.beta_pos = self.alpha_pos.copy()
            self.alpha_score = fitness
            self.alpha_pos = position.copy()
            
        elif fitness < self.beta_score:
            self.delta_score = self.beta_score
            self.delta_pos = self.beta_pos.copy()
            self.beta_score = fitness
            self.beta_pos = position.copy()
            
        elif fitness < self.delta_score:
            self.delta_score = fitness
            self.delta_pos = position.copy()
    
    def update_position(self, wolf_idx, a):
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
        
        self.positions[wolf_idx] = (X1 + X2 + X3) / 3.0
        self.positions[wolf_idx] = np.clip(self.positions[wolf_idx], -10, 10)
    
    def optimize(self, X_train, X_test, y_train, y_test):
        self.dim = X_train.shape[1]
        self.initialize_population()
        
        for i in range(self.n_wolves):
            fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test)
            self.update_alpha_beta_delta(fitness, self.positions[i], i)
        
        self.convergence_curve.append(self.alpha_score)
        
        for iteration in range(self.max_iter):
            a = 2 - 2 * iteration / self.max_iter
            
            for i in range(self.n_wolves):
                self.update_position(i, a)
                fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test)
                self.update_alpha_beta_delta(fitness, self.positions[i], i)
            
            self.convergence_curve.append(self.alpha_score)
        
        best_binary = self.position_to_binary(self.alpha_pos)
        return best_binary, self.alpha_score

# ========================================
# COMPARISON FRAMEWORK
# ========================================

class SVMOptimizationComparison:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.convergence_curves = {}
        self.execution_times = {}
        self.feature_selections = {}
        
    def run_original_svm(self):
        """Run original SVM with all features"""
        print("\n" + "="*60)
        print("RUNNING ORIGINAL SVM (BASELINE)")
        print("="*60)
        
        start_time = time.time()
        
        # Grid search parameters
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        
        svm_model = SVR()
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, 
                                 scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        execution_time = time.time() - start_time
        
        # Store results
        self.results['Original_SVM'] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Features_Used': self.X_train.shape[1],
            'Feature_Reduction': 0.0,
            'Best_Params': grid_search.best_params_,
            'Model': best_model,
            'Predictions': y_pred
        }
        
        self.execution_times['Original_SVM'] = execution_time
        self.feature_selections['Original_SVM'] = list(range(self.X_train.shape[1]))
        
        print(f"Original SVM Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Features: {self.X_train.shape[1]}")
        print(f"  Time: {execution_time:.2f} seconds")
        
        return best_model, y_pred
    
    def run_metaheuristic_optimization(self, algorithm_name, optimizer_class, **kwargs):
        """Run metaheuristic optimization"""
        print(f"\n" + "="*60)
        print(f"RUNNING {algorithm_name.upper()} OPTIMIZATION")
        print("="*60)
        
        start_time = time.time()
        
        # Initialize and run optimizer
        optimizer = optimizer_class(**kwargs)
        selected_features_binary, best_fitness = optimizer.optimize(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        # Get selected features
        selected_indices = np.where(selected_features_binary == 1)[0]
        
        if len(selected_indices) == 0:
            print(f"Warning: No features selected by {algorithm_name}!")
            return None, None
        
        # Train model with selected features
        X_train_selected = self.X_train[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]
        
        # Grid search with selected features
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        }
        
        svm_model = SVR()
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, 
                                 scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train_selected, self.y_train)
        
        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_selected)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        execution_time = time.time() - start_time
        feature_reduction = (1 - len(selected_indices) / self.X_train.shape[1]) * 100
        
        # Store results
        self.results[algorithm_name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Features_Used': len(selected_indices),
            'Feature_Reduction': feature_reduction,
            'Best_Params': grid_search.best_params_,
            'Model': best_model,
            'Predictions': y_pred,
            'Best_Fitness': best_fitness
        }
        
        self.execution_times[algorithm_name] = execution_time
        self.convergence_curves[algorithm_name] = optimizer.convergence_curve
        self.feature_selections[algorithm_name] = selected_indices.tolist()
        
        print(f"{algorithm_name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Features: {len(selected_indices)}/{self.X_train.shape[1]}")
        print(f"  Reduction: {feature_reduction:.1f}%")
        print(f"  Time: {execution_time:.2f} seconds")
        
        return best_model, y_pred
    
    def run_all_comparisons(self):
        """Run all four approaches"""
        print("\n" + "#"*80)
        print("COMPREHENSIVE SVM FEATURE SELECTION COMPARISON")
        print("4 APPROACHES: Original SVM, WOA, PSO, GWO")
        print("#"*80)
        
        # 1. Original SVM
        self.run_original_svm()
        
        # 2. WOA Optimization
        self.run_metaheuristic_optimization('WOA', WhaleOptimizationAlgorithm, 
                                           n_whales=30, max_iter=50)
        
        # 3. PSO Optimization
        self.run_metaheuristic_optimization('PSO', ParticleSwarmOptimization, 
                                           n_particles=30, max_iter=50)
        
        # 4. GWO Optimization
        self.run_metaheuristic_optimization('GWO', GreyWolfOptimizer, 
                                           n_wolves=30, max_iter=50)
    
    def create_comparison_summary(self):
        """Create comprehensive comparison summary"""
        print("\n" + "#"*80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("#"*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for method, results in self.results.items():
            comparison_data.append({
                'Method': method,
                'MSE': results['MSE'],
                'MAE': results['MAE'],
                'R²': results['R2'],
                'Features': results['Features_Used'],
                'Feature_Reduction_%': results['Feature_Reduction'],
                'Execution_Time_s': self.execution_times[method]
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('MSE').reset_index(drop=True)
        
        print("\nRESULTS SUMMARY (Sorted by MSE - Lower is Better):")
        print("="*80)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Find best performing method
        best_method = df_comparison.iloc[0]['Method']
        best_mse = df_comparison.iloc[0]['MSE']
        baseline_mse = self.results['Original_SVM']['MSE']
        
        print(f"\n🏆 BEST PERFORMING METHOD: {best_method}")
        print(f"   MSE Improvement over Baseline: {((baseline_mse - best_mse)/baseline_mse)*100:.2f}%")
        
        return df_comparison
    
    def visualize_comparisons(self, df_comparison):
        """Create three separate figures with 4 plots each"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        methods = df_comparison['Method']
        
        # =====================================
        # FIGURE 1: Performance Metrics (2x2)
        # =====================================
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        axes1 = axes1.flatten()
        
        # Plot 1: MSE Comparison
        mse_values = df_comparison['MSE']
        bars1 = axes1[0].bar(methods, mse_values, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
        axes1[0].set_title('Mean Squared Error Comparison', fontsize=12, fontweight='bold')
        axes1[0].set_ylabel('MSE')
        axes1[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(mse_values):
            axes1[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Plot 2: MAE Comparison
        mae_values = df_comparison['MAE']
        bars2 = axes1[1].bar(methods, mae_values, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
        axes1[1].set_title('Mean Absolute Error Comparison', fontsize=12, fontweight='bold')
        axes1[1].set_ylabel('MAE')
        axes1[1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(mae_values):
            axes1[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Plot 3: R² Comparison
        r2_values = df_comparison['R²']
        bars3 = axes1[2].bar(methods, r2_values, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
        axes1[2].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        axes1[2].set_ylabel('R² Score')
        axes1[2].tick_params(axis='x', rotation=45)
        for i, v in enumerate(r2_values):
            axes1[2].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Plot 4: Feature Count Comparison
        feature_counts = df_comparison['Features']
        bars4 = axes1[3].bar(methods, feature_counts, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
        axes1[3].set_title('Features Used Comparison', fontsize=12, fontweight='bold')
        axes1[3].set_ylabel('Number of Features')
        axes1[3].tick_params(axis='x', rotation=45)
        for i, v in enumerate(feature_counts):
            axes1[3].text(i, v, f'{v}', ha='center', va='bottom')
        
        plt.suptitle('SVM Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # =====================================
        # FIGURE 2: Feature Analysis & Convergence (2x2)
        # =====================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        axes2 = axes2.flatten()
        
        # Plot 1: Feature Reduction Percentage
        reduction_pct = df_comparison['Feature_Reduction_%']
        bars5 = axes2[0].bar(methods, reduction_pct, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
        axes2[0].set_title('Feature Reduction Percentage', fontsize=12, fontweight='bold')
        axes2[0].set_ylabel('Reduction %')
        axes2[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(reduction_pct):
            axes2[0].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Execution Time Comparison
        exec_times = df_comparison['Execution_Time_s']
        bars6 = axes2[1].bar(methods, exec_times, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
        axes2[1].set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
        axes2[1].set_ylabel('Time (seconds)')
        axes2[1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(exec_times):
            axes2[1].text(i, v, f'{v:.1f}s', ha='center', va='bottom')
        
        # Plot 3: Convergence Curves
        for method in ['WOA', 'PSO', 'GWO']:
            if method in self.convergence_curves:
                axes2[2].plot(self.convergence_curves[method], label=method, linewidth=2)
        axes2[2].set_title('Optimization Convergence Curves', fontsize=12, fontweight='bold')
        axes2[2].set_xlabel('Iteration')
        axes2[2].set_ylabel('Fitness (MSE + Penalty)')
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3)
        
        # Plot 4: Feature Selection Heatmap
        feature_matrix = []
        method_names = []
        
        for method in ['WOA', 'PSO', 'GWO']:
            if method in self.feature_selections:
                feature_vector = np.zeros(self.X_train.shape[1])
                selected_features = self.feature_selections[method]
                feature_vector[selected_features] = 1
                feature_matrix.append(feature_vector[:50])  # Show first 50 features
                method_names.append(method)
        
        if feature_matrix:
            sns.heatmap(feature_matrix, xticklabels=False, yticklabels=method_names, 
                       cmap='RdYlBu_r', cbar_kws={'label': 'Feature Selected'}, ax=axes2[3])
            axes2[3].set_title('Feature Selection Patterns\n(First 50 Features)', fontsize=12, fontweight='bold')
        
        plt.suptitle('SVM Feature Analysis & Algorithm Convergence', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # =====================================
        # FIGURE 3: Prediction Analysis (2x2)
        # =====================================
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
        axes3 = axes3.flatten()
        
        # Actual vs Predicted plots for each method
        for idx, method in enumerate(methods):
            y_pred = self.results[method]['Predictions']
            axes3[idx].scatter(self.y_test, y_pred, alpha=0.6, s=20)
            axes3[idx].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes3[idx].set_title(f'Actual vs Predicted ({method})', fontsize=12, fontweight='bold')
            axes3[idx].set_xlabel('Actual Magnitude')
            axes3[idx].set_ylabel('Predicted Magnitude')
            axes3[idx].grid(True, alpha=0.3)
            
            # Add R² score to plot
            r2_score_val = self.results[method]['R2']
            axes3[idx].text(0.05, 0.95, f'R² = {r2_score_val:.4f}', 
                           transform=axes3[idx].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
        
        plt.suptitle('SVM: Actual vs Predicted Earthquake Magnitude', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def save_results(self):
        """Save all results to files"""
        # Save comparison summary
        comparison_data = []
        for method, results in self.results.items():
            comparison_data.append({
                'Method': method,
                'MSE': results['MSE'],
                'MAE': results['MAE'],
                'R²': results['R2'],
                'Features_Used': results['Features_Used'],
                'Feature_Reduction_%': results['Feature_Reduction'],
                'Execution_Time_s': self.execution_times[method]
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv('svm_optimization_comparison_results.csv', index=False)
        
        # Save selected features for each method
        for method, features in self.feature_selections.items():
            if method != 'Original_SVM':
                pd.DataFrame({
                    'Feature_Index': features,
                    'Feature_Name': [f'Feature_{i}' for i in features]
                }).to_csv(f'{method.lower()}_svm_selected_features.csv', index=False)
        
        # Save convergence curves
        convergence_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.convergence_curves.items()]))
        convergence_df.to_csv('svm_convergence_curves.csv', index=False)
        
        # Save models
        for method, results in self.results.items():
            joblib.dump(results['Model'], f'{method.lower()}_svm_model.pkl')
        
        print("\n" + "="*60)
        print("RESULTS SAVED:")
        print("  • svm_optimization_comparison_results.csv")
        print("  • [algorithm]_svm_selected_features.csv (for WOA, PSO, GWO)")
        print("  • svm_convergence_curves.csv")
        print("  • [method]_svm_model.pkl (all models)")
        print("="*60)

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    # Load and preprocess data
    print("Loading and preprocessing earthquake data...")
    
    # Update this path to match your file location
    data = pd.read_csv("/Users/rishitadhulipalla/Desktop/progsvs/earthquake_1995-2023.csv")
    data = data.drop(columns=['title','date_time'])

    string_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'magnitude' in string_cols:
        string_cols.remove('magnitude')

    data = pd.get_dummies(data, columns=string_cols)

    X = data.drop(columns=['magnitude'])
    y = data['magnitude']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Initialize comparison framework
    comparison = SVMOptimizationComparison(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Run all comparisons
    comparison.run_all_comparisons()
    
    # Create summary and visualizations
    df_results = comparison.create_comparison_summary()
    comparison.visualize_comparisons(df_results)
    
    # Save results
    comparison.save_results()
    
    print("\n" + "#"*80)
    print("COMPREHENSIVE SVM COMPARISON COMPLETE!")
    print("Check the generated CSV files and saved models for detailed results.")
    print("#"*80)

if __name__ == "__main__":
    main()
