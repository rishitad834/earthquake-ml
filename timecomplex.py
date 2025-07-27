import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import re
warnings.filterwarnings('ignore')

# ========================================
# METAHEURISTIC ALGORITHM CLASSES
# ========================================

class WhaleOptimizationAlgorithm:
    def __init__(self, n_whales=20, max_iter=30, dim=None):  # Reduced iterations for time analysis
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
    
    def fitness_function(self, solution, X_train, X_test, y_train, y_test, model_type='rf'):
        selected_features = np.where(solution == 1)[0]
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            X_train_selected = X_train.iloc[:, selected_features] if hasattr(X_train, 'iloc') else X_train[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features] if hasattr(X_test, 'iloc') else X_test[:, selected_features]
            
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            elif model_type == 'lgbm':
                model = lgbm.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVR(kernel='rbf', C=1.0)
            elif model_type == 'dt':
                model = DecisionTreeRegressor(random_state=42, max_depth=10)
            
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness = mse + feature_penalty
            
            return fitness
        except:
            return float('inf')
    
    def optimize(self, X_train, X_test, y_train, y_test, model_type='rf'):
        self.dim = X_train.shape[1]
        population = self.initialize_population()
        fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test, model_type) 
                                 for whale in population])
        
        best_idx = np.argmin(fitness_values)
        self.best_whale = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        
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
            
            fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test, model_type) 
                                     for whale in population])
            
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < self.best_fitness:
                self.best_whale = population[current_best_idx].copy()
                self.best_fitness = fitness_values[current_best_idx]
        
        return self.best_whale, self.best_fitness

class ParticleSwarmOptimization:
    def __init__(self, n_particles=20, max_iter=30, w=0.9, c1=2.0, c2=2.0, dim=None):
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
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test, model_type='rf'):
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            X_train_selected = X_train.iloc[:, selected_features] if hasattr(X_train, 'iloc') else X_train[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features] if hasattr(X_test, 'iloc') else X_test[:, selected_features]
            
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            elif model_type == 'lgbm':
                model = lgbm.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVR(kernel='rbf', C=1.0)
            elif model_type == 'dt':
                model = DecisionTreeRegressor(random_state=42, max_depth=10)
            
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
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
    
    def optimize(self, X_train, X_test, y_train, y_test, model_type='rf'):
        self.dim = X_train.shape[1]
        self.initialize_swarm()
        
        for i in range(self.n_particles):
            fitness = self.fitness_function(self.particles_pos[i], X_train, X_test, y_train, y_test, model_type)
            self.particles_best_fitness[i] = fitness
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_pos = self.particles_pos[i].copy()
        
        for iteration in range(self.max_iter):
            self.w = 0.9 - (0.9 - 0.4) * iteration / self.max_iter
            
            for i in range(self.n_particles):
                self.update_velocity(i)
                self.update_position(i)
                
                fitness = self.fitness_function(self.particles_pos[i], X_train, X_test, y_train, y_test, model_type)
                
                if fitness < self.particles_best_fitness[i]:
                    self.particles_best_fitness[i] = fitness
                    self.particles_best_pos[i] = self.particles_pos[i].copy()
                    
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_pos = self.particles_pos[i].copy()
        
        best_binary = self.position_to_binary(self.global_best_pos)
        return best_binary, self.global_best_fitness

class GreyWolfOptimizer:
    def __init__(self, n_wolves=20, max_iter=30, dim=None):
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
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test, model_type='rf'):
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            X_train_selected = X_train.iloc[:, selected_features] if hasattr(X_train, 'iloc') else X_train[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features] if hasattr(X_test, 'iloc') else X_test[:, selected_features]
            
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            elif model_type == 'lgbm':
                model = lgbm.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVR(kernel='rbf', C=1.0)
            elif model_type == 'dt':
                model = DecisionTreeRegressor(random_state=42, max_depth=10)
            
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
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
    
    def optimize(self, X_train, X_test, y_train, y_test, model_type='rf'):
        self.dim = X_train.shape[1]
        self.initialize_population()
        
        for i in range(self.n_wolves):
            fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test, model_type)
            self.update_alpha_beta_delta(fitness, self.positions[i], i)
        
        for iteration in range(self.max_iter):
            a = 2 - 2 * iteration / self.max_iter
            
            for i in range(self.n_wolves):
                self.update_position(i, a)
                fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test, model_type)
                self.update_alpha_beta_delta(fitness, self.positions[i], i)
        
        best_binary = self.position_to_binary(self.alpha_pos)
        return best_binary, self.alpha_score

# ========================================
# TIME COMPLEXITY ANALYSIS FRAMEWORK
# ========================================

class TimeComplexityAnalyzer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.time_results = {}
        
    def clean_column_name(self, name):
        """Clean column names for LightGBM compatibility"""
        return re.sub(r'[^\w]+', '_', name)
    
    def run_baseline_model(self, model_type):
        """Run baseline model and measure time"""
        print(f"Running {model_type.upper()} baseline...")
        start_time = time.time()
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            param_grid = {'max_depth': [10, 20], 'min_samples_split': [2, 5]}
        elif model_type == 'lgbm':
            model = lgbm.LGBMRegressor(random_state=42, verbose=-1)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        elif model_type == 'svm':
            model = SVR()
            param_grid = {'kernel': ['rbf'], 'C': [1, 10], 'gamma': ['scale']}
        elif model_type == 'dt':
            model = DecisionTreeRegressor(random_state=42)
            param_grid = {'max_depth': [10, 20], 'min_samples_split': [2, 5]}
        
        # Use smaller CV for time analysis
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
        
        if model_type == 'lgbm':
            # Clean column names for LightGBM
            X_train_clean = self.X_train.copy()
            X_train_clean.columns = [self.clean_column_name(col) for col in X_train_clean.columns]
            grid_search.fit(X_train_clean, self.y_train)
        else:
            grid_search.fit(self.X_train, self.y_train)
        
        execution_time = time.time() - start_time
        return execution_time
    
    def run_metaheuristic_model(self, model_type, algorithm_name, optimizer_class, **kwargs):
        """Run metaheuristic optimization and measure time"""
        print(f"Running {model_type.upper()} with {algorithm_name}...")
        start_time = time.time()
        
        # Prepare data based on model type
        if model_type == 'lgbm':
            X_train_proc = self.X_train.copy()
            X_test_proc = self.X_test.copy()
            X_train_proc.columns = [self.clean_column_name(col) for col in X_train_proc.columns]
            X_test_proc.columns = [self.clean_column_name(col) for col in X_test_proc.columns]
        elif model_type == 'svm':
            # SVM needs scaled data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            X_train_proc = pd.DataFrame(X_train_scaled, columns=self.X_train.columns)
            X_test_proc = pd.DataFrame(X_test_scaled, columns=self.X_test.columns)
        else:
            X_train_proc = self.X_train
            X_test_proc = self.X_test
        
        # Run optimization
        optimizer = optimizer_class(**kwargs)
        selected_features_binary, best_fitness = optimizer.optimize(
            X_train_proc, X_test_proc, self.y_train, self.y_test, model_type
        )
        
        # Train final model with selected features
        selected_indices = np.where(selected_features_binary == 1)[0]
        
        if len(selected_indices) > 0:
            X_train_selected = X_train_proc.iloc[:, selected_indices]
            
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            elif model_type == 'lgbm':
                model = lgbm.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVR(kernel='rbf', C=1.0)
            elif model_type == 'dt':
                model = DecisionTreeRegressor(random_state=42, max_depth=10)
            
            model.fit(X_train_selected, self.y_train)
        
        execution_time = time.time() - start_time
        return execution_time
    
    def analyze_all_models(self):
        """Analyze time complexity for all models and algorithms"""
        models = ['rf', 'lgbm', 'svm', 'dt']
        algorithms = {
            'WOA': WhaleOptimizationAlgorithm,
            'PSO': ParticleSwarmOptimization,
            'GWO': GreyWolfOptimizer
        }
        
        print("=" * 80)
        print("TIME COMPLEXITY ANALYSIS FOR ALL MODELS")
        print("=" * 80)
        
        for model_type in models:
            print(f"\n--- Analyzing {model_type.upper()} ---")
            
            # Initialize results for this model
            self.time_results[model_type] = {}
            
            # Run baseline
            baseline_time = self.run_baseline_model(model_type)
            self.time_results[model_type]['Original'] = baseline_time
            print(f"Original {model_type.upper()}: {baseline_time:.2f} seconds")
            
            # Run each metaheuristic
            for alg_name, alg_class in algorithms.items():
                try:
                    if alg_name == 'WOA':
                        exec_time = self.run_metaheuristic_model(model_type, alg_name, alg_class, 
                                                               n_whales=20, max_iter=30)
                    elif alg_name == 'PSO':
                        exec_time = self.run_metaheuristic_model(model_type, alg_name, alg_class, 
                                                               n_particles=20, max_iter=30)
                    elif alg_name == 'GWO':
                        exec_time = self.run_metaheuristic_model(model_type, alg_name, alg_class, 
                                                               n_wolves=20, max_iter=30)
                    
                    self.time_results[model_type][alg_name] = exec_time
                    print(f"{alg_name} + {model_type.upper()}: {exec_time:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error with {alg_name} + {model_type.upper()}: {e}")
                    self.time_results[model_type][alg_name] = 0
    
    def create_time_complexity_plots(self):
        """Create time complexity visualization plots"""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        model_names = {
            'rf': 'Random Forest',
            'lgbm': 'LightGBM', 
            'svm': 'Support Vector Machine',
            'dt': 'Decision Tree'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Create 4 separate plots - one for each model
        for idx, (model_key, model_name) in enumerate(model_names.items()):
            if model_key not in self.time_results:
                continue
                
            plt.figure(figsize=(10, 6))
            
            methods = list(self.time_results[model_key].keys())
            times = list(self.time_results[model_key].values())
            
            # Create bar plot
            bars = plt.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Customize the plot
            plt.title(f'Time Complexity Analysis: {model_name}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Method', fontsize=12, fontweight='bold')
            plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                         f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
            
            # Customize grid and layout
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.xticks(rotation=0, fontsize=11)
            plt.yticks(fontsize=11)
            
            # Add some styling
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(0.5)
            plt.gca().spines['bottom'].set_linewidth(0.5)
            
            plt.tight_layout()
            plt.show()
    
    def create_summary_comparison(self):
        """Create a comprehensive summary comparison"""
        print("\n" + "=" * 80)
        print("TIME COMPLEXITY SUMMARY")
        print("=" * 80)
        
        # Create summary DataFrame
        summary_data = []
        
        for model_type, times in self.time_results.items():
            for method, exec_time in times.items():
                summary_data.append({
                    'Model': model_type.upper(),
                    'Method': method,
                    'Execution_Time': exec_time,
                    'Relative_Speed': exec_time / times.get('Original', 1.0)
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Print summary table
        pivot_table = df_summary.pivot(index='Method', columns='Model', values='Execution_Time')
        print("\nExecution Times (seconds):")
        print(pivot_table.round(2))
        
        # Find fastest and slowest for each model
        print("\n" + "-" * 50)
        print("PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        for model in pivot_table.columns:
            model_times = pivot_table[model].dropna()
            fastest_method = model_times.idxmin()
            slowest_method = model_times.idxmax()
            fastest_time = model_times.min()
            slowest_time = model_times.max()
            
            print(f"\n{model}:")
            print(f"  Fastest: {fastest_method} ({fastest_time:.2f}s)")
            print(f"  Slowest: {slowest_method} ({slowest_time:.2f}s)")
            print(f"  Speed Difference: {slowest_time/fastest_time:.2f}x")
        
        # Create overall comparison plot
        plt.figure(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        models = pivot_table.columns
        methods = pivot_table.index
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, method in enumerate(methods):
            times = [pivot_table.loc[method, model] if not pd.isna(pivot_table.loc[method, model]) else 0 
                    for model in models]
            plt.bar(x + i*width, times, width, label=method, alpha=0.8)
        
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Time Complexity Comparison: All Models vs All Methods', fontsize=14, fontweight='bold')
        plt.xticks(x + width * 1.5, models)
        plt.legend(title='Methods', title_fontsize=11, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df_summary

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

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize time complexity analyzer
    analyzer = TimeComplexityAnalyzer(X_train, X_test, y_train, y_test)
    
    # Run analysis for all models
    analyzer.analyze_all_models()
    
    # Create individual model plots
    analyzer.create_time_complexity_plots()
    
    # Create summary comparison
    summary_df = analyzer.create_summary_comparison()
    
    # Save results
    summary_df.to_csv('time_complexity_analysis_results.csv', index=False)
    print(f"\nTime complexity analysis results saved to 'time_complexity_analysis_results.csv'")

if __name__ == "__main__":
    main()
