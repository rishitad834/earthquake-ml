import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
import lightgbm as lgbm
import joblib
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
    def __init__(self, n_whales=20, max_iter=30, dim=None):
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
            if model_type == 'rf':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            elif model_type == 'lgbm':
                model = lgbm.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVC(kernel='rbf', probability=True, random_state=42)
            elif model_type == 'dt':
                model = DecisionTreeClassifier(random_state=42)
            
            if model_type in ['rf', 'lgbm', 'dt']:
                X_train_selected = X_train.iloc[:, selected_features]
                X_test_selected = X_test.iloc[:, selected_features]
            else:  # SVM with numpy arrays
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]
            
            model.fit(X_train_selected, y_train)
            y_pred_proba = model.predict_proba(X_test_selected)
            
            # Calculate multi-class AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                fitness = 1 - auc_score  # Minimize (1 - AUC)
            except:
                fitness = 1.0  # Fallback if AUC calculation fails
            
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness += feature_penalty
            
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
            
            fitness_values = np.array([self.fitness_function(whale, X_train, X_test, y_train, y_test, model_type) 
                                     for whale in population])
            
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < self.best_fitness:
                self.best_whale = population[current_best_idx].copy()
                self.best_fitness = fitness_values[current_best_idx]
            
            self.convergence_curve.append(self.best_fitness)
        
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
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test, model_type='rf'):
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            if model_type == 'rf':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            elif model_type == 'lgbm':
                model = lgbm.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVC(kernel='rbf', probability=True, random_state=42)
            elif model_type == 'dt':
                model = DecisionTreeClassifier(random_state=42)
            
            if model_type in ['rf', 'lgbm', 'dt']:
                X_train_selected = X_train.iloc[:, selected_features]
                X_test_selected = X_test.iloc[:, selected_features]
            else:  # SVM with numpy arrays
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]
            
            model.fit(X_train_selected, y_train)
            y_pred_proba = model.predict_proba(X_test_selected)
            
            try:
                from sklearn.metrics import roc_auc_score
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                fitness = 1 - auc_score
            except:
                fitness = 1.0
            
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness += feature_penalty
            
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
        
        self.convergence_curve.append(self.global_best_fitness)
        
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
            
            self.convergence_curve.append(self.global_best_fitness)
        
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
    
    def fitness_function(self, position, X_train, X_test, y_train, y_test, model_type='rf'):
        binary_pos = self.position_to_binary(position)
        selected_features = np.where(binary_pos == 1)[0]
        
        if len(selected_features) == 0:
            return float('inf')
        
        try:
            if model_type == 'rf':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            elif model_type == 'lgbm':
                model = lgbm.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            elif model_type == 'svm':
                model = SVC(kernel='rbf', probability=True, random_state=42)
            elif model_type == 'dt':
                model = DecisionTreeClassifier(random_state=42)
            
            if model_type in ['rf', 'lgbm', 'dt']:
                X_train_selected = X_train.iloc[:, selected_features]
                X_test_selected = X_test.iloc[:, selected_features]
            else:  # SVM with numpy arrays
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]
            
            model.fit(X_train_selected, y_train)
            y_pred_proba = model.predict_proba(X_test_selected)
            
            try:
                from sklearn.metrics import roc_auc_score
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                fitness = 1 - auc_score
            except:
                fitness = 1.0
            
            feature_penalty = len(selected_features) / self.dim * 0.1
            fitness += feature_penalty
            
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
        
        self.convergence_curve.append(self.alpha_score)
        
        for iteration in range(self.max_iter):
            a = 2 - 2 * iteration / self.max_iter
            
            for i in range(self.n_wolves):
                self.update_position(i, a)
                fitness = self.fitness_function(self.positions[i], X_train, X_test, y_train, y_test, model_type)
                self.update_alpha_beta_delta(fitness, self.positions[i], i)
            
            self.convergence_curve.append(self.alpha_score)
        
        best_binary = self.position_to_binary(self.alpha_pos)
        return best_binary, self.alpha_score

# ========================================
# ROC/AUC COMPARISON FRAMEWORK
# ========================================

def clean_column_name(name):
    """Clean column names to remove special characters that LightGBM can't handle"""
    return re.sub(r'[^\w]+', '_', name)

class ROCAUCComparison:
    def __init__(self, X_train, X_test, y_train, y_test, X_train_scaled=None, X_test_scaled=None):
        self.X_train = X_train
        self.X_test = X_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions_proba = {}
        
    def run_baseline_models(self):
        """Train baseline models for all types"""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['rf_baseline'] = rf_model
        self.predictions_proba['rf_baseline'] = rf_model.predict_proba(self.X_test)
        
        # LightGBM
        print("Training LightGBM...")
        try:
            lgbm_model = lgbm.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            lgbm_model.fit(self.X_train, self.y_train)
            self.models['lgbm_baseline'] = lgbm_model
            self.predictions_proba['lgbm_baseline'] = lgbm_model.predict_proba(self.X_test)
        except Exception as e:
            print(f"LightGBM error: {e}")
            self.models['lgbm_baseline'] = None
            self.predictions_proba['lgbm_baseline'] = None
        
        # SVM
        print("Training SVM...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(self.X_train_scaled, self.y_train)
        self.models['svm_baseline'] = svm_model
        self.predictions_proba['svm_baseline'] = svm_model.predict_proba(self.X_test_scaled)
        
        # Decision Tree
        print("Training Decision Tree...")
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt_model.fit(self.X_train, self.y_train)
        self.models['dt_baseline'] = dt_model
        self.predictions_proba['dt_baseline'] = dt_model.predict_proba(self.X_test)
        
    def run_optimized_models(self):
        """Train optimized models for all types and algorithms"""
        print("\n" + "="*60)
        print("TRAINING OPTIMIZED MODELS")
        print("="*60)
        
        algorithms = {
            'woa': WhaleOptimizationAlgorithm(n_whales=20, max_iter=20),
            'pso': ParticleSwarmOptimization(n_particles=20, max_iter=20),
            'gwo': GreyWolfOptimizer(n_wolves=20, max_iter=20)
        }
        
        model_types = {
            'rf': (self.X_train, self.X_test),
            'lgbm': (self.X_train, self.X_test),
            'svm': (self.X_train_scaled, self.X_test_scaled),
            'dt': (self.X_train, self.X_test)
        }
        
        for model_name, (X_tr, X_te) in model_types.items():
            print(f"\nOptimizing {model_name.upper()} models...")
            
            for algo_name, optimizer in algorithms.items():
                print(f"  Running {algo_name.upper()}...")
                
                try:
                    selected_features, _ = optimizer.optimize(X_tr, X_te, self.y_train, self.y_test, model_name)
                    selected_indices = np.where(selected_features == 1)[0]
                    
                    if len(selected_indices) > 0:
                        # Train final model with selected features
                        if model_name == 'rf':
                            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                            X_train_sel = X_tr.iloc[:, selected_indices]
                            X_test_sel = X_te.iloc[:, selected_indices]
                        elif model_name == 'lgbm':
                            model = lgbm.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                            X_train_sel = X_tr.iloc[:, selected_indices]
                            X_test_sel = X_te.iloc[:, selected_indices]
                        elif model_name == 'svm':
                            model = SVC(kernel='rbf', probability=True, random_state=42)
                            X_train_sel = X_tr[:, selected_indices]
                            X_test_sel = X_te[:, selected_indices]
                        elif model_name == 'dt':
                            model = DecisionTreeClassifier(random_state=42, max_depth=10)
                            X_train_sel = X_tr.iloc[:, selected_indices]
                            X_test_sel = X_te.iloc[:, selected_indices]
                        
                        model.fit(X_train_sel, self.y_train)
                        y_pred_proba = model.predict_proba(X_test_sel)
                        
                        self.models[f'{model_name}_{algo_name}'] = model
                        self.predictions_proba[f'{model_name}_{algo_name}'] = y_pred_proba
                        
                        print(f"    Features selected: {len(selected_indices)}/{X_tr.shape[1]}")
                    else:
                        print(f"    No features selected by {algo_name.upper()}")
                        
                except Exception as e:
                    print(f"    Error with {algo_name.upper()}: {e}")
                    self.models[f'{model_name}_{algo_name}'] = None
                    self.predictions_proba[f'{model_name}_{algo_name}'] = None
    
    def plot_roc_curves(self):
        """Generate ROC/AUC curves for each model type"""
        plt.style.use('default')
        
        model_types = ['rf', 'lgbm', 'svm', 'dt']
        model_names = ['Random Forest', 'LightGBM', 'SVM', 'Decision Tree']
        methods = ['baseline', 'woa', 'pso', 'gwo']
        method_names = ['Baseline', 'WOA', 'PSO', 'GWO']
        colors = ['red', 'blue', 'green', 'orange']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (model_type, model_name) in enumerate(zip(model_types, model_names)):
            ax = axes[idx]
            
            # Get unique classes and create one-vs-rest ROC curves
            classes = np.unique(self.y_test)
            n_classes = len(classes)
            
            legend_entries = []
            
            for method_idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
                key = f'{model_type}_{method}'
                
                if key in self.predictions_proba and self.predictions_proba[key] is not None:
                    y_pred_proba = self.predictions_proba[key]
                    
                    try:
                        # Calculate micro-average ROC curve and AUC
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import roc_curve, auc
                        
                        # Binarize the output
                        y_test_bin = label_binarize(self.y_test, classes=classes)
                        
                        if n_classes == 2:
                            # Binary classification
                            fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba[:, 1])
                            roc_auc = auc(fpr, tpr)
                        else:
                            # Multi-class: compute micro-average
                            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                            roc_auc = auc(fpr, tpr)
                        
                        ax.plot(fpr, tpr, color=color, lw=2, 
                               label=f'{method_name} (AUC = {roc_auc:.3f})')
                        legend_entries.append(method_name)
                        
                    except Exception as e:
                        print(f"Error plotting ROC for {key}: {e}")
                        continue
                else:
                    print(f"No predictions available for {key}")
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curves')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('ROC/AUC Curves: Baseline vs Metaheuristic Optimization', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def create_auc_comparison_table(self):
        """Create a comparison table of AUC scores"""
        from sklearn.metrics import roc_auc_score
        
        model_types = ['rf', 'lgbm', 'svm', 'dt']
        model_names = ['Random Forest', 'LightGBM', 'SVM', 'Decision Tree']
        methods = ['baseline', 'woa', 'pso', 'gwo']
        
        auc_data = []
        
        for model_type, model_name in zip(model_types, model_names):
            row = {'Model': model_name}
            
            for method in methods:
                key = f'{model_type}_{method}'
                
                if key in self.predictions_proba and self.predictions_proba[key] is not None:
                    try:
                        auc_score = roc_auc_score(self.y_test, self.predictions_proba[key], 
                                                multi_class='ovr', average='weighted')
                        row[method.upper()] = f"{auc_score:.3f}"
                    except:
                        row[method.upper()] = "N/A"
                else:
                    row[method.upper()] = "N/A"
            
            auc_data.append(row)
        
        auc_df = pd.DataFrame(auc_data)
        print("\n" + "="*60)
        print("AUC SCORES COMPARISON TABLE")
        print("="*60)
        print(auc_df.to_string(index=False))
        
        return auc_df

# ========================================
# MAIN EXECUTION
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
    
    # Clean column names for LightGBM compatibility
    data.columns = [clean_column_name(col) for col in data.columns]
    
    # Convert regression to classification by binning magnitudes
    X = data.drop(columns=['magnitude'])
    y_continuous = data['magnitude']
    
    # Create magnitude bins for classification
    bins = [0, 3, 5, 7, 10]  # Magnitude ranges
    bin_labels = ['Low (0-3)', 'Medium (3-5)', 'High (5-7)', 'Very High (7+)']
    y_binned = pd.cut(y_continuous, bins=bins, labels=bin_labels, include_lowest=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_binned)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classification bins: {dict(zip(range(len(bin_labels)), bin_labels))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    
    # Feature Scaling for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize comparison framework
    comparison = ROCAUCComparison(X_train, X_test, y_train, y_test, 
                                 X_train_scaled, X_test_scaled)
    
    # Run baseline models
    comparison.run_baseline_models()
    
    # Run optimized models
    comparison.run_optimized_models()
    
    # Generate ROC/AUC plots
    comparison.plot_roc_curves()
    
    # Create AUC comparison table
    auc_df = comparison.create_auc_comparison_table()
    
    # Save results
    auc_df.to_csv('auc_comparison_results.csv', index=False)
    print(f"\nAUC comparison table saved to 'auc_comparison_results.csv'")
    
    print("\n" + "#"*80)
    print("ROC/AUC ANALYSIS COMPLETE!")
    print("Generated ROC curves for all models with baseline vs optimization comparison.")
    print("#"*80)

if __name__ == "__main__":
    main()
