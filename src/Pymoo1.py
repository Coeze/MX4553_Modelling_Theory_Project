# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
from scipy.stats import pearsonr

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Add src directory to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import the model
from model import CA

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define available fire datasets
fires = ['alabama', 'arizona']

# Define parameter ranges
PARAM_RANGES = {
    'p0': (0.1, 0.9),   # Base ignition probability
    'c1': (0.1, 1.0),   # Wind effect parameter 1
    'c2': (0.1, 1.0)    # Wind effect parameter 2
}

def init_model_from_fire_data(fire, grid_size=(100, 100)):
    """Initialize a CA model from fire data"""
    # Create a new CA model with initial parameters
    model = CA(grid_size=grid_size)
    
    # Initialize from MTBS data
    success = model.initialise_ndvi_from_data(fire)
    if success:
        model.load_mtbs_fire_data(fire)
    
    return model if success else None

def evaluate_model_performance(model, simulation_steps=20, params=None):
    """Run a model simulation and evaluate performance against actual fire data"""
    if params:
        # Set model parameters
        model.p0 = params.get('p0', 0.5)
        model.c1 = params.get('c1', 0.5)
        model.c2 = params.get('c2', 0.5)

    # Run the simulation
    history = model.run_simulation(simulation_steps)

    # Compare with actual burned area
    if model.actual_burned_area is not None:
        simulated_burned = (model.grid == 2).astype(int)  # Cells with state 2 are burnt

        # Calculate Sørensen index (Dice coefficient)
        true_positives = np.sum((simulated_burned == 1) & (model.actual_burned_area == 1))
        false_positives = np.sum((simulated_burned == 1) & (model.actual_burned_area == 0))
        false_negatives = np.sum((simulated_burned == 0) & (model.actual_burned_area == 1))

        sorensen = 2 * true_positives / (2 * true_positives + false_positives + false_negatives) if (2 * true_positives + false_positives + false_negatives) > 0 else 0

        return sorensen
    else:
        return 0.0  # No actual data to compare with

# Define Pymoo Problem class for the optimization
class FireModelCalibration(Problem):
    def __init__(self, fire_folder, simulation_steps=20):
        self.fire_folder = fire_folder
        self.simulation_steps = simulation_steps
        
        # Define parameter ranges (3 parameters: p0, c1, c2)
        n_var = 3
        xl = np.array([PARAM_RANGES['p0'][0], PARAM_RANGES['c1'][0], PARAM_RANGES['c2'][0]])  # Lower bounds
        xu = np.array([PARAM_RANGES['p0'][1], PARAM_RANGES['c1'][1], PARAM_RANGES['c2'][1]])  # Upper bounds
        
        # Single objective: maximize Sørensen index
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        n_points = x.shape[0]
        f = np.zeros((n_points, 1))
        
        for i in range(n_points):
            # Convert design variables to parameter dictionary
            params = {
                'p0': x[i, 0],
                'c1': x[i, 1],
                'c2': x[i, 2]
            }
            
            # Initialize model
            model = init_model_from_fire_data(self.fire_folder)
            if not model:
                f[i, 0] = 1.0  # Penalty for failed initialization (to be minimized)
                continue
                
            # Evaluate with these parameters
            sorensen = evaluate_model_performance(model, self.simulation_steps, params)
            
            # Since Pymoo minimizes by default, we negate the Sørensen index
            f[i, 0] = -sorensen
            
        out["F"] = f

def run_genetic_algorithm(fire_folder, pop_size=30, n_gen=10, simulation_steps=20):
    """Run Pymoo genetic algorithm to optimize model parameters"""
    print(f"Running genetic algorithm optimization for {fire_folder}...")
    
    # Create problem
    problem = FireModelCalibration(fire_folder, simulation_steps)
    
    # Define algorithm
    algorithm = GA(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        selection=TournamentSelection(pressure=3),
        eliminate_duplicates=True
    )
    
    # Define termination
    termination = DefaultSingleObjectiveTermination(
        x_tol=1e-4,
        cv_tol=1e-6,
        f_tol=0.0025,
        n_max_gen=n_gen,
        n_max_evals=n_gen * pop_size
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        save_history=True,
        verbose=True
    )
    
    # Extract best parameters
    best_params = {
        'p0': res.X[0],
        'c1': res.X[1],
        'c2': res.X[2]
    }
    
    # Fitness is negated in the problem, so we negate it back
    best_fitness = -res.F[0]
    
    print(f"\nOptimization complete.")
    print(f"Best parameters: {best_params}")
    print(f"Best Sørensen index: {best_fitness:.4f}")
    
    # Plot convergence
    if len(res.history) > 0:
        n_gen = len(res.history)
        hist_F = np.array([-e.opt.get("F")[0] for e in res.history])
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(n_gen), hist_F, 'o-')
        plt.title("Genetic Algorithm Optimization Progress")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Sørensen Index)")
        plt.grid(True)
        plt.show()
    
    return best_params, best_fitness, res

def cross_validate_parameters(params, fire_folders, simulation_steps=20):
    """Cross-validate parameters on multiple fire datasets"""
    print(f"Cross-validating parameters on {len(fire_folders)} fire datasets...")

    results = {}
    for fire_folder in fire_folders:
        print(f"\nValidating on {fire_folder}...")

        # Initialize model
        model = init_model_from_fire_data(fire_folder)
        if not model:
            print(f"Could not initialize model for {fire_folder}")
            continue

        # Evaluate with given parameters
        sorensen = evaluate_model_performance(model, simulation_steps, params)
        print(f"Sørensen index: {sorensen:.4f}")

        results[fire_folder] = sorensen

    # Calculate average performance
    if results:
        avg_sorensen = sum(results.values()) / len(results)
        print(f"\nAverage Sørensen index across all datasets: {avg_sorensen:.4f}")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.axhline(y=avg_sorensen, color='r', linestyle='--', label=f"Average: {avg_sorensen:.4f}")
        plt.xlabel("Fire Dataset")
        plt.ylabel("Sørensen Index")
        plt.title("Parameter Performance Across Fire Datasets")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results, avg_sorensen if results else 0.0

def parameter_sensitivity_analysis(fire_folder, base_params, parameter_name, value_range, steps=10, simulation_steps=20):
    """Analyze the sensitivity of model to a specific parameter"""
    print(f"Running sensitivity analysis for {parameter_name} on {fire_folder}...")

    values = np.linspace(value_range[0], value_range[1], steps)
    scores = []

    for value in values:
        # Create new parameter set with the changed parameter
        params = base_params.copy()
        params[parameter_name] = value

        # Initialize model
        model = init_model_from_fire_data(fire_folder)
        if not model:
            print(f"Could not initialize model")
            scores.append(0.0)
            continue

        # Evaluate with this parameter value
        sorensen = evaluate_model_performance(model, simulation_steps, params)
        scores.append(sorensen)
        print(f"{parameter_name} = {value:.3f} -> Sørensen = {sorensen:.4f}")

    # Plot sensitivity curve
    plt.figure(figsize=(10, 6))
    plt.plot(values, scores, 'o-')
    plt.axvline(x=base_params[parameter_name], color='r', linestyle='--',
                label=f"Optimized value: {base_params[parameter_name]:.3f}")
    plt.xlabel(parameter_name)
    plt.ylabel("Sørensen Index")
    plt.title(f"Sensitivity Analysis for {parameter_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return values, scores

def save_parameters(params, filename="optimized_parameters.json"):
    """Save parameters to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {filename}")

def load_parameters(filename="optimized_parameters.json"):
    """Load parameters from a JSON file"""
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        print(f"Parameters loaded from {filename}")
        return params
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

# Main execution block
if __name__ == "__main__":
    # Select fire dataset for calibration
    calibration_fire = fires[0]  # alabama
    
    # Run Pymoo optimization with genetic algorithm
    best_params, best_fitness, res = run_genetic_algorithm(
        fire_folder=calibration_fire,
        pop_size=20,  # Small population for demonstration
        n_gen=5,      # Few generations for demonstration
        simulation_steps=20
    )
    
    # Cross-validate optimized parameters on all fire datasets
    results, avg_performance = cross_validate_parameters(
        params=best_params,
        fire_folders=fires,
        simulation_steps=20
    )
    
    # Run sensitivity analysis for each parameter
    for param_name, param_range in PARAM_RANGES.items():
        values, scores = parameter_sensitivity_analysis(
            fire_folder=calibration_fire,
            base_params=best_params,
            parameter_name=param_name,
            value_range=param_range,
            steps=8,  # Fewer steps for demonstration
            simulation_steps=20
        )
    
    # Save the optimized parameters
    save_parameters(best_params, filename="optimized_parameters.json")