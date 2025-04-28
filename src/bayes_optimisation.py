
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
import pandas as pd
from tqdm.notebook import tqdm
import warnings
from .models import CA


class BayesianParameterEstimation:
    def __init__(self, fire_name, grid_size=(100, 100)):
        """Initialize the Bayesian parameter estimation for CA wildfire model
        
        Parameters:
        - fire_name: str, name of the fire dataset to use (e.g., 'arizona', 'alabama')
        - grid_size: tuple, dimensions of the grid (rows, cols)
        """
        self.fire_name = fire_name
        self.grid_size = grid_size
        self.ca_model = None
        self.trace = None
        self.parameter_ranges = {
            'c1': (0.1, 1.0),  # Wind direction effect parameter
            'c2': (0.1, 1.0),  # Wind speed effect parameter
            'p0': (0.1, 1.0),  # Base ignition probability 
            'p1': (0.1, 0.5),  # Percolation fuel density parameter
            'p2': (0.05, 0.2)  # Percolation vegetation diversity parameter
        }
        
    def setup_ca_model(self, params):
        """Set up the CA model with given parameters
        
        Parameters:
        - params: dict, model parameters (c1, c2, p0, p1, p2)
        
        Returns:
        - CA model instance
        """
        # Initialize CA model with parameters
        ca_model = CA(grid_size=self.grid_size, params=params)
        
        # Load data for the specified fire
        ca_model.load_mtbs_fire_data(self.fire_name)
        
        # Set environmental conditions (default values, can be adjusted)
        ca_model.set_environmental_data(
            wind_speed=5.0,       # m/s
            wind_direction=90.0,  # degrees
            temperature=85.0,     # degrees F
            humidity=20.0,        # percent
            fire_direction=90.0   # degrees
        )
        
        return ca_model
    
    def evaluate_model(self, params, simulation_steps=50):
        """Evaluate model with given parameters
        
        Parameters:
        - params: dict containing model parameters
        - simulation_steps: int, number of time steps to simulate
        
        Returns:
        - metrics: dict, evaluation metrics including sorensen index, accuracy, etc.
        """
        # Set up CA model with parameters
        ca_model = self.setup_ca_model(params)
        
        # Run simulation
        ca_model.run_simulation(steps=simulation_steps, stochastic=True)
        
        # Calculate evaluation metrics
        accuracy, precision, recall, sorensen = ca_model.evaluate_simulation()
        
        return {
            'sorensen': sorensen,
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall
        }
    
    def log_likelihood(self, observed, params, simulation_steps=50):
        """Calculate log likelihood of parameters given observed data
        
        Parameters:
        - observed: The observed burned area
        - params: dict, model parameters
        - simulation_steps: int, number of simulation steps
        
        Returns:
        - log_likelihood: float, log likelihood of parameters
        """
        try:
            metrics = self.evaluate_model(params, simulation_steps)
            
            # Use Sorensen index (Dice coefficient) as the basis for likelihood
            # Higher sorensen = higher likelihood
            sorensen = metrics['sorensen']
            
            # Convert Sorensen index to a likelihood value
            # We'll use a beta distribution centered on the Sorensen score
            # This gives higher probability to parameter sets that produce
            # simulations more similar to observed data
            alpha = 2.0
            beta = 1.0 - sorensen
            
            # Avoid numerical issues
            if beta < 0.01:
                beta = 0.01
                
            # Calculate log likelihood using beta distribution
            log_like = stats.beta.logpdf(0.9, alpha, beta)
            
            return log_like
        except Exception as e:
            print(f"Error in log_likelihood: {str(e)}")
            return -np.inf  # Return negative infinity for failed simulations
    
    def run_mcmc(self, num_samples=1000, tune=500, chains=2, cores=1, simulation_steps=50):
        """Run MCMC to estimate posterior distribution of parameters
        
        Parameters:
        - num_samples: int, number of samples to draw
        - tune: int, number of tuning steps
        - chains: int, number of MCMC chains
        - cores: int, number of CPU cores to use
        - simulation_steps: int, number of simulation steps for each evaluation
        
        Returns:
        - trace: PyMC trace object containing parameter samples
        """
        with pm.Model() as model:
            # Define priors for model parameters
            c1 = pm.Uniform('c1', 
                         lower=self.parameter_ranges['c1'][0], 
                         upper=self.parameter_ranges['c1'][1])
            
            c2 = pm.Uniform('c2', 
                         lower=self.parameter_ranges['c2'][0], 
                         upper=self.parameter_ranges['c2'][1])
            
            p0 = pm.Uniform('p0', 
                         lower=self.parameter_ranges['p0'][0], 
                         upper=self.parameter_ranges['p0'][1])
            
            p1 = pm.Uniform('p1', 
                         lower=self.parameter_ranges['p1'][0], 
                         upper=self.parameter_ranges['p1'][1])
            
            p2 = pm.Uniform('p2', 
                         lower=self.parameter_ranges['p2'][0], 
                         upper=self.parameter_ranges['p2'][1])
            
            # Create a dummy observed variable (we'll use likelihood function directly)
            observed = pm.MutableData('observed', np.zeros(1))
            
            # Define likelihood function using a PyMC Potential
            def likelihood_function(c1, c2, p0, p1, p2):
                params = {'c1': c1, 'c2': c2, 'p0': p0, 'p1': p1, 'p2': p2}
                return self.log_likelihood(observed, params, simulation_steps)
            
            # Add potential (log-likelihood contribution)
            pm.Potential('likelihood', likelihood_function(c1, c2, p0, p1, p2))
            
            # Sample from the posterior
            self.trace = pm.sample(
                draws=num_samples,
                tune=tune,
                chains=chains,
                cores=cores,
                return_inferencedata=True
            )
            
        return self.trace
    
    def run_sequential_monte_carlo(self, n_particles=1000, steps=5, simulation_steps=50):
        """Run Sequential Monte Carlo for parameter estimation
        
        This is an alternative to MCMC that can be more efficient for complex models
        
        Parameters:
        - n_particles: int, number of particles (parameter sets)
        - steps: int, number of SMC steps
        - simulation_steps: int, number of simulation steps for each evaluation
        
        Returns:
        - best_params: dict, best parameter set found
        - results: pandas DataFrame with all parameter sets and scores
        """
        # Initialize particles from prior distributions
        particles = {
            'c1': np.random.uniform(
                self.parameter_ranges['c1'][0],
                self.parameter_ranges['c1'][1],
                n_particles
            ),
            'c2': np.random.uniform(
                self.parameter_ranges['c2'][0],
                self.parameter_ranges['c2'][1], 
                n_particles
            ),
            'p0': np.random.uniform(
                self.parameter_ranges['p0'][0],
                self.parameter_ranges['p0'][1],
                n_particles
            ),
            'p1': np.random.uniform(
                self.parameter_ranges['p1'][0],
                self.parameter_ranges['p1'][1],
                n_particles
            ),
            'p2': np.random.uniform(
                self.parameter_ranges['p2'][0],
                self.parameter_ranges['p2'][1],
                n_particles
            ),
            'score': np.zeros(n_particles)
        }
        
        results = []
        best_score = -np.inf
        best_params = None
        
        # Run SMC iterations
        for step in range(steps):
            print(f"SMC Step {step+1}/{steps}")
            
            # Evaluate each particle
            for i in tqdm(range(n_particles)):
                params = {
                    'c1': particles['c1'][i],
                    'c2': particles['c2'][i],
                    'p0': particles['p0'][i],
                    'p1': particles['p1'][i],
                    'p2': particles['p2'][i]
                }
                
                try:
                    metrics = self.evaluate_model(params, simulation_steps)
                    particles['score'][i] = metrics['sorensen']
                    
                    # Record this parameter set and its score
                    param_record = params.copy()
                    param_record.update(metrics)
                    param_record['step'] = step
                    results.append(param_record)
                    
                    # Update best parameters
                    if metrics['sorensen'] > best_score:
                        best_score = metrics['sorensen']
                        best_params = params.copy()
                        print(f"New best score: {best_score:.4f} with params: {best_params}")
                
                except Exception as e:
                    print(e)
            
            # Stop early if we've reached a very good fit
            if best_score > 0.95:
                print(f"Stopping early with very good fit: {best_score:.4f}")
                break
                
            # For all steps except the last, update particles through resampling and perturbation
            if step < steps - 1:
                # Calculate weights based on scores
                weights = np.exp(particles['score'] * 10)  # Exponentiate to emphasize good scores
                weights = weights / np.sum(weights)  # Normalize
                
                # Resample particles based on weights
                indices = np.random.choice(n_particles, size=n_particles, p=weights)
                
                # Create new particle set
                new_particles = {
                    'c1': particles['c1'][indices],
                    'c2': particles['c2'][indices],
                    'p0': particles['p0'][indices],
                    'p1': particles['p1'][indices],
                    'p2': particles['p2'][indices],
                    'score': np.zeros(n_particles)
                }
                
                # Add perturbation
                for param in ['c1', 'c2', 'p0', 'p1', 'p2']:
                    # Calculate perturbation scale based on parameter range
                    param_range = self.parameter_ranges[param][1] - self.parameter_ranges[param][0]
                    scale = param_range * 0.1 * (1 - (step / steps))  # Reduce perturbation over time
                    
                    # Add Gaussian noise
                    new_particles[param] += np.random.normal(0, scale, n_particles)
                    
                    # Keep within bounds
                    new_particles[param] = np.clip(
                        new_particles[param],
                        self.parameter_ranges[param][0],
                        self.parameter_ranges[param][1]
                    )
                
                particles = new_particles
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        return best_params, results_df
    
    def visualize_results(self, results_df=None, trace=None):
        """Visualize parameter estimation results
        
        Parameters:
        - results_df: pandas DataFrame from SMC method
        - trace: PyMC trace from MCMC method
        """
        if results_df is not None:
            # Create scatter plots for each parameter vs score
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, param in enumerate(['c1', 'c2', 'p0', 'p1', 'p2']):
                axes[i].scatter(results_df[param], results_df['sorensen'], alpha=0.5)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Sorensen Index')
                axes[i].set_title(f'{param} vs. Sorensen Index')
                
            # Remove the empty subplot
            fig.delaxes(axes[5])
            
            plt.tight_layout()
            plt.show()
            
            # Plot parameter pairs
            param_pairs = [('c1', 'c2'), ('p0', 'p1'), ('p1', 'p2')]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, (p1, p2) in enumerate(param_pairs):
                scatter = axes[i].scatter(
                    results_df[p1], 
                    results_df[p2], 
                    c=results_df['sorensen'], 
                    cmap='viridis', 
                    alpha=0.7
                )
                axes[i].set_xlabel(p1)
                axes[i].set_ylabel(p2)
                axes[i].set_title(f'{p1} vs {p2} (color = Sorensen Index)')
                
                fig.colorbar(scatter, ax=axes[i])
            
            plt.tight_layout()
            plt.show()
            
            # Plot progression of best score over SMC steps
            best_per_step = results_df.groupby('step')['sorensen'].max().reset_index()
            plt.figure(figsize=(10, 6))
            plt.plot(best_per_step['step'], best_per_step['sorensen'], marker='o')
            plt.xlabel('SMC Step')
            plt.ylabel('Best Sorensen Index')
            plt.title('Convergence of Parameter Estimation')
            plt.grid(True)
            plt.show()
        
        if trace is not None:
            # Plot MCMC trace and posterior distributions
            az.plot_trace(trace)
            plt.tight_layout()
            plt.show()
            
            # Plot posterior pair plots
            az.plot_pair(trace)
            plt.show()
    
    def evaluate_best_parameters(self, params, simulation_steps=100):
        """Evaluate the model with the best parameters and visualize results
        
        Parameters:
        - params: dict, best parameters found
        - simulation_steps: int, number of simulation steps
        
        Returns:
        - metrics: dict, evaluation metrics
        """
        print(f"Evaluating model with parameters: {params}")
        
        # Initialize CA model with best parameters
        ca_model = self.setup_ca_model(params)
        
        # Run the simulation
        history = ca_model.run_simulation(steps=simulation_steps)
        print(f"Simulation completed in {len(history)} steps")
        
        # Visualize the simulation
        ca_model.visualize_simulation(history)
        
        # Compare with actual burned area
        ca_model.overlay_simulation_with_actual(self.fire_name)
        
        # Calculate metrics
        metrics = self.evaluate_model(params, simulation_steps)
        print(f"Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
