import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from src.model import CA



def apply_direct_attack_strategy(model, steps):
    """
    Apply a direct attack strategy targeting the fire perimeter
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Setup - first let fire establish slightly
    history = model.run_simulation(2)
    
    # For each step, apply direct attack to the fire perimeter
    for step in range(steps - 2):
        # Find cells that are currently burning
        burning_cells = np.where(model.grid == 1)
        
        if len(burning_cells[0]) == 0:
            # No more burning cells, simulation can end
            break
            
        # Identify the fire perimeter (burning cells adjacent to unburnt cells)
        perimeter_cells = []
        for i in range(len(burning_cells[0])):
            row, col = burning_cells[0][i], burning_cells[1][i]
            
            # Check if this burning cell is adjacent to any unburnt cells
            is_perimeter = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < model.rows and 0 <= nc < model.cols and model.grid[nr, nc] == 0:
                        is_perimeter = True
                        break
                if is_perimeter:
                    break
                    
            if is_perimeter:
                perimeter_cells.append((row, col))
        
        # Resources capacity - can address up to 10 cells per step
        resources_capacity = min(len(perimeter_cells), 10)
        
        if resources_capacity > 0:
            # If we don't have a current focus area, select a starting point
            if 'focus_point' not in locals() or focus_point is None:
                # Choose a starting point (could prioritize based on wind direction, threat to structures, etc.)
                focus_idx = np.random.randint(len(perimeter_cells))
                focus_point = perimeter_cells[focus_idx]
            
            # Find cells closest to the focus point
            distances = [(focus_point[0] - r)**2 + (focus_point[1] - c)**2 for r, c in perimeter_cells]
            closest_indices = np.argsort(distances)[:resources_capacity]
            
            # Apply suppressant to these nearby cells
            for idx in closest_indices:
                row, col = perimeter_cells[idx]
                
                # 70% chance of successfully extinguishing a burning cell
                if np.random.random() < 0.7:
                    model.grid[row, col] = 2  # Directly extinguish to burnt state
                    
                # Update focus point to be the last cell worked on
                focus_point = (row, col)
        else:
            focus_point = None
        
        # Update the model for this step
        model.update()
        history.append(np.copy(model.grid))
        
        # Stop if no more burning cells
        if not np.any(model.grid == 1):
            break
    
    return history




import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class FirefightingTourGA:
    """
    Genetic Algorithm for finding the optimal tour for a single firefighting resource
    to combat multiple fire ignitions, as described in the paper 'A Genetic Algorithm for 
    Forest Firefighting Optimization'.
    """
    def __init__(self, population_size=20, max_generations=200, 
                 crossover_rate=0.8, mutation_rate=0.2):
        """
        Initialize the Genetic Algorithm parameters
        
        Parameters:
        - population_size: number of chromosomes in population
        - max_generations: maximum number of generations for evolution
        - crossover_rate: probability of crossover
        - mutation_rate: probability of mutation
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_fitness = 0
        
    def _initialize_population(self, num_fires):
        """Create initial random population of permutations"""
        population = []
        for _ in range(self.population_size):
            # Create a random permutation of fire indices
            chromosome = np.random.permutation(num_fires)
            population.append(chromosome)
        return population
    
    def _calculate_travel_times(self, fire_locations, base_location=None):
        """
        Calculate travel time matrix between fire locations
        
        Parameters:
        - fire_locations: list of (x, y) coordinates of fire ignitions
        - base_location: (x, y) coordinates of the firefighting resource starting point
                        (if None, will use first fire location)
        
        Returns:
        - travel_times: matrix of travel times between locations
        """
        n_locations = len(fire_locations)
        all_locations = list(fire_locations)
        
        # Add base location if provided
        has_base = base_location is not None
        if has_base:
            all_locations = [base_location] + all_locations
        
        # Calculate distance matrix
        dist_matrix = distance.cdist(all_locations, all_locations, 'euclidean')
        
        # Convert distances to travel times (assuming 1 unit distance = 0.1 hours)
        travel_times = dist_matrix * 0.1
        
        return travel_times
    
    def _evaluate_fitness(self, chromosome, travel_times, fire_params, base_included=True):
        """
        Evaluate the fitness of a chromosome (tour sequence) using the formulas from the paper:
        V[i] = V[i]0 - a[i]C[i]^2 where C[i] is the completion time
        
        Parameters:
        - chromosome: sequence of fire indices to visit
        - travel_times: matrix of travel times between locations
        - fire_params: parameters for each fire (initial area, deterioration rate, etc.)
        - base_included: whether the travel_times matrix includes the base location
        
        Returns:
        - fitness: total unburned area after the tour
        """
        offset = 1 if base_included else 0  # Offset for indexing travel_times if base is included
        
        n_fires = len(chromosome)
        total_unburned = 0
        current_time = 0
        
        # If base is included, start from base (index 0)
        current_location = 0 if base_included else chromosome[0]
        
        # Process each fire in the sequence
        for i, fire_idx in enumerate(chromosome):
            # Get travel time to this fire
            target_idx = fire_idx + offset if base_included else fire_idx
            travel_time = travel_times[current_location, target_idx]
            current_time += travel_time
            
            # Get fire parameters
            v0 = fire_params[fire_idx]['initial_area']  # V[i]0 - initial area
            a = fire_params[fire_idx]['deterioration_rate']  # a[i] - deterioration rate
            
            # Calculate processing time based on formula from paper (eq. 4)
            processing_time = self._calculate_processing_time(
                fire_params[fire_idx], 
                current_time
            )
            
            # Update completion time (C[i] in the paper)
            completion_time = current_time + processing_time
            
            # Calculate unburned area using the formula from the paper (eq. 2)
            # V[i] = V[i]0 - a[i] * C[i]^2
            unburned = max(0, v0 - a * (completion_time ** 2))
            total_unburned += unburned
            
            # Update current location and time
            current_location = target_idx
            current_time += processing_time
        
        return total_unburned
    
    def _calculate_processing_time(self, fire_param, time_elapsed):
        """
        Calculate processing time for a fire based on the paper's formula (eq. 4)
        
        Parameters:
        - fire_param: parameters for this fire
        - time_elapsed: time elapsed since fire ignition
        
        Returns:
        - processing_time: time needed to suppress this fire
        """
        alpha = fire_param.get('alpha', 0.5)   # α[i] in paper
        beta = fire_param.get('beta', 3.0)     # β[i] in paper
        gamma = fire_param.get('gamma', 0.25)  # γ[i] in paper
        delta = fire_param.get('delta', 0.006) # δ[i] in paper
        d = fire_param.get('containment_time', 0.5)  # d[i] in paper - containment escape time limit
        x = fire_param.get('additional_resources', 1.0)  # X[i] in paper
        
        # Formula from paper (eq. 4)
        if time_elapsed <= d:
            # First case: α[i]t / (β[i] + γ[i]t + δ[i])
            processing_time = (alpha * time_elapsed) / (beta + gamma * time_elapsed + delta)
        else:
            # Second case: X[i] - additional resources required
            processing_time = x
            
        return processing_time
    
    def _selection(self, population, fitness_values):
        """
        Select parents for reproduction using tournament selection as described in the paper
        
        Parameters:
        - population: list of chromosomes
        - fitness_values: fitness value for each chromosome
        
        Returns:
        - parents: selected chromosomes for reproduction
        """
        parents = []
        for _ in range(self.population_size):
            # Select 3 random individuals for tournament
            tournament_idx = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_idx]
            # Select the best one (maximize unburned area)
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())
        return parents
    
    def _order_crossover(self, parent1, parent2):
        """
        Order-based crossover for permutation representation as described in the paper
        
        Parameters:
        - parent1, parent2: parent chromosomes
        
        Returns:
        - child1, child2: offspring chromosomes
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        # Select random segment
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        
        # Create children starting with the selected segment
        child1 = np.full(size, -1)
        child2 = np.full(size, -1)
        
        # Copy selected segment
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # Fill remaining positions with values from other parent
        # For child1
        idx = (end + 1) % size
        for i in range(size):
            pos = (end + 1 + i) % size
            if child1[pos] == -1:  # If position not filled yet
                while parent2[idx] in child1:
                    idx = (idx + 1) % size
                child1[pos] = parent2[idx]
                idx = (idx + 1) % size
        
        # For child2
        idx = (end + 1) % size
        for i in range(size):
            pos = (end + 1 + i) % size
            if child2[pos] == -1:  # If position not filled yet
                while parent1[idx] in child2:
                    idx = (idx + 1) % size
                child2[pos] = parent1[idx]
                idx = (idx + 1) % size
            
        return child1, child2
    
    def _inverse_mutation(self, chromosome):
        """
        Inverse mutation for permutation representation as described in the paper
        Reverses the order of a randomly selected segment
        
        Parameters:
        - chromosome: chromosome to mutate
        
        Returns:
        - mutated: mutated chromosome
        """
        if np.random.random() > self.mutation_rate:
            return chromosome
        
        size = len(chromosome)
        # Select random segment
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        
        # Create mutated chromosome
        mutated = chromosome.copy()
        # Reverse the selected segment
        mutated[start:end+1] = mutated[start:end+1][::-1]
        
        return mutated
    
    def optimize(self, fire_locations, fire_params, base_location=None, verbose=True):
        """
        Run the genetic algorithm to find optimal firefighting tour
        
        Parameters:
        - fire_locations: list of (x, y) coordinates of fire ignitions
        - fire_params: list of dictionaries with parameters for each fire
        - base_location: (x, y) coordinates of the firefighting resource starting point
        - verbose: whether to print progress information
        
        Returns:
        - best_sequence: optimal firefighting sequence
        - best_fitness: fitness value of best solution (unburned area)
        """
        num_fires = len(fire_locations)
        
        # Calculate travel times
        travel_times = self._calculate_travel_times(fire_locations, base_location)
        base_included = base_location is not None
        
        # Initialize population
        population = self._initialize_population(num_fires)
        
        # Evaluate initial population
        fitness_values = []
        for chromosome in population:
            fitness = self._evaluate_fitness(chromosome, travel_times, fire_params, base_included)
            fitness_values.append(fitness)
            
            # Update best solution if better
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = chromosome.copy()
        
        # Evolution process
        for generation in range(self.max_generations):
            # Selection
            parents = self._selection(population, fitness_values)
            
            # Create new population with crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    child1, child2 = self._order_crossover(parents[i], parents[i+1])
                    child1 = self._inverse_mutation(child1)
                    child2 = self._inverse_mutation(child2)
                    new_population.extend([child1, child2])
                else:
                    child = self._inverse_mutation(parents[i].copy())
                    new_population.append(child)
            
            # Replace population
            population = new_population
            
            # Evaluate new population
            fitness_values = []
            for chromosome in population:
                fitness = self._evaluate_fitness(chromosome, travel_times, fire_params, base_included)
                fitness_values.append(fitness)
                
                # Update best solution if better
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = chromosome.copy()
            
            # Print progress every 10 generations if verbose
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness}")
        
        if verbose:
            print(f"Final solution: {self.best_solution}")
            print(f"Unburned area: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness


def apply_ga_tour_strategy(model, steps):
    """
    Apply the genetic algorithm to find the optimal tour for a single firefighting resource
    to combat multiple fires, based on the approach from Matos et al.
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # First, identify fire ignition points
    burning_cells = np.where(model.grid == 1)
    fire_locations = list(zip(burning_cells[1], burning_cells[0]))  # (x, y) format
    
    # If only one fire or no fires, use a simple direct attack approach
    if len(fire_locations) <= 1:
        print("Only one fire detected. Using simple direct attack.")
        return apply_direct_attack_strategy(model, steps)
    
    # Prepare fire parameters according to the paper's approach
    fire_params = []
    for i, (x, y) in enumerate(fire_locations):
        # Extract local conditions at each fire location
        local_ndvi = model.ndvi[y, x]
        local_humidity = model.humidity[y, x]
        local_slope = model.slope[y, x]
        
        # Using values inspired by the paper's Table 2 but adjusted based on local conditions
        alpha = 0.4 + 0.2 * local_ndvi  # Higher NDVI -> faster initial spread
        beta = 3.0 - 0.5 * local_ndvi    # Higher NDVI -> lower denominator base
        gamma = 0.2 + 0.1 * local_ndvi   # Higher NDVI -> faster denominator growth
        delta = 0.006 - 0.001 * local_humidity/100  # Higher humidity -> slower growth
        
        # Containment time estimate (d[i] in the paper)
        containment_time = 0.3 + 0.5 * local_ndvi
        
        # Additional resources required if containment time is exceeded (X[i] in the paper)
        additional_resources = 0.5 + local_ndvi
        
        # Initial area (V[i]0 in the paper) - use a fraction of the grid size 
        initial_area = 100 + 50 * local_ndvi
        
        # Deterioration rate (a[i] in the paper)
        deterioration_rate = 0.1 * (1 - local_humidity/100)
        
        # Combine all parameters into a dictionary
        fire_params.append({
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta,
            'containment_time': containment_time,
            'additional_resources': additional_resources,
            'initial_area': initial_area,
            'deterioration_rate': deterioration_rate
        })
    
    # Define base location (fire station) - use (0,0) as default or set to your preferred location
    base_location = (0, 0)
    
    # Create and run GA optimizer with parameters matching the paper
    ga = FirefightingTourGA(
        population_size=20,  # Paper used 20
        max_generations=200, # Paper used 200
        crossover_rate=0.8,  # Common value
        mutation_rate=0.2    # Common value
    )
    
    # Find optimal sequence
    best_sequence, best_fitness = ga.optimize(fire_locations, fire_params, base_location)
    
    print(f"Optimal firefighting sequence found. Unburned area: {best_fitness:.2f} hectares")
    
    # Apply the optimal tour to the model
    return apply_optimal_sequence_to_model(model, steps, fire_locations, best_sequence, base_location)

def apply_optimal_sequence_to_model(model, steps, fire_locations, sequence, base_location):
    """
    Apply the optimal firefighting sequence to the model
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    - fire_locations: list of (x, y) coordinates of fire ignitions
    - sequence: optimal sequence to visit fires
    - base_location: starting location for firefighting resource
    
    Returns:
    - history: list of grid states at each time step
    """
    # Initialize history with current state
    history = [np.copy(model.grid)]
    
    # Current location of firefighting resource
    current_location = base_location
    
    # Time step counter
    time_step = 0
    
    # Process each fire in the optimal sequence
    for fire_idx in sequence:
        # Location of this fire
        fire_location = fire_locations[fire_idx]
        
        # Calculate travel time (simple Euclidean distance)
        travel_distance = np.sqrt((current_location[0] - fire_location[0])**2 + 
                                 (current_location[1] - fire_location[1])**2)
        travel_time_steps = int(np.ceil(travel_distance * 0.1 * 10))  # Convert hours to steps
        
        # Simulate fire spread during travel
        for _ in range(travel_time_steps):
            if time_step < steps:
                model.update()
                history.append(np.copy(model.grid))
                time_step += 1
            else:
                break
        
        # Apply firefighting at this location (similar to direct attack)
        if time_step < steps:
            x, y = fire_location
            radius = 5  # Firefighting impact radius
            
            # Apply suppression in the area
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dx**2 + dy**2 <= radius**2:  # Circular area
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < model.cols and 0 <= ny < model.rows:
                            # If cell is burning, extinguish with 70% probability
                            if model.grid[ny, nx] == 1 and np.random.random() < 0.7:
                                model.grid[ny, nx] = 2  # Set to burnt state
                            
                            # Increase humidity to slow fire spread
                            model.humidity[ny, nx] = min(95, model.humidity[ny, nx] + 20)
            
            # Update model after firefighting (simulates time spent fighting this fire)
            firefighting_time_steps = 5  # Assume 5 steps to fight a fire
            for _ in range(firefighting_time_steps):
                if time_step < steps:
                    model.update()
                    history.append(np.copy(model.grid))
                    time_step += 1
                else:
                    break
            
            # Update current location
            current_location = fire_location
        
    # If we haven't reached the requested number of steps, continue simulation
    while time_step < steps:
        model.update()
        history.append(np.copy(model.grid))
        time_step += 1
    
    return history[:steps]

