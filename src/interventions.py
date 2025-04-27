import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from src.model import CA

def apply_firebreak_strategy(model, steps):
    """
    Apply a firebreak strategy by creating gaps in fuel continuity
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Create firebreaks (set fuel-related values to very low)
    # Create two strategic firebreak lines
    # Horizontal firebreak
    y_pos = 40
    for x in range(0, 100):
        for y in range(y_pos, min(y_pos + 3, 100)):
            model.ndvi[y, x] = 0.05  # reduce vegetable density
            model.fuel_type[y, x] = 5  # Change fuel type to not fire prone forest
    
    # Vertical firebreak
    x_pos = 60
    for y in range(0, 100):
        for x in range(x_pos, min(x_pos + 3, 100)):
            model.ndvi[y, x] = 0.05 # reduce vegetable density
            model.fuel_type[y, x] = 5  # Change fuel type to not fire prone forest
    
    # Run the simulation
    return model.run_simulation(steps)

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
        
        # Apply suppressant to a percentage of burning cells (simulating firefighting resources)
        resources_capacity = min(len(burning_cells[0]), 10)  # Can address 10 cells per step
        
        if resources_capacity > 0:
            # Select random burning cells to target
            indices = np.random.choice(len(burning_cells[0]), resources_capacity, replace=False)
            
            for idx in indices:
                row, col = burning_cells[0][idx], burning_cells[1][idx]
                
                # 70% chance of successfully extinguishing a burning cell
                if np.random.random() < 0.7:
                    model.grid[row, col] = 2  # Directly extinguish to burnt state
        
        # Update the model for this step
        model.update()
        history.append(np.copy(model.grid))
        
        # Stop if no more burning cells
        if not np.any(model.grid == 1):
            break
    
    return history


def apply_point_protection_strategy(model, steps):
    """
    Apply a point protection strategy focusing on high-value areas
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Define "high-value" points to protect (e.g., hypothetical structures)
    high_value_points = [
        (25, 25), (25, 75), (75, 25), (75, 75), (50, 50),  # Five points of interest
    ]
    
    # Create protection zones around high-value points
    for center_y, center_x in high_value_points:
        # Create a protection zone with radius 5
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dx**2 + dy**2 <= 25:  # Circular zone with radius 5
                    y = center_y + dy
                    x = center_x + dx
                    
                    if 0 <= y < model.rows and 0 <= x < model.cols:
                        # Apply protection (increased humidity, reduced vegetation)
                        model.humidity[y, x] = 90
                        model.ndvi[y, x] = 0.1
    
    # Run the simulation
    history = model.run_simulation(steps)
    
    # Assess protection effectiveness
    burn_status = []
    for center_y, center_x in high_value_points:
        # Check if the center point was burned
        if 0 <= center_y < model.rows and 0 <= center_x < model.cols:
            burned = model.grid[center_y, center_x] == 2
            burn_status.append(burned)
    
    # Calculate protection effectiveness
    protected_count = len(burn_status) - sum(burn_status)
    print(f"  Point protection effectiveness: {protected_count}/{len(high_value_points)} points protected")
    
    return history

def apply_early_detection_strategy(model, steps):
    """
    Apply an early detection and rapid response strategy
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Start with immediate response (no delay in intervention)
    history = [np.copy(model.grid)]  # Save initial state
    
    # For each step, apply rapid response
    for step in range(steps):
        # Update the model first
        model.update()
        history.append(np.copy(model.grid))
        
        # Find new ignitions (burning cells)
        burning_cells = np.where(model.grid == 1)
        
        # Early detection means we can respond to almost all new ignitions
        # With high effectiveness rate (90%)
        for i in range(len(burning_cells[0])):
            row, col = burning_cells[0][i], burning_cells[1][i]
            
            # 90% chance of catching and extinguishing a fire early
            if np.random.random() < 0.9:
                model.grid[row, col] = 2  # Extinguish to burnt state
        
        # Stop if no more burning cells
        if not np.any(model.grid == 1):
            break
    
    return history
    
