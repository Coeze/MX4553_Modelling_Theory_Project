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
            model.ndvi[y, x] = 0.05  # Very low vegetation
            model.humidity[y, x] = 90  # Very high humidity
    
    # Vertical firebreak
    x_pos = 60
    for y in range(0, 100):
        for x in range(x_pos, min(x_pos + 3, 100)):
            model.ndvi[y, x] = 0.05
            model.humidity[y, x] = 90
    
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



def apply_aerial_attack_strategy(model, steps):
    """
    Apply an aerial attack strategy with water/retardant drops
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Run first few steps to establish the fire
    history = model.run_simulation(2)
    
    # For each remaining step, apply aerial drops
    for step in range(steps - 2):
        # Find cells that are currently burning
        burning_cells = np.where(model.grid == 1)
        
        if len(burning_cells[0]) > 0:
            # Calculate the center of the fire for targeting
            center_y = int(np.mean(burning_cells[0]))
            center_x = int(np.mean(burning_cells[1]))
            
            # Simulate dropping water/retardant in a rectangular pattern
            drop_length = 20
            drop_width = 10
            
            # Align drop with wind direction for maximum effectiveness
            wind_direction_rad = np.radians(model.wind_direction)
            
            # Apply water/retardant effect
            for dy in range(-drop_width//2, drop_width//2):
                for dx in range(-drop_length//2, drop_length//2):
                    # Rotate coordinates according to wind direction
                    rot_x = int(dx * np.cos(wind_direction_rad) - dy * np.sin(wind_direction_rad))
                    rot_y = int(dx * np.sin(wind_direction_rad) + dy * np.cos(wind_direction_rad))
                    
                    target_y = center_y + rot_y
                    target_x = center_x + rot_x
                    
                    if 0 <= target_y < model.rows and 0 <= target_x < model.cols:
                        # Increase humidity significantly to simulate water/retardant
                        model.humidity[target_y, target_x] = min(100, model.humidity[target_y, target_x] + 40)
                        
                        # Directly extinguish some burning cells (80% effectiveness on direct hits)
                        if model.grid[target_y, target_x] == 1 and np.random.random() < 0.8:
                            model.grid[target_y, target_x] = 2  # Change to burnt state
        
        # Update the model for this step
        model.update()
        history.append(np.copy(model.grid))
        
        # Stop if no more burning cells
        if not np.any(model.grid == 1):
            break
    
    return history

def apply_burnout_strategy(model, steps):
    """
    Apply a burnout strategy with controlled burns between control lines and the fire
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Let fire establish
    history = model.run_simulation(2)
    
    # Calculate expected fire direction based on wind
    wind_direction_rad = np.radians(model.wind_direction)
    
    # Find the fire
    burning_cells = np.where(model.grid == 1)
    
    if len(burning_cells[0]) > 0:
        # Calculate fire centroid
        centroid_y = int(np.mean(burning_cells[0]))
        centroid_x = int(np.mean(burning_cells[1]))
        
        # Create a control line ahead of the fire
        dx = int(round(15 * np.sin(wind_direction_rad)))
        dy = int(round(-15 * np.cos(wind_direction_rad)))
        
        # Perpendicular to wind direction for the control line
        perp_rad = wind_direction_rad + np.pi/2
        
        # Create control line
        for d in range(-25, 26):
            x = centroid_x + dx + int(round(d * np.cos(perp_rad)))
            y = centroid_y + dy + int(round(d * np.sin(perp_rad)))
            
            for w in range(3):
                line_x = x + w
                line_y = y
                
                if 0 <= line_y < model.rows and 0 <= line_x < model.cols:
                    # Create fuel break
                    model.humidity[line_y, line_x] = 95
                    model.ndvi[line_y, line_x] = 0.0  # No fuel
        
        # Create a burnout zone (mark as already burnt) between the control line and the fire
        for y in range(model.rows):
            for x in range(model.cols):
                # Check if point is between fire and control line
                vector_to_point = (x - centroid_x, y - centroid_y)
                distance = np.sqrt(vector_to_point[0]**2 + vector_to_point[1]**2)
                
                # Skip if too far away
                if distance > 25:
                    continue
                
                # Calculate angle to point
                angle = np.arctan2(vector_to_point[1], vector_to_point[0])
                
                # Normalize angle difference
                angle_diff = abs(angle - wind_direction_rad)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # Check if in the forward direction of the fire
                if angle_diff < np.pi/2 and 5 < distance < 15:
                    # Mark as burnt (controlled burn area)
                    if model.grid[y, x] == 0:  # Only if not already burning or burnt
                        model.grid[y, x] = 2
    
    # Continue simulation
    for step in range(steps - 2):
        model.update()
        history.append(np.copy(model.grid))
        
        # Stop if no more burning cells
        if not np.any(model.grid == 1):
            break
    
    return history

def apply_wet_line_strategy(model, steps):
    """
    Apply a wet line strategy to stop fire spread
    
    Parameters:
    - model: CA model
    - steps: number of simulation steps
    
    Returns:
    - history: list of grid states at each time step
    """
    # Run first few steps to establish the fire
    history = model.run_simulation(3)
    
    # Find the fire perimeter
    grid_with_buffer = np.zeros((model.rows + 2, model.cols + 2))
    grid_with_buffer[1:-1, 1:-1] = model.grid > 0  # Both burning and burnt
    
    # Get the edge cells using a convolution
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Remove center
    
    perimeter_with_buffer = convolve(grid_with_buffer, kernel) * (1 - grid_with_buffer) > 0
    perimeter = perimeter_with_buffer[1:-1, 1:-1]
    
    # Get perimeter cell coordinates
    perimeter_cells = np.where(perimeter)
    
    # Create a ring of wet cells just outside the fire perimeter
    for i in range(len(perimeter_cells[0])):
        y, x = perimeter_cells[0][i], perimeter_cells[1][i]
        
        # Create a thicker wet line
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny = y + dy
                nx = x + dx
                
                if 0 <= ny < model.rows and 0 <= nx < model.cols and model.grid[ny, nx] == 0:
                    # Increase humidity significantly (water application)
                    model.humidity[ny, nx] = 95
    
    # Continue simulation
    for step in range(steps - 3):
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
    
