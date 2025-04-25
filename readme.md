# LSSVM-CA: Forest Fire Spread Model

The LSSVM-CA model is a forest fire spread simulation system that combines Least Squares Support Vector Machines with Cellular Automata to predict fire behavior based on environmental conditions.

## Core Components

### Class Structure

The model is organized into several key classes:

- **LSSVMCA**: Main model class that handles simulation logic
- **SimulationUI**: User interface for controlling simulations

### Environmental Factors

The model incorporates multiple environmental layers:

- **Topographic**: Slope, aspect, elevation
- **Meteorological**: Wind speed, direction, temperature, humidity
- **Vegetation**: NDVI (Normalized Difference Vegetation Index)

## Code Overview

### Model Initialization

```python
def __init__(self, grid_size=(100, 100), cell_size=30, gamma=1.0):
    # Grid configuration
    self.rows, self.cols = grid_size
    self.cell_size = cell_size
    self.gamma = gamma
    
    # CA grid states: 0-unburnt, 1-burning, 2-burnt
    self.grid = np.zeros(grid_size, dtype=int)
    self.next_grid = np.zeros_like(self.grid)
    
    # Environmental factors
    self.slope = np.zeros(grid_size)
    self.aspect = np.zeros(grid_size)
    self.elevation = np.zeros(grid_size)
    self.humidity = np.zeros(grid_size)
    self.ndvi = np.zeros(grid_size)
    
    # Weather parameters
    self.wind_speed = 0
    self.wind_direction = 0
    self.temperature = 20
    
    # LSSVM classifier
    self.classifier = SVC(kernel='rbf', gamma='auto', probability=True)


def load_environmental_data(self, slope, aspect, elevation, humidity, ndvi):
    # Validation
    assert slope.shape == (self.rows, self.cols), "Slope array shape mismatch"
    assert aspect.shape == (self.rows, self.cols), "Aspect array shape mismatch"
    assert elevation.shape == (self.rows, self.cols), "Elevation array shape mismatch"
    assert humidity.shape == (self.rows, self.cols), "Humidity array shape mismatch"
    assert ndvi.shape == (self.rows, self.cols), "NDVI array shape mismatch"
    
    # Load data
    self.slope = slope
    self.aspect = aspect
    self.elevation = elevation
    self.humidity = humidity
    self.ndvi = ndvi
```
