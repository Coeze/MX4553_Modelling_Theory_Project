import fiona
import rasterio
from rasterio import features
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.animation import FuncAnimation
import math
from sklearn.svm import SVC
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import os
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping
import random
from deap import base, creator, tools, algorithms
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, HTML
from skimage.transform import resize




class CA:
    """
    CA: A Cellular Automata wildfire spread Model 
    """
    
    def __init__(self, grid_size=(100, 100), cell_size=1, params=None):
        """
        Initialize the CA model
        
        Parameters:
        - grid_size: tuple, dimensions of the grid (rows, cols)
        - cell_size: float, size of each cell in meters (default 30m as in the paper)
        - gamma: float, regularization parameter for LSSVM
        """
        self.rows, self.cols = grid_size
        self.cell_size = cell_size
        
        # CA grid states: 0-unburnt, 1-burning, 2-burnt
        self.grid = np.zeros(grid_size, dtype=int)
        self.next_grid = np.zeros_like(self.grid)
        
        # Environmental factors
        self.slope = np.zeros(grid_size)
        self.aspect = np.zeros(grid_size)
        self.elevation = np.zeros(grid_size)
        self.ndvi = np.zeros(grid_size)
        self.fuel_type = np.zeros(grid_size)  # veg/fuel type for each cell

        self.fuel_type = np.random.randint(0, 6, grid_size)

        self.fuel_model = np.array([
        # Broadleaves, Shrubs, Grassland, Fire-Prone Conifers, Agro-Forestry Areas, Not Fire-Prone Forest
        [0.300, 0.375, 0.250, 0.275, 0.250, 0.250],  # Broadleaves
        [0.375, 0.375, 0.350, 0.400, 0.300, 0.375],  # Shrubs
        [0.450, 0.475, 0.475, 0.475, 0.375, 0.475],  # Grassland
        [0.225, 0.325, 0.250, 0.350, 0.200, 0.350],  # Fire-prone conifers
        [0.250, 0.250, 0.300, 0.475, 0.350, 0.250],  # Agro-forestry areas
        [0.075, 0.100, 0.075, 0.275, 0.075, 0.075]   # Not fire-prone forest
    ]) 
        #Environmental factors
        self.wind_speed = 0
        self.wind_direction = 0  # in degrees
        self.fire_direction = 10
        self.temperature = 80
        self.precipitation = 0.0
        self.humidity = 5
        
        # Geospatial parameters for raster alignment
        self.transform = None
        self.crs = None
        self.actual_burned_area = None
        self.risk_map = None
        
        self.c1 = params['c1'] if params else 0.5
        self.c2 = params['c2'] if params else 0.5
        self.p0 = params['p0'] if params else 0.5
    
    def load_terrain_data(self, slope, aspect, elevation, humidity):
        """
        Load environmental data for the grid
        
        Parameters:
        - slope: 2D numpy array of slope values for each cell
        - aspect: 2D numpy array of aspect values for each cell
        - elevation: 2D numpy array of elevation values for each cell
        - humidity: 2D numpy array of relative humidity values for each cell
        - ndvi: 2D numpy array of normalized vegetation index values for each cell
        """
        
        assert slope.shape == (self.rows, self.cols), "Slope array shape mismatch"
        assert aspect.shape == (self.rows, self.cols), "Aspect array shape mismatch"
        assert elevation.shape == (self.rows, self.cols), "Elevation array shape mismatch"
        assert humidity.shape == (self.rows, self.cols), "Humidity array shape mismatch"
        
        self.slope = slope
        self.aspect = aspect
        self.elevation = elevation
        self.humidity = humidity
    
    def set_environmental_data(self, wind_speed, wind_direction, temperature, humidity, fire_direction):
        """
        Set wind parameters for simulation
        
        Parameters:
        - wind_speed: float, wind speed in m/s
        - wind_direction: float, wind direction in degrees (0 is north, 90 is east)
        """
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.temperature = temperature
        self.humidity = humidity
        self.fire_direction = fire_direction
    
    
    def set_initial_fire(self, fire_points):
        """
        Set initial fire points
        
        Parameters:
        - fire_points: list of tuples containing (row, col) coordinates of initial fire points
        """
        for row, col in fire_points:
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.grid[row, col] = 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def wind_effect(self, c1, c2):
        """
        Calculate the wind effect factor based on wind speed and direction.
        """
        # wind_factor = np.exp(self.wind_speed * 0.1783)
        diff = np.abs(self.wind_direction - self.fire_direction)
        wind_factor = np.exp(self.wind_speed * (c1 + c2 * (np.cos(np.radians(diff)) - 1)))
        return self.sigmoid(wind_factor)

    def topography_effect(self, slope):
        """
        Adjust fire spread probability based on terrain slope.
        """
        slope_factor = np.exp((3.533 * (np.tan(slope))))
        return self.sigmoid(slope_factor)

    def humidity_effect(self, humidity, h=0.2036):
        """
        Adjust fire spread probability based on humidity.
        """
        humidity_factor = np.exp((abs(h)) * humidity)
        return self.sigmoid(humidity_factor)

    def temperature_effect(self, temperature, t=0.0194):
        """
        Adjust fire spread probability based on temperature.
        """   
        temperature_factor = np.exp(t * temperature)
        return self.sigmoid(temperature_factor)
        
    def precipitation_effect(self, precipitation, p=-0.3473):
        """
        Adjust fire spread probability based on precipitation.
        """   
        precipitation_factor = np.exp((abs(p)) * precipitation)
        return self.sigmoid(precipitation_factor)

    
    def calculate_ignition_probability(self, row, col):
        """
        Calculate the probability of a cell igniting using LSSVM
        As described in Section 2.3 of the paper
        
        Parameters:
        - row, col: coordinates of the cell
        
        Returns:
        - probability: ignition probability [0, 1]
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return 0.0
        
        if self.grid[row, col] != 0:  # Already burning or burnt
            return 0.0
        
        highest_veg_prob = 0.0
            
        # Check if the cell has any burning neighbors
        has_burning_neighbors = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 1:
                    highest_veg_prob = max(highest_veg_prob, self.fuel_model[self.fuel_type[nr, nc], self.fuel_type[row, col]])
                    has_burning_neighbors = True
            if has_burning_neighbors:
                break
        
        if not has_burning_neighbors:
            return 0.0
        
      
        wind_effects = self.wind_effect(0.5, 0.5)
        topography_effects = self.topography_effect(self.slope[row, col])
        humidity_effects = self.humidity_effect(humidity=self.humidity)
        temperature_effects = self.temperature_effect(temperature=self.temperature)
        precipitation_effect = self.precipitation_effect(self.precipitation)
        p_density = self.ndvi[row, col] * 0.5 + 0.5
        # * (temperature_effects)  / ((humidity_effects) * (precipitation_effect))
        adjusted_probability = self.p0 * (1+highest_veg_prob) * (1+p_density) * (wind_effects) * (topography_effects) 
        print(f" prob: {self.p0}, we: {wind_effects}, a_prob: {adjusted_probability}, tp: {highest_veg_prob}, p_density: {p_density}, humidity: {humidity_effects}, temperature: {temperature_effects}, precipitation: {precipitation_effect}")
        
        # Ensure probability is in [0, 1] range
        return min(1, adjusted_probability)
        
    def update(self):
        """
        Update the CA grid for one time step
        """
        self.next_grid = np.copy(self.grid)
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 0:  # Unburnt cell
                    # Calculate ignition probability
                    p_ignite = self.calculate_ignition_probability(row, col)                    
                    # Probabilistic ignition
                    if np.random.random() < p_ignite:
                        self.next_grid[row, col] = 1  # Cell ignites
                
                elif self.grid[row, col] == 1:  # Burning cell
                    # Cells burn for one time step, then become burnt
                    self.next_grid[row, col] = 2
        
        self.grid = np.copy(self.next_grid)
    
    def initialise_ndvi_from_data(self, fire):
        """
        Initialize the grid and environmental data from MTBS fire data
        
        Parameters:
        - fire_folder_path: path to folder containing MTBS fire data
        - ignition_date: YYYYMMDD string for ignition date, if known. If None, will estimate based on data
        - use_hotspots: boolean, whether to use hotspot data for initial fire points (if available)
        
        Returns:
        - success: boolean indicating if initialization was successful
        """
        print(f"Initializing from {fire} fire")
        
        # Load burn perimeter as raster
        if fire == "arizona":
            burn_bndy_path = 'data/az3698311211020200729/az3698311211020200729_20200714_20210717_burn_bndy.shp'
            dnbr_path = 'data/az3698311211020200729/az3698311211020200729_20200714_20210717_dnbr.tif'
        elif fire == "alabama":
            burn_bndy_path = 'data/al3039808817220190514/al3039808817220190514_20190513_20190528_burn_bndy.shp'
            dnbr_path = 'data/al3039808817220190514/al3039808817220190514_20190513_20190528_dnbr.tif'
        try:
            self.actual_burned_area = self.load_shapefile_as_raster(burn_bndy_path)
            print(f"Loaded burn perimeter successfully")
        except Exception as e:
            print(f"Error loading burn perimeter: {str(e)}")
            return False
        
    
        try:
            with rasterio.open(dnbr_path) as src:
                dnbr_data = src.read(1)
                # Save transform for georeferencing
                self.transform = src.transform
                self.crs = src.crs
                
                if dnbr_data.shape != (self.rows, self.cols):
                    print(f"Resampling DNBR data from {dnbr_data.shape} to {(self.rows, self.cols)}")
                    dnbr_data = resize(dnbr_data, (self.rows, self.cols), preserve_range=True)
                
                dnbr_min = np.nanmin(dnbr_data)
                dnbr_max = np.nanmax(dnbr_data)
                if dnbr_max > dnbr_min:
                    normalized_dnbr = (dnbr_data - dnbr_min) / (dnbr_max - dnbr_min)
                    self.ndvi = 0.2 + 0.6 * normalized_dnbr  # Scale to 0.2-0.8 range
                else:
                    self.ndvi = np.ones((self.rows, self.cols)) * 0.5
                    
                print("Estimated NDVI from DNBR data")
        except Exception as e:
            print(f"Error processing DNBR data: {str(e)}")
            # Generate random NDVI if DNBR processing fails
            self.ndvi = np.random.uniform(0.1, 0.8, (self.rows, self.cols))
        
        self.slope = np.random.uniform(0, 30, (self.rows, self.cols))  # 0-30 degree slopes
        self.aspect = np.random.uniform(0, 360, (self.rows, self.cols))  # 0-360 degree aspects
        
        self.elevation = np.random.uniform(100, 1000, (self.rows, self.cols))  # 100-1000m elevation
        
        self.humidity = np.random.uniform(20, 80, (self.rows, self.cols))  # 20-80% humidity
        
        return True
    
    def run_simulation(self, steps):
        """
        Run the simulation for a given number of steps
        
        Parameters:
        - steps: int, number of time steps to simulate
        
        Returns:
        - history: list of grid states at each time step
        """
        history = [np.copy(self.grid)]
        
        for _ in range(steps):
            self.update()
            history.append(np.copy(self.grid))
            
            # Stop if no more burning cells
            if not np.any(self.grid == 1):
                break
        
        return history
    
    def visualize_simulation(self, history, interval_ms=500):
        """
        Visualize the simulation history
        
        Parameters:
        - history: list of grid states at each time step
        - interval_ms: interval between frames in milliseconds
        """
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.colors.ListedColormap(['green', 'red', 'black'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        def update_plot(frame):
            plt.clf()
            plt.imshow(history[frame], cmap=cmap, norm=norm)
            plt.title(f"Forest Fire Spread Simulation - Step {frame}")
            plt.axis('off')
        
        anim = FuncAnimation(
            plt.gcf(), update_plot, frames=len(history), 
            interval=interval_ms, repeat=False
        )
        
        plt.tight_layout()
        
        display(HTML(anim.to_jshtml()))
        
        return True
    
    def evaluate_simulation(self):
        """
        Evaluate the simulation results against actual burned area
        
        Parameters:
        - actual_burned_area: 2D numpy array of the same shape as grid, 
          with 1s indicating burned cells and 0s for unburned
          
        Returns:
        - accuracy: float, percentage of correctly predicted cells
        - precision: float, precision of fire prediction
        - recall: float, recall of fire prediction
        - sorensen: float, Sørensen index (Dice coefficient)
        """
        # Convert grid to binary (burned/unburned)
        simulated_burned = (self.grid == 2).astype(int)
        
        # Calculate total cells
        total_cells = self.rows * self.cols
        
        # Calculate correctly predicted cells
        correct_cells = np.sum(simulated_burned == self.actual_burned_area)
        accuracy = correct_cells / total_cells
        
        # Calculate true positives, false positives, and false negatives
        true_positives = np.sum((simulated_burned == 1) & (self.actual_burned_area == 1))
        false_positives = np.sum((simulated_burned == 1) & (self.actual_burned_area == 0))
        false_negatives = np.sum((simulated_burned == 0) & (self.actual_burned_area == 1))
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate Sørensen index (Dice coefficient)
        sorensen = 2 * true_positives / (2 * true_positives + false_positives + false_negatives) if (2 * true_positives + false_positives + false_negatives) > 0 else 0
        
        return accuracy, precision, recall, sorensen
    
    def load_shapefile_as_raster(self, shapefile_path, grid_size=None, cell_size=None):
        """
        Load a shapefile (burn perimeter) and rasterize it to match the model grid
        
        Parameters:
        - shapefile_path: path to the shapefile
        - grid_size: tuple (rows, cols) or None to use model's grid_size
        - cell_size: cell size in meters or None to use model's cell_size
        
        Returns:
        - rasterized: 2D numpy array where 1 = burned area, 0 = unburned area
        """
        if grid_size is None:
            grid_size = (self.rows, self.cols)
        if cell_size is None:
            cell_size = self.cell_size
            
        rows, cols = grid_size
    
        # Open the shapefile
        with fiona.open(shapefile_path, 'r') as shapefile:
            # Get the bounds of the shapefile
            bounds = shapefile.bounds
            # Read all geometries
            geometries = [feature["geometry"] for feature in shapefile]

        # Define the dimensions of the output array
        height = self.rows 
        width = self.cols

        # Calculate the resolution of each pixel
        x_res = (bounds[2] - bounds[0]) / width
        y_res = (bounds[3] - bounds[1]) / height

        # Define the transform for the raster
        transform = rasterio.transform.from_origin(
            west=bounds[0],
            north=bounds[3],
            xsize=x_res,
            ysize=y_res
        )

        # Rasterize the geometries
        raster = features.rasterize(
            ((geom, 1) for geom in geometries),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        self.actual_burned_area = raster
        
        return raster
    
    def load_raster_data(self, raster_path, grid_size=None):
        """
        Load raster data (e.g., dnbr.tif) and resample it to match the model grid
        
        Parameters:
        - raster_path: path to the raster file
        - grid_size: tuple (rows, cols) or None to use model's grid_size
        
        Returns:
        - resampled_data: 2D numpy array of raster values resampled to match the model grid
        """
        if grid_size is None:
            grid_size = (self.rows, self.cols)
            
        with rasterio.open(raster_path) as src:
            # If we don't have a CRS yet, get it from the raster
            if self.crs is None and src.crs:
                self.crs = src.crs
                
            data = src.read(1)
            
            return data
    
    def load_mtbs_fire_data(self, folder_path, grid_size=None, cell_size=None):
        """
        Load MTBS fire data from a folder, including burn perimeter and dnbr
        
        Parameters:
        - folder_path: path to the folder containing MTBS data
        - grid_size: tuple (rows, cols) or None to use model's grid_size
        - cell_size: cell size in meters or None to use model's cell_size
        """
        if grid_size is None:
            grid_size = (self.rows, self.cols)
        if cell_size is None:
            cell_size = self.cell_size

        if folder_path == "arizona":
            burn_bndy_path = 'data/az3698311211020200729/az3698311211020200729_20200714_20210717_burn_bndy.shp'
            dnbr_path = 'data/az3698311211020200729/az3698311211020200729_20200714_20210717_dnbr.tif'
            self.set_initial_fire([(int((self.rows)*0.5), int((self.cols)*0.5))])
        elif folder_path == "alabama":
            burn_bndy_path = 'data/al3039808817220190514/al3039808817220190514_20190513_20190528_burn_bndy.shp'
            dnbr_path = 'data/al3039808817220190514/al3039808817220190514_20190513_20190528_dnbr.tif'
            ignition_points = (int((self.rows)*0.75), int((self.cols)*0.25))
            print(ignition_points)
            self.set_initial_fire([ignition_points])
        else:
            raise ValueError('Please select a valid fire either: alabama or arizona')

        self.actual_burned_area = self.load_shapefile_as_raster(burn_bndy_path, grid_size, cell_size)
        print(f"Loaded burn perimeter shapefile: {burn_bndy_path}")
        
        dnbr_data = self.load_raster_data(dnbr_path, grid_size)
        print(f"Loaded DNBR raster: {dnbr_path}")
    
    def overlay_simulation_with_actual(self, fire, figsize=(15, 8)):
        """
        Create a side-by-side visualization comparing simulated vs actual burn area
        
        Parameters:
        - figsize: tuple, figure size in inches
        
        Returns:
        - fig: matplotlib figure
        """
        if self.actual_burned_area is None:
            raise ValueError("No actual burned area data has been loaded.")
            
        # Convert grid to binary (burned/unburned)
        simulated_burned = (self.grid == 2).astype(int)
        
        # Create a difference map
        # 0: Neither burned (correct negative)
        # 1: Actually burned but not simulated (false negative)
        # 2: Simulated burned but not actually (false positive)
        # 3: Both burned (correct positive)
        difference = simulated_burned * 2 + self.actual_burned_area
        
        # Create visualization
        fig, axs = plt.subplots(1, 4, figsize=(20, 8))
        
        # Plot actual burned area
        axs[0].imshow(self.actual_burned_area, cmap='Reds', interpolation='none')
        axs[0].set_title('Actual Burned Area')
        axs[0].axis('off')
        
        # Plot simulated burned area
        axs[1].imshow(simulated_burned, cmap='Reds', interpolation='none')
        axs[1].set_title('Simulated Burned Area')
        axs[1].axis('off')
        
        # Plot difference map
        cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red', 'black'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        im = axs[2].imshow(difference, cmap=cmap, norm=norm, interpolation='none')
        axs[2].set_title('Difference Map')
        axs[2].axis('off')
        
        # Add colorbar to the difference map
        cbar = plt.colorbar(im, ax=axs[2], ticks=[0, 1, 2, 3])
        cbar.set_label('Comparison')
        cbar.set_ticklabels(['Unburned (Correct)', 'Missed (False Neg)', 'Overestimated (False Pos)', 'Burned (Correct)'])
        if fire == "arizona":
            burn_bndy_path = 'data/az3698311211020200729/az3698311211020200729_20200714_20210717_burn_bndy.shp'
            dnbr_path = 'data/az3698311211020200729/az3698311211020200729_20200714_20210717_dnbr.tif'
        elif fire == "alabama":
            burn_bndy_path = 'data/al3039808817220190514/al3039808817220190514_20190513_20190528_burn_bndy.shp'
            dnbr_path = 'data/al3039808817220190514/al3039808817220190514_20190513_20190528_dnbr.tif'
        
        # NEW: Create overlay visualization with simulated burn on actual shapefile
        try:
            burn_gdf = gpd.read_file(burn_bndy_path)
            
            # Plot the actual shapefile
            burn_gdf.plot(ax=axs[3], color='none', edgecolor='red', linewidth=2)
            
            # Create a transparent overlay of the simulated burned area
            alpha_sim = np.zeros((self.rows, self.cols, 4))
            # Set RGBA for simulated burned areas (black with 60% opacity)
            alpha_sim[simulated_burned == 1, :] = [0, 0, 0, 0.6]
            
            # Plot the simulated burned area on top of the shapefile
            axs[3].imshow(alpha_sim, origin='upper')
            axs[3].set_title('Simulated Grid Overlay on Shapefile')
                
            # Add a scale bar if transform is available
            if self.transform:
                scale_bar = ScaleBar(self.cell_size, units="m", location="lower right")
                axs[3].add_artist(scale_bar)
                
        except Exception as e:
            axs[3].text(0.5, 0.5, f"Error creating overlay: {str(e)}", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[3].transAxes)
        
        axs[3].axis('off')
        
        # Calculate metrics
        accuracy, precision, recall, sorensen = self.evaluate_simulation()
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.suptitle(
            f"Fire Spread Comparison\n"
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, Sørensen: {sorensen:.4f}",
            fontsize=14
        )
        
        plt.tight_layout()
        return fig


