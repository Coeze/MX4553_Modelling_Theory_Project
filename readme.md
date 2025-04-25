# Wildfire Spread Modeling and Intervention Strategy Analysis

## Overview

This project implements and evaluates a cellular automata (CA) model for wildfire spread simulation, optimized using genetic algorithms. The model incorporates environmental factors such as topography, vegetation patterns, wind conditions, and humidity to predict fire behavior. Additionally, it evaluates different firefighting and intervention strategies to identify the most effective approaches for containing wildfires.

## Repository Structure

- **src/**
  - **model.py**: Core cellular automata model for fire spread simulation
  - **interventions.py**: Implementation of different firefighting strategies
- **data/**: Contains MTBS (Monitoring Trends in Burn Severity) fire datasets
  - Each subfolder contains data for a specific historical fire event
- **main.ipynb**: Main notebook for running simulations and evaluating intervention strategies
- **calibration.ipynb**: Notebook for optimizing model parameters using genetic algorithms [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Coeze/MX4553_Modelling_Theory_Project/blob/main/calibration.ipynb)

- **cnn_detection.ipynb**: Experimental machine learning approach for fire detection
- **optimal_parameters.json**: Stores optimized model parameters

## Core Model Features

The wildfire spread model is built on several key components:

1. **Cellular Automata Framework**: Simulates fire spread through a grid-based approach where each cell's state (unburnt, burning, burnt) is updated based on neighborhood conditions.

2. **Environmental Factor Integration**: Incorporates:
   - Topography (slope, aspect, elevation)
   - Vegetation (NDVI)
   - Weather conditions (wind speed, wind direction, humidity, temperature)
   - Fuel types and characteristics

3. **Parameter Optimization**: Uses genetic algorithms to optimize model parameters (base spread probability, wind effects) based on the Sørensen index (Dice coefficient) between simulated and actual burn areas.

4. **Intervention Strategy Evaluation**: Simulates and compares various firefighting strategies:
   - Firebreaks
   - Direct/indirect attack
   - Aerial operations
   - Burnout operations
   - Point protection

## Model Parameters

The model uses three key parameters that are optimized using genetic algorithms:

1. **p0**: Base probability of fire spread (0.1-0.9)
2. **c1**: Wind effect coefficient - controls how much wind speed affects spread probability
3. **c2**: Wind direction coefficient - controls the directional influence of wind

## Data Sources

The model uses data from the Monitoring Trends in Burn Severity (MTBS) program, which provides:
- Burn perimeter shapefiles
- Differenced Normalized Burn Ratio (dNBR) rasters
- Pre-fire and post-fire reflectance data
- Fire metadata

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, matplotlib, pandas, deap, scikit-learn, fiona, rasterio, geopandas

### Running the Model

1. **Parameter Optimization**:
   ```
   jupyter notebook calibration.ipynb
   ```
   This will find optimal parameters using genetic algorithms and save them to `optimal_parameters.json`.

2. **Simulation and Intervention Analysis**:
   ```
   jupyter notebook main.ipynb
   ```
   This will load the optimized parameters and run simulations to evaluate different intervention strategies.

## Evaluation Metrics

The model's performance is evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of correctly predicted burned areas among all predicted burn areas
- **Recall**: Proportion of actual burned areas that were correctly predicted
- **F1 Score**: Harmonic mean of precision and recall
- **Sørensen Index**: Similarity measure between predicted and actual burned areas

## Intervention Strategies

The model evaluates several intervention strategies:

1. **Strategic Firebreaks**: Creating fuel-free barriers to stop fire spread
2. **Direct Attack**: Directly applying fire retardant to the fire front
3. **Indirect Attack**: Creating control lines away from the fire edge
4. **Combined Attack**: Mixing direct and indirect approaches
5. **Aerial Attack**: Water and retardant drops from aircraft
6. **Burnout Operations**: Controlled burning between control lines and fire
7. **Wet Line Strategy**: Creating lines of wet fuels to stop spread
8. **Point Protection**: Focusing resources on protecting high-value areas
9. **Early Detection & Response**: Rapid initial attack on new ignitions

## Future Improvements

- Integration with real-time weather data
- Machine learning for real-time prediction of fire behavior
- Web-based visualization interface
- Integration with remote sensing data for early detection

## Contributors

- [Your Name] - Initial work and development

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MTBS program for providing fire data
- USGS and USDA Forest Service for environmental datasets
