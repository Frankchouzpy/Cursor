 # Back Analysis UI for Double Tunnels Settlement

This program performs back-analysis of tunnel parameters using measured settlement data. It estimates unknown parameters by fitting the theoretical settlement curve to the observed data.

## Features

- Interactive UI for parameter input
- Support for partial parameter input (some parameters can be unknown)
- Automatic parameter estimation using optimization algorithms
- Visualization of fitting results
- Export of results

## Required Parameters

### Required Input Parameters
- HF, HS: Tunnel depths (5-40)
- RF, RS: Tunnel radii (2-6)
- betaF, betaS: Strata influence angles (0-90)

### Optional Input Parameters
- VlF, VlS: Ground loss rates (0.001-0.04)

### Parameters to be Estimated
- gama1F, gama1S: First bias parameters (0.001-0.01)
- gama3F, gama3S: Third bias parameters (0.001-0.01)
- thetaF, thetaS: Bias angles (0-90)

## Installation

1. Ensure Python 3.7+ is installed
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the program:
   ```
   python main.py
   ```
2. Input known parameters
3. Enter 'N' for unknown parameters
4. Click "Start Analysis" to begin parameter estimation
5. View results and fitted curve
6. Export results if needed

## Data Format

The program uses train_data.xlsx which should contain:
- Column 1: x coordinates
- Column 2: Measured settlement values TotalT(x)

## Optimization Method

The program uses a two-phase optimization approach:
1. Global optimization (PSO/GA) for initial parameter estimation
2. Local optimization (Levenberg-Marquardt) for fine-tuning

## Output

- Fitted parameter values
- RMSE, MAE, and R² statistics
- Plot of measured vs. fitted settlement curves
- Parameter sensitivity analysis