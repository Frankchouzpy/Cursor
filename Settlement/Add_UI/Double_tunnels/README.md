# Double Tunnels Settlement Calculator

This application calculates and visualizes the settlement profile of double tunnels based on various geological and geometric parameters.

## Features

- User-friendly interface for inputting parameters for two tunnels
- Real-time calculation of settlement profiles for both tunnels and their combined effect
- Interactive plot with zoom, pan, and point marking capabilities
- Ability to save plots as PNG, PDF, or SVG files
- Automatic conversion of angles from degrees to radians
- Error handling for invalid inputs
- Display of minimum settlement points for each curve

## Installation

1. Make sure you have Python 3.7 or higher installed
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Input the required parameters:
   - Tunnel spacing (L) in meters
   - First Tunnel Parameters:
     * Tunnel Depth (HF) in meters
     * Tunnel Radius (RF) in meters
     * Ground Loss Rate (VlF)
     * First Bias Parameter (gama1F)
     * Third Bias Parameter (gama3F)
     * Bias Angle (thetaF) in degrees
     * Ground Influence Angle (betaF) in degrees
   - Second Tunnel Parameters:
     * Tunnel Depth (HS) in meters
     * Tunnel Radius (RS) in meters
     * Ground Loss Rate (VlS)
     * First Bias Parameter (gama1S)
     * Third Bias Parameter (gama3S)
     * Bias Angle (thetaS) in degrees
     * Ground Influence Angle (betaS) in degrees

3. Click "Calculate Settlement" to generate the settlement profiles

4. Use the toolbar buttons to:
   - Zoom in/out
   - Pan the plot
   - Mark points and view coordinates
   - Save the plot as an image file

## Notes

- All angle inputs should be in degrees (the program automatically converts to radians)
- The calculation may take a few seconds depending on the input parameters
- The plot shows three curves:
  * First Tunnel settlement profile
  * Second Tunnel settlement profile
  * Total settlement profile (superposition of both tunnels)
- Minimum points are automatically marked on each curve 