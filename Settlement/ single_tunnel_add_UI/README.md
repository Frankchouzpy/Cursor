# Single Tunnel Settlement Calculator

This application calculates and visualizes the settlement profile of a single tunnel based on various geological and geometric parameters.

## Features

- User-friendly interface for inputting tunnel parameters
- Real-time calculation of settlement profiles
- Interactive plot with zoom, pan, and point marking capabilities
- Ability to save plots as PNG files
- Automatic conversion of angles from degrees to radians
- Error handling for invalid inputs

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
   - Tunnel Depth (H) in meters
   - Tunnel Radius (R) in meters
   - Ground Loss Rate (Vl)
   - First Bias Parameter (gama1)
   - Third Bias Parameter (gama3)
   - Bias Angle (theta) in degrees
   - Ground Influence Angle (beta) in degrees

3. Click "Calculate Settlement" to generate the settlement profile

4. Use the toolbar buttons to:
   - Zoom in/out
   - Pan the plot
   - Mark points and view coordinates
   - Save the plot as a PNG file

## Notes

- All angle inputs should be in degrees (the program automatically converts to radians)
- The calculation may take a few seconds depending on the input parameters
- The plot shows the settlement profile from -50m to 50m from the tunnel center 