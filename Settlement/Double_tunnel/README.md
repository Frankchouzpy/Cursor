# Double Tunnel Settlement Analysis

This program calculates and visualizes the settlement curves for double tunnels using the superposition principle.

## Features
- Calculates settlement curves for individual tunnels
- Combines settlements using superposition principle
- Visualizes settlement curves with minimum points marked
- Supports custom tunnel parameters

## Requirements
- Python 3.7+
- Required packages listed in requirements.txt

## Usage
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the program:
```bash
python main.py
```

3. Follow the prompts to input tunnel parameters:
   - Tunnel spacing (L)
   - First tunnel parameters (HF, RF, VlF, gama1F, gama3F, thetaF, faiF, kF)
   - Second tunnel parameters (HS, RS, VlS, gama1S, gama3S, thetaS, faiS, kS)

The program will generate a plot showing the settlement curves and save it as 'settlement_curve.png'. 