import numpy as np
import matplotlib.pyplot as plt
from settlement_calculator import calculate_settlement
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tunnel_parameters(tunnel_name):
    """
    Get tunnel parameters from user input.
    
    Args:
        tunnel_name (str): Name of the tunnel (First/Second)
        
    Returns:
        tuple: (H, R, Vl, gama1, gama3, theta, fai, k)
    """
    print(f"\nEnter parameters for {tunnel_name} tunnel:")
    H = float(input("Tunnel depth (H): "))
    R = float(input("Tunnel radius (R): "))
    Vl = float(input("Ground loss rate (Vl): "))
    gama1 = float(input("First bias parameter (gama1): "))
    gama3 = float(input("Third bias parameter (gama3): "))
    theta = float(input("Bias angle (theta): "))
    fai = float(input("Average internal friction angle (fai): "))
    k = float(input("Soil parameter (k): "))
    return H, R, Vl, gama1, gama3, theta, fai, k

def calculate_settlement_curves(x_range, L, first_params, second_params):
    """
    Calculate settlement curves for both tunnels and their superposition.
    
    Args:
        x_range (numpy.ndarray): Array of x values
        L (float): Tunnel spacing
        first_params (tuple): Parameters for first tunnel
        second_params (tuple): Parameters for second tunnel
        
    Returns:
        tuple: (first_tunnel, second_tunnel, total_settlement)
    """
    first_tunnel = []
    second_tunnel = []
    total_settlement = []
    n = len(x_range)
    for i, x in enumerate(x_range):
        if i % max(1, n // 10) == 0:
            print(f"Progress: {i/n*100:.1f}%")
            sys.stdout.flush()
        ft = calculate_settlement(x + L, *first_params)
        st = calculate_settlement(x - L, *second_params)
        first_tunnel.append(ft)
        second_tunnel.append(st)
        total_settlement.append(ft + st)
    return np.array(first_tunnel), np.array(second_tunnel), np.array(total_settlement)

def plot_settlement_curves(x_range, first_tunnel, second_tunnel, total_settlement):
    """
    Plot settlement curves and save the figure.
    
    Args:
        x_range (numpy.ndarray): Array of x values
        first_tunnel (numpy.ndarray): First tunnel settlement values
        second_tunnel (numpy.ndarray): Second tunnel settlement values
        total_settlement (numpy.ndarray): Total settlement values
    """
    plt.figure(figsize=(12, 8))
    
    # Plot curves
    plt.plot(x_range, first_tunnel, 'b-', label='First Tunnel')
    plt.plot(x_range, second_tunnel, 'r-', label='Second Tunnel')
    plt.plot(x_range, total_settlement, 'g-', label='Total Settlement')
    
    # Find and mark minimum points
    min_first = x_range[np.argmin(first_tunnel)]
    min_second = x_range[np.argmin(second_tunnel)]
    min_total = x_range[np.argmin(total_settlement)]
    
    plt.plot(min_first, np.min(first_tunnel), 'bo', label=f'Min First: ({min_first:.2f}, {np.min(first_tunnel):.2f})')
    plt.plot(min_second, np.min(second_tunnel), 'ro', label=f'Min Second: ({min_second:.2f}, {np.min(second_tunnel):.2f})')
    plt.plot(min_total, np.min(total_settlement), 'go', label=f'Min Total: ({min_total:.2f}, {np.min(total_settlement):.2f})')
    
    # Customize plot
    plt.grid(True)
    plt.xlabel('Distance from Center (m)')
    plt.ylabel('Settlement (m)')
    plt.title('Tunnel Settlement Curves')
    plt.legend()
    
    # Save plot
    plt.savefig('settlement_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Get tunnel spacing
        L = float(input("Enter tunnel spacing (L): "))
        
        # Get parameters for both tunnels
        first_params = get_tunnel_parameters("First")
        second_params = get_tunnel_parameters("Second")
        
        # Calculate settlement curves
        x_range = np.linspace(-50, 50, 1000)
        first_tunnel, second_tunnel, total_settlement = calculate_settlement_curves(
            x_range, L, first_params, second_params
        )
        
        # Plot and save results
        plot_settlement_curves(x_range, first_tunnel, second_tunnel, total_settlement)
        logger.info("Settlement curves have been calculated and saved as 'settlement_curve.png'")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 