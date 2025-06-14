import numpy as np
from scipy import integrate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_initial_parameters(H, R, Vl, gama1):
    """
    Calculate initial parameters for settlement calculation.
    
    Args:
        H (float): Tunnel depth
        R (float): Tunnel radius
        Vl (float): Ground loss rate
        gama1 (float): First bias parameter
        
    Returns:
        tuple: (u0, u2, Z, A, B)
    """
    u0 = R - R * np.sqrt(1 - Vl)
    u2 = R * np.sqrt(1 - Vl) - R * (1 - Vl) / (np.sqrt(1 - Vl) + gama1)
    Z = H - R
    A = H - R
    B = H + R
    return u0, u2, Z, A, B

def calculate_integration_limits(H, R, mu):
    """
    Calculate integration limits C and D.
    
    Args:
        H (float): Tunnel depth
        R (float): Tunnel radius
        mu (float): Integration variable
        
    Returns:
        tuple: (C, D)
    """
    C = -np.sqrt(R**2 - (mu - H)**2)
    D = -C
    return C, D

def calculate_f_function(x, H, R, fai, k, mu, sigma):
    """
    Calculate the f(x) function for integration.
    
    Args:
        x (float): Distance from tunnel center
        H (float): Tunnel depth
        R (float): Tunnel radius
        fai (float): Average internal friction angle
        k (float): Soil parameter
        mu (float): First integration variable
        sigma (float): Second integration variable
        
    Returns:
        float: f(x) value
    """
    tanb = H / (np.sqrt(2 * np.pi) * (1 - 0.02 * fai) * (H - k * (H - R)))
    f_1 = tanb / mu
    f_2 = -np.pi * tanb**2 * (x - sigma)**2 / mu**2
    return f_1 * np.exp(f_2)

def calculate_g_function(x, H, R, gama1, gama3, theta, fai, k, muhat, sigma_hat):
    """
    Calculate the g(x) function for integration.
    
    Args:
        x (float): Distance from tunnel center
        H (float): Tunnel depth
        R (float): Tunnel radius
        gama1 (float): First bias parameter
        gama3 (float): Third bias parameter
        theta (float): Bias angle in degrees
        fai (float): Average internal friction angle
        k (float): Soil parameter
        muhat (float): First integration variable
        sigma_hat (float): Second integration variable
        
    Returns:
        float: g(x) value
    """
    # Convert theta from degrees to radians
    theta_rad = np.radians(theta)
    
    tanb = H / (np.sqrt(2 * np.pi) * (1 - 0.02 * fai) * (H - k * (H - R)))
    g_11 = H + (gama3 * R + muhat) * np.cos(theta_rad) + sigma_hat * np.sin(theta_rad)
    g_1 = tanb / g_11
    g_21 = -np.pi * tanb**2 * (x + (gama3 * R + muhat) * np.sin(theta_rad) - sigma_hat * np.cos(theta_rad))**2
    g_22 = g_11**2
    g_2 = g_21 / g_22
    return g_1 * np.exp(g_2)

def calculate_settlement(x, H, R, Vl, gama1, gama3, theta, fai, k):
    """
    Calculate the settlement W(x) for a single tunnel.
    
    Args:
        x (float): Distance from tunnel center
        H (float): Tunnel depth
        R (float): Tunnel radius
        Vl (float): Ground loss rate
        gama1 (float): First bias parameter
        gama3 (float): Third bias parameter
        theta (float): Bias angle
        fai (float): Average internal friction angle
        k (float): Soil parameter
        
    Returns:
        float: Settlement value W(x)
    """
    try:
        u0, u2, Z, A, B = calculate_initial_parameters(H, R, Vl, gama1)
        
        def f_integrand(sigma, mu):
            C, D = calculate_integration_limits(H, R, mu)
            return calculate_f_function(x, H, R, fai, k, mu, sigma)
        
        def g_integrand(sigma_hat, muhat):
            A_1 = -R + u0 + gama1 * R
            B_1 = -A_1
            C_1 = -(R - u0 + u2) * np.sqrt(1 - (muhat / (R - u0 - gama1 * R))**2)
            D_1 = -C_1
            return calculate_g_function(x, H, R, gama1, gama3, theta, fai, k, muhat, sigma_hat)
        
        # Calculate F(x) using double integration
        F_x, _ = integrate.dblquad(
            lambda sigma, mu: f_integrand(sigma, mu),
            A, B,
            lambda mu: calculate_integration_limits(H, R, mu)[0],
            lambda mu: calculate_integration_limits(H, R, mu)[1]
        )
        
        # Calculate G(x) using double integration
        G_x, _ = integrate.dblquad(
            lambda sigma_hat, muhat: g_integrand(sigma_hat, muhat),
            -R + u0 + gama1 * R, R - u0 - gama1 * R,
            lambda muhat: -(R - u0 + u2) * np.sqrt(1 - (muhat / (R - u0 - gama1 * R))**2),
            lambda muhat: (R - u0 + u2) * np.sqrt(1 - (muhat / (R - u0 - gama1 * R))**2)
        )
        
        return -(F_x - G_x) / 1000
        
    except Exception as e:
        logger.error(f"Error calculating settlement: {str(e)}")
        return 0.0 