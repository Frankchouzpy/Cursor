import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QGroupBox, QMessageBox, QFileDialog, QStatusBar,
                            QDockWidget, QTextEdit, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.optimize import minimize, differential_evolution
from scipy import integrate
import math
import warnings
warnings.filterwarnings('ignore')

# Define the AnalysisThread class to run calculations in a separate thread
class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, known_params, unknown_params_list, param_ranges, measured_data, parent=None):
        super().__init__(parent)
        self.known_params = known_params
        self.unknown_params_list = unknown_params_list # List of parameter names to be optimized
        self.param_ranges = param_ranges
        self.measured_data = measured_data

    def calculate_w(self, x, H, R, Vl, gama1, gama3, theta, beta):
        """Calculate settlement for a single tunnel"""
        # Convert angles from degrees to radians
        theta_rad = math.radians(theta)
        beta_rad = math.radians(beta)
        
        # Calculate intermediate variables
        try:
            u0 = R - R * math.sqrt(1 - Vl)
            u2 = R * math.sqrt(1 - Vl) - R * (1 - Vl) / (math.sqrt(1 - Vl) + gama1)
        except ValueError as e:
            print(f"Initial calculation error in calculate_w (u0/u2): {e}, Vl={Vl}, gama1={gama1}")
            return np.nan
        except ZeroDivisionError:
            print(f"ZeroDivisionError in calculate_w (u2): Vl={Vl}, gama1={gama1}")
            return np.nan

        A = H - R
        B = H + R
        
        def f_integrand(mu, sigma, x_val):
            mu_safe = mu
            if abs(mu) < 1e-9: 
                mu_safe = 1e-9 if mu >= 0 else -1e-9 

            try:
                sqrt_arg = R**2 - (mu - H)**2
                if sqrt_arg < 0:
                    raise ValueError(f"sqrt_arg < 0 in f_integrand: {sqrt_arg}")
                C = -math.sqrt(sqrt_arg)
            except ValueError as ve:
                print(f"Error in f_integrand (C calculation): {ve}, mu={mu}, H={H}, R={R}")
                return np.nan 

            f1 = math.tan(beta_rad) / mu_safe
            f2 = -math.pi * math.tan(beta_rad)**2 * (x_val - sigma)**2 / mu_safe**2
            return f1 * math.exp(f2)
            
        def g_integrand(muhat, sigma_hat, x_val):
            denominator = (R - u0 - gama1 * R)
            if abs(denominator) < 1e-9:
                print(f"ZeroDivisionError in g_integrand (denominator): R={R}, u0={u0}, gama1={gama1}")
                return np.nan

            try:
                sqrt_arg_g = 1 - (muhat/denominator)**2
                if sqrt_arg_g < 0:
                    raise ValueError(f"sqrt_arg_g < 0 in g_integrand: {sqrt_arg_g}")
                C1 = -(R - u0 + u2) * math.sqrt(sqrt_arg_g)
            except ValueError as ve:
                print(f"Error in g_integrand (C1 calculation): {ve}, muhat={muhat}, R={R}, u0={u0}, gama1={gama1}, u2={u2}")
                return np.nan 

            g11_val = H + (gama3 * R + muhat) * math.cos(theta_rad) + sigma_hat * math.sin(theta_rad)
            if abs(g11_val) < 1e-9: 
                g11_val = 1e-9 if g11_val >= 0 else -1e-9

            g1 = math.tan(beta_rad) / g11_val
            g21 = -math.pi * math.tan(beta_rad)**2 * (x_val + (gama3 * R + muhat) * math.sin(theta_rad) - sigma_hat * math.cos(theta_rad))**2
            g22 = g11_val**2
            g2 = g21 / g22
            return g1 * math.exp(g2)
            
        def F_integrand(mu, sigma):
            return f_integrand(mu, sigma, x)
            
        try:
            F, _ = integrate.dblquad(
                lambda sigma, mu: F_integrand(mu, sigma),
                A, B,
                lambda mu: -math.sqrt(max(0, R**2 - (mu - H)**2)), 
                lambda mu: math.sqrt(max(0, R**2 - (mu - H)**2))   
            )
        except Exception as e:
            print(f"Error during F integral calculation: {e}, A={A}, B={B}, H={H}, R={R}")
            F = np.nan 
        
        def G_integrand(muhat, sigma_hat):
            return g_integrand(muhat, sigma_hat, x)
            
        try:
            G, _ = integrate.dblquad(
                lambda sigma_hat, muhat: G_integrand(muhat, sigma_hat),
                -R + u0 + gama1 * R,
                R - u0 - gama1 * R,
                lambda muhat: -(R - u0 + u2) * math.sqrt(max(0, 1 - (muhat/(R - u0 - gama1 * R))**2)), 
                lambda muhat: (R - u0 + u2) * math.sqrt(max(0, 1 - (muhat/(R - u0 - gama1 * R))**2))  
            )
        except Exception as e:
            print(f"Error during G integral calculation: {e}, R={R}, u0={u0}, gama1={gama1}, u2={u2}")
            G = np.nan 
        
        W = -1000 * (F - G)
        if np.isnan(W) or np.isinf(W):
            raise ValueError("Settlement calculation resulted in NaN/Inf. Check parameters and integrals.")
        return W

    def settlement_function(self, x, params):
        L = params['L']
        
        tunnel1_args = {
            'H': params['HF'], 'R': params['RF'], 'Vl': params['VlF'],
            'gama1': params['gama1F'], 'gama3': params['gama3F'],
            'theta': params['thetaF'], 'beta': params['betaF']
        }
        tunnel2_args = {
            'H': params['HS'], 'R': params['RS'], 'Vl': params['VlS'],
            'gama1': params['gama1S'], 'gama3': params['gama3S'],
            'theta': params['thetaS'], 'beta': params['betaS']
        }
        
        first_tunnel_w = []
        second_tunnel_w = []
        for xi in x:
            try:
                first_tunnel_w.append(self.calculate_w(xi + L, **tunnel1_args))
            except (ValueError, ZeroDivisionError):
                first_tunnel_w.append(np.nan) 
            
            try:
                second_tunnel_w.append(self.calculate_w(xi - L, **tunnel2_args))
            except (ValueError, ZeroDivisionError):
                second_tunnel_w.append(np.nan) 

        total_w = [f + s for f, s in zip(first_tunnel_w, second_tunnel_w)]
        return np.array(total_w)

    def objective_function(self, unknown_values, known_params_snapshot, unknown_params_names):
        params = known_params_snapshot.copy()
        for param_name, value in zip(unknown_params_names, unknown_values):
            params[param_name] = value
        
        x = self.measured_data.iloc[:, 0].values
        y_pred = self.settlement_function(x, params)
        y_true = self.measured_data.iloc[:, 1].values
        
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print(f"Warning: NaN/Inf encountered for params: {params}")
            return 1e15

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    def run(self):
        try:
            self.progress.emit("Starting optimization...")
            
            bounds = [self.param_ranges[param] for param in self.unknown_params_list]
            x0 = [(b[0] + b[1])/2 for b in bounds]
            
            # Global optimization
            self.progress.emit("Performing global optimization (Differential Evolution)... This may take a while.")
            print("Starting Differential Evolution...")
            result_global = differential_evolution(
                self.objective_function,
                bounds=bounds,
                args=(self.known_params, self.unknown_params_list),
                maxiter=200,
                popsize=30,
                tol=1e-4,
                disp=True
            )
            print(f"Differential Evolution finished with RMSE: {result_global.fun:.6f}")
            
            # Local optimization
            self.progress.emit("Performing local optimization (L-BFGS-B)...")
            print("Starting L-BFGS-B optimization...")
            result_local = minimize(
                self.objective_function,
                x0=result_global.x,
                args=(self.known_params, self.unknown_params_list),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-7, 'disp': True}
            )
            print(f"L-BFGS-B finished with RMSE: {result_local.fun:.6f}")
            
            if not result_local.success:
                self.progress.emit(f"Optimization did not fully converge: {result_local.message}")
                print(f"Optimization warning: {result_local.message}")

            final_params = self.known_params.copy()
            for param, value in zip(self.unknown_params_list, result_local.x):
                final_params[param] = value
            
            self.finished.emit({
                'final_params': final_params,
                'rmse': result_local.fun,
                'unknown_estimated': dict(zip(self.unknown_params_list, result_local.x))
            })
            
        except Exception as e:
            print(f"Critical error in AnalysisThread.run: {e}")
            self.error.emit(f"Optimization failed: {str(e)}. Check console for details.")

class BackAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tunnel Settlement Back Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        self.measured_data = None
        self.param_ranges = {
            'HF': (5, 40), 'HS': (5, 40),
            'RF': (2, 6), 'RS': (2, 6),
            'VlF': (0.001, 0.04), 'VlS': (0.001, 0.04),
            'gama1F': (0.001, 0.01), 'gama1S': (0.001, 0.01),
            'gama3F': (0.001, 0.01), 'gama3S': (0.001, 0.01),
            'thetaF': (0, 89.999), 'thetaS': (0, 89.999),
            'betaF': (0, 89.999), 'betaS': (0, 89.999)    
        }
        
        self.init_ui()
        self.load_data()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        spacing_group = QGroupBox("Tunnel Spacing")
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("L:"))
        self.L_input = QLineEdit()
        spacing_layout.addWidget(self.L_input)
        spacing_group.setLayout(spacing_layout)
        left_layout.addWidget(spacing_group)
        
        first_tunnel_group = QGroupBox("First Tunnel Parameters")
        first_tunnel_layout = QVBoxLayout()
        self.first_tunnel_inputs = {}
        for param in ['HF', 'RF', 'VlF', 'gama1F', 'gama3F', 'thetaF', 'betaF']:
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param}:"))
            input_field = QLineEdit()
            self.first_tunnel_inputs[param] = input_field
            param_layout.addWidget(input_field)
            first_tunnel_layout.addLayout(param_layout)
        first_tunnel_group.setLayout(first_tunnel_layout)
        left_layout.addWidget(first_tunnel_group)
        
        second_tunnel_group = QGroupBox("Second Tunnel Parameters")
        second_tunnel_layout = QVBoxLayout()
        self.second_tunnel_inputs = {}
        for param in ['HS', 'RS', 'VlS', 'gama1S', 'gama3S', 'thetaS', 'betaS']:
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param}:"))
            input_field = QLineEdit()
            self.second_tunnel_inputs[param] = input_field
            param_layout.addWidget(input_field)
            second_tunnel_layout.addLayout(param_layout)
        second_tunnel_group.setLayout(second_tunnel_layout)
        left_layout.addWidget(second_tunnel_group)
        
        self.analyze_button = QPushButton("Start Analysis")
        self.analyze_button.clicked.connect(self.start_analysis_thread)
        left_layout.addWidget(self.analyze_button)
        
        main_layout.addWidget(left_panel, stretch=1)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        
        main_layout.addWidget(right_panel, stretch=2)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # New members for coordinate display and point marking toggle
        self.coordinate_display_active = False
        self.motion_cid = None
        
        self.mark_points_active = False
        self.click_cid = None
        
        # Data points for marking
        self.data_points = []
        
        self.create_custom_plot_toolbar()
        self.create_data_dock()
        
    def create_custom_plot_toolbar(self):
        """Create a custom toolbar for additional plot actions."""
        custom_toolbar = self.addToolBar("Custom Plot Tools")
        custom_toolbar.setObjectName("CustomPlotToolbar")

        self.toggle_coords_action = QAction(QIcon(), "Enable Coords Hover", self)
        self.toggle_coords_action.setToolTip("Toggle display of mouse coordinates on hover")
        self.toggle_coords_action.setCheckable(True)
        self.toggle_coords_action.setChecked(self.coordinate_display_active)
        self.toggle_coords_action.triggered.connect(self.toggle_coordinate_display)
        custom_toolbar.addAction(self.toggle_coords_action)

        self.toggle_mark_action = QAction(QIcon(), "Enable Mark Points", self)
        self.toggle_mark_action.setToolTip("Toggle marking points on plot with a click")
        self.toggle_mark_action.setCheckable(True)
        self.toggle_mark_action.setChecked(self.mark_points_active)
        self.toggle_mark_action.triggered.connect(self.toggle_mark_points)
        custom_toolbar.addAction(self.toggle_mark_action)
        
        self.save_plot_action = QAction(QIcon(), "Save Plot", self)
        self.save_plot_action.setToolTip("Save the current plot")
        self.save_plot_action.triggered.connect(self.save_plot)
        custom_toolbar.addAction(self.save_plot_action)

    def create_data_dock(self):
        """Create a dock widget to display marked points"""
        dock = QDockWidget("Marked Points", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.points_text = QTextEdit()
        self.points_text.setReadOnly(True)
        dock.setWidget(self.points_text)
        
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def toggle_coordinate_display(self):
        """Toggles the display of mouse coordinates in the status bar on hover."""
        if self.coordinate_display_active:
            if self.motion_cid:
                self.canvas.mpl_disconnect(self.motion_cid)
                self.motion_cid = None
            self.statusBar.clearMessage()
        else:
            self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.update_status_bar)
        
        self.coordinate_display_active = not self.coordinate_display_active
        self.toggle_coords_action.setChecked(self.coordinate_display_active)
        self.toggle_coords_action.setText("Disable Coords Hover" if self.coordinate_display_active else "Enable Coords Hover")

    def toggle_mark_points(self):
        """Toggles the ability to mark points on the plot with a click."""
        if self.mark_points_active:
            if self.click_cid:
                self.canvas.mpl_disconnect(self.click_cid)
                self.click_cid = None
        else:
            self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.mark_points_active = not self.mark_points_active
        self.toggle_mark_action.setChecked(self.mark_points_active)
        self.toggle_mark_action.setText("Disable Mark Points" if self.mark_points_active else "Enable Mark Points")

    def update_status_bar(self, event):
        """Update status bar with current mouse coordinates"""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.statusBar.showMessage(f'x: {x:.3f}, y: {y:.3f}')
        else:
            self.statusBar.clearMessage()
            
    def on_click(self, event):
        """Handle mouse clicks for marking points"""
        if event.inaxes and self.mark_points_active:
            x, y = event.xdata, event.ydata
            ax = self.figure.gca() # Get current active axes
            ax.plot(x, y, 'ro')
            ax.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, ha='left', va='bottom')
            self.canvas.draw()
            
            # Store point data
            self.data_points.append((x, y))
            self.update_points_display()
            
    def update_points_display(self):
        """Update the dock widget with marked points"""
        text = "Marked Points:\n"
        for i, (x, y) in enumerate(self.data_points, 1):
            text += f"{i}. x: {x:.3f}, y: {y:.3f}\n"
        self.points_text.setText(text)
        
    def save_plot(self):
        """Save the current plot with enhanced options"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        if file_name:
            self.figure.savefig(file_name, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", "Plot saved successfully!")
            

    def load_data(self):
        try:
            self.measured_data = pd.read_excel('train_data.xlsx')
            if len(self.measured_data.columns) != 2:
                raise ValueError("Data file must contain exactly two columns")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.measured_data = None

    def get_parameters(self):
        params = {}
        unknown_names = []
        
        try:
            params['L'] = float(self.L_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid tunnel spacing value")
            return None, None
        
        all_input_fields = {**self.first_tunnel_inputs, **self.second_tunnel_inputs}
        for param, input_field in all_input_fields.items():
            value = input_field.text().strip().upper()
            if value == 'N':
                unknown_names.append(param)
            else:
                try:
                    params[param] = float(value)
                except ValueError:
                    QMessageBox.warning(self, "Warning", f"Invalid value for {param}")
                    return None, None
        
        return params, unknown_names

    def start_analysis_thread(self):
        known_params, unknown_params_list = self.get_parameters()
        if known_params is None or self.measured_data is None:
            return
        
        if not unknown_params_list:
            QMessageBox.warning(self, "Warning", "No unknown parameters to estimate. Please enter 'N' for parameters to be fitted.")
            return
        
        self.analyze_button.setEnabled(False) # Disable button during analysis
        self.statusBar.showMessage("Initializing analysis... Please wait.")

        self.thread = AnalysisThread(known_params, unknown_params_list, self.param_ranges, self.measured_data)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.progress.connect(self.on_analysis_progress)
        self.thread.start()

    def on_analysis_finished(self, results):
        self.analyze_button.setEnabled(True) # Re-enable button
        self.statusBar.clearMessage()
        
        final_params = results['final_params']
        rmse = results['rmse']
        unknown_estimated = results['unknown_estimated']

        self.plot_results(final_params)

        result_text = "Optimization Results:\n\n"
        for param, value in unknown_estimated.items():
            result_text += f"{param}: {value:.6f}\n"
        result_text += f"\nRMSE: {rmse:.6f}"
        
        QMessageBox.information(self, "Results", result_text)

    def on_analysis_error(self, message):
        self.analyze_button.setEnabled(True) # Re-enable button
        self.statusBar.clearMessage()
        QMessageBox.critical(self, "Error", message)

    def on_analysis_progress(self, message):
        self.statusBar.showMessage(message)

    def plot_results(self, params):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        x = self.measured_data.iloc[:, 0].values
        y_true = self.measured_data.iloc[:, 1].values
        ax.scatter(x, y_true, label='Measured Data', color='blue', alpha=0.5)
        
        # Use a range for plotting the fitted curve
        x_plot = np.linspace(min(x) - 10, max(x) + 10, 200)
        
        try:
            y_pred = self.thread.settlement_function(x_plot, params)
        except Exception as e:
            print(f"Error during plotting settlement_function call: {e}")
            y_pred = np.full_like(x_plot, np.nan) # Plot NaNs if calculation fails
            
        ax.plot(x_plot, y_pred, label='Fitted Curve', color='red')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Settlement (mm)')
        ax.set_title('Settlement Curve Fitting Results')
        ax.legend()
        ax.grid(True)
        
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = BackAnalysisUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
