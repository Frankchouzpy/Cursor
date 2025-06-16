import sys
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QMessageBox, QFileDialog, QStatusBar, QToolBar,
                            QAction, QDockWidget, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon
import math

class DoubleTunnelCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Tunnels Settlement Calculator")
        self.setMinimumSize(1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create input sections
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # Create spacing input
        spacing_group = QGroupBox("Tunnel Spacing")
        spacing_layout = QHBoxLayout()
        spacing_layout.setSpacing(5) # Smaller spacing
        spacing_layout.addWidget(QLabel("L (m):"))
        self.spacing_input = QLineEdit()
        self.spacing_input.setFixedWidth(100)
        spacing_layout.addWidget(self.spacing_input)
        spacing_layout.addStretch() # Push widgets to the left
        spacing_group.setLayout(spacing_layout)
        input_layout.addWidget(spacing_group)
        
        # Create first tunnel parameters group
        first_tunnel_group = QGroupBox("First Tunnel Parameters")
        first_tunnel_layout = QHBoxLayout()
        
        # Parameters for first tunnel
        self.first_tunnel_params = {
            'HF': 'Tunnel Depth (m)',
            'RF': 'Tunnel Radius (m)',
            'VlF': 'Ground Loss Rate',
            'gama1F': 'First Bias Parameter',
            'gama3F': 'Third Bias Parameter',
            'thetaF': 'Bias Angle (degrees)',
            'betaF': 'Ground Influence Angle (degrees)'
        }
        
        self.first_tunnel_inputs = {}
        for param, label in self.first_tunnel_params.items():
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.addWidget(QLabel(label))
            input_field = QLineEdit()
            input_field.setFixedWidth(100)
            param_layout.addWidget(input_field)
            self.first_tunnel_inputs[param] = input_field
            first_tunnel_layout.addWidget(param_widget)
        
        first_tunnel_group.setLayout(first_tunnel_layout)
        input_layout.addWidget(first_tunnel_group)
        
        # Create second tunnel parameters group
        second_tunnel_group = QGroupBox("Second Tunnel Parameters")
        second_tunnel_layout = QHBoxLayout()
        
        # Parameters for second tunnel
        self.second_tunnel_params = {
            'HS': 'Tunnel Depth (m)',
            'RS': 'Tunnel Radius (m)',
            'VlS': 'Ground Loss Rate',
            'gama1S': 'First Bias Parameter',
            'gama3S': 'Third Bias Parameter',
            'thetaS': 'Bias Angle (degrees)',
            'betaS': 'Ground Influence Angle (degrees)'
        }
        
        self.second_tunnel_inputs = {}
        for param, label in self.second_tunnel_params.items():
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.addWidget(QLabel(label))
            input_field = QLineEdit()
            input_field.setFixedWidth(100)
            param_layout.addWidget(input_field)
            self.second_tunnel_inputs[param] = input_field
            second_tunnel_layout.addWidget(param_widget)
        
        second_tunnel_group.setLayout(second_tunnel_layout)
        input_layout.addWidget(second_tunnel_group)
        
        # Add calculate button
        calculate_btn = QPushButton("Calculate Settlement")
        calculate_btn.clicked.connect(self.calculate_settlement)
        input_layout.addWidget(calculate_btn)
        
        layout.addWidget(input_widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        # Create status bar for coordinates
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Create dock widget for data points
        self.create_data_dock()
        
        # New members for coordinate display and point marking toggle
        self.coordinate_display_active = False
        self.motion_cid = None
        
        self.mark_points_active = False
        self.click_cid = None
        
        # Add custom plot toolbar with toggleable features
        self.create_custom_plot_toolbar()
        
        # Initialize data storage
        self.data_points = []
        self.original_limits = None
        
    def create_custom_plot_toolbar(self):
        """Create a custom toolbar for additional plot actions."""
        custom_toolbar = self.addToolBar("Custom Plot Tools")
        custom_toolbar.setObjectName("CustomPlotToolbar")

        # Action for toggling coordinate display on hover
        self.toggle_coords_action = QAction(QIcon(), "Enable Coords Hover", self)
        self.toggle_coords_action.setToolTip("Toggle display of mouse coordinates on hover")
        self.toggle_coords_action.setCheckable(True)
        self.toggle_coords_action.setChecked(self.coordinate_display_active)
        self.toggle_coords_action.triggered.connect(self.toggle_coordinate_display)
        custom_toolbar.addAction(self.toggle_coords_action)

        # Action for toggling point marking on click
        self.toggle_mark_action = QAction(QIcon(), "Enable Mark Points", self)
        self.toggle_mark_action.setToolTip("Toggle marking points on plot with a click")
        self.toggle_mark_action.setCheckable(True)
        self.toggle_mark_action.setChecked(self.mark_points_active)
        self.toggle_mark_action.triggered.connect(self.toggle_mark_points)
        custom_toolbar.addAction(self.toggle_mark_action)
        
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

    def create_data_dock(self):
        """Create a dock widget to display marked points"""
        dock = QDockWidget("Marked Points", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.points_text = QTextEdit()
        self.points_text.setReadOnly(True)
        dock.setWidget(self.points_text)
        
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def update_status_bar(self, event):
        """Update status bar with current mouse coordinates"""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.statusBar.showMessage(f'x: {x:.3f}, y: {y:.3f}')
        else:
            self.statusBar.clearMessage()
            
    def on_click(self, event):
        """Handle mouse clicks for marking points"""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.figure.gca().plot(x, y, 'ro')
            self.figure.gca().text(x, y, f'({x:.2f}, {y:.2f})')
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
            
    def calculate_w(self, x, H, R, Vl, gama1, gama3, theta, beta):
        """Calculate settlement for a single tunnel"""
        # Calculate intermediate variables
        u0 = R - R * math.sqrt(1 - Vl)
        u2 = R * math.sqrt(1 - Vl) - R * (1 - Vl) / (math.sqrt(1 - Vl) + gama1)
        A = H - R
        B = H + R
        
        def f_integrand(mu, sigma, x):
            C = -math.sqrt(R**2 - (mu - H)**2)
            D = -C
            f1 = math.tan(beta) / mu
            f2 = -math.pi * math.tan(beta)**2 * (x - sigma)**2 / mu**2
            return f1 * math.exp(f2)
            
        def g_integrand(muhat, sigma_hat, x):
            A1 = -R + u0 + gama1 * R
            B1 = -A1
            C1 = -(R - u0 + u2) * math.sqrt(1 - (muhat/(R - u0 - gama1 * R))**2)
            D1 = -C1
            g11 = H + (gama3 * R + muhat) * math.cos(theta) + sigma_hat * math.sin(theta)
            g1 = math.tan(beta) / g11
            g21 = -math.pi * math.tan(beta)**2 * (x + (gama3 * R + muhat) * math.sin(theta) - sigma_hat * math.cos(theta))**2
            g22 = (H + (gama3 * R + muhat) * math.cos(theta) + sigma_hat * math.sin(theta))**2
            g2 = g21 / g22
            return g1 * math.exp(g2)
            
        # Calculate F(x) using double integration
        def F_integrand(mu, sigma):
            return f_integrand(mu, sigma, x)
            
        F, _ = integrate.dblquad(
            lambda sigma, mu: F_integrand(mu, sigma),
            A, B,
            lambda mu: -math.sqrt(R**2 - (mu - H)**2),
            lambda mu: math.sqrt(R**2 - (mu - H)**2)
        )
        
        # Calculate G(x) using double integration
        def G_integrand(muhat, sigma_hat):
            return g_integrand(muhat, sigma_hat, x)
            
        G, _ = integrate.dblquad(
            lambda sigma_hat, muhat: G_integrand(muhat, sigma_hat),
            -R + u0 + gama1 * R,
            R - u0 - gama1 * R,
            lambda muhat: -(R - u0 + u2) * math.sqrt(1 - (muhat/(R - u0 - gama1 * R))**2),
            lambda muhat: (R - u0 + u2) * math.sqrt(1 - (muhat/(R - u0 - gama1 * R))**2)
        )
        
        # Calculate final settlement
        W = -1000 * (F - G)
        return W
            
    def calculate_settlement(self):
        try:
            self.statusBar.showMessage("Calculation in progress...")
            QApplication.processEvents() # Process events to update UI immediately

            # Get spacing value
            L = float(self.spacing_input.text())
            
            # Get first tunnel parameters
            first_params = {}
            for param, field in self.first_tunnel_inputs.items():
                value = float(field.text())
                if param in ['thetaF', 'betaF']:
                    value = math.radians(value)
                first_params[param] = value
                
            # Get second tunnel parameters
            second_params = {}
            for param, field in self.second_tunnel_inputs.items():
                value = float(field.text())
                if param in ['thetaS', 'betaS']:
                    value = math.radians(value)
                second_params[param] = value
                
            # Calculate settlement
            x = np.linspace(-100, 100, 1000)  # x range for plotting
            
            # Calculate first tunnel settlement
            tunnel1_args = {
                'H': first_params['HF'],
                'R': first_params['RF'],
                'Vl': first_params['VlF'],
                'gama1': first_params['gama1F'],
                'gama3': first_params['gama3F'],
                'theta': first_params['thetaF'],
                'beta': first_params['betaF']
            }
            first_tunnel_w = [self.calculate_w(xi + L, **tunnel1_args) for xi in x]
            
            # Calculate second tunnel settlement
            tunnel2_args = {
                'H': second_params['HS'],
                'R': second_params['RS'],
                'Vl': second_params['VlS'],
                'gama1': second_params['gama1S'],
                'gama3': second_params['gama3S'],
                'theta': second_params['thetaS'],
                'beta': second_params['betaS']
            }
            second_tunnel_w = [self.calculate_w(xi - L, **tunnel2_args) for xi in x]
            
            # Calculate total settlement
            total_w = [f + s for f, s in zip(first_tunnel_w, second_tunnel_w)]
            
            # Plot results
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot all three curves
            ax.plot(x, first_tunnel_w, label='First Tunnel', color='blue')
            ax.plot(x, second_tunnel_w, label='Second Tunnel', color='red')
            ax.plot(x, total_w, label='Total Settlement', color='green')
            
            # Find and mark minimum points
            min_first_idx = np.argmin(first_tunnel_w)
            min_second_idx = np.argmin(second_tunnel_w)
            min_total_idx = np.argmin(total_w)
            
            # Mark minimum points
            ax.plot(x[min_first_idx], first_tunnel_w[min_first_idx], 'bo')
            ax.plot(x[min_second_idx], second_tunnel_w[min_second_idx], 'ro')
            ax.plot(x[min_total_idx], total_w[min_total_idx], 'go')
            
            # Add annotations for minimum points
            ax.annotate(f'({x[min_first_idx]:.1f}, {first_tunnel_w[min_first_idx]:.1f})',
                       (x[min_first_idx], first_tunnel_w[min_first_idx]),
                       xytext=(10, 10), textcoords='offset points')
            ax.annotate(f'({x[min_second_idx]:.1f}, {second_tunnel_w[min_second_idx]:.1f})',
                       (x[min_second_idx], second_tunnel_w[min_second_idx]),
                       xytext=(10, -10), textcoords='offset points')
            ax.annotate(f'({x[min_total_idx]:.1f}, {total_w[min_total_idx]:.1f})',
                       (x[min_total_idx], total_w[min_total_idx]),
                       xytext=(10, 10), textcoords='offset points')
            
            # Add grid and labels
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Distance from Center (m)')
            ax.set_ylabel('Settlement (mm)')
            ax.set_title('Double Tunnels Settlement Profile')
            
            # Add legend
            ax.legend()
            
            # Store original limits
            self.original_limits = {
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim()
            }
            
            # Clear previous data points
            self.data_points = []
            self.update_points_display()
            
            # Redraw canvas
            self.canvas.draw()
            
            QMessageBox.information(self, "Calculation Complete", "Settlement calculation finished successfully!")
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for all parameters.")
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"An error occurred: {str(e)}")
        finally:
            self.statusBar.clearMessage() # Clear message after calculation is done or an error occurs

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DoubleTunnelCalculator()
    window.show()
    sys.exit(app.exec_()) 