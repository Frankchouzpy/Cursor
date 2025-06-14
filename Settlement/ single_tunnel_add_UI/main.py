import sys
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QMessageBox, QFileDialog, QStatusBar, QToolBar,
                            QAction, QDockWidget, QTextEdit)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon
import math

class SettlementCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single Tunnel Settlement Calculator")
        self.setMinimumSize(1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create input fields
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setSpacing(20)
        
        # Parameters to input
        self.parameters = {
            'H': 'Tunnel Depth (m)',
            'R': 'Tunnel Radius (m)',
            'Vl': 'Ground Loss Rate',
            'gama1': 'First Bias Parameter',
            'gama3': 'Third Bias Parameter',
            'theta': 'Bias Angle (degrees)',
            'beta': 'Ground Influence Angle (degrees)'
        }
        
        self.input_fields = {}
        for param, label in self.parameters.items():
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.addWidget(QLabel(label))
            input_field = QLineEdit()
            input_field.setFixedWidth(100)
            param_layout.addWidget(input_field)
            self.input_fields[param] = input_field
            input_layout.addWidget(param_widget)
        
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
        self.motion_cid = None # To store connection ID for motion event

        self.mark_points_active = False # New flag for point marking
        self.click_cid = None # To store connection ID for click event
        
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
            # Disconnect the motion event
            if self.motion_cid:
                self.canvas.mpl_disconnect(self.motion_cid)
                self.motion_cid = None
            self.statusBar.clearMessage() # Clear message when disabled
        else:
            # Connect the motion event
            self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.update_status_bar)
        
        self.coordinate_display_active = not self.coordinate_display_active
        self.toggle_coords_action.setChecked(self.coordinate_display_active)
        self.toggle_coords_action.setText("Disable Coords Hover" if self.coordinate_display_active else "Enable Coords Hover")

    def toggle_mark_points(self):
        """Toggles the ability to mark points on the plot with a click."""
        if self.mark_points_active:
            # Disconnect the click event
            if self.click_cid:
                self.canvas.mpl_disconnect(self.click_cid)
                self.click_cid = None
        else:
            # Connect the click event
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
            
    def calculate_settlement(self):
        try:
            # Get input values
            params = {}
            for param, field in self.input_fields.items():
                value = float(field.text())
                if param in ['theta', 'beta']:
                    value = math.radians(value)  # Convert to radians
                params[param] = value
                
            # Calculate settlement
            x = np.linspace(-50, 50, 1000)  # x range for plotting
            w = [self.calculate_w(xi, **params) for xi in x]
            
            # Plot results
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            line = ax.plot(x, w, label='Settlement Profile')[0]
            
            # Add grid and labels
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Distance from Tunnel Center (m)')
            ax.set_ylabel('Settlement (m)')
            ax.set_title('Tunnel Settlement Profile')
            
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
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for all parameters.")
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"An error occurred: {str(e)}")
            
    def calculate_w(self, x, H, R, Vl, gama1, gama3, theta, beta):
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SettlementCalculator()
    window.show()
    sys.exit(app.exec()) 