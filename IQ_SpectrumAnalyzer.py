import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QAction, QFileDialog, QToolBar, QLabel, QComboBox, QSpinBox,
                           QPushButton, QStatusBar, QSplitter, QSizePolicy, QSlider, QGroupBox,
                           QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


class IQPlotCanvas(FigureCanvas):
    """Canvas for plotting IQ waveform data with FFT region highlighting"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        self.axes = self.fig.add_subplot(111, facecolor='#2d2d2d')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # Store reference to the FFT region rectangle
        self.fft_region_rect = None
        # Store current axis limits to maintain zoom
        self.current_xlim = None
        self.current_ylim = None
        # Store plot lines for efficient updates
        self.plot_lines = []


class FFTPlotCanvas(FigureCanvas):
    """Canvas for plotting FFT spectrum with high-end dark styling and performance optimizations"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.fig.add_subplot(111, facecolor='#111111')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # Store current axis limits to maintain zoom for FFT plot
        self.current_xlim = None
        self.current_ylim = None
        
        # Store fixed y-axis limits when y-axis is locked
        self.fixed_ylim = None
        
        # Performance optimization: store plot line for fast updates
        self.fft_line = None
        self.plot_initialized = False
        
        # Setup high-end styling
        self.setup_dark_theme()
    
    def setup_dark_theme(self):
        """Configure dark theme styling for the FFT plot"""
        # Set background colors
        self.fig.patch.set_facecolor('#0a0a0a')
        self.axes.set_facecolor('#111111')
        
        # Configure grid
        self.axes.grid(True, alpha=0.15, color='#444444', linewidth=0.5, linestyle='-')
        
        # Set spine colors
        for spine in self.axes.spines.values():
            spine.set_color('#555555')
            spine.set_linewidth(0.8)
        
        # Configure tick colors
        self.axes.tick_params(colors='#cccccc', which='both', labelsize=9)
        
        # Configure label colors
        self.axes.xaxis.label.set_color('#cccccc')
        self.axes.yaxis.label.set_color('#cccccc')
        self.axes.title.set_color('#ffffff')
    
    def update_fft_data_fast(self, freq_data, mag_data, window_type, fix_y_axis=False):
        """Fast FFT plot update using existing line object"""
        if self.fft_line is None or not self.plot_initialized:
            # Initialize plot on first call
            self.initialize_fft_plot(freq_data, mag_data, window_type, fix_y_axis)
        else:
            # Fast update: only change data, don't recreate plot
            self.fft_line.set_data(freq_data, mag_data)
            
            if fix_y_axis and self.fixed_ylim is not None:
                # Use fixed y-axis limits
                self.axes.set_ylim(self.fixed_ylim)
            elif not fix_y_axis:
                # Auto-scale y-axis
                max_magnitude = np.max(mag_data)
                new_ylim = (max_magnitude - 80, max_magnitude + 10)
                
                # Only update y-limits if there's a significant change (reduces redraws)
                current_ylim = self.axes.get_ylim()
                if abs(current_ylim[0] - new_ylim[0]) > 5 or abs(current_ylim[1] - new_ylim[1]) > 5:
                    self.axes.set_ylim(new_ylim)
            
            # Use fast drawing method
            self.draw_idle()
    
    def initialize_fft_plot(self, freq_data, mag_data, window_type, fix_y_axis=False):
        """Initialize the FFT plot with proper styling"""
        self.axes.clear()
        
        # Create the line plot
        self.fft_line, = self.axes.plot(freq_data, mag_data, linewidth=1, color='#00ff00')
        
        # Format frequency axis for better readability
        if max(abs(freq_data)) > 1e6:
            self.axes.set_xlabel('Frequency (MHz)', color='#cccccc', fontsize=11)
        elif max(abs(freq_data)) > 1e3:
            self.axes.set_xlabel('Frequency (kHz)', color='#cccccc', fontsize=11)
        else:
            self.axes.set_xlabel('Frequency (Hz)', color='#cccccc', fontsize=11)

        self.axes.set_ylabel('Magnitude (dB)', color='#cccccc', fontsize=11)
        self.axes.set_title(f'FFT Spectrum ({window_type.title()} Window)', 
                           color='#ffffff', fontsize=12, fontweight='bold')
        
        # Set y-axis limits
        if fix_y_axis and self.fixed_ylim is not None:
            # Use fixed y-axis limits
            self.axes.set_ylim(self.fixed_ylim)
        else:
            # Auto-scale y-axis
            max_magnitude = np.max(mag_data)
            self.axes.set_ylim(bottom=max_magnitude - 80, top=max_magnitude + 10)
        
        # Add fine grids (both major and minor)
        self.axes.grid(True, which='major', alpha=0.4, color='#666666', linewidth=0.8)
        self.axes.grid(True, which='minor', alpha=0.2, color='#444444', linewidth=0.4)
        self.axes.minorticks_on()
        
        # Restore x-axis zoom if it exists
        if self.current_xlim is not None:
            self.axes.set_xlim(self.current_xlim)
        
        self.plot_initialized = True
        self.draw()
    
    def lock_y_axis(self):
        """Lock the current y-axis limits"""
        self.fixed_ylim = self.axes.get_ylim()
    
    def unlock_y_axis(self):
        """Unlock the y-axis to allow auto-scaling"""
        self.fixed_ylim = None
    
    def reset_zoom_state(self):
        """Reset the zoom state (called when home button is pressed or data is loaded)"""
        self.current_xlim = None
        self.current_ylim = None
        self.fixed_ylim = None


class IQAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Program state
        self.iq_data = None
        self.sampling_rate = 240.0  # Default sampling rate changed to 240 MHz
        self.fft_start_sample = 0  # Start sample for FFT (changed from center)
        self.fft_size = 1024  # Current FFT size
        self.dragging = False
        self.time_plot_mode = "20*log10(abs(I+Q))"  # Default time domain plot mode
        
        # Performance optimization settings - always real-time
        self.fft_update_timer = QTimer()
        self.fft_update_timer.timeout.connect(self.delayed_fft_update)
        self.fft_update_timer.setSingleShot(True)
        self.fft_update_delay = 16  # ~60 FPS for smooth updates
        self.pending_fft_update = False
        
        # Cache for FFT computation to avoid recalculation
        self.fft_cache = {}
        self.last_fft_params = None
        
        # Setup dark theme for the application
        self.setup_dark_theme()
        self.initUI()
    
    def setup_dark_theme(self):
        """Setup dark theme for the entire application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
                background-color: #3a3a3a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            QComboBox {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
                min-width: 80px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                border: none;
            }
            QSpinBox {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #404040;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #005a9e;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #404040;
                border: 1px solid #555555;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #005a9e;
            }
            QStatusBar {
                background-color: #333333;
                color: #ffffff;
                border-top: 1px solid #555555;
            }
            QMenuBar {
                background-color: #333333;
                color: #ffffff;
                border-bottom: 1px solid #555555;
            }
            QMenuBar::item:selected {
                background-color: #0078d4;
            }
            QMenu {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
        """)
    
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('IQ Spectrum Analyzer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Create splitter for resizable plot areas
        splitter = QSplitter(Qt.Vertical)
        
        # Time domain plot
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        
        time_label = QLabel("Time Domain - IQ Waveform")
        time_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #ffffff;")
        time_layout.addWidget(time_label)
        
        self.iq_canvas = IQPlotCanvas(time_widget)
        self.iq_toolbar = NavigationToolbar(self.iq_canvas, time_widget)
        
        # Connect to toolbar home button to ensure proper auto-scaling
        self.iq_toolbar.home_original = self.iq_toolbar.home
        self.iq_toolbar.home = self.custom_home_function
        
        time_layout.addWidget(self.iq_toolbar)
        time_layout.addWidget(self.iq_canvas)
        time_widget.setLayout(time_layout)
        
        # FFT plot
        fft_widget = QWidget()
        fft_layout = QVBoxLayout(fft_widget)
        
        fft_label = QLabel("Frequency Domain - FFT Spectrum")
        fft_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #ffffff;")
        fft_layout.addWidget(fft_label)
        
        self.fft_canvas = FFTPlotCanvas(fft_widget)
        self.fft_toolbar = NavigationToolbar(self.fft_canvas, fft_widget)
        
        # Connect to FFT toolbar home button to ensure proper auto-scaling
        self.fft_toolbar.home_original = self.fft_toolbar.home
        self.fft_toolbar.home = self.custom_fft_home_function
        
        fft_layout.addWidget(self.fft_toolbar)
        fft_layout.addWidget(self.fft_canvas)
        fft_widget.setLayout(fft_layout)
        
        # Add widgets to splitter
        splitter.addWidget(time_widget)
        splitter.addWidget(fft_widget)
        splitter.setSizes([450, 450])  # Initial sizes
        
        main_layout.addWidget(splitter)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready - Load IQ data to begin analysis')
        
        # Connect mouse events for FFT region selection
        self.iq_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.iq_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.iq_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        self.show()
    
    def create_control_panel(self):
        """Create the control panel with all settings"""
        control_group = QGroupBox("Analysis Controls")
        control_layout = QHBoxLayout()
        
        # Time domain plot controls
        plot_group = QGroupBox("Time Domain Plot")
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("Plot:"))
        self.plot_mode_combo = QComboBox()
        plot_modes = ["20*log10(abs(I+Q))", "abs(I+Q)", "I+Q", "I only", "Q only"]
        self.plot_mode_combo.addItems(plot_modes)
        self.plot_mode_combo.setCurrentText("20*log10(abs(I+Q))")  # Default to dB magnitude
        self.plot_mode_combo.currentTextChanged.connect(self.update_plot_mode)
        plot_layout.addWidget(self.plot_mode_combo)
        plot_group.setLayout(plot_layout)
        
        # Sampling rate controls
        sr_group = QGroupBox("Sampling Rate")
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Rate (MHz):"))
        self.sampling_rate_input = QSpinBox()
        self.sampling_rate_input.setRange(1, 100000)
        self.sampling_rate_input.setValue(240)  # Default changed to 240 MHz
        self.sampling_rate_input.setSuffix(" MHz")
        # Sampling rate affects FFT plot frequency axis
        self.sampling_rate_input.valueChanged.connect(self.update_sampling_rate) 
        sr_layout.addWidget(self.sampling_rate_input)
        sr_group.setLayout(sr_layout)
        
        # FFT size controls
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QVBoxLayout()
        
        fft_size_layout = QHBoxLayout()
        fft_size_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        fft_sizes = ["64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384"]
        self.fft_size_combo.addItems(fft_sizes)
        self.fft_size_combo.setCurrentIndex(4)  # Default to 1024
        self.fft_size_combo.currentIndexChanged.connect(self.update_fft_size)
        fft_size_layout.addWidget(self.fft_size_combo)
        
        # Window function
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        window_types = ["Rectangle", "Hamming", "Hanning", "Blackman", "Bartlett"]
        self.window_combo.addItems(window_types)
        self.window_combo.setCurrentIndex(1)  # Default to Hamming
        self.window_combo.currentIndexChanged.connect(self.invalidate_fft_cache)
        window_layout.addWidget(self.window_combo)
        
        # Y-axis lock checkbox
        y_axis_layout = QHBoxLayout()
        self.fix_y_axis_checkbox = QCheckBox("Fix Y-Axis Scale")
        self.fix_y_axis_checkbox.setToolTip("Lock the current FFT Y-axis scale to prevent auto-scaling")
        self.fix_y_axis_checkbox.stateChanged.connect(self.toggle_y_axis_lock)
        y_axis_layout.addWidget(self.fix_y_axis_checkbox)
        
        fft_layout.addLayout(fft_size_layout)
        fft_layout.addLayout(window_layout)
        fft_layout.addLayout(y_axis_layout)
        fft_group.setLayout(fft_layout)
        
        # FFT position controls
        pos_group = QGroupBox("FFT Region")
        pos_layout = QVBoxLayout()
        
        pos_info_layout = QHBoxLayout()
        self.position_label = QLabel("Start Sample: 0")
        pos_info_layout.addWidget(self.position_label)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(1000)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.position_slider.valueChanged.connect(self.update_fft_position)
        
        pos_layout.addLayout(pos_info_layout)
        pos_layout.addWidget(self.position_slider)
        pos_group.setLayout(pos_layout)
        
        # Add all groups to control panel
        control_layout.addWidget(plot_group)
        control_layout.addWidget(sr_group)
        control_layout.addWidget(fft_group)
        control_layout.addWidget(pos_group)
        control_layout.addStretch()
        
        control_group.setLayout(control_layout)
        return control_group
    
    def toggle_y_axis_lock(self, state):
        """Toggle Y-axis lock for FFT plot"""
        if state == Qt.Checked:
            # Lock the current y-axis scale
            self.fft_canvas.lock_y_axis()
            self.status_bar.showMessage('FFT Y-axis scale locked')
        else:
            # Unlock the y-axis to allow auto-scaling
            self.fft_canvas.unlock_y_axis()
            # Force update to apply auto-scaling
            if self.iq_data is not None:
                self.update_fft()
            self.status_bar.showMessage('FFT Y-axis scale unlocked - auto-scaling enabled')
    
    def invalidate_fft_cache(self):
        """Clear FFT cache when parameters change"""
        self.fft_cache.clear()
        self.last_fft_params = None
        if self.iq_data is not None:
            self.update_fft()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open IQ Data...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_image_action = QAction('Save Plots as Image...', self)
        save_image_action.setShortcut('Ctrl+S')
        save_image_action.triggered.connect(self.save_image)
        file_menu.addAction(save_image_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def show_about(self):
        """Show about dialog"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(self, "About IQ Spectrum Analyzer", 
                         "IQ Spectrum Analyzer v1.4\n\n"
                         "A high-performance tool for analyzing IQ waveform data.\n\n"
                         "Performance Optimizations:\n"
                         "• Fast line plot updates without full redraw\n"
                         "• Optimized rectangle highlighting\n"
                         "• Efficient zoom preservation\n\n"
                         "Features:\n"
                         "• Real-time FFT region selection\n"
                         "• Multiple window functions\n"
                         "• Professional dark theme interface\n"
                         "• Multiple time domain plot modes\n"
                         "• FFT Y-axis scale lock control")
    
    def custom_fft_home_function(self):
        """Custom home function for FFT plot that properly auto-scales for current sampling rate"""
        if self.iq_data is not None:
            # Clear FFT zoom state to allow full auto-scaling
            self.fft_canvas.reset_zoom_state()
            # Uncheck the y-axis lock checkbox
            self.fix_y_axis_checkbox.setChecked(False)
            # Force reinitialize plot
            self.fft_canvas.plot_initialized = False
            # Replot with auto-scaling for the current sampling rate
            self.update_fft_with_auto_scale()
        else:
            # If no data, use the default home function
            self.fft_toolbar.home_original()
    
    def custom_home_function(self):
        """Custom home function that properly auto-scales the y-axis for current plot mode"""
        if self.iq_data is not None:
            # Clear zoom state to allow full auto-scaling
            self.iq_canvas.current_xlim = None
            self.iq_canvas.current_ylim = None
            # Replot with auto-scaling for the current plot mode
            # Pass auto_scale=True to ensure proper handling
            self.plot_iq_data(auto_scale=True)
        else:
            # If no data, use the default home function
            self.iq_toolbar.home_original()
    
    def update_plot_mode(self):
        """Update the time domain plot mode"""
        self.time_plot_mode = self.plot_mode_combo.currentText()
        if self.iq_data is not None:
            # Don't clear zoom state - we want to preserve x-axis zoom
            # Only auto-scale the y-axis when switching plot modes
            self.plot_iq_data(auto_scale=True)
    
    def store_current_zoom(self):
        """Store current axis limits to preserve zoom level"""
        # Check if axes exist and have valid limits
        if self.iq_canvas.axes.lines: # Check if any lines have been plotted
            self.iq_canvas.current_xlim = self.iq_canvas.axes.get_xlim()
            self.iq_canvas.current_ylim = self.iq_canvas.axes.get_ylim()
        else:
            self.iq_canvas.current_xlim = None
            self.iq_canvas.current_ylim = None
    
    def restore_zoom(self):
        """Restore previously stored zoom level"""
        if self.iq_canvas.current_xlim is not None and self.iq_canvas.current_ylim is not None:
            self.iq_canvas.axes.set_xlim(self.iq_canvas.current_xlim)
            self.iq_canvas.axes.set_ylim(self.iq_canvas.current_ylim)
    
    def store_fft_zoom(self):
        """Store current FFT axis limits to preserve zoom level"""
        # Check if FFT axes exist and have valid limits
        if self.fft_canvas.axes.lines: # Check if any lines have been plotted
            self.fft_canvas.current_xlim = self.fft_canvas.axes.get_xlim()
            self.fft_canvas.current_ylim = self.fft_canvas.axes.get_ylim()
        else:
            self.fft_canvas.current_xlim = None
            self.fft_canvas.current_ylim = None
    
    def restore_fft_zoom(self):
        """Restore previously stored FFT zoom level"""
        if self.fft_canvas.current_xlim is not None and self.fft_canvas.current_ylim is not None:
            self.fft_canvas.axes.set_xlim(self.fft_canvas.current_xlim)
            self.fft_canvas.axes.set_ylim(self.fft_canvas.current_ylim)
    
    def compute_fft_data(self, start_sample, fft_size, window_type, sampling_rate):
        """Compute FFT data with caching for performance"""
        # Create cache key
        cache_key = (start_sample, fft_size, window_type, sampling_rate)
        
        # Check cache first
        if cache_key in self.fft_cache:
            return self.fft_cache[cache_key]
        
        # Calculate FFT region
        end_sample = min(len(self.iq_data), start_sample + fft_size)
        data_segment = self.iq_data[start_sample:end_sample]
        
        # Apply window function
        window_type_lower = window_type.lower()
        if window_type_lower == 'rectangle':
            windowed_data = data_segment
        elif window_type_lower == 'hamming':
            windowed_data = data_segment * np.hamming(len(data_segment))
        elif window_type_lower == 'hanning':
            windowed_data = data_segment * np.hanning(len(data_segment))
        elif window_type_lower == 'blackman':
            windowed_data = data_segment * np.blackman(len(data_segment))
        elif window_type_lower == 'bartlett':
            windowed_data = data_segment * np.bartlett(len(data_segment))
        else:
            windowed_data = data_segment
        
        # Zero-padding if necessary
        if len(windowed_data) < fft_size:
            padded_data = np.zeros(fft_size, dtype=complex)
            padded_data[:len(windowed_data)] = windowed_data
            windowed_data = padded_data
        
        # Perform FFT and shift
        fft_result = np.fft.fftshift(np.fft.fft(windowed_data, fft_size))
        fft_mag = 20 * np.log10(np.abs(fft_result) + 1e-12)  # dB scale
        
        # Calculate frequency axis
        freq_axis = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/(sampling_rate * 1e6)))
        
        # Format frequency axis for better readability
        if max(abs(freq_axis)) > 1e6:
            freq_data = freq_axis / 1e6
        elif max(abs(freq_axis)) > 1e3:
            freq_data = freq_axis / 1e3
        else:
            freq_data = freq_axis
        
        result = (freq_data, fft_mag, end_sample - start_sample)
        
        # Cache result (limit cache size to prevent memory issues)
        if len(self.fft_cache) > 50:  # Limit cache size
            # Remove oldest entry
            oldest_key = next(iter(self.fft_cache))
            del self.fft_cache[oldest_key]
        
        self.fft_cache[cache_key] = result
        return result
    
    def open_file(self):
        """Open and load IQ data file"""
        from PyQt5.QtWidgets import QMessageBox
        
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open IQ Data File", "", 
            "Text Files (*.txt);;All Supported (*.txt *.bin *.csv *.npy);;Binary Files (*.bin);;CSV Files (*.csv);;NPY Files (*.npy);;All Files (*)", 
            options=options
        )
        
        if filename:
            try:
                # Try to determine file type and load accordingly
                if filename.endswith('.npy'):
                    self.iq_data = np.load(filename)
                elif filename.endswith('.csv'):
                    data = np.genfromtxt(filename, delimiter=',')
                    # Assume complex data with I,Q columns
                    if len(data.shape) > 1 and data.shape[1] >= 2:
                        self.iq_data = data[:, 0] + 1j * data[:, 1]
                    else:
                        self.status_bar.showMessage('CSV file must have at least 2 columns (I and Q)')
                        return
                elif filename.endswith('.bin'):
                    # Assume binary file with interleaved I,Q float32 values
                    data = np.fromfile(filename, dtype=np.float32)
                    if len(data) % 2 == 0:
                        self.iq_data = data[::2] + 1j * data[1::2]
                    else:
                        self.status_bar.showMessage('Binary file must have even number of values for I/Q pairs')
                        return
                elif filename.endswith('.txt'):
                    # Handle .txt files - assume space or tab delimited with I,Q columns
                    data = np.loadtxt(filename)
                    if len(data.shape) > 1 and data.shape[1] >= 2:
                        self.iq_data = data[:, 0] + 1j * data[:, 1]
                    else:
                        self.status_bar.showMessage('Text file must have at least 2 columns (I and Q)')
                        return
                else:
                    # Try to guess format for other extensions
                    try:
                        data = np.loadtxt(filename)
                        if len(data.shape) > 1 and data.shape[1] >= 2:
                            self.iq_data = data[:, 0] + 1j * data[:, 1]
                        else:
                            self.status_bar.showMessage('File must have at least 2 columns (I and Q)')
                            return
                    except:
                        self.status_bar.showMessage('Unsupported file format')
                        return
                
                # Validate data
                if len(self.iq_data) < 64:
                    QMessageBox.warning(self, "Warning", "Data file contains less than 64 samples. Analysis may be limited.")
                
                # Initialize FFT parameters
                self.fft_start_sample = 0  # Start from beginning
                self.fft_size = int(self.fft_size_combo.currentText())
                
                # Clear caches
                self.fft_cache.clear()
                self.last_fft_params = None
                
                # Update position slider
                self.position_slider.setEnabled(True)
                # Ensure maximum is at least 0 to prevent errors with very small data
                self.position_slider.setMaximum(max(0, len(self.iq_data) - self.fft_size))
                self.position_slider.setValue(self.fft_start_sample)
                
                # Reset zoom state for both plots
                self.iq_canvas.current_xlim = None
                self.iq_canvas.current_ylim = None
                self.fft_canvas.reset_zoom_state()
                self.fft_canvas.plot_initialized = False
                
                # Reset y-axis lock checkbox
                self.fix_y_axis_checkbox.setChecked(False)
                
                self.plot_iq_data()
                self.update_fft()
                
                self.status_bar.showMessage(f'Loaded {len(self.iq_data)} IQ samples from {filename}')
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f'Error loading file: {str(e)}')
                self.status_bar.showMessage(f'Error loading file: {str(e)}')
    
    def plot_iq_data(self, auto_scale=False):
        """Plot the IQ data in time domain with FFT region highlighting"""
        if self.iq_data is None:
            return
        
        # Store current zoom level before clearing
        self.store_current_zoom()
        
        self.iq_canvas.axes.clear()
        
        # IMPORTANT: Reset the rectangle reference after clearing axes
        self.iq_canvas.fft_region_rect = None        
            
        # Create sample axis
        sample_axis = np.arange(len(self.iq_data))
        
        # Plot based on selected mode
        if self.time_plot_mode == "20*log10(abs(I+Q))":
            magnitude_db = 20 * np.log10(np.abs(self.iq_data) + 1e-12)  # Add small value to avoid log(0)
            line, = self.iq_canvas.axes.plot(sample_axis, magnitude_db, label='20*log10(|I+jQ|)', 
                                    linewidth=1.2, color='#ffff00', alpha=0.8)
            ylabel = 'Magnitude (dB)'
            title_suffix = 'Magnitude (dB)'
            self.iq_canvas.plot_lines = [line]
        elif self.time_plot_mode == "abs(I+Q)":
            magnitude = np.abs(self.iq_data)
            line, = self.iq_canvas.axes.plot(sample_axis, magnitude, label='|I+jQ|', 
                                    linewidth=1.2, color='#00ff00', alpha=0.8)
            ylabel = 'Magnitude'
            title_suffix = 'Magnitude'
            self.iq_canvas.plot_lines = [line]
        elif self.time_plot_mode == "I+Q":
            # Plot I and Q components with enhanced styling
            line1, = self.iq_canvas.axes.plot(sample_axis, np.real(self.iq_data), label='I', 
                                    linewidth=1.2, color='#00bfff', alpha=0.8)
            line2, = self.iq_canvas.axes.plot(sample_axis, np.imag(self.iq_data), label='Q', 
                                    linewidth=1.2, color='#ff6b6b', alpha=0.8)
            ylabel = 'Amplitude'
            title_suffix = 'I and Q Components'
            self.iq_canvas.plot_lines = [line1, line2]
        elif self.time_plot_mode == "I only":
            line, = self.iq_canvas.axes.plot(sample_axis, np.real(self.iq_data), label='I', 
                                    linewidth=1.2, color='#00bfff', alpha=0.8)
            ylabel = 'Amplitude'
            title_suffix = 'I Component Only'
            self.iq_canvas.plot_lines = [line]
        elif self.time_plot_mode == "Q only":
            line, = self.iq_canvas.axes.plot(sample_axis, np.imag(self.iq_data), label='Q', 
                                    linewidth=1.2, color='#ff6b6b', alpha=0.8)
            ylabel = 'Amplitude'
            title_suffix = 'Q Component Only'
            self.iq_canvas.plot_lines = [line]
        
        # Set labels and legend with dark theme styling
        self.iq_canvas.axes.set_xlabel('Sample Number', color='#cccccc', fontsize=11)
        self.iq_canvas.axes.set_ylabel(ylabel, color='#cccccc', fontsize=11)
        self.iq_canvas.axes.set_title(f'IQ Waveform - {title_suffix} (Cyan bar shows FFT analysis region)', 
                                     color='#ffffff', fontsize=12, fontweight='bold')
        
        # Configure legend with dark theme
        legend = self.iq_canvas.axes.legend(loc='upper right', framealpha=0.8)
        legend.get_frame().set_facecolor('#333333')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('#ffffff')
        
        # Configure grid
        self.iq_canvas.axes.grid(True, alpha=0.2, color='#555555', linewidth=0.5)
        
        # Configure spines and ticks
        for spine in self.iq_canvas.axes.spines.values():
            spine.set_color('#555555')
        self.iq_canvas.axes.tick_params(colors='#cccccc')
        
        # Handle zoom restoration based on auto_scale mode
        if auto_scale and self.iq_canvas.current_xlim is not None:
            # Auto-scale mode: preserve x-axis zoom, auto-scale y-axis
            self.iq_canvas.axes.autoscale_view()  # Let it autoscale both first
            self.iq_canvas.axes.set_xlim(self.iq_canvas.current_xlim)  # Then restore x-axis
        elif not auto_scale and (self.iq_canvas.current_xlim is not None or self.iq_canvas.current_ylim is not None):
            # Normal mode with existing zoom: let matplotlib autoscale first, then restore both axes
            self.iq_canvas.axes.autoscale_view()
            self.restore_zoom()
        else:
            # No previous zoom or both zoom states are None: just autoscale everything
            self.iq_canvas.axes.autoscale_view()
            # Special handling for abs(I+Q) mode - set y-axis minimum to 0
            if self.time_plot_mode == "abs(I+Q)":
                current_xlim = self.iq_canvas.axes.get_xlim()
                current_ylim = self.iq_canvas.axes.get_ylim()
                self.iq_canvas.axes.set_ylim(bottom=0, top=current_ylim[1])
        
        # Add FFT region highlighting AFTER setting axis limits
        self.highlight_fft_region()
        
        self.iq_canvas.draw()
    
    def highlight_fft_region(self):
        """Add a colored rectangle to highlight the FFT analysis region - optimized version"""
        if self.iq_data is None:
            return
        
        # Calculate FFT region boundaries using sample indices
        start_sample = self.fft_start_sample
        end_sample = min(len(self.iq_data), start_sample + self.fft_size)
        
        # Get current y-axis limits (after data has been plotted and autoscaled)
        y_min, y_max = self.iq_canvas.axes.get_ylim()
        
        # If rectangle exists, just update its position instead of recreating
        if self.iq_canvas.fft_region_rect:
            self.iq_canvas.fft_region_rect.set_x(start_sample)
            self.iq_canvas.fft_region_rect.set_width(end_sample - start_sample)
            self.iq_canvas.fft_region_rect.set_y(y_min)
            self.iq_canvas.fft_region_rect.set_height(y_max - y_min)
        else:
            # Create new rectangle only if it doesn't exist
            rect = Rectangle((start_sample, y_min), end_sample - start_sample, y_max - y_min,
                            facecolor='#00ffff', alpha=0.15, edgecolor='#00bfff', linewidth=2.0)
            self.iq_canvas.axes.add_patch(rect)
            self.iq_canvas.fft_region_rect = rect
    
    def update_fft_region_only(self):
        """Update only the FFT region rectangle position - very fast"""
        if self.iq_data is None or not self.iq_canvas.fft_region_rect:
            return
        
        # Calculate FFT region boundaries
        start_sample = self.fft_start_sample
        end_sample = min(len(self.iq_data), start_sample + self.fft_size)
        
        # Update rectangle position
        self.iq_canvas.fft_region_rect.set_x(start_sample)
        self.iq_canvas.fft_region_rect.set_width(end_sample - start_sample)
        
        # Use idle drawing for fast update
        self.iq_canvas.draw_idle()
    
    def on_mouse_press(self, event):
        """Handle mouse press for FFT region selection"""
        if event.inaxes != self.iq_canvas.axes or self.iq_data is None:
            return
        
        # Check if navigation toolbar is in zoom or pan mode
        if self.iq_toolbar.mode != '':
            return  # Don't handle mouse events when toolbar is active
        
        if event.button == 1:  # Left mouse button
            self.dragging = True
            # Convert click position to sample index (xdata is already sample number)
            sample_clicked = int(event.xdata)
            
            # Update FFT start position (ensure it doesn't exceed valid range)
            self.fft_start_sample = np.clip(sample_clicked, 0, 
                                          max(0, len(self.iq_data) - self.fft_size))
            
            self.update_position_controls()
            # Update region immediately
            self.update_fft_region_only()
            # Schedule FFT update - always enabled now
            self.schedule_fft_update()
    
    def on_mouse_move(self, event):
        """Handle mouse move for dragging FFT region"""
        if not self.dragging or event.inaxes != self.iq_canvas.axes or self.iq_data is None:
            return
        
        # Check if navigation toolbar is in zoom or pan mode
        if self.iq_toolbar.mode != '':
            return  # Don't handle mouse events when toolbar is active
        
        # Convert mouse position to sample index (xdata is already sample number)
        sample_pos = int(event.xdata)
        
        # Update FFT start position (ensure it doesn't exceed valid range)
        new_fft_start_sample = np.clip(sample_pos, 0, 
                                      max(0, len(self.iq_data) - self.fft_size))
        
        if new_fft_start_sample != self.fft_start_sample:
            self.fft_start_sample = new_fft_start_sample
            self.update_position_controls()
            # Only update the rectangle position, not the entire plot
            self.update_fft_region_only()
            # Schedule FFT update - always enabled now
            self.schedule_fft_update()
    
    def on_mouse_release(self, event):
        """Handle mouse release"""
        # Check if navigation toolbar is in zoom or pan mode
        if self.iq_toolbar.mode != '':
            self.dragging = False  # Reset dragging state but don't update plots
            return
        
        was_dragging = self.dragging
        self.dragging = False
        # Ensure FFT is updated one last time after release
        if self.iq_data is not None and was_dragging:
            # Cancel any pending timer and update immediately
            self.fft_update_timer.stop()
            self.update_fft()
    
    def schedule_fft_update(self):
        """Schedule an FFT update with a delay to avoid too frequent updates"""
        self.pending_fft_update = True
        self.fft_update_timer.stop()
        self.fft_update_timer.start(self.fft_update_delay)
    
    def delayed_fft_update(self):
        """Perform the delayed FFT update"""
        if self.pending_fft_update:
            self.pending_fft_update = False
            self.update_fft()
    
    def update_position_controls(self):
        """Update position slider and label"""
        # Ensure slider max is correctly set based on current data length and FFT size
        if self.iq_data is not None:
            self.position_slider.setMaximum(max(0, len(self.iq_data) - self.fft_size))
        else:
            self.position_slider.setMaximum(0) # No data, slider max is 0

        self.position_slider.blockSignals(True)  # Prevent recursive updates
        self.position_slider.setValue(self.fft_start_sample)
        self.position_slider.blockSignals(False)
        self.position_label.setText(f"Start Sample: {self.fft_start_sample}")
    
    def update_fft_position(self):
        """Update FFT position from slider"""
        if self.iq_data is None:
            return
        
        new_fft_start_sample = self.position_slider.value()
        new_fft_start_sample = np.clip(new_fft_start_sample, 0, 
                                      max(0, len(self.iq_data) - self.fft_size))
        
        if new_fft_start_sample != self.fft_start_sample:
            self.fft_start_sample = new_fft_start_sample
            self.position_label.setText(f"Start Sample: {self.fft_start_sample}")
            # Only update rectangle for slider movement
            self.update_fft_region_only()
            # Always use delayed FFT update for smooth slider response
            self.schedule_fft_update()
    
    def update_sampling_rate(self):
        """Update sampling rate and refresh FFT plot"""
        self.sampling_rate = self.sampling_rate_input.value()
        if self.iq_data is not None:
            # Clear FFT zoom state and cache when sampling rate changes
            self.fft_canvas.reset_zoom_state()
            self.fft_canvas.plot_initialized = False
            self.fft_cache.clear()
            # Reset y-axis lock checkbox when sampling rate changes
            self.fix_y_axis_checkbox.setChecked(False)
            self.update_fft_with_auto_scale() # Update FFT with new frequency axis and auto-scale
    
    def update_fft_with_auto_scale(self):
        """Update the FFT plot with automatic scaling (used when sampling rate changes)"""
        if self.iq_data is None:
            return

        # Update self.sampling_rate from the input spinbox
        self.sampling_rate = self.sampling_rate_input.value()

        # Get window function
        window_type = self.window_combo.currentText()

        # Compute FFT data using optimized function
        freq_data, fft_mag, actual_samples = self.compute_fft_data(
            self.fft_start_sample, self.fft_size, window_type, self.sampling_rate)

        # Force reinitialize plot for auto-scaling
        self.fft_canvas.plot_initialized = False
        self.fft_canvas.initialize_fft_plot(freq_data, fft_mag, window_type, fix_y_axis=False)

        # Update status bar with FFT details
        start_sample = self.fft_start_sample
        end_sample = start_sample + actual_samples
        freq_resolution = self.sampling_rate * 1e6 / self.fft_size
        self.status_bar.showMessage(
            f'FFT: samples {start_sample}-{end_sample} ({actual_samples} samples), '
            f'{self.fft_size} FFT points, Resolution: {freq_resolution:.2f} Hz')
    
    def update_fft_size(self):
        """Update FFT size and refresh analysis"""
        self.fft_size = int(self.fft_size_combo.currentText())
        
        # Clear cache when FFT size changes
        self.fft_cache.clear()
        
        if self.iq_data is not None:
            # Update position slider range
            self.position_slider.setMaximum(max(0, len(self.iq_data) - self.fft_size))
            
            # Ensure FFT start is within valid range
            self.fft_start_sample = np.clip(self.fft_start_sample, 0, 
                                          max(0, len(self.iq_data) - self.fft_size))
            
            self.update_position_controls()
            self.plot_iq_data() # Redraw IQ plot to update highlight with new FFT size
            self.update_fft()
    
    def update_fft(self):
        """Update the FFT plot based on the selected region - optimized version"""
        if self.iq_data is None:
            return

        # Store current FFT zoom level before updating
        self.store_fft_zoom()

        # Update self.sampling_rate from the input spinbox
        self.sampling_rate = self.sampling_rate_input.value()

        # Get window function
        window_type = self.window_combo.currentText()

        # Compute FFT data using optimized function with caching
        freq_data, fft_mag, actual_samples = self.compute_fft_data(
            self.fft_start_sample, self.fft_size, window_type, self.sampling_rate)

        # Check if y-axis should be fixed
        fix_y_axis = self.fix_y_axis_checkbox.isChecked()

        # Use fast update method
        self.fft_canvas.update_fft_data_fast(freq_data, fft_mag, window_type, fix_y_axis)

        # Update status bar with FFT details
        start_sample = self.fft_start_sample
        end_sample = start_sample + actual_samples
        freq_resolution = self.sampling_rate * 1e6 / self.fft_size
        self.status_bar.showMessage(
            f'FFT: samples {start_sample}-{end_sample} ({actual_samples} samples), '
            f'{self.fft_size} FFT points, Resolution: {freq_resolution:.2f} Hz')
    
    def save_image(self):
        """Save both plots as a single image"""
        if self.iq_data is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Info", "No data to save. Please load IQ data first.")
            return
        
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plots as Image", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
            options=options
        )
        
        if filename:
            try:
                # Create a new figure with both plots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Copy IQ plot (sample-based x-axis) based on current plot mode
                sample_axis = np.arange(len(self.iq_data))
                
                if self.time_plot_mode == "20*log10(abs(I+Q))":
                    magnitude_db = 20 * np.log10(np.abs(self.iq_data) + 1e-12)
                    ax1.plot(sample_axis, magnitude_db, label='20*log10(|I+jQ|)', linewidth=0.8, color='orange')
                    ylabel = 'Magnitude (dB)'
                    title_suffix = 'Magnitude (dB)'
                elif self.time_plot_mode == "abs(I+Q)":
                    magnitude = np.abs(self.iq_data)
                    ax1.plot(sample_axis, magnitude, label='|I+jQ|', linewidth=0.8, color='green')
                    ylabel = 'Magnitude'
                    title_suffix = 'Magnitude'
                elif self.time_plot_mode == "I+Q":
                    ax1.plot(sample_axis, np.real(self.iq_data), label='I', linewidth=0.8, color='blue')
                    ax1.plot(sample_axis, np.imag(self.iq_data), label='Q', linewidth=0.8, color='red')
                    ylabel = 'Amplitude'
                    title_suffix = 'I and Q Components'
                elif self.time_plot_mode == "I only":
                    ax1.plot(sample_axis, np.real(self.iq_data), label='I', linewidth=0.8, color='blue')
                    ylabel = 'Amplitude'
                    title_suffix = 'I Component Only'
                elif self.time_plot_mode == "Q only":
                    ax1.plot(sample_axis, np.imag(self.iq_data), label='Q', linewidth=0.8, color='red')
                    ylabel = 'Amplitude'
                    title_suffix = 'Q Component Only'
                
                # Add FFT region highlight to saved image (sample-based)
                start_sample = self.fft_start_sample
                end_sample = min(len(self.iq_data), start_sample + self.fft_size)
                
                # Let matplotlib autoscale first
                ax1.autoscale_view()
                y_min, y_max = ax1.get_ylim()
                rect = Rectangle((start_sample, y_min), end_sample - start_sample, y_max - y_min,
                               facecolor='cyan', alpha=0.2, edgecolor='blue', linewidth=1.5)
                ax1.add_patch(rect)
                
                # Add annotation for saved image
                mid_sample = (start_sample + end_sample) / 2
                text_y = y_min + 0.85 * (y_max - y_min)
                ax1.annotate(f'FFT Region\n{end_sample - start_sample} samples\nStart: {start_sample}', 
                             xy=(mid_sample, text_y), 
                             ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='blue'))

                ax1.set_xlabel('Sample Number')
                ax1.set_ylabel(ylabel)
                ax1.set_title(f'IQ Waveform - {title_suffix} (Blue bar shows FFT analysis region)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Copy FFT plot
                for line in self.fft_canvas.axes.lines:
                    ax2.plot(line.get_xdata(), line.get_ydata(), linewidth=1)
                ax2.set_xlabel(self.fft_canvas.axes.get_xlabel())
                ax2.set_ylabel(self.fft_canvas.axes.get_ylabel())
                ax2.set_title(self.fft_canvas.axes.get_title())
                ax2.grid(True, alpha=0.3)
                
                fig.tight_layout()
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                self.status_bar.showMessage(f'Saved plots to {filename}')
            
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f'Error saving image: {str(e)}')


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("IQ Spectrum Analyzer")
    app.setApplicationVersion("1.4")
    
    analyzer = IQAnalyzer()
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass


if __name__ == '__main__':
    main()