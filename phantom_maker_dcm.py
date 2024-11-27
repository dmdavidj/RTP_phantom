# hello world
# Thank you for visiting. 
# I hope this module will be helpful to you in your research or practice.
#
# MIT License
#
# Best Regards
# dmJ from Republic of Korea
# 2024.11

import sys
import os
from datetime import datetime
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QSlider, QProgressDialog, QMessageBox)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.ndimage import gaussian_filter

# Standard CT constants
HU_MIN = -1024  # Standard air HU
HU_MAX = 3072   # Maximum HU for standard CT
HU_RANGE = HU_MAX - HU_MIN

# Window presets
WINDOW_PRESETS = {
    "Default": (40, 400),
    "Brain": (40, 80),
    "Lung": (-600, 1500),
    "Bone": (400, 2000),
    "Abdomen": (40, 400),
    "Wide": (0, 2000)
}

# Material HU presets
HU_PRESETS = {
    "Custom": None,
    "Air": -1000,
    "Fat": -120,
    "Water": 0,
    "Soft Tissue": 40,
    "Bone": 1000,
    "Dense Bone": 2000,
    "Metal": 3000
}

class PhantomGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Phantom Generator_dmJ_2024.11")
        self.setMinimumSize(1200, 800)
        
        self.current_volume = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create left and right panels
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 2)
        
        self.update_phantom_size_limit()
        self.update_position_limits()
        
        # Create status bar for HU display
        self.statusBar().showMessage("Ready")
        
        
        
    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # CT Parameters Group
        ct_group = QGroupBox("CT Parameters")
        ct_layout = QVBoxLayout()
        
        # Image size parameters
        size_layout = QHBoxLayout()
        self.x_size = QSpinBox()
        self.x_size.setRange(64, 1024)
        self.x_size.setValue(512)
        self.y_size = QSpinBox()
        self.y_size.setRange(64, 1024)
        self.y_size.setValue(512)
        
        size_layout.addWidget(QLabel("Image Size (pixels):"))
        size_layout.addWidget(QLabel("X:"))
        size_layout.addWidget(self.x_size)
        size_layout.addWidget(QLabel("Y:"))
        size_layout.addWidget(self.y_size)
        ct_layout.addLayout(size_layout)
        
        self.x_size.valueChanged.connect(self.update_phantom_size_limit)
        self.y_size.valueChanged.connect(self.update_phantom_size_limit)
        
        # Slice thickness
        thickness_layout = QHBoxLayout()
        self.slice_thickness = QDoubleSpinBox()
        self.slice_thickness.setRange(0.1, 10.0)
        self.slice_thickness.setValue(1.0)
        self.slice_thickness.setSingleStep(0.1)
        
        thickness_layout.addWidget(QLabel("Slice Thickness (mm):"))
        thickness_layout.addWidget(self.slice_thickness)
        ct_layout.addLayout(thickness_layout)
        
        # Number of slices
        slices_layout = QHBoxLayout()
        self.num_slices = QSpinBox()
        self.num_slices.setRange(1, 1000)
        self.num_slices.setValue(50)
        self.num_slices.valueChanged.connect(self.update_slice_slider_range)
        
        slices_layout.addWidget(QLabel("Number of Slices:"))
        slices_layout.addWidget(self.num_slices)
        ct_layout.addLayout(slices_layout)
        
        ct_group.setLayout(ct_layout)
        left_layout.addWidget(ct_group)
        
        # Phantom Parameters Group
        phantom_group = QGroupBox("Phantom Parameters")
        phantom_layout = QVBoxLayout()
        
        # Phantom shape
        shape_layout = QHBoxLayout()
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Cube", "Cylinder", "Sphere"])
        
        shape_layout.addWidget(QLabel("Shape:"))
        shape_layout.addWidget(self.shape_combo)
        phantom_layout.addLayout(shape_layout)
        
        # Material selection
        material_layout = QHBoxLayout()
        self.material_combo = QComboBox()
        self.material_combo.addItems(list(HU_PRESETS.keys()))
        self.material_combo.currentTextChanged.connect(self.update_hu_from_material)
        
        material_layout.addWidget(QLabel("Material:"))
        material_layout.addWidget(self.material_combo)
        phantom_layout.addLayout(material_layout)
        
        # Edge smoothing
        smooth_layout = QHBoxLayout()
        self.edge_smoothing = QDoubleSpinBox()
        self.edge_smoothing.setRange(0, 5.0)
        self.edge_smoothing.setValue(0.5)
        self.edge_smoothing.setSingleStep(0.1)
        self.edge_smoothing.setDecimals(1)
        
        smooth_layout.addWidget(QLabel("Edge Smoothing:"))
        smooth_layout.addWidget(self.edge_smoothing)
        phantom_layout.addLayout(smooth_layout)
        
        # HU value
        hu_layout = QHBoxLayout()
        self.hu_value = QDoubleSpinBox()
        self.hu_value.setRange(HU_MIN, HU_MAX)
        self.hu_value.setValue(0.0)
        self.hu_value.setDecimals(2)
        self.hu_value.setSingleStep(0.01)
        
        hu_layout.addWidget(QLabel("HU Value:"))
        hu_layout.addWidget(self.hu_value)
        phantom_layout.addLayout(hu_layout)
        
        # Phantom size
        phantom_size_layout = QHBoxLayout()
        self.phantom_size = QSpinBox()
        self.phantom_size.setRange(10, 400)
        self.phantom_size.setValue(100)
        
        phantom_size_layout.addWidget(QLabel("Size (mm):"))
        phantom_size_layout.addWidget(self.phantom_size)
        phantom_layout.addLayout(phantom_size_layout)
        # Phantom position
        position_group = QGroupBox("Position (mm)")
        position_layout = QVBoxLayout()
        
        # X position
        x_pos_layout = QHBoxLayout()
        self.x_position = QSpinBox()
        self.x_position.setRange(-256, 256)
        x_pos_layout.addWidget(QLabel("X:"))
        x_pos_layout.addWidget(self.x_position)
        position_layout.addLayout(x_pos_layout)
        
        # Y position
        y_pos_layout = QHBoxLayout()
        self.y_position = QSpinBox()
        self.y_position.setRange(-256, 256)
        y_pos_layout.addWidget(QLabel("Y:"))
        y_pos_layout.addWidget(self.y_position)
        position_layout.addLayout(y_pos_layout)
        
        # Z position
        z_pos_layout = QHBoxLayout()
        self.z_position = QSpinBox()
        self.z_position.setRange(-256, 256)
        z_pos_layout.addWidget(QLabel("Z:"))
        z_pos_layout.addWidget(self.z_position)
        position_layout.addLayout(z_pos_layout)
        
        position_group.setLayout(position_layout)
        phantom_layout.addWidget(position_group)
        
        phantom_group.setLayout(phantom_layout)
        left_layout.addWidget(phantom_group)
        
        # Output Settings Group
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Output directory selection
        dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setReadOnly(True)
        select_dir_btn = QPushButton("Select Directory")
        select_dir_btn.clicked.connect(self.select_output_directory)
        
        dir_layout.addWidget(QLabel("Output Directory:"))
        dir_layout.addWidget(self.output_dir)
        dir_layout.addWidget(select_dir_btn)
        output_layout.addLayout(dir_layout)
        
        # Series info
        series_layout = QHBoxLayout()
        self.series_name = QLineEdit()
        self.series_name.setText("PHANTOM_SERIES")
        series_layout.addWidget(QLabel("Series Name:"))
        series_layout.addWidget(self.series_name)
        output_layout.addLayout(series_layout)
        
        # Patient information
        patient_group = QGroupBox("Patient Information")
        patient_layout = QVBoxLayout()
        
        # Patient ID
        patient_id_layout = QHBoxLayout()
        self.patient_id = QLineEdit()
        self.patient_id.setText("synthetic_phantom")
        patient_id_layout.addWidget(QLabel("Patient ID:"))
        patient_id_layout.addWidget(self.patient_id)
        patient_layout.addLayout(patient_id_layout)
        
        # Patient Name
        patient_name_layout = QHBoxLayout()
        self.patient_name = QLineEdit()
        self.patient_name.setText("synthetic_phantom")
        patient_name_layout.addWidget(QLabel("Patient Name:"))
        patient_name_layout.addWidget(self.patient_name)
        patient_layout.addLayout(patient_name_layout)
        
        patient_group.setLayout(patient_layout)
        output_layout.addWidget(patient_group)
        
        output_group.setLayout(output_layout)
        left_layout.addWidget(output_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_phantom)
        generate_btn = QPushButton("Generate DICOM")
        generate_btn.clicked.connect(self.generate_dicom)
        
        buttons_layout.addWidget(preview_btn)
        buttons_layout.addWidget(generate_btn)
        left_layout.addLayout(buttons_layout)
        
        return left_panel
    
    def create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Preview area with toolbar
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_figure, (self.preview_ax, self.colorbar_ax) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [20, 1]})
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.toolbar = NavigationToolbar(self.preview_canvas, self)
        
        # Slice slider
        slice_layout = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.num_slices.value() - 1)
        self.slice_slider.setValue((self.num_slices.value() - 1) // 2)
        self.slice_slider.valueChanged.connect(self.update_preview)
        
        self.slice_label = QLabel(f"Slice: {self.slice_slider.value() + 1}")
        slice_layout.addWidget(QLabel("Z Position:"))
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(self.slice_label)
        
        # Window/Level controls
        window_group = QGroupBox("Window/Level Controls")
        window_layout = QVBoxLayout()
        
        # Window presets
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(WINDOW_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_window_preset)
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)
        window_layout.addLayout(preset_layout)
        
        # Window center (level) slider
        level_layout = QHBoxLayout()
        self.window_level_slider = QSlider(Qt.Horizontal)
        self.window_level_slider.setRange(HU_MIN, HU_MAX)
        self.window_level_slider.setValue(40)
        self.window_level_label = QLabel("Level: 40 HU")
        self.window_level_slider.valueChanged.connect(self.update_level_label)
        self.window_level_slider.valueChanged.connect(self.update_preview)
        
        level_layout.addWidget(QLabel("Level:"))
        level_layout.addWidget(self.window_level_slider)
        level_layout.addWidget(self.window_level_label)
        window_layout.addLayout(level_layout)
        
        # Window width slider
        width_layout = QHBoxLayout()
        self.window_width_slider = QSlider(Qt.Horizontal)
        self.window_width_slider.setRange(1, 4000)
        self.window_width_slider.setValue(400)
        self.window_width_label = QLabel("Width: 400 HU")
        self.window_width_slider.valueChanged.connect(self.update_width_label)
        self.window_width_slider.valueChanged.connect(self.update_preview)
        
        width_layout.addWidget(QLabel("Width:"))
        width_layout.addWidget(self.window_width_slider)
        width_layout.addWidget(self.window_width_label)
        window_layout.addLayout(width_layout)
        
        window_group.setLayout(window_layout)
        
        # Add components to preview layout
        preview_layout.addWidget(self.toolbar)
        preview_layout.addWidget(self.preview_canvas)
        preview_layout.addLayout(slice_layout)
        preview_layout.addWidget(window_group)
        
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)
        
        # Connect mouse movement to HU display
        self.preview_canvas.mpl_connect('motion_notify_event', self.update_hu_display)
        
        return right_panel

    def update_hu_display(self, event):
        if event.inaxes == self.preview_ax and self.current_volume is not None:
            try:
                x, y = int(round(event.xdata)), int(round(event.ydata))
                if (0 <= y < self.current_volume.shape[1] and 
                    0 <= x < self.current_volume.shape[2]):
                    hu_value = self.current_volume[self.slice_slider.value(), y, x]
                    self.statusBar().showMessage(
                        f"Position: (x={x}, y={y}) HU: {hu_value:.2f}"
                    )
                else:
                    self.statusBar().showMessage("")
            except (TypeError, ValueError):
                self.statusBar().showMessage("")

    def update_level_label(self, value):
        self.window_level_label.setText(f"Level: {value} HU")

    def update_width_label(self, value):
        self.window_width_label.setText(f"Width: {value} HU")

    def apply_window_preset(self, preset_name):
        if preset_name in WINDOW_PRESETS:
            level, width = WINDOW_PRESETS[preset_name]
            self.window_level_slider.setValue(level)
            self.window_width_slider.setValue(width)
            self.update_preview()

    def update_hu_from_material(self, material):
        if material in HU_PRESETS and HU_PRESETS[material] is not None:
            self.hu_value.setValue(HU_PRESETS[material])
            self.material_combo.setCurrentText(material)

    def update_phantom_size_limit(self):
        max_size = min(self.x_size.value(), self.y_size.value())
        self.phantom_size.setMaximum(max_size)
        if self.phantom_size.value() > max_size:
            self.phantom_size.setValue(max_size)
        self.update_position_limits()
    
    def update_position_limits(self):
        phantom_radius = self.phantom_size.value() // 2
        x_max = self.x_size.value() - phantom_radius
        y_max = self.y_size.value() - phantom_radius
        z_max = self.num_slices.value() - phantom_radius
        
        self.x_position.setRange(-phantom_radius, phantom_radius)
        self.y_position.setRange(-phantom_radius, phantom_radius)
        self.z_position.setRange(-phantom_radius, phantom_radius)
        
    def update_slice_slider_range(self):
        current_slice = min(self.slice_slider.value(), self.num_slices.value() - 1)
        self.slice_slider.setMaximum(self.num_slices.value() - 1)
        self.slice_slider.setValue(current_slice)
        self.update_position_limits()
    
    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir.setText(directory)
    
    def create_phantom_volume(self):
        x_size = self.x_size.value()
        y_size = self.y_size.value()
        z_size = self.num_slices.value()
        shape = self.shape_combo.currentText()
        phantom_size = self.phantom_size.value()
        hu_value = self.hu_value.value()
        sigma = self.edge_smoothing.value()
        
        pos_x = self.x_position.value() + x_size // 2
        pos_y = self.y_position.value() + y_size // 2
        pos_z = self.z_position.value() + z_size // 2
        
        # Initialize volume with air HU value
        volume = np.full((z_size, y_size, x_size), HU_MIN, dtype=np.float32)
        radius = phantom_size // 2

        # Create phantom shape with proper HU values
        if shape == "Cube":
            volume[
                max(0, pos_z - radius):min(z_size, pos_z + radius),
                max(0, pos_y - radius):min(y_size, pos_y + radius),
                max(0, pos_x - radius):min(x_size, pos_x + radius)
            ] = hu_value
        
        elif shape == "Cylinder":
            y, x = np.ogrid[-pos_y:y_size-pos_y, -pos_x:x_size-pos_x]
            mask = x*x + y*y <= radius*radius
            for z in range(max(0, pos_z - radius), min(z_size, pos_z + radius)):
                volume[z][mask] = hu_value
        
        elif shape == "Sphere":
            z, y, x = np.ogrid[-pos_z:z_size-pos_z,
                              -pos_y:y_size-pos_y,
                              -pos_x:x_size-pos_x]
            distances = np.sqrt(x*x + y*y + z*z)
            mask = distances <= radius
            volume[mask] = hu_value

            # Add gradual transition at the edges if smoothing is enabled
            if sigma > 0:
                transition = np.exp(-(distances - radius)**2 / (2*sigma**2))
                transition[distances <= radius] = 1
                transition[distances > radius + 3*sigma] = 0
                volume = HU_MIN + (hu_value - HU_MIN) * transition
        
        # Apply Gaussian smoothing if enabled and not already applied
        if sigma > 0 and shape != "Sphere":
            volume = gaussian_filter(volume, sigma=sigma)
        
        return volume
    
    def apply_window_level(self, image):
        window_center = self.window_level_slider.value()
        window_width = self.window_width_slider.value()
        
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        
        normalized_image = np.clip((image - min_value) / (max_value - min_value), 0, 1)
        return normalized_image
    
    def update_preview(self):
        if self.current_volume is None:
            return
        
        self.preview_ax.clear()
        self.colorbar_ax.clear()
        
        current_slice = self.slice_slider.value()
        slice_image = self.current_volume[current_slice]
        
        # Display image with proper window/level
        im = self.preview_ax.imshow(
            slice_image,
            cmap='gray',
            vmin=self.window_level_slider.value() - self.window_width_slider.value()//2,
            vmax=self.window_level_slider.value() + self.window_width_slider.value()//2
        )
        
        # Remove axis ticks but keep the image aspect ratio
        self.preview_ax.set_xticks([])
        self.preview_ax.set_yticks([])
        self.preview_ax.set_aspect('equal')
        
        # Update colorbar
        plt.colorbar(im, cax=self.colorbar_ax, label='HU')
        
        # Update title
        self.preview_ax.set_title(f'Slice {current_slice + 1}/{self.current_volume.shape[0]}')
        self.slice_label.setText(f"Slice: {current_slice + 1}")
        
        # Update canvas
        self.preview_canvas.draw()
    
    def preview_phantom(self):
        self.current_volume = self.create_phantom_volume()
        self.update_preview()
        
        
    def generate_dicom(self):
        if not self.output_dir.text():
            QMessageBox.warning(self, "Warning", "Please select output directory first.")
            return

        volume = self.create_phantom_volume()
        num_slices = volume.shape[0]

        # Create progress dialog
        progress = QProgressDialog("Generating DICOM files...", "Cancel", 0, num_slices, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Generate UIDs
        study_instance_uid = generate_uid()
        series_instance_uid = generate_uid()
        frame_of_reference_uid = generate_uid()

        # Get current date and time
        now = datetime.now()
        series_name = self.series_name.text()
        output_dir = self.output_dir.text()
        series_dir = os.path.join(output_dir, f"{series_name}_{now.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(series_dir, exist_ok=True)

        try:
            for i in range(num_slices):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                ds = pydicom.Dataset()
                
                # Basic DICOM Information
                ds.file_meta = pydicom.Dataset()
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
                ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
                ds.file_meta.ImplementationClassUID = generate_uid()
                
                # Patient and Study Information
                ds.PatientName = self.patient_name.text()
                ds.PatientID = self.patient_id.text()
                ds.PatientBirthDate = ""
                ds.PatientSex = ""
                ds.PatientPosition = "HFS"  # Head First-Supine
                ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
                ds.StudyDescription = "Phantom Study"
                ds.SeriesDescription = series_name
                
                # Study and Series Information
                ds.StudyInstanceUID = study_instance_uid
                ds.SeriesInstanceUID = series_instance_uid
                ds.FrameOfReferenceUID = frame_of_reference_uid
                ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
                ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
                ds.Modality = "CT"
                ds.Manufacturer = "PHANTOM_GENERATOR"
                
                # Date and Time
                ds.StudyDate = now.strftime("%Y%m%d")
                ds.SeriesDate = now.strftime("%Y%m%d")
                ds.AcquisitionDate = now.strftime("%Y%m%d")
                ds.ContentDate = now.strftime("%Y%m%d")
                ds.StudyTime = now.strftime("%H%M%S")
                ds.SeriesTime = now.strftime("%H%M%S")
                ds.AcquisitionTime = now.strftime("%H%M%S")
                ds.ContentTime = now.strftime("%H%M%S")
                
                # Image Specific Information
                ds.SeriesNumber = 1
                ds.AcquisitionNumber = 1
                ds.InstanceNumber = i + 1
                ds.ImagePositionPatient = [0, 0, i * self.slice_thickness.value()]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.SliceLocation = i * self.slice_thickness.value()
                ds.SliceThickness = self.slice_thickness.value()
                ds.PixelSpacing = [1.0, 1.0]
                ds.KVP = "120"
                
                # Image Pixel Characteristics
                ds.Rows = self.y_size.value()
                ds.Columns = self.x_size.value()
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"

                # Window Settings
                ds.WindowCenter = str(self.window_level_slider.value())
                ds.WindowWidth = str(self.window_width_slider.value())
                
                # Convert HU values to stored pixel values
                # For CT: stored_value = (HU_value + 1024)
                pixel_data = volume[i].astype(np.float32)
                rescaled_pixels = ((pixel_data + 1024) * 1).astype(np.int16)
                ds.PixelData = rescaled_pixels.tobytes()
                
                # Rescale Intercept and Slope for correct HU values
                ds.RescaleIntercept = -1024
                ds.RescaleSlope = 1
                ds.RescaleType = "HU"
                
                ds.is_little_endian = True
                ds.is_implicit_VR = False

                # Save as DICOM file
                filename = f"slice_{i+1:04d}.dcm"
                ds.save_as(os.path.join(series_dir, filename), write_like_original=False)

            progress.setValue(num_slices)
            QMessageBox.information(self, "Success", 
                                  f"DICOM series generated successfully!\n"
                                  f"Location: {series_dir}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            progress.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhantomGenerator()
    window.show()
    sys.exit(app.exec())