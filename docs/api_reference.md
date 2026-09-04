# API Reference: Non-Cooperative Target Measurement Experimental System

## Overview

This document provides a comprehensive API reference for the experimental system modules, supporting the paper writing process by documenting all key classes, methods, and their usage.

## Module Structure

```
experiment_system/
├── config.py          # Configuration classes and utilities
├── sensors.py         # Sensor management and data acquisition
├── temperature_compensation.py  # Temperature compensation modules
├── filtering.py       # Kalman filtering and adaptive algorithms
├── inversion.py       # Material property inversion models
├── storage.py         # Data storage and management
├── database.py        # Database operations
├── visualization.py   # Data visualization utilities
├── validation.py      # Data validation and quality assessment
├── progress.py        # Progress tracking and status reporting
├── main.py            # Main experiment system class
└── data_generator.py  # Synthetic data generation
```

## Configuration Module

### SystemConfig

```python
class SystemConfig:
    vibration: VibrationSensorConfig
    laser_detector: LaserDetectorConfig
    qcl: QCLConfig
    temperature: TemperatureSensorConfig
    experiment: ExperimentConfig
    processing: ProcessingConfig
    validation: ValidationConfig
```

### ExperimentConfig

```python
class ExperimentConfig:
    experiment_name: str          # Experiment identifier
    output_dir: str               # Output directory for data
    duration_seconds: int         # Experiment duration
    use_simulator: bool           # Use hardware simulator
    data_source: str              # Data source ('database' or 'file')
    temperature_range: List[float]  # [min, max] temperature range
    distance_range: List[float]     # [min, max] distance range
    angle_range: List[float]        # [min, max] angle range
```

### ProcessingConfig

```python
class ProcessingConfig:
    apply_temperature_compensation: bool
    apply_filtering: bool
    filter_type: str              # 'ns_arkf', 'ekf', 'uif'
    apply_inversion: bool
    inversion_model: str          # 'ssm_pinn', 'bayesian', 'ifhbfnn'
    noise_type: str               # Noise injection type
    noise_level: float            # Noise intensity
    noise_seed: int               # Random seed for reproducibility
```

## Sensor Module

### SensorManager

```python
class SensorManager:
    def setup_sensors(config: SystemConfig) -> None
    def connect_sensors() -> None
    def disconnect_sensors() -> None
    def read_data() -> Dict[str, np.ndarray]
    def get_sensor_status() -> Dict[str, str]
```

### VibrationSensor

```python
class VibrationSensor:
    def __init__(config: VibrationSensorConfig)
    def connect() -> bool
    def disconnect() -> None
    def read() -> np.ndarray  # Returns [x, y, z] acceleration
```

### LaserDetector

```python
class LaserDetector:
    def __init__(config: LaserDetectorConfig)
    def connect() -> bool
    def disconnect() -> None
    def read() -> np.ndarray  # Returns [intensity_nw, intensity_w, detector_temp_c]
```

### QCLController

```python
class QCLController:
    def __init__(config: QCLConfig)
    def connect() -> bool
    def set_power(mw: float) -> None
    def get_power() -> float
    def read() -> np.ndarray  # Returns [wavelength, power_mw, temperature]
```

### TemperatureSensor

```python
class TemperatureSensor:
    def __init__(config: TemperatureSensorConfig)
    def connect() -> bool
    def read() -> float  # Returns temperature in Celsius
```

## Filtering Module

### NSARKF

```python
class NSARKF:
    def __init__(dim_x: int, dim_z: int)
    def predict(F: np.ndarray) -> None
    def update(z: np.ndarray, H: np.ndarray) -> None
    def get_state() -> FilterState
```

### ExtendedKalmanFilter

```python
class ExtendedKalmanFilter:
    def __init__(dim_x: int, dim_z: int)
    def predict(F: np.ndarray, B: np.ndarray = None, u: np.ndarray = None) -> None
    def update(z: np.ndarray, H: np.ndarray, h: callable = None) -> None
```

### UnknownInputFilter

```python
class UnknownInputFilter:
    def __init__(dim_x: int, dim_z: int, dim_d: int = 1)
    def predict(F: np.ndarray) -> None
    def update(z: np.ndarray, H: np.ndarray) -> None
```

### HBKFO (Hippo-Black Kite Fusion Optimizer)

```python
class HBKFO:
    def __init__(dim: int, bounds: List[Tuple[float, float]], pop_size: int = 50)
    def optimize(objective: callable, max_iter: int = 100, p_switch: float = 0.7) -> np.ndarray
```

### FilteringManager

```python
class FilteringManager:
    def add_filter(name: str, filter_obj: object) -> None
    def apply_filter(filter_name: str, measurements: np.ndarray, F: np.ndarray, H: np.ndarray) -> np.ndarray
```

## Inversion Module

### SSMPINN

```python
class SSMPINN:
    def __init__(config: InversionConfig = None)
    def predict(X: np.ndarray) -> np.ndarray  # Returns [emissivity, reflectivity]
```

### BayesianInversion

```python
class BayesianInversion:
    def __init__(config: InversionConfig = None)
    def predict(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]  # (mean, std)
```

### IFHBFNN

```python
class IFHBFNN:
    def __init__(config: InversionConfig = None)
    def predict(X: np.ndarray) -> np.ndarray  # Returns [emissivity, reflectivity]
```

### InversionConfig

```python
class InversionConfig:
    method: str                      # 'ssm_pinn', 'bayesian', 'ifhbfnn'
    enforce_hard_constraint: bool    # Enforce Kirchhoff's law
    constraint_tolerance: float
    learning_rate: float
    max_iterations: int
    regularization_weight: float
```

### InversionResult

```python
class InversionResult:
    emissivity_pred: float
    emissivity_true: Optional[float]
    emissivity_std: float
    reflectivity_pred: float
    reflectivity_true: Optional[float]
    reflectivity_std: float
    constraint_satisfied: bool
    method: str
```

## Storage Module

### DataStorage

```python
class DataStorage:
    def __init__(db_path: str, use_database: bool = True)
    def start_experiment(name: str, config: SystemConfig, num_samples: int) -> int
    def end_experiment(status: str) -> None
    def save_raw_data(data: Dict, name: str) -> str
    def save_processed_data(data: Dict, name: str) -> str
    def save_metadata(metadata: Dict, name: str) -> str
    def add_material(**kwargs) -> int
    def get_materials() -> List[Dict]
    def add_inversion_result(**kwargs) -> int
    def add_inversion_results_batch(results: List[Dict]) -> None
```

### DataAnalyzer

```python
class DataAnalyzer:
    def compute_statistics(data: Dict) -> Dict
    def analyze_filter_performance(filtered: np.ndarray, reference: np.ndarray) -> Dict
    def generate_report(results: Dict) -> str
```

## Database Module

### ExperimentDatabase

```python
class ExperimentDatabase:
    def __init__(db_path: str)
    def add_experiment(**kwargs) -> int
    def get_experiment(id: int) -> Dict
    def add_sensor_data(experiment_id: int, sensor_name: str, data: np.ndarray) -> None
    def add_material(**kwargs) -> int
    def add_materials_batch(materials: List[Dict]) -> None
    def get_materials() -> List[Dict]
    def get_material_by_id(id: int) -> Dict
    def add_inversion_result(**kwargs) -> int
    def add_evaluation_metric(**kwargs) -> int
    def add_noise_configuration(**kwargs) -> int
    def add_measurement_conditions(**kwargs) -> int
```

## Main ExperimentSystem Class

```python
class ExperimentSystem:
    def __init__(config: SystemConfig = DEFAULT_CONFIG)
    def initialize_material_database() -> bool
    def setup_sensors() -> None
    def connect_sensors() -> None
    def disconnect_sensors() -> None
    def calibrate_temperature_compensation() -> None
    def setup_filters() -> None
    def start_experiment() -> None
    def process_data() -> None
    def save_data() -> None
    def analyze_data() -> Dict
    def perform_inversion(material_id: int = None) -> None
    def visualize_data() -> None
    def validate_data(reference_data: Dict = None) -> List
    def add_noise_configuration(**kwargs) -> int
    def add_measurement_conditions(**kwargs) -> int
```

## Progress Module

### ProgressBar

```python
class ProgressBar:
    def __init__(total: int, description: str = "")
    def update(current: int, message: str = "") -> None
    def finish() -> None
```

### Status Functions

```python
def log_status(message: str, level: str = "INFO") -> None
def log_progress(current: int, total: int, message: str = "") -> None
```

## Data Generator Module

### DataGenerator

```python
class DataGenerator:
    def __init__(config: SimulationConfig = None)
    def generate_all() -> Dict[str, np.ndarray]
    def generate_vibration_data(t: np.ndarray) -> np.ndarray
    def generate_laser_signal(t: np.ndarray) -> np.ndarray
    def generate_qcl_data(t: np.ndarray) -> np.ndarray
    def generate_temperature_data(t: np.ndarray) -> np.ndarray
```

### NoiseInjector

```python
class NoiseInjector:
    def __init__(config: NoiseConfig = None)
    def inject(data: np.ndarray) -> np.ndarray
    def inject_gaussian(data: np.ndarray, level: float) -> np.ndarray
    def inject_uniform(data: np.ndarray, level: float) -> np.ndarray
    def inject_poisson(data: np.ndarray) -> np.ndarray
    def inject_salt_pepper(data: np.ndarray, level: float) -> np.ndarray
    def inject_impulse(data: np.ndarray, level: float, probability: float) -> np.ndarray
```

## Usage Examples

### Basic Experiment Workflow

```python
from experiment_system import ExperimentSystem, DEFAULT_CONFIG

config = DEFAULT_CONFIG
config.experiment.duration_seconds = 30
config.processing.filter_type = 'ns_arkf'
config.processing.inversion_model = 'ssm_pinn'

system = ExperimentSystem(config)
system.initialize_material_database()
system.setup_filters()
system.calibrate_temperature_compensation()
system.storage.start_experiment("test_exp", config, 100)
system.setup_sensors()
system.connect_sensors()
system.start_experiment()
system.process_data()
system.perform_inversion(material_id=1)
system.analyze_data()
system.storage.end_experiment('completed')
```

### Material Database Operations

```python
from experiment_system import ExperimentDatabase

db = ExperimentDatabase("./data/experiment.db")

materials = db.get_materials()
for mat in materials:
    print(f"{mat['material_name']}: ε={mat['emissivity_mean']:.4f}")

new_material = db.add_material(
    category="Test Material",
    material_name="Sample Material",
    emissivity_mean=0.85,
    emissivity_std=0.02,
    reflectivity_mean=0.13,
    reflectivity_std=0.01
)
```

### Filtering Usage

```python
from experiment_system import create_ns_arkf_filter, FilteringManager

filter_manager = FilteringManager()
filter_manager.add_filter('ns_arkf', create_ns_arkf_filter(dim_x=6, dim_z=4))

F = np.eye(6)
H = np.eye(4, 6)

filtered_states = filter_manager.apply_filter('ns_arkf', measurements, F, H)
```

### Inversion Usage

```python
from experiment_system import create_ssm_pinn_model, InversionConfig

config = InversionConfig(
    method='ssm_pinn',
    enforce_hard_constraint=True,
    max_iterations=500
)

model = create_ssm_pinn_model(config)
predictions = model.predict(X)
```
