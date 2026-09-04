# Experimental Workflow Guide: Non-Cooperative Target Measurement

## Overview

This guide provides a step-by-step workflow for conducting experiments using the non-cooperative target measurement system, supporting the paper writing process by documenting the complete experimental protocol.

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Experiment Workflow                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │ 1. Setup     │ -> │ 2. Calibrate │ -> │ 3. Acquire   │             │
│  │              │    │              │    │              │             │
│  │ • Configure  │    │ • Temperature│    │ • Sensors    │             │
│  │ • Sensors    │    │ • Filters    │    │ • Data       │             │
│  │ • Database   │    │ • Reference  │    │ • Metadata   │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                   │                   │                       │
│         v                   v                   v                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │ 6. Document  │ <- │ 5. Validate  │ <- │ 4. Process   │             │
│  │              │    │              │    │              │             │
│  │ • Reports    │    │ • QC Checks  │    │ • Compensation│            │
│  │ • Figures    │    │ • Statistics │    │ • Filtering  │             │
│  │ • Logs       │    │ • Comparison │    │ • Inversion  │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Step 1: System Setup

### 1.1 Configuration

```python
from experiment_system import DEFAULT_CONFIG, save_config

config = DEFAULT_CONFIG
config.experiment.experiment_name = "exp_2024_01_01"
config.experiment.output_dir = "./data/experiments"
config.experiment.duration_seconds = 300
config.experiment.use_simulator = True

config.processing.filter_type = "ns_arkf"
config.processing.inversion_model = "ssm_pinn"
config.processing.noise_type = "gaussian"
config.processing.noise_level = 0.01

save_config(config, "./configs/exp_config.yaml")
```

### 1.2 Material Database Initialization

```python
from experiment_system import ExperimentSystem

system = ExperimentSystem(config)
system.initialize_material_database()

materials = system.storage.get_materials()
print(f"Loaded {len(materials)} materials")
```

### 1.3 Sensor Setup

```python
system.setup_sensors()
system.connect_sensors()

status = system.sensor_manager.get_sensor_status()
for sensor, state in status.items():
    print(f"{sensor}: {state}")
```

## Step 2: Calibration

### 2.1 Temperature Compensation Calibration

```python
system.calibrate_temperature_compensation()
```

### 2.2 Filter Setup

```python
system.setup_filters()
```

### 2.3 Reference Measurement

```python
system.storage.start_experiment("reference_measurement", config, 100)

reference_material_id = 1
system.perform_inversion(material_id=reference_material_id)

system.storage.end_experiment('completed')
```

## Step 3: Data Acquisition

### 3.1 Start Experiment

```python
experiment_id = system.storage.start_experiment(
    name="material_characterization",
    config=config,
    num_samples=1000
)
```

### 3.2 Add Experimental Conditions

```python
system.add_noise_configuration(
    noise_type="gaussian",
    param_1=0.01,
    param_2=42,
    description="Low-level Gaussian noise"
)

system.add_measurement_conditions(
    temperature=25.0,
    distance=5.0,
    angle=30.0,
    vibration_level=10.0,
    laser_power_mw=20.0
)
```

### 3.3 Data Collection

```python
system.start_experiment()
```

## Step 4: Data Processing

### 4.1 Temperature Compensation

```python
system.process_data()
```

### 4.2 Filtering

The filtering step is automatically applied during `process_data()` if `apply_filtering=True` in the configuration.

### 4.3 Material Property Inversion

```python
target_material_id = 5
system.perform_inversion(material_id=target_material_id)
```

## Step 5: Validation

### 5.1 Data Quality Check

```python
validation_results = system.validate_data()
report = system.validator.generate_validation_report(validation_results)
print(report)
```

### 5.2 Performance Analysis

```python
analysis_results = system.analyze_data()

filter_perf = analysis_results.get('filter_performance', {})
print(f"RMSE: {filter_perf.get('rmse', 'N/A'):.4f}")
print(f"MAE: {filter_perf.get('mae', 'N/A'):.4f}")
print(f"Correlation: {filter_perf.get('correlation', 'N/A'):.4f}")
```

## Step 6: Documentation

### 6.1 Generate Visualizations

```python
system.visualize_data()
```

### 6.2 Save Results

```python
system.save_data()
```

### 6.3 End Experiment

```python
system.storage.end_experiment('completed')
```

## Complete Experiment Example

```python
from experiment_system import ExperimentSystem, DEFAULT_CONFIG
import os

config = DEFAULT_CONFIG
config.experiment.experiment_name = "full_validation_test"
config.experiment.output_dir = "./data/full_test"
config.experiment.duration_seconds = 60
config.experiment.use_simulator = True

config.processing.filter_type = "ns_arkf"
config.processing.inversion_model = "ssm_pinn"
config.processing.apply_temperature_compensation = True
config.processing.apply_filtering = True
config.processing.apply_inversion = True

os.makedirs(config.experiment.output_dir, exist_ok=True)

system = ExperimentSystem(config)

print("Step 1: Initializing material database...")
system.initialize_material_database()

print("Step 2: Setting up filters...")
system.setup_filters()

print("Step 3: Calibrating temperature compensation...")
system.calibrate_temperature_compensation()

print("Step 4: Starting experiment...")
system.storage.start_experiment("full_test", config, 100)

print("Step 5: Connecting sensors...")
system.setup_sensors()
system.connect_sensors()

print("Step 6: Adding experimental conditions...")
system.add_noise_configuration(noise_type="gaussian", param_1=0.01)
system.add_measurement_conditions()

print("Step 7: Collecting data...")
system.start_experiment()

print("Step 8: Processing data...")
system.process_data()

print("Step 9: Performing inversion...")
system.perform_inversion(material_id=1)

print("Step 10: Analyzing data...")
system.analyze_data()

print("Step 11: Saving data...")
system.save_data()

print("Step 12: Visualizing data...")
system.visualize_data()

print("Step 13: Ending experiment...")
system.storage.end_experiment('completed')

print("\nExperiment completed successfully!")
```

## Batch Experiment Workflow

```python
from experiment_system import ExperimentSystem, DEFAULT_CONFIG
import numpy as np

materials = [1, 2, 3, 4, 5]
temperatures = [25, 40, 55, 70]
distances = [3, 5, 7]

for material_id in materials:
    for temp in temperatures:
        for distance in distances:
            config = DEFAULT_CONFIG
            config.experiment.experiment_name = f"mat{material_id}_temp{temp}_dist{distance}"
            config.experiment.duration_seconds = 30
            
            system = ExperimentSystem(config)
            system.initialize_material_database()
            
            system.storage.start_experiment(
                config.experiment.experiment_name,
                config,
                50
            )
            
            system.add_measurement_conditions(
                temperature=temp,
                distance=distance
            )
            
            system.setup_sensors()
            system.connect_sensors()
            system.start_experiment()
            system.process_data()
            system.perform_inversion(material_id=material_id)
            system.analyze_data()
            system.save_data()
            system.storage.end_experiment('completed')
            
            print(f"Completed: mat{material_id}_temp{temp}_dist{distance}")
```

## Noise Injection Test Workflow

```python
from experiment_system import ExperimentSystem, DEFAULT_CONFIG
from experiment_system.data_generator import NoiseInjector, NoiseConfig

noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
noise_types = ["gaussian", "uniform", "impulse", "mixed"]

for noise_type in noise_types:
    for noise_level in noise_levels:
        config = DEFAULT_CONFIG
        config.experiment.experiment_name = f"noise_{noise_type}_{noise_level}"
        config.processing.noise_type = noise_type
        config.processing.noise_level = noise_level
        
        system = ExperimentSystem(config)
        system.initialize_material_database()
        system.storage.start_experiment(config.experiment.experiment_name, config, 100)
        system.setup_sensors()
        system.connect_sensors()
        system.start_experiment()
        system.process_data()
        system.analyze_data()
        system.storage.end_experiment('completed')
        
        print(f"Completed noise test: {noise_type} at level {noise_level}")
```

## Performance Benchmark Workflow

```python
from experiment_system import ExperimentSystem, DEFAULT_CONFIG
import time

filter_methods = ["ns_arkf", "ekf", "uif"]
inversion_methods = ["ssm_pinn", "bayesian", "ifhbfnn"]

results = []

for filter_method in filter_methods:
    for inversion_method in inversion_methods:
        config = DEFAULT_CONFIG
        config.processing.filter_type = filter_method
        config.processing.inversion_model = inversion_method
        
        system = ExperimentSystem(config)
        system.initialize_material_database()
        system.storage.start_experiment("benchmark", config, 100)
        
        start_time = time.time()
        
        system.setup_sensors()
        system.connect_sensors()
        system.start_experiment()
        system.process_data()
        system.perform_inversion(material_id=1)
        
        elapsed_time = time.time() - start_time
        
        analysis = system.analyze_data()
        filter_perf = analysis.get('filter_performance', {})
        
        results.append({
            'filter': filter_method,
            'inversion': inversion_method,
            'time': elapsed_time,
            'rmse': filter_perf.get('rmse', float('inf'))
        })
        
        system.storage.end_experiment('completed')

print("\nBenchmark Results:")
print("-" * 60)
for r in results:
    print(f"{r['filter']:10s} | {r['inversion']:15s} | Time: {r['time']:6.2f}s | RMSE: {r['rmse']:8.4f}")
```

## Error Handling and Recovery

### Try-Except Pattern

```python
from experiment_system import ExperimentSystem

try:
    system = ExperimentSystem()
    system.initialize_material_database()
    system.setup_sensors()
    system.connect_sensors()
    
    try:
        system.start_experiment()
        system.process_data()
    except Exception as e:
        print(f"Experiment error: {e}")
        system.storage.end_experiment('failed')
        
    system.save_data()
    system.analyze_data()
    
except Exception as e:
    print(f"System error: {e}")
    if 'system' in locals():
        system.disconnect_sensors()
```

## Logging Best Practices

```python
from experiment_system import log_status, ProgressBar

log_status("Starting experiment setup...", "INFO")

try:
    log_status("Initializing database...", "INFO")
    
    log_status("Database initialized successfully", "SUCCESS")
except Exception as e:
    log_status(f"Database initialization failed: {e}", "ERROR")
    raise

progress = ProgressBar(total=100, description="Processing")
for i in range(100):
    progress.update(i + 1, f"Processing sample {i + 1}")
progress.finish()
```

## Data Retrieval for Analysis

```python
from experiment_system import ExperimentDatabase

db = ExperimentDatabase("./data/experiment.db")

experiments = db.get_all_experiments()
for exp in experiments:
    print(f"Experiment {exp['id']}: {exp['name']} ({exp['status']})")

experiment_data = db.get_experiment(1)
sensor_data = db.get_sensor_data(1)
inversion_results = db.get_inversion_results(1)

print(f"Number of inversion results: {len(inversion_results)}")
```

## Conclusion

This workflow guide provides the complete protocol for conducting experiments with the non-cooperative target measurement system. By following these steps, researchers can systematically collect, process, analyze, and document experimental data in support of the paper writing process.
