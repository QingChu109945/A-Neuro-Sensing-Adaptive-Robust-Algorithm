"""
数据模拟脚本 - 用于测试和验证
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_system.storage import DataStorage, DataAnalyzer
from experiment_system.visualization import DataVisualizer
from experiment_system.validation import DataValidator

def generate_simulated_vibration_data(duration_seconds: int = 300, 
                                      sampling_rate: int = 100) -> np.ndarray:
    """生成模拟振动数据"""
    num_samples = duration_seconds * sampling_rate
    time = np.arange(num_samples) / sampling_rate
    
    base_freq = 5.0
    amplitude = 0.5
    
    x = amplitude * np.sin(2 * np.pi * base_freq * time) + \
        0.1 * np.sin(2 * np.pi * 15 * time) + \
        np.random.normal(0, 0.02, num_samples)
    
    y = amplitude * np.cos(2 * np.pi * base_freq * time) + \
        0.1 * np.cos(2 * np.pi * 12 * time) + \
        np.random.normal(0, 0.02, num_samples)
    
    z = 0.3 * np.sin(2 * np.pi * 8 * time) + \
        np.random.normal(0, 0.02, num_samples)
    
    return np.column_stack([x, y, z])

def generate_simulated_laser_data(duration_seconds: int = 300,
                                  sampling_rate: int = 100) -> np.ndarray:
    """生成模拟激光回波数据"""
    num_samples = duration_seconds * sampling_rate
    time = np.arange(num_samples) / sampling_rate
    
    base_intensity = 1000.0
    modulation_freq = 2.0
    
    intensity = base_intensity * (1 + 0.2 * np.sin(2 * np.pi * modulation_freq * time)) + \
                np.random.normal(0, 50, num_samples)
    
    detector_temp = 25.0 + 5 * np.sin(2 * np.pi * 0.1 * time) + \
                    np.random.normal(0, 0.5, num_samples)
    
    return np.column_stack([intensity, intensity * 1e-9, detector_temp])

def generate_simulated_temperature_data(duration_seconds: int = 300,
                                        sampling_rate: int = 100,
                                        target_temp: float = 25.0) -> np.ndarray:
    """生成模拟温度数据"""
    num_samples = duration_seconds * sampling_rate
    
    temperature = target_temp + np.random.normal(0, 0.3, num_samples)
    
    return temperature

def generate_simulated_qcl_data(duration_seconds: int = 300,
                                sampling_rate: int = 100,
                                power_mw: float = 20.0) -> np.ndarray:
    """生成模拟QCL数据"""
    num_samples = duration_seconds * sampling_rate
    
    power = power_mw + np.random.normal(0, 0.5, num_samples)
    temp = 25.0 + np.random.normal(0, 1.0, num_samples)
    current = power * 10 + np.random.normal(0, 5, num_samples)
    
    return np.column_stack([power, temp, current])

def add_temperature_bias(data: np.ndarray, temperature: float) -> np.ndarray:
    """添加温度偏差"""
    bias = 0.002 * (temperature - 25) + 0.0001 * (temperature - 25)**2
    return data + bias

def simulate_experiment(temperature: float = 25.0, duration: int = 300):
    """模拟完整实验"""
    print(f"Simulating experiment at {temperature}°C...")
    
    vibration = generate_simulated_vibration_data(duration)
    laser = generate_simulated_laser_data(duration)
    temperature_data = generate_simulated_temperature_data(duration, target_temp=temperature)
    qcl = generate_simulated_qcl_data(duration)
    
    vibration_with_bias = add_temperature_bias(vibration, temperature)
    laser_with_bias = add_temperature_bias(laser[:, 0], temperature)
    laser_with_bias = np.column_stack([laser_with_bias, laser[:, 1], laser[:, 2]])
    
    data = {
        'vibration': vibration_with_bias,
        'laser_detector': laser_with_bias,
        'temperature': temperature_data,
        'qcl': qcl,
        'vibration_true': vibration,
        'laser_true': laser[:, 0]
    }
    
    storage = DataStorage('./data/simulation')
    analyzer = DataAnalyzer(storage)
    visualizer = DataVisualizer(storage)
    validator = DataValidator()
    
    storage.save_processed_data(data, f"simulation_{temperature}c")
    
    stats = analyzer.compute_statistics(data)
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    rmse = analyzer.compute_rmse(vibration_with_bias[:, 0], vibration[:, 0])
    print(f"\nVibration RMSE (before compensation): {rmse:.6f}")
    
    visualizer.generate_overview_dashboard(data, save_path=f"simulation_{temperature}c_dashboard.png")
    visualizer.plot_vibration_data(vibration_with_bias, save_path=f"simulation_{temperature}c_vibration.png")
    visualizer.plot_laser_intensity(laser_with_bias[:, 0], save_path=f"simulation_{temperature}c_laser.png")
    
    results = validator.run_complete_validation(data)
    report = validator.generate_validation_report(results)
    print("\nValidation Report:")
    print(report)
    
    return data

if __name__ == "__main__":
    print("Data Simulation Tool")
    print("=" * 60)
    
    temperatures = [20, 30, 40, 50, 60, 70, 80]
    
    for temp in temperatures:
        simulate_experiment(temp, duration=120)
    
    print("\nAll simulations completed!")