import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from .plot_config import get_plot_config, finalize_figure, apply_style

class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, storage):
        self.storage = storage
        apply_style()
    
    def plot_time_series(self, data: Dict[str, np.ndarray], title: str = "Time Series Data",
                        save_path: str = None, show: bool = None):
        """绘制时间序列图"""
        num_sensors = len(data)
        fig, axes = plt.subplots(num_sensors, 1, figsize=(12, 4 * num_sensors), sharex=True)
        
        if num_sensors == 1:
            axes = [axes]
        
        for i, (sensor_name, sensor_data) in enumerate(data.items()):
            time = np.arange(len(sensor_data)) / 100.0
            
            if sensor_data.ndim == 2:
                for j in range(sensor_data.shape[1]):
                    axes[i].plot(time, sensor_data[:, j], label=f'Channel {j+1}')
            else:
                axes[i].plot(time, sensor_data)
            
            axes[i].set_title(sensor_name.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_vibration_data(self, vibration_data: np.ndarray, title: str = "Vibration Data",
                           save_path: str = None, show: bool = None):
        """绘制振动数据"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        time = np.arange(len(vibration_data)) / 100.0
        
        axes[0].plot(time, vibration_data[:, 0], color='r', label='X-axis')
        axes[0].set_title('X-axis Vibration')
        axes[0].set_ylabel('Acceleration (g)')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(time, vibration_data[:, 1], color='g', label='Y-axis')
        axes[1].set_title('Y-axis Vibration')
        axes[1].set_ylabel('Acceleration (g)')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].plot(time, vibration_data[:, 2], color='b', label='Z-axis')
        axes[2].set_title('Z-axis Vibration')
        axes[2].set_ylabel('Acceleration (g)')
        axes[2].legend()
        axes[2].grid(True)
        
        axes[3].plot(time, np.sqrt(vibration_data[:, 0]**2 + vibration_data[:, 1]**2 + 
                                  vibration_data[:, 2]**2), color='k', label='Amplitude')
        axes[3].set_title('Vibration Amplitude')
        axes[3].set_ylabel('Amplitude (g)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()
        axes[3].grid(True)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_laser_intensity(self, intensity_data: np.ndarray, title: str = "Laser Echo Intensity",
                           save_path: str = None, show: bool = None):
        """绘制激光回波强度"""
        fig, ax = plt.subplots(figsize=(12, 6))
        time = np.arange(len(intensity_data)) / 100.0
        
        ax.plot(time, intensity_data, color='purple', label='Echo Intensity')
        ax.fill_between(time, intensity_data - np.std(intensity_data), 
                       intensity_data + np.std(intensity_data), 
                       alpha=0.2, color='purple', label='±1σ')
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Intensity (nW)')
        ax.legend()
        ax.grid(True)
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_temperature(self, temperature_data: np.ndarray, title: str = "Temperature",
                        save_path: str = None, show: bool = None):
        """绘制温度数据"""
        fig, ax = plt.subplots(figsize=(12, 6))
        time = np.arange(len(temperature_data)) / 100.0
        
        ax.plot(time, temperature_data, color='orange', label='Temperature')
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_filter_comparison(self, measurements: np.ndarray, filtered: np.ndarray,
                              ground_truth: np.ndarray = None, title: str = "Filter Comparison",
                              save_path: str = None, show: bool = None):
        """绘制滤波对比图"""
        fig, axes = plt.subplots(min(measurements.shape[1], 3), 1, figsize=(12, 4 * min(measurements.shape[1], 3)), 
                                sharex=True)
        
        if measurements.shape[1] == 1:
            axes = [axes]
        
        time = np.arange(len(measurements)) / 100.0
        
        for i in range(min(measurements.shape[1], 3)):
            axes[i].plot(time, measurements[:, i], color='gray', alpha=0.5, label='Raw')
            axes[i].plot(time, filtered[:, i], color='blue', label='Filtered')
            
            if ground_truth is not None:
                axes[i].plot(time, ground_truth[:, i], color='red', alpha=0.8, label='Ground Truth')
            
            axes[i].set_title(f'Channel {i+1}')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_inversion_results(self, emissivity_pred: np.ndarray, emissivity_true: np.ndarray,
                             reflectivity_pred: np.ndarray = None, reflectivity_true: np.ndarray = None,
                             title: str = "Inversion Results", save_path: str = None, show: bool = None):
        """绘制反演结果"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].scatter(emissivity_true, emissivity_pred, alpha=0.6, color='blue', s=20)
        axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect')
        axes[0].set_xlabel('True Emissivity')
        axes[0].set_ylabel('Predicted Emissivity')
        axes[0].set_title('Emissivity Inversion')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        
        if reflectivity_pred is not None and reflectivity_true is not None:
            axes[1].scatter(reflectivity_true, reflectivity_pred, alpha=0.6, color='green', s=20)
            axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect')
            axes[1].set_xlabel('True Reflectivity')
            axes[1].set_ylabel('Predicted Reflectivity')
            axes[1].set_title('Reflectivity Inversion')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_uncertainty(self, predictions: np.ndarray, uncertainty: np.ndarray,
                        ground_truth: np.ndarray = None, title: str = "Uncertainty Visualization",
                        save_path: str = None, show: bool = None):
        """绘制不确定性可视化"""
        fig, ax = plt.subplots(figsize=(12, 6))
        time = np.arange(len(predictions)) / 100.0
        
        ax.plot(time, predictions, color='blue', label='Prediction')
        ax.fill_between(time, predictions - 1.96 * uncertainty, 
                       predictions + 1.96 * uncertainty, 
                       alpha=0.2, color='blue', label='95% CI')
        
        if ground_truth is not None:
            ax.plot(time, ground_truth, color='red', alpha=0.8, label='Ground Truth')
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_error_histogram(self, errors: np.ndarray, title: str = "Error Distribution",
                           save_path: str = None, show: bool = None):
        """绘制误差直方图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
        
        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2), 
               'r--', label=f'N({mu:.4f}, {sigma:.4f})')
        
        ax.set_title(title)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def plot_calibration_curve(self, temperatures: np.ndarray, raw_values: np.ndarray,
                              compensated_values: np.ndarray, title: str = "Temperature Compensation",
                              save_path: str = None, show: bool = None):
        """绘制温度补偿曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(temperatures, raw_values, 'ro-', label='Raw Data')
        ax.plot(temperatures, compensated_values, 'bo-', label='Compensated Data')
        
        ax.set_title(title)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Sensor Value')
        ax.legend()
        ax.grid(True)
        
        return finalize_figure(fig, save_path=save_path, show=show)
    
    def generate_overview_dashboard(self, data: Dict, save_path: str = None, show: bool = None):
        """生成数据概览仪表盘"""
        fig = plt.figure(figsize=(20, 12))
        
        gs = fig.add_gridspec(3, 3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[2, :])
        
        sample_data = data.get('vibration', np.zeros((100, 3)))
        time = np.arange(len(sample_data)) / 100.0
        
        if 'vibration' in data:
            vib_data = data['vibration']
            if vib_data.ndim >= 2:
                ax1.plot(time[:min(500, len(vib_data))], vib_data[:min(500, len(vib_data)), 0], 'r', label='X')
                ax1.plot(time[:min(500, len(vib_data))], vib_data[:min(500, len(vib_data)), 1], 'g', label='Y')
                ax1.plot(time[:min(500, len(vib_data))], vib_data[:min(500, len(vib_data)), 2], 'b', label='Z')
            ax1.set_title('Vibration (First 5s)')
            ax1.legend()
            ax1.grid(True)
        
        if 'laser_detector' in data:
            laser_data = data['laser_detector']
            if laser_data.ndim >= 2:
                laser_intensity = laser_data[:, 0]
            else:
                laser_intensity = laser_data
            laser_time = np.arange(len(laser_intensity)) / 100.0
            ax2.plot(laser_time[:min(500, len(laser_intensity))], laser_intensity[:min(500, len(laser_intensity))], 'purple')
            ax2.set_title('Laser Intensity (First 5s)')
            ax2.grid(True)
        
        if 'temperature' in data:
            temp_data = data['temperature']
            temp_time = np.arange(len(temp_data)) / 100.0
            ax3.plot(temp_time[:min(500, len(temp_data))], temp_data[:min(500, len(temp_data))], 'orange')
            ax3.set_title('Temperature (First 5s)')
            ax3.grid(True)
        
        if 'vibration' in data:
            vib_data = data['vibration']
            if vib_data.ndim >= 2:
                ax4.hist(vib_data[:, 0], bins=50, alpha=0.5, label='X')
                ax4.hist(vib_data[:, 1], bins=50, alpha=0.5, label='Y')
                ax4.hist(vib_data[:, 2], bins=50, alpha=0.5, label='Z')
            ax4.set_title('Vibration Distribution')
            ax4.legend()
        
        if 'laser_detector' in data:
            laser_data = data['laser_detector']
            if laser_data.ndim >= 2:
                laser_intensity = laser_data[:, 0]
            else:
                laser_intensity = laser_data
            ax5.hist(laser_intensity, bins=50, color='purple', alpha=0.7)
            ax5.set_title('Laser Intensity Distribution')
        
        if 'temperature' in data:
            temp_data = data['temperature']
            temp_time = np.arange(len(temp_data)) / 100.0
            ax6.plot(temp_time, temp_data, 'orange')
            ax6.axhline(y=np.mean(temp_data), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(temp_data):.2f}°C')
            ax6.set_title('Temperature Over Time')
            ax6.legend()
            ax6.grid(True)
        
        fig.suptitle('Experimental Data Overview Dashboard', fontsize=16, y=0.98)
        fig.tight_layout()
        
        return finalize_figure(fig, save_path=save_path, show=show)

def create_default_visualizer(storage = None) -> DataVisualizer:
    """创建默认可视化器"""
    from .storage import create_default_storage
    if storage is None:
        storage = create_default_storage()
    return DataVisualizer(storage)