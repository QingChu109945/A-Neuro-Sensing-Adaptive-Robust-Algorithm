import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CompensationParams:
    """温度补偿参数"""
    sensor_type: str
    reference_temp: float = 25.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    calibrated: bool = False

class TemperatureCompensator:
    """温度补偿器基类"""
    
    def __init__(self, params: CompensationParams):
        self.params = params
        self._calibration_data: List[Tuple[float, float, float]] = []
    
    def calibrate(self, calibration_data: List[Tuple[float, float, float]]):
        """校准补偿参数"""
        self._calibration_data = calibration_data
        self._fit_params()
        self.params.calibrated = True
    
    def _fit_params(self):
        """拟合补偿参数"""
        raise NotImplementedError
    
    def compensate(self, raw_value: float, current_temp: float) -> float:
        """应用温度补偿"""
        if not self.params.calibrated:
            return raw_value
        return self._compensate(raw_value, current_temp)
    
    def _compensate(self, raw_value: float, current_temp: float) -> float:
        """补偿计算"""
        raise NotImplementedError

class LinearTemperatureCompensator(TemperatureCompensator):
    """线性温度补偿器"""
    
    def _fit_params(self):
        """线性拟合补偿参数"""
        if len(self._calibration_data) < 2:
            return
        
        temps = np.array([d[0] for d in self._calibration_data])
        raw_vals = np.array([d[1] for d in self._calibration_data])
        true_vals = np.array([d[2] for d in self._calibration_data])
        
        errors = raw_vals - true_vals
        self.params.alpha = np.polyfit(temps, errors, 1)[0]
        self.params.beta = np.polyfit(temps, errors, 1)[1]
    
    def _compensate(self, raw_value: float, current_temp: float) -> float:
        """线性补偿公式"""
        error = self.params.alpha * (current_temp - self.params.reference_temp) + self.params.beta
        return raw_value - error

class QuadraticTemperatureCompensator(TemperatureCompensator):
    """二次温度补偿器"""
    
    def _fit_params(self):
        """二次拟合补偿参数"""
        if len(self._calibration_data) < 3:
            return
        
        temps = np.array([d[0] for d in self._calibration_data])
        raw_vals = np.array([d[1] for d in self._calibration_data])
        true_vals = np.array([d[2] for d in self._calibration_data])
        
        errors = raw_vals - true_vals
        coeffs = np.polyfit(temps, errors, 2)
        self.params.alpha = coeffs[0]
        self.params.beta = coeffs[1]
        self.params.gamma = coeffs[2]
    
    def _compensate(self, raw_value: float, current_temp: float) -> float:
        """二次补偿公式"""
        delta_temp = current_temp - self.params.reference_temp
        error = self.params.alpha * delta_temp**2 + self.params.beta * delta_temp + self.params.gamma
        return raw_value - error

class VibrationTemperatureCompensator(TemperatureCompensator):
    """振动传感器温度补偿器"""
    
    def __init__(self, params: CompensationParams):
        super().__init__(params)
        self._axis_params = {
            'x': CompensationParams('vibration_x'),
            'y': CompensationParams('vibration_y'),
            'z': CompensationParams('vibration_z')
        }
    
    def _fit_params(self):
        """拟合振动补偿参数（为所有轴拟合相同的参数）"""
        if len(self._calibration_data) < 2:
            return
        
        temps = np.array([d[0] for d in self._calibration_data])
        raw_vals = np.array([d[1] for d in self._calibration_data])
        true_vals = np.array([d[2] for d in self._calibration_data])
        
        errors = raw_vals - true_vals
        coeffs = np.polyfit(temps, errors, 1)
        
        for axis in ['x', 'y', 'z']:
            self._axis_params[axis].alpha = coeffs[0]
            self._axis_params[axis].beta = coeffs[1]
            self._axis_params[axis].calibrated = True
    
    def calibrate_axis(self, axis: str, calibration_data: List[Tuple[float, float, float]]):
        """校准单个轴的补偿参数"""
        if axis in self._axis_params:
            compensator = LinearTemperatureCompensator(self._axis_params[axis])
            compensator.calibrate(calibration_data)
    
    def compensate_vibration(self, vibration_data: Dict[str, float], current_temp: float) -> Dict[str, float]:
        """补偿振动数据"""
        compensated = {}
        for axis in ['x', 'y', 'z']:
            if axis in vibration_data:
                raw_val = vibration_data[axis]
                if self._axis_params[axis].calibrated:
                    delta_temp = current_temp - self.params.reference_temp
                    error = self._axis_params[axis].alpha * delta_temp + self._axis_params[axis].beta
                    compensated[axis] = raw_val - error
                else:
                    compensated[axis] = raw_val
        
        if 'amplitude' in vibration_data:
            compensated['amplitude'] = (compensated.get('x', 0)**2 + 
                                      compensated.get('y', 0)**2 + 
                                      compensated.get('z', 0)**2)**0.5
        
        return compensated

class LaserIntensityCompensator(TemperatureCompensator):
    """激光回波强度温度补偿器"""
    
    def _fit_params(self):
        """拟合激光探测器温度补偿参数"""
        if len(self._calibration_data) < 3:
            return
        
        temps = np.array([d[0] for d in self._calibration_data])
        raw_vals = np.array([d[1] for d in self._calibration_data])
        true_vals = np.array([d[2] for d in self._calibration_data])
        
        ratios = true_vals / raw_vals
        coeffs = np.polyfit(temps, ratios, 2)
        self.params.alpha = coeffs[0]
        self.params.beta = coeffs[1]
        self.params.gamma = coeffs[2]
    
    def _compensate(self, raw_value: float, current_temp: float) -> float:
        """激光强度补偿公式"""
        delta_temp = current_temp - self.params.reference_temp
        ratio = self.params.alpha * delta_temp**2 + self.params.beta * delta_temp + self.params.gamma
        return raw_value * ratio

class CompensationManager:
    """温度补偿管理器"""
    
    def __init__(self):
        self._compensators: Dict[str, TemperatureCompensator] = {}
    
    def add_compensator(self, name: str, compensator: TemperatureCompensator):
        """添加补偿器"""
        self._compensators[name] = compensator
    
    def get_compensator(self, name: str):
        """获取补偿器"""
        return self._compensators.get(name)
    
    def compensate_all(self, sensor_data: Dict[str, Dict[str, float]], 
                       temperature: float) -> Dict[str, Dict[str, float]]:
        """批量补偿所有传感器数据"""
        compensated_data = {}
        
        for sensor_name, data in sensor_data.items():
            if sensor_name in self._compensators:
                compensator = self._compensators[sensor_name]
                if sensor_name == 'vibration':
                    compensated_data[sensor_name] = compensator.compensate_vibration(data, temperature)
                elif sensor_name == 'laser_detector':
                    compensated_data[sensor_name] = data.copy()
                    if 'intensity_nw' in data:
                        compensated_data[sensor_name]['intensity_nw'] = \
                            compensator.compensate(data['intensity_nw'], temperature)
                        compensated_data[sensor_name]['intensity_w'] = \
                            compensated_data[sensor_name]['intensity_nw'] * 1e-9
                else:
                    compensated_data[sensor_name] = {
                        key: compensator.compensate(val, temperature) 
                        for key, val in data.items()
                    }
            else:
                compensated_data[sensor_name] = data
        
        return compensated_data
    
    def load_calibration_from_file(self, file_path: str):
        """从文件加载校准数据"""
        import json
        with open(file_path, 'r') as f:
            calibration_data = json.load(f)
        
        for sensor_name, data in calibration_data.items():
            if sensor_name in self._compensators:
                self._compensators[sensor_name].calibrate(data)
    
    def save_calibration_to_file(self, file_path: str):
        """保存校准数据到文件"""
        import json
        calibration_data = {}
        for sensor_name, compensator in self._compensators.items():
            calibration_data[sensor_name] = compensator._calibration_data
        
        with open(file_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

def create_default_compensators() -> CompensationManager:
    """创建默认补偿器配置"""
    manager = CompensationManager()
    
    vibration_params = CompensationParams('vibration', reference_temp=25.0)
    vibration_compensator = VibrationTemperatureCompensator(vibration_params)
    manager.add_compensator('vibration', vibration_compensator)
    
    laser_params = CompensationParams('laser_detector', reference_temp=25.0)
    laser_compensator = LaserIntensityCompensator(laser_params)
    manager.add_compensator('laser_detector', laser_compensator)
    
    return manager

def generate_calibration_data(temperature_range: Tuple[float, float] = (20.0, 80.0), 
                             num_points: int = 7) -> Dict[str, List[Tuple[float, float, float]]]:
    """生成模拟校准数据"""
    temps = np.linspace(temperature_range[0], temperature_range[1], num_points)
    
    vibration_data = []
    laser_data = []
    
    for temp in temps:
        base_vibration = 1.0
        true_vibration = base_vibration
        raw_vibration = true_vibration + 0.002 * (temp - 25) + 0.0001 * (temp - 25)**2
        vibration_data.append((temp, raw_vibration, true_vibration))
        
        base_intensity = 1000.0
        true_intensity = base_intensity
        raw_intensity = true_intensity * (1.0 + 0.005 * (temp - 25) + 0.0002 * (temp - 25)**2)
        laser_data.append((temp, raw_intensity, true_intensity))
    
    return {
        'vibration': vibration_data,
        'laser_detector': laser_data
    }