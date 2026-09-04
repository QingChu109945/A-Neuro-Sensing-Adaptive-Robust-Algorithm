import time
import threading
import struct
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from .config import VibrationSensorConfig, LaserDetectorConfig, QCLConfig, TemperatureSensorConfig

try:
    import serial
except ImportError:
    serial = None

try:
    import bluetooth
except ImportError:
    bluetooth = None

@dataclass
class SensorData:
    """传感器数据类"""
    timestamp: float
    sensor_name: str
    values: Dict[str, float]
    raw_data: Optional[bytes] = None

class BaseSensor:
    """传感器基类"""
    def __init__(self, config):
        self.config = config
        self._running = False
        self._data_buffer: List[SensorData] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
    
    def connect(self) -> bool:
        """连接传感器"""
        raise NotImplementedError
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
    
    def start_sampling(self):
        """开始采样"""
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
    
    def stop_sampling(self):
        """停止采样"""
        self._running = False
    
    def _sampling_loop(self):
        """采样循环"""
        raise NotImplementedError
    
    def get_data(self) -> List[SensorData]:
        """获取缓存数据"""
        with self._lock:
            data = self._data_buffer.copy()
            self._data_buffer.clear()
            return data
    
    def get_latest_data(self) -> Optional[SensorData]:
        """获取最新数据"""
        with self._lock:
            return self._data_buffer[-1] if self._data_buffer else None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def _add_data(self, values: Dict[str, float], raw_data: Optional[bytes] = None):
        """添加数据到缓冲区"""
        with self._lock:
            self._data_buffer.append(SensorData(
                timestamp=time.time(),
                sensor_name=self.config.name,
                values=values,
                raw_data=raw_data
            ))

class VibrationSensor(BaseSensor):
    """维特智能WTVB01-BT50振动传感器"""
    
    def __init__(self, config: VibrationSensorConfig, use_simulator: bool = True):
        super().__init__(config)
        self._socket: Optional[bluetooth.BluetoothSocket] = None
        self._use_simulator = use_simulator or (bluetooth is None)
        self._sim_time = 0.0
    
    def connect(self) -> bool:
        """连接蓝牙振动传感器"""
        if self._use_simulator:
            print("Vibration sensor: Using simulator mode (hardware disabled)")
            return True
        try:
            self._socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self._socket.connect((self.config.bluetooth_mac, 1))
            self._socket.settimeout(1.0)
            return True
        except Exception as e:
            print(f"Failed to connect to vibration sensor: {e}")
            print("Vibration sensor: Falling back to simulator mode")
            self._use_simulator = True
            return True
    
    def disconnect(self):
        """断开连接"""
        super().disconnect()
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def _sampling_loop(self):
        """振动数据采样循环"""
        sample_interval = 1.0 / self.config.sampling_rate
        while self._running:
            try:
                if self._use_simulator:
                    values = self._generate_simulated_data()
                    self._add_data(values)
                else:
                    data = self._socket.recv(20)
                    if len(data) >= 11:
                        values = self._parse_vibration_data(data)
                        self._add_data(values, data)
            except Exception as e:
                print(f"Vibration sensor error: {e}")
            time.sleep(sample_interval)
    
    def _generate_simulated_data(self) -> Dict[str, float]:
        """生成模拟振动数据"""
        self._sim_time += 1.0 / self.config.sampling_rate
        base_freq = 5.0
        x = 0.5 * np.sin(2 * np.pi * base_freq * self._sim_time) + \
            0.1 * np.sin(2 * np.pi * 15 * self._sim_time) + \
            np.random.normal(0, 0.02) + self.config.offset_x
        y = 0.5 * np.cos(2 * np.pi * base_freq * self._sim_time) + \
            0.1 * np.cos(2 * np.pi * 12 * self._sim_time) + \
            np.random.normal(0, 0.02) + self.config.offset_y
        z = 0.3 * np.sin(2 * np.pi * 8 * self._sim_time) + \
            np.random.normal(0, 0.02) + self.config.offset_z
        return {"x": x, "y": y, "z": z, "amplitude": (x**2 + y**2 + z**2)**0.5}
    
    def _parse_vibration_data(self, raw_data: bytes) -> Dict[str, float]:
        """解析WTVB01-BT50数据格式"""
        if len(raw_data) < 11:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        
        x_raw = struct.unpack('<h', raw_data[2:4])[0]
        y_raw = struct.unpack('<h', raw_data[4:6])[0]
        z_raw = struct.unpack('<h', raw_data[6:8])[0]
        
        x = x_raw / 32768.0 * self.config.range_g + self.config.offset_x
        y = y_raw / 32768.0 * self.config.range_g + self.config.offset_y
        z = z_raw / 32768.0 * self.config.range_g + self.config.offset_z
        
        return {"x": x, "y": y, "z": z, "amplitude": (x**2 + y**2 + z**2)**0.5}

class LaserDetector(BaseSensor):
    """labM-10.6激光探测器"""
    
    def __init__(self, config: LaserDetectorConfig, use_simulator: bool = True):
        super().__init__(config)
        self._serial: Optional[serial.Serial] = None
        self._use_simulator = use_simulator or (serial is None)
        self._sim_time = 0.0
    
    def connect(self) -> bool:
        """连接激光探测器"""
        if self._use_simulator:
            print("Laser detector: Using simulator mode (hardware disabled)")
            return True
        try:
            self._serial = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            return True
        except Exception as e:
            print(f"Failed to connect to laser detector: {e}")
            print("Laser detector: Falling back to simulator mode")
            self._use_simulator = True
            return True
    
    def disconnect(self):
        """断开连接"""
        super().disconnect()
        if self._serial:
            try:
                self._serial.close()
            except:
                pass
            self._serial = None
    
    def _sampling_loop(self):
        """激光回波采样循环"""
        sample_interval = 1.0 / self.config.sampling_rate
        while self._running:
            try:
                if self._use_simulator:
                    values = self._generate_simulated_data()
                    self._add_data(values)
                else:
                    line = self._serial.readline().decode('utf-8').strip()
                    if line:
                        values = self._parse_laser_data(line)
                        self._add_data(values, line.encode())
            except Exception as e:
                print(f"Laser detector error: {e}")
            time.sleep(sample_interval)
    
    def _generate_simulated_data(self) -> Dict[str, float]:
        """生成模拟激光数据"""
        self._sim_time += 1.0 / self.config.sampling_rate
        base_intensity = 1000.0 * self.config.gain
        intensity_nw = base_intensity * (1 + 0.2 * np.sin(2 * np.pi * 2 * self._sim_time)) + \
                        np.random.normal(0, 50)
        detector_temp = 25.0 + 5 * np.sin(2 * np.pi * 0.1 * self._sim_time) + \
                        np.random.normal(0, 0.5)
        return {
            "intensity_nw": intensity_nw,
            "intensity_w": intensity_nw * 1e-9,
            "detector_temp_c": detector_temp
        }
    
    def _parse_laser_data(self, line: str) -> Dict[str, float]:
        """解析激光探测器数据"""
        try:
            parts = line.split(',')
            if len(parts) >= 2:
                intensity_nw = float(parts[0])
                temperature = float(parts[1]) if len(parts) > 1 else 25.0
                return {
                    "intensity_nw": intensity_nw,
                    "intensity_w": intensity_nw * 1e-9,
                    "detector_temp_c": temperature
                }
        except:
            pass
        return {"intensity_nw": 0.0, "intensity_w": 0.0, "detector_temp_c": 25.0}

class QCLController(BaseSensor):
    """QCL量子级联激光器控制器"""
    
    def __init__(self, config: QCLConfig, use_simulator: bool = True):
        super().__init__(config)
        self._serial: Optional[serial.Serial] = None
        self._current_power = config.current_power_mw
        self._use_simulator = use_simulator or (serial is None)
    
    def connect(self) -> bool:
        """连接QCL控制器"""
        if self._use_simulator:
            print("QCL controller: Using simulator mode (hardware disabled)")
            return True
        try:
            self._serial = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            self.set_power(self.config.current_power_mw)
            return True
        except Exception as e:
            print(f"Failed to connect to QCL controller: {e}")
            print("QCL controller: Falling back to simulator mode")
            self._use_simulator = True
            return True
    
    def disconnect(self):
        """断开连接"""
        super().disconnect()
        self.set_power(0)
        if self._serial:
            try:
                self._serial.close()
            except:
                pass
            self._serial = None
    
    def set_power(self, power_mw: float):
        """设置激光功率"""
        clamped_power = max(self.config.min_power_mw, min(power_mw, self.config.max_power_mw))
        self._current_power = clamped_power
        if self._serial:
            cmd = f"POWER:{clamped_power:.1f}\r\n"
            self._serial.write(cmd.encode())
    
    def get_power(self) -> float:
        """获取当前功率"""
        return self._current_power
    
    def _sampling_loop(self):
        """QCL状态采样循环"""
        sample_interval = 1.0 / self.config.sampling_rate
        while self._running:
            try:
                if self._use_simulator:
                    values = self._generate_simulated_data()
                    self._add_data(values)
                elif self._serial:
                    self._serial.write(b"STATUS\r\n")
                    response = self._serial.readline().decode('utf-8').strip()
                    values = self._parse_qcl_status(response)
                    self._add_data(values, response.encode())
            except Exception as e:
                print(f"QCL controller error: {e}")
            time.sleep(sample_interval)
    
    def _generate_simulated_data(self) -> Dict[str, float]:
        """生成模拟QCL数据"""
        power = self._current_power + np.random.normal(0, 0.5)
        temp = 25.0 + np.random.normal(0, 1.0) + self._current_power * 0.1
        current = self._current_power * 10 + np.random.normal(0, 5)
        return {
            "power_mw": power,
            "temperature_c": temp,
            "current_ma": current
        }
    
    def _parse_qcl_status(self, line: str) -> Dict[str, float]:
        """解析QCL状态"""
        try:
            parts = line.split(',')
            if len(parts) >= 3:
                return {
                    "power_mw": float(parts[0]),
                    "temperature_c": float(parts[1]),
                    "current_ma": float(parts[2])
                }
        except:
            pass
        return {
            "power_mw": self._current_power,
            "temperature_c": 25.0,
            "current_ma": 0.0
        }

class TemperatureSensor(BaseSensor):
    """DS18B20温度传感器"""
    
    def __init__(self, config: TemperatureSensorConfig, use_simulator: bool = True):
        super().__init__(config)
        self._use_simulator = use_simulator
        self._sim_time = 0.0
    
    def connect(self) -> bool:
        """连接温度传感器"""
        if self._use_simulator:
            print("Temperature sensor: Using simulator mode (hardware disabled)")
            return True
        try:
            import os
            os.system(f"modprobe w1-gpio")
            os.system(f"modprobe w1-therm")
            self._use_simulator = False
            return True
        except:
            print("Temperature sensor: Using simulator mode (Linux sysfs not available)")
            self._use_simulator = True
            return True
    
    def _sampling_loop(self):
        """温度采样循环"""
        sample_interval = 1.0 / self.config.sampling_rate
        while self._running:
            try:
                if self._use_simulator:
                    temp_c = self._generate_simulated_data()
                else:
                    temp_c = self._read_temperature()
                self._add_data({"temperature_c": temp_c + self.config.offset})
            except Exception as e:
                print(f"Temperature sensor error: {e}")
            time.sleep(sample_interval)
    
    def _generate_simulated_data(self) -> float:
        """生成模拟温度数据"""
        self._sim_time += 1.0 / self.config.sampling_rate
        return 25.0 + 5 * np.sin(2 * np.pi * 0.01 * self._sim_time) + np.random.normal(0, 0.3)
    
    def _read_temperature(self) -> float:
        """读取温度值"""
        try:
            with open(f"/sys/bus/w1/devices/{self.config.device_id}/w1_slave", 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2 and "YES" in lines[0]:
                    temp_str = lines[1].split('=')[-1]
                    return float(temp_str) / 1000.0
        except:
            pass
        return 25.0

class SensorManager:
    """传感器管理器"""
    
    def __init__(self):
        self._sensors: Dict[str, BaseSensor] = {}
    
    def add_sensor(self, name: str, sensor: BaseSensor):
        """添加传感器"""
        self._sensors[name] = sensor
    
    def remove_sensor(self, name: str):
        """移除传感器"""
        if name in self._sensors:
            self._sensors[name].disconnect()
            del self._sensors[name]
    
    def connect_all(self) -> bool:
        """连接所有传感器"""
        results = [sensor.connect() for sensor in self._sensors.values()]
        return all(results)
    
    def disconnect_all(self):
        """断开所有传感器"""
        for sensor in self._sensors.values():
            sensor.disconnect()
    
    def start_all(self):
        """启动所有传感器采样"""
        for sensor in self._sensors.values():
            sensor.start_sampling()
    
    def stop_all(self):
        """停止所有传感器采样"""
        for sensor in self._sensors.values():
            sensor.stop_sampling()
    
    def get_all_data(self) -> Dict[str, List[SensorData]]:
        """获取所有传感器数据"""
        return {name: sensor.get_data() for name, sensor in self._sensors.items()}
    
    def get_latest_data(self) -> Dict[str, Optional[SensorData]]:
        """获取所有传感器最新数据"""
        return {name: sensor.get_latest_data() for name, sensor in self._sensors.items()}
    
    def is_all_running(self) -> bool:
        """检查所有传感器是否运行"""
        return all(sensor.is_running for sensor in self._sensors.values())
    
    def get_sensor(self, name: str) -> Optional[BaseSensor]:
        """获取传感器"""
        return self._sensors.get(name)