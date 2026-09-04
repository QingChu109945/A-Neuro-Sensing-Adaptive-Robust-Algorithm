import os
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class SensorConfig:
    """传感器配置基类"""
    name: str
    enabled: bool = True
    sampling_rate: float = 100.0
    calibration_file: str = None

@dataclass
class VibrationSensorConfig(SensorConfig):
    """维特智能WTVB01-BT50振动传感器配置"""
    bluetooth_mac: str = "00:00:00:00:00:00"
    range_g: float = 8.0
    sensitivity: float = 0.01
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0

@dataclass
class LaserDetectorConfig(SensorConfig):
    """labM-10.6激光探测器配置 (论文Section 5.1.2: Nd:YAG 1.064μm)"""
    serial_port: str = "COM1"
    baud_rate: int = 115200
    wavelength: float = 1.064  # 论文要求: Nd:YAG 1.064μm
    sensitivity_nw: float = 1.0
    gain: int = 10

@dataclass
class QCLConfig(SensorConfig):
    """QCL量子级联激光器配置"""
    serial_port: str = "COM2"
    baud_rate: int = 9600
    wavelength: float = 9.6
    min_power_mw: float = 1.0
    max_power_mw: float = 50.0
    current_power_mw: float = 20.0

@dataclass
class TemperatureSensorConfig(SensorConfig):
    """DS18B20温度传感器配置"""
    device_id: str = "28-000000000000"
    precision: float = 0.5
    offset: float = 0.0

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "default_experiment"
    output_dir: str = "./data"
    duration_seconds: int = 300
    use_simulator: bool = True
    data_source: str = "public"
    load_experiment_id: int = None
    temperature_range: List[float] = field(default_factory=lambda: [20.0, 80.0])
    temperature_step: float = 10.0
    distance_range: List[float] = field(default_factory=lambda: [1.0, 10.0])
    distance_step: float = 1.0
    angle_range: List[float] = field(default_factory=lambda: [0.0, 75.0])
    angle_step: float = 15.0
    vibration_range: List[float] = field(default_factory=lambda: [0.0, 50.0])
    vibration_step: float = 10.0
    laser_power_range: List[float] = field(default_factory=lambda: [10.0, 50.0])
    laser_power_step: float = 10.0

@dataclass
class ProcessingConfig:
    """数据处理配置"""
    apply_temperature_compensation: bool = True
    apply_filtering: bool = True
    filter_type: str = "ns_arkf"
    apply_inversion: bool = True
    inversion_model: str = "ssm_pinn"
    save_raw_data: bool = True
    save_processed_data: bool = True
    noise_type: str = "gaussian"
    noise_level: float = 0.01
    noise_seed: int = 42

@dataclass
class ValidationConfig:
    """数据验证配置"""
    enable_validation: bool = True
    consistency_threshold: float = 0.005
    reliability_threshold: float = 0.02
    outlier_threshold: float = 3.0

@dataclass
class SystemConfig:
    """系统配置"""
    vibration: VibrationSensorConfig = field(default_factory=lambda: VibrationSensorConfig(name="WTVB01-BT50"))
    laser_detector: LaserDetectorConfig = field(default_factory=lambda: LaserDetectorConfig(name="labM-10.6"))
    qcl: QCLConfig = field(default_factory=lambda: QCLConfig(name="QCL-9.6"))
    temperature: TemperatureSensorConfig = field(default_factory=lambda: TemperatureSensorConfig(name="DS18B20"))
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

def load_config(config_path: str = None) -> SystemConfig:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return SystemConfig(**config_dict)
    return SystemConfig()

def save_config(config: SystemConfig, config_path: str):
    """保存配置文件"""
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)

DEFAULT_CONFIG = SystemConfig(
    vibration=VibrationSensorConfig(
        name="WTVB01-BT50",
        bluetooth_mac="00:1A:7D:DA:71:13",
        range_g=8.0,
        sensitivity=0.01,
        sampling_rate=100.0
    ),
    laser_detector=LaserDetectorConfig(
        name="labM-10.6",
        serial_port="COM3",
        baud_rate=115200,
        wavelength=10.6,
        gain=10,
        sampling_rate=100.0
    ),
    qcl=QCLConfig(
        name="QCL-9.6",
        serial_port="COM4",
        baud_rate=9600,
        wavelength=9.6,
        current_power_mw=20.0,
        sampling_rate=100.0
    ),
    temperature=TemperatureSensorConfig(
        name="DS18B20",
        device_id="28-3C01D6075628",
        precision=0.5,
        sampling_rate=100.0
    ),
    experiment=ExperimentConfig(
            experiment_name="non_cooperative_target_measurement",
            output_dir="./data",
            duration_seconds=300,
            use_simulator=True,
            data_source="public",
            load_experiment_id=None
        ),
    processing=ProcessingConfig(
        apply_temperature_compensation=True,
        apply_filtering=True,
        apply_inversion=True
    ),
    validation=ValidationConfig(
        enable_validation=True,
        consistency_threshold=0.005
    )
)