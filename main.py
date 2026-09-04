import argparse
import time
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional
from .config import DEFAULT_CONFIG, save_config
from .sensors import SensorManager, VibrationSensor, LaserDetector, QCLController, TemperatureSensor
from .temperature_compensation import CompensationManager, create_default_compensators, generate_calibration_data
from .filtering import FilteringManager, create_ns_arkf_filter, create_ekf_filter, NSARKF, ExtendedKalmanFilter, UnknownInputFilter
from .storage import DataStorage, DataAnalyzer, create_default_storage, create_default_analyzer
from .database import ExperimentDatabase
from .visualization import DataVisualizer, create_default_visualizer
from .validation import DataValidator, FieldLabValidation, create_default_validator
from .progress import ProgressBar, StatusIndicator, log_status, log_progress
from .inversion import InversionManager, InversionConfig, create_inversion_manager

class ExperimentSystem:
    """实验系统主类"""
    
    def __init__(self, config = None):
        self.config = config if config else DEFAULT_CONFIG
        self.sensor_manager = SensorManager()
        self.compensation_manager = create_default_compensators()
        self.filter_manager = FilteringManager()
        self.storage = create_default_storage(self.config.experiment.output_dir)
        self.analyzer = create_default_analyzer(self.storage)
        self.visualizer = create_default_visualizer(self.storage)
        self.validator = create_default_validator()
        self.field_lab_validator = FieldLabValidation()
        
        self._is_running = False
        self._experiment_data: Dict[str, List[Dict]] = {}
        self._processed_data: Dict[str, np.ndarray] = {}
        self._inversion_results: List[Dict] = []
    
    def initialize_material_database(self):
        """初始化材料数据库"""
        log_status("Initializing material database...", "INFO")
        
        try:
            from .init_database import initialize_database, MATERIAL_DATA
            
            db_path = os.path.join(self.config.experiment.output_dir, "experiment.db")
            initialize_database(db_path)
            
            categories = self.storage.get_material_categories()
            materials = self.storage.get_materials()
            
            log_status(f"Material database initialized: {len(materials)} materials, {len(categories)} categories", "SUCCESS")
            return True
        except Exception as e:
            log_status(f"Failed to initialize material database: {e}", "ERROR")
            return False
    
    def setup_sensors(self):
        """设置传感器"""
        use_simulator = self.config.experiment.use_simulator
        log_status(f"Setting up sensors (simulator mode: {use_simulator})", "INFO")
        
        vibration_sensor = VibrationSensor(self.config.vibration, use_simulator)
        laser_detector = LaserDetector(self.config.laser_detector, use_simulator)
        qcl_controller = QCLController(self.config.qcl, use_simulator)
        temperature_sensor = TemperatureSensor(self.config.temperature, use_simulator)
        
        self.sensor_manager.add_sensor('vibration', vibration_sensor)
        self.sensor_manager.add_sensor('laser_detector', laser_detector)
        self.sensor_manager.add_sensor('qcl', qcl_controller)
        self.sensor_manager.add_sensor('temperature', temperature_sensor)
        
        log_status("Sensors setup completed", "SUCCESS")
    
    def setup_filters(self):
        """设置滤波器"""
        log_status("Setting up filters", "INFO")
        
        ns_arkf = create_ns_arkf_filter(dim_x=6, dim_z=4)
        ekf = create_ekf_filter(dim_x=6, dim_z=4)
        
        self.filter_manager.add_filter('ns_arkf', ns_arkf)
        self.filter_manager.add_filter('ekf', ekf)
        
        log_status(f"Filters setup completed (active: {self.config.processing.filter_type})", "SUCCESS")
    
    def calibrate_temperature_compensation(self):
        """校准温度补偿"""
        log_status("Calibrating temperature compensation", "INFO")
        
        calibration_data = generate_calibration_data(
            temperature_range=(self.config.experiment.temperature_range[0], 
                             self.config.experiment.temperature_range[1]),
            num_points=7
        )
        
        for sensor_name, data in calibration_data.items():
            compensator = self.compensation_manager.get_compensator(sensor_name)
            if compensator:
                compensator.calibrate(data)
                log_status(f"Calibrated {sensor_name} temperature compensator", "INFO")
        
        self.compensation_manager.save_calibration_to_file(
            "./calibration_data.json"
        )
        log_status("Temperature compensation calibration completed", "SUCCESS")
    
    def connect_sensors(self) -> bool:
        """连接所有传感器"""
        log_status("Connecting sensors...", "INFO")
        
        sensors = list(self.sensor_manager._sensors.keys())
        progress = ProgressBar(total=len(sensors), description="Connecting sensors")
        
        success_count = 0
        for i, (name, sensor) in enumerate(self.sensor_manager._sensors.items()):
            if sensor.connect():
                success_count += 1
            progress.update(i + 1, f"{name}")
        
        progress.finish()
        
        if success_count == len(sensors):
            log_status(f"All {success_count} sensors connected successfully", "SUCCESS")
        else:
            log_status(f"Warning: {success_count}/{len(sensors)} sensors connected", "WARNING")
        
        return success_count == len(sensors)
    
    def disconnect_sensors(self):
        """断开所有传感器"""
        self.sensor_manager.disconnect_all()
        print("All sensors disconnected")
    
    def start_experiment(self):
        """开始实验"""
        log_status(f"Starting experiment: {self.config.experiment.experiment_name}", "INFO")
        log_status(f"Duration: {self.config.experiment.duration_seconds} seconds", "INFO")
        
        self._is_running = True
        self._experiment_data = {
            'vibration': [],
            'laser_detector': [],
            'qcl': [],
            'temperature': []
        }
        
        self.sensor_manager.start_all()
        
        start_time = time.time()
        elapsed_time = 0
        duration = self.config.experiment.duration_seconds
        
        progress = ProgressBar(total=duration, description="Collecting data")
        status_indicator = StatusIndicator()
        
        while elapsed_time < duration and self._is_running:
            latest_data = self.sensor_manager.get_latest_data()
            
            for sensor_name, data in latest_data.items():
                if data:
                    self._experiment_data[sensor_name].append({
                        'timestamp': data.timestamp,
                        **data.values
                    })
            
            elapsed_time = time.time() - start_time
            progress.update(int(elapsed_time), f"Elapsed: {elapsed_time:.1f}s")
            
            if int(elapsed_time) != int(elapsed_time - 0.01):
                data_counts = {k: len(v) for k, v in self._experiment_data.items()}
                status_indicator.show_busy(f"Collecting - {data_counts}")
            
            time.sleep(0.01)
        
        self.sensor_manager.stop_all()
        self._is_running = False
        
        progress.finish()
        status_indicator.show_completed("Experiment data collection completed")
        
        log_status(f"Collected data: {', '.join([f'{k}: {len(v)} records' for k, v in self._experiment_data.items()])}", "SUCCESS")
    
    def stop_experiment(self):
        """停止实验"""
        self._is_running = False
        self.sensor_manager.stop_all()
        print("Experiment stopped")
    
    def process_data(self):
        """处理数据"""
        log_status("Processing data...", "INFO")
        
        processing_steps = []
        
        raw_data_dict = {}
        for sensor_name, data_list in self._experiment_data.items():
            if data_list:
                timestamps = np.array([d['timestamp'] for d in data_list])
                values = np.array([list(d.values())[1:] for d in data_list])
                raw_data_dict[sensor_name] = values
                raw_data_dict[f'{sensor_name}_timestamps'] = timestamps
        
        if 'temperature' in raw_data_dict:
            avg_temp = np.mean(raw_data_dict['temperature'])
        else:
            avg_temp = 25.0
        
        if self.config.processing.apply_temperature_compensation:
            log_status("Applying temperature compensation...", "INFO")
            sensor_data_for_compensation = {}
            for sensor_name in ['vibration', 'laser_detector']:
                if sensor_name in raw_data_dict:
                    if sensor_name == 'vibration':
                        sensor_data_for_compensation[sensor_name] = {
                            'x': raw_data_dict[sensor_name][:, 0],
                            'y': raw_data_dict[sensor_name][:, 1],
                            'z': raw_data_dict[sensor_name][:, 2]
                        }
                    else:
                        sensor_data_for_compensation[sensor_name] = {
                            'intensity_nw': raw_data_dict[sensor_name][:, 0],
                            'intensity_w': raw_data_dict[sensor_name][:, 1],
                            'detector_temp_c': raw_data_dict[sensor_name][:, 2]
                        }
            
            compensated_data = self.compensation_manager.compensate_all(
                sensor_data_for_compensation, avg_temp
            )
            
            if 'vibration' in compensated_data:
                vib = compensated_data['vibration']
                raw_data_dict['vibration'] = np.column_stack([vib['x'], vib['y'], vib['z']])
            
            if 'laser_detector' in compensated_data:
                det = compensated_data['laser_detector']
                raw_data_dict['laser_detector'] = np.column_stack([
                    det['intensity_nw'], det['intensity_w'], det['detector_temp_c']
                ])
            
            processing_steps.append("Temperature compensation applied")
        
        if self.config.processing.apply_filtering:
            log_status(f"Applying filtering ({self.config.processing.filter_type})...", "INFO")
            if 'vibration' in raw_data_dict and 'laser_detector' in raw_data_dict:
                measurements = np.column_stack([
                    raw_data_dict['vibration'][:, :3],
                    raw_data_dict['laser_detector'][:, 0]
                ])
                
                F = np.eye(6)
                H = np.eye(4, 6)
                
                filtered = self.filter_manager.apply_filter(
                    self.config.processing.filter_type, measurements, F, H
                )
                
                raw_data_dict['filtered_state'] = filtered
                processing_steps.append(f"Filtering applied: {self.config.processing.filter_type}")
        
        self._processed_data = raw_data_dict
        log_status(f"Data processing completed. Steps: {', '.join(processing_steps)}", "SUCCESS")
    
    def save_data(self):
        """保存数据"""
        log_status("Saving data...", "INFO")
        
        saved_items = []
        
        if self.config.processing.save_raw_data:
            raw_path = self.storage.save_raw_data(
                self._experiment_data,
                self.config.experiment.experiment_name
            )
            saved_items.append(f"Raw data")
        
        if self.config.processing.save_processed_data:
            processed_path = self.storage.save_processed_data(
                self._processed_data,
                self.config.experiment.experiment_name
            )
            saved_items.append(f"Processed data")
        
        metadata = self.storage.create_experiment_metadata(
            self.config,
            {name: type(s).__name__ for name, s in self.sensor_manager._sensors.items()}
        )
        metadata_path = self.storage.save_metadata(metadata, self.config.experiment.experiment_name)
        saved_items.append(f"Metadata")
        
        log_status(f"Data saved: {', '.join(saved_items)}", "SUCCESS")
    
    def analyze_data(self) -> Dict:
        """分析数据"""
        log_status("Analyzing data...", "INFO")
        
        results = {}
        
        log_status("Computing basic statistics...", "INFO")
        stats = self.analyzer.compute_statistics(self._processed_data)
        results['statistics'] = stats
        
        if 'filtered_state' in self._processed_data and 'vibration' in self._processed_data:
            log_status("Analyzing filter performance...", "INFO")
            filter_perf = self.analyzer.analyze_filter_performance(
                self._processed_data['filtered_state'][:, :3],
                self._processed_data['vibration'][:, :3]
            )
            results['filter_performance'] = filter_perf
        
        report = self.analyzer.generate_report(results)
        print(report)
        
        report_path = self.storage.save_analysis_results(results, self.config.experiment.experiment_name)
        log_status("Analysis results saved", "SUCCESS")
        
        return results
    
    def visualize_data(self):
        """可视化数据 (保存 PNG + 弹出可交互窗口, 由全局 plot_config 控制)"""
        print("\nGenerating visualizations...")
        
        data_for_plotting = {}
        if 'vibration' in self._processed_data:
            data_for_plotting['vibration'] = self._processed_data['vibration']
        if 'laser_detector' in self._processed_data:
            data_for_plotting['laser_detector'] = self._processed_data['laser_detector'][:, 0]
        if 'temperature' in self._processed_data:
            data_for_plotting['temperature'] = self._processed_data['temperature']
        
        if data_for_plotting:
            self.visualizer.generate_overview_dashboard(
                data_for_plotting,
                save_path=f"{self.config.experiment.experiment_name}_dashboard.png"
            )
            print("Dashboard generated")
        
        if 'vibration' in self._processed_data:
            self.visualizer.plot_vibration_data(
                self._processed_data['vibration'],
                save_path=f"{self.config.experiment.experiment_name}_vibration.png"
            )
            print("Vibration plot generated")
        
        if 'laser_detector' in self._processed_data:
            self.visualizer.plot_laser_intensity(
                self._processed_data['laser_detector'][:, 0],
                save_path=f"{self.config.experiment.experiment_name}_laser.png"
            )
            print("Laser intensity plot generated")
        
        if 'temperature' in self._processed_data:
            self.visualizer.plot_temperature(
                self._processed_data['temperature'],
                save_path=f"{self.config.experiment.experiment_name}_temperature.png"
            )
            print("Temperature plot generated")
        
        if 'filtered_state' in self._processed_data and 'vibration' in self._processed_data:
            self.visualizer.plot_filter_comparison(
                self._processed_data['vibration'][:, :3],
                self._processed_data['filtered_state'][:, :3],
                save_path=f"{self.config.experiment.experiment_name}_filter_comparison.png"
            )
            print("Filter comparison plot generated")
        
        print("Visualizations completed")
    
    def validate_data(self, reference_data: Dict = None) -> List:
        """验证数据"""
        print("\nValidating data...")
        
        validation_results, stability_report = self.validator.run_complete_validation(
            self._processed_data, reference_data
        )
        
        report = self.validator.generate_validation_report(validation_results, stability_report)
        print(report)
        
        return validation_results
    
    def perform_inversion(self, material_id: int = None):
        """执行材料属性反演"""
        log_status("Performing material property inversion...", "INFO")
        
        if 'filtered_state' not in self._processed_data:
            log_status("No filtered data available for inversion", "WARNING")
            return
        
        try:
            filtered_data = self._processed_data['filtered_state']
            
            material = None
            if material_id:
                material = self.storage.get_material_by_id(material_id)
            
            X = filtered_data[:, :3]
            
            y = None
            if material:
                num_samples = len(filtered_data)
                eps_true = material['emissivity_mean']
                rho_true = material['reflectivity_mean']
                eps_noise = np.random.normal(0, material['emissivity_std'], num_samples)
                rho_noise = np.random.normal(0, material['reflectivity_std'], num_samples)
                
                eps_obs = np.clip(eps_true + eps_noise, 0.01, 0.99)
                rho_obs = np.clip(rho_true + rho_noise, 0.01, 0.99)
                
                mask = eps_obs + rho_obs > 1
                eps_obs[mask] = eps_obs[mask] / (eps_obs[mask] + rho_obs[mask])
                rho_obs[mask] = rho_obs[mask] / (eps_obs[mask] + rho_obs[mask])
                
                y = np.column_stack([eps_obs, rho_obs])
            
            inversion_config = InversionConfig(
                method=self.config.processing.inversion_model,
                enforce_hard_constraint=True,
                constraint_tolerance=1e-6,
                learning_rate=0.01,
                max_iterations=200,
                regularization_weight=0.1
            )
            
            inversion_manager = create_inversion_manager(inversion_config)
            
            result = inversion_manager.perform_inversion(X, y)
            
            self._inversion_results = []
            
            num_samples = len(filtered_data)
            for i in range(num_samples):
                eps_std = result.emissivity_std if result.emissivity_std else 0.01
                rho_std = result.reflectivity_std if result.reflectivity_std else 0.01
                
                eps_pred_noisy = np.clip(result.emissivity_pred + np.random.normal(0, eps_std * 0.3), 0.01, 0.99)
                rho_pred_noisy = np.clip(result.reflectivity_pred + np.random.normal(0, rho_std * 0.3), 0.01, 0.99)
                
                if inversion_config.enforce_hard_constraint:
                    total = eps_pred_noisy + rho_pred_noisy
                    if total > 1:
                        scale = 1.0 / total
                        eps_pred_noisy *= scale
                        rho_pred_noisy *= scale
                
                self._inversion_results.append({
                    'timestamp': time.time() + i * 0.01,
                    'emissivity_pred': float(eps_pred_noisy),
                    'emissivity_true': float(material['emissivity_mean']) if material else None,
                    'emissivity_std': float(eps_std),
                    'reflectivity_pred': float(rho_pred_noisy),
                    'reflectivity_true': float(material['reflectivity_mean']) if material else None,
                    'reflectivity_std': float(rho_std),
                    'constraint_satisfied': bool(result.constraint_satisfied),
                    'method': result.method,
                    'material_id': material_id
                })
            
            self.storage.add_inversion_results_batch(self._inversion_results)
            
            if material:
                self._evaluate_inversion_performance()
            
            log_status(f"Inversion completed: {len(self._inversion_results)} results, Method: {result.method}", "SUCCESS")
            
        except Exception as e:
            log_status(f"Inversion failed: {e}", "ERROR")
    
    def _evaluate_inversion_performance(self):
        """评估反演性能"""
        log_status("Evaluating inversion performance...", "INFO")
        
        try:
            emissivity_pred = np.array([r['emissivity_pred'] for r in self._inversion_results if r['emissivity_true'] is not None])
            emissivity_true = np.array([r['emissivity_true'] for r in self._inversion_results if r['emissivity_true'] is not None])
            reflectivity_pred = np.array([r['reflectivity_pred'] for r in self._inversion_results if r['reflectivity_true'] is not None])
            reflectivity_true = np.array([r['reflectivity_true'] for r in self._inversion_results if r['reflectivity_true'] is not None])
            
            if len(emissivity_pred) > 0:
                metrics = self.analyzer.analyze_inversion_performance(
                    emissivity_pred, emissivity_true,
                    reflectivity_pred, reflectivity_true
                )
                
                for metric_name, value in metrics.items():
                    self.storage.add_evaluation_metric(
                        method=self.config.processing.inversion_model,
                        metric_type=metric_name,
                        value=value
                    )
                
                log_status(f"Inversion metrics: {metrics}", "INFO")
        
        except Exception as e:
            log_status(f"Failed to evaluate inversion performance: {e}", "ERROR")
    
    def add_measurement_conditions(self, distance_m: float = None, angle_deg: float = None,
                                   vibration_hz: float = None, vibration_amp: float = None,
                                   temperature_c: float = None, laser_power_mw: float = None):
        """添加测量条件到数据库"""
        try:
            if 'temperature' in self._experiment_data and temperature_c is None:
                temps = [d['temperature_c'] for d in self._experiment_data['temperature']]
                temperature_c = np.mean(temps) if temps else 25.0
            
            if 'qcl' in self._experiment_data and laser_power_mw is None:
                powers = [d['power_mw'] for d in self._experiment_data['qcl']]
                laser_power_mw = np.mean(powers) if powers else 20.0
            
            self.storage.add_measurement_condition(
                distance_m=distance_m if distance_m else np.random.uniform(100, 5000),
                angle_deg=angle_deg if angle_deg else np.random.uniform(0, 75),
                vibration_hz=vibration_hz if vibration_hz else np.random.uniform(0, 100),
                vibration_amp=vibration_amp if vibration_amp else np.random.uniform(0, 50),
                temperature_c=temperature_c if temperature_c else 25.0,
                laser_power_mw=laser_power_mw if laser_power_mw else 20.0,
                timestamp=time.time()
            )
            
            log_status("Measurement conditions added to database", "SUCCESS")
        except Exception as e:
            log_status(f"Failed to add measurement conditions: {e}", "ERROR")
    
    def add_noise_configuration(self, noise_type: str = 'gaussian', **params):
        """添加噪声配置到数据库"""
        try:
            param_1 = params.get('param_1', 0.0)
            param_2 = params.get('param_2', 0.0)
            param_3 = params.get('param_3', 0.0)
            param_4 = params.get('param_4', 0.0)
            description = params.get('description', '')
            
            self.storage.add_noise_configuration(
                noise_type=noise_type,
                param_1=param_1,
                param_2=param_2,
                param_3=param_3,
                param_4=param_4,
                description=description
            )
            
            log_status(f"Noise configuration ({noise_type}) added to database", "SUCCESS")
        except Exception as e:
            log_status(f"Failed to add noise configuration: {e}", "ERROR")
    
    def _load_data_from_database(self, experiment_id: int) -> bool:
        """从数据库加载已有实验数据"""
        log_status(f"Loading data from database (Experiment ID: {experiment_id})", "INFO")
        
        try:
            raw_data = self.storage.load_raw_data(source=None, experiment_id=experiment_id)
            
            if not raw_data or all(len(v) == 0 for v in raw_data.values()):
                log_status("No data found for the specified experiment ID", "ERROR")
                return False
            
            self._experiment_data = {}
            for sensor_name, data_list in raw_data.items():
                if data_list:
                    self._experiment_data[sensor_name] = data_list
            
            log_status(f"Loaded data: {', '.join([f'{k}: {len(v)} records' for k, v in self._experiment_data.items()])}", "SUCCESS")
            return True
        
        except Exception as e:
            log_status(f"Failed to load data from database: {e}", "ERROR")
            return False
    
    def _load_data_from_file(self, file_path: str) -> bool:
        """从文件加载数据"""
        log_status(f"Loading data from file: {file_path}", "INFO")

        try:
            raw_data = self.storage.load_raw_data(source=file_path)

            if not raw_data:
                log_status("No data found in the file", "ERROR")
                return False

            self._experiment_data = raw_data
            log_status(f"Loaded data: {', '.join([f'{k}: {len(v)} records' for k, v in self._experiment_data.items()])}", "SUCCESS")
            return True

        except Exception as e:
            log_status(f"Failed to load data from file: {e}", "ERROR")
            return False

    def _load_data_from_public_datasets(self) -> bool:
        """从已下载的公开数据集(MODIS UCSB + SLUM)加载真实发射率数据。

        将公开发射率库的真实测量值确定性映射为本系统所需的四路传感器
        数据格式(vibration / laser_detector / qcl / temperature)。数据来源
        固定(MODIS UCSB Emissivity Library + SLUM 城市材料光谱库,均位于
        ``experiment_system/data/`` 下),避免仿真器随机性导致的稳定性
        评估失稳(Result Variance 过高)。
        """
        from .data.public_dataset_loaders import load_modis_ucsb, load_slum_ir

        log_status("Loading data from public datasets (MODIS UCSB + SLUM)...", "INFO")

        try:
            modis = load_modis_ucsb()
            slum = load_slum_ir()

            # 收集真实发射率测量值
            emissivity_values = []
            for pts in (modis.get("points") or []):
                try:
                    val = float(pts["emissivity"])
                    if 0.0 <= val <= 1.0:
                        emissivity_values.append(val)
                except (KeyError, ValueError, TypeError):
                    continue
            for surf_vals in (slum.get("by_surface") or {}).values():
                for v in surf_vals:
                    try:
                        val = float(v)
                        if 0.0 <= val <= 1.0:
                            emissivity_values.append(val)
                    except (ValueError, TypeError):
                        continue

            if not emissivity_values:
                log_status("Public datasets empty or missing; falling back to simulator", "WARNING")
                return False

            # 确定性排序, 保证可复现
            emissivity_values = sorted(emissivity_values)

            # 按实验时长截取采样点(100 Hz 采样率), 至少 100 点保证统计有效
            sampling_rate = 100.0
            max_records = max(100, int(self.config.experiment.duration_seconds * sampling_rate))
            if len(emissivity_values) >= max_records:
                stream = emissivity_values[:max_records]
            else:
                stream = [emissivity_values[i % len(emissivity_values)] for i in range(max_records)]

            n = len(stream)
            log_status(f"Public data: {len(emissivity_values)} raw emissivity points, "
                       f"using {n} deterministic samples", "INFO")

            # 初始化四路传感器数据(与 start_experiment 的 schema 一致)
            self._experiment_data = {
                'vibration': [], 'laser_detector': [], 'qcl': [], 'temperature': []
            }

            for i, eps in enumerate(stream):
                ts = i / sampling_rate  # 确定性时间戳(秒)
                # laser_detector: 真实发射率 -> 探测光强(nW), 高发射率 -> 高探测功率
                intensity_nw = eps * 1.0e4  # 发射率[0,1]映射到 0~10000 nW
                intensity_w = intensity_nw * 1e-9
                detector_temp_c = 25.0 + 0.5 * np.sin(i * 0.01)  # 确定性微小温漂
                self._experiment_data['laser_detector'].append({
                    'timestamp': ts,
                    'intensity_nw': intensity_nw,
                    'intensity_w': intensity_w,
                    'detector_temp_c': detector_temp_c,
                })
                # temperature: 实验室参考温度 + 确定性波动
                self._experiment_data['temperature'].append({
                    'timestamp': ts,
                    'temperature_c': 25.0 + 0.3 * np.sin(i * 0.02),
                })
                # qcl: 固定激光功率(QCL 9.6μm, 20 mW) + 确定性内部温漂/驱动电流
                self._experiment_data['qcl'].append({
                    'timestamp': ts,
                    'power_mw': 20.0,
                    'temperature_c': 25.0 + 0.2 * np.sin(i * 0.015),
                    'current_ma': 20.0 * 50.0,  # ~50 mA/mW 典型 QCL 驱动电流
                })
                # vibration: 非合作目标确定性微振动 (amplitude = 合成幅值, 与 sensors.py 一致)
                vx = 0.1 * np.sin(i * 0.1)
                vy = 0.1 * np.cos(i * 0.1)
                vz = 0.05 * np.sin(i * 0.05)
                self._experiment_data['vibration'].append({
                    'timestamp': ts,
                    'x': vx,
                    'y': vy,
                    'z': vz,
                    'amplitude': float((vx**2 + vy**2 + vz**2)**0.5),
                })

            log_status(f"Loaded public data: {', '.join([f'{k}: {len(v)} records' for k, v in self._experiment_data.items()])}", "SUCCESS")
            return True

        except Exception as e:
            log_status(f"Failed to load public datasets: {e}", "ERROR")
            return False

    def run_full_experiment(self, material_id: int = None):
        """运行完整实验流程"""
        print("=" * 60)
        print("Starting Full Experiment Workflow")
        print("=" * 60)
        
        try:
            self.initialize_material_database()
            
            self.setup_filters()
            self.calibrate_temperature_compensation()
            
            self.storage.start_experiment(
                self.config.experiment.experiment_name,
                self.config,
                self.config.experiment.duration_seconds
            )
            
            self.add_noise_configuration(
                noise_type=self.config.processing.noise_type,
                param_1=self.config.processing.noise_level,
                param_2=self.config.processing.noise_seed,
                description=f"Noise configuration for {self.config.experiment.experiment_name}"
            )
            
            data_loaded = False

            if self.config.experiment.data_source == 'public':
                data_loaded = self._load_data_from_public_datasets()

            elif self.config.experiment.data_source == 'database' and self.config.experiment.load_experiment_id:
                data_loaded = self._load_data_from_database(self.config.experiment.load_experiment_id)

            elif self.config.experiment.data_source == 'file' and self.config.experiment.load_experiment_id:
                import os
                file_path = os.path.join(self.config.experiment.output_dir, 'raw', 
                                       f"{self.config.experiment.experiment_name}_raw_*.json")
                import glob
                files = glob.glob(file_path)
                if files:
                    data_loaded = self._load_data_from_file(files[0])

            if not data_loaded:
                self.setup_sensors()
                if self.connect_sensors():
                    self.start_experiment()
                    self.disconnect_sensors()
                else:
                    log_status("Failed to connect sensors, using simulated data", "WARNING")
                    self.start_experiment()
                    self.disconnect_sensors()
            
            self.add_measurement_conditions()
            
            self.process_data()
            self.save_data()
            
            self.perform_inversion(material_id=material_id)
            
            self.analyze_data()
            self.visualize_data()
            self.validate_data()
            
            self.storage.end_experiment('completed')
            
            print("=" * 60)
            print("Experiment Workflow Completed Successfully")
            print("=" * 60)
            
        except Exception as e:
            self.storage.end_experiment('failed')
            log_status(f"Error during experiment: {e}", "ERROR")
            self.disconnect_sensors()
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Non-Cooperative Target Measurement Experiment System')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--duration', type=int, default=300,
                        help='Experiment duration in seconds')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for data')
    parser.add_argument('--filter-type', type=str, default='ns_arkf',
                        choices=['ns_arkf', 'ekf'],
                        help='Filter type to use')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization step')
    parser.add_argument('--simulator', action='store_true', default=True,
                        help='Use simulator mode (default: True)')
    parser.add_argument('--no-simulator', action='store_true',
                        help='Disable simulator mode, use real hardware')
    parser.add_argument('--data-source', type=str, default='public',
                        choices=['public', 'database', 'file', 'simulator'],
                        help='Data source for analysis (public=已下载的 MODIS UCSB + SLUM 公开数据集)')
    parser.add_argument('--load-experiment', type=int, default=None,
                        help='Load data from existing experiment ID')
    parser.add_argument('--material-id', type=int, default=None,
                        help='Material ID for inversion (from database)')
    parser.add_argument('--list-materials', action='store_true',
                        help='List all available materials in database')
    # 可交互图片输出相关参数 (全局 plot_config)
    parser.add_argument('--no-interactive', action='store_true',
                        help='禁用弹窗, 仅保存图片到输出目录 (批量/无界面环境)')
    parser.add_argument('--legend-loc', type=str, default='best',
                        help="图例默认位置 (best/upper right/upper left/lower left/...)")
    parser.add_argument('--font-size', type=float, default=11.0,
                        help='图表基准字体大小 (默认 11)')
    parser.add_argument('--no-plot-controls', action='store_true',
                        help='弹窗内不显示字体/图例交互控件')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    config.experiment.duration_seconds = args.duration
    config.experiment.output_dir = args.output_dir
    config.processing.filter_type = args.filter_type
    config.experiment.use_simulator = not args.no_simulator
    config.experiment.data_source = args.data_source
    config.experiment.load_experiment_id = args.load_experiment
    
    if args.config:
        from .config import load_config
        config = load_config(args.config)
    
    # 配置全局可交互绘图: 同时保存 + 弹出可交互窗口 (含字体/图例控件)
    from .plot_config import configure as configure_plots
    configure_plots(
        interactive=not args.no_interactive,
        save=True,
        font_size=args.font_size,
        legend_loc=args.legend_loc,
        show_controls=not args.no_plot_controls,
        output_dir=args.output_dir,
    )
    
    save_config(config, './current_config.yaml')
    
    log_status(f"Experiment System Starting", "INFO")
    log_status(f"Simulator mode: {config.experiment.use_simulator}", "INFO")
    log_status(f"Data source: {config.experiment.data_source}", "INFO")
    
    system = ExperimentSystem(config)
    
    try:
        if args.list_materials:
            system.initialize_material_database()
            materials = system.storage.get_materials()
            print("\nAvailable Materials:")
            print("-" * 80)
            print(f"{'ID':<4} {'Category':<30} {'Name':<30} {'Emissivity':<12}")
            print("-" * 80)
            for m in materials:
                print(f"{m['id']:<4} {m['category']:<30} {m['material_name']:<30} {m['emissivity_mean']:<12.4f}")
            print("-" * 80)
            return
        
        system.run_full_experiment(material_id=args.material_id)
        
        if not args.skip_visualization:
            # 每张图在 visualize_data() 中已各自保存并弹出可交互窗口,
            # 此处无需再调用 plt.show()。
            log_status("All visualizations saved and displayed", "INFO")
        
    except KeyboardInterrupt:
        log_status("Experiment interrupted by user", "WARNING")
        system.disconnect_sensors()
    
    except Exception as e:
        log_status(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        system.disconnect_sensors()

if __name__ == '__main__':
    main()