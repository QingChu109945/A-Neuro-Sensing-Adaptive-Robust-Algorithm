import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict

class DataStorage:
    """数据存储管理器 - 支持文件和数据库双模式"""
    
    def __init__(self, base_dir: str = "./data", use_database: bool = True):
        self.base_dir = base_dir
        self.use_database = use_database
        
        if use_database:
            from .database import ExperimentDatabase
            abs_base = os.path.abspath(base_dir)
            if os.path.basename(abs_base) != 'data':
                abs_base = os.path.dirname(abs_base)
            db_path = os.path.join(abs_base, "experiment.db")
            self.db = ExperimentDatabase(db_path)
            self._current_experiment_id = None
        
        self._ensure_directory(base_dir)
        self._ensure_directory(os.path.join(base_dir, "raw"))
        self._ensure_directory(os.path.join(base_dir, "processed"))
        self._ensure_directory(os.path.join(base_dir, "analysis"))
        self._ensure_directory(os.path.join(base_dir, "figures"))
    
    def _ensure_directory(self, path: str):
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def start_experiment(self, experiment_name: str, config = None, 
                         duration: float = None) -> int:
        """开始实验，返回实验ID"""
        if self.use_database:
            self._current_experiment_id = self.db.add_experiment(
                name=experiment_name,
                config=config,
                duration=duration,
                output_dir=self.base_dir
            )
            
            if config:
                sensor_configs = {
                    'vibration': config.vibration,
                    'laser_detector': config.laser_detector,
                    'qcl': config.qcl,
                    'temperature': config.temperature
                }
                self.db.add_sensors(self._current_experiment_id, sensor_configs)
        
        return self._current_experiment_id
    
    def end_experiment(self, status: str = 'completed'):
        """结束实验"""
        if self.use_database and self._current_experiment_id:
            self.db.update_experiment_status(self._current_experiment_id, status)
    
    def save_raw_data(self, data: Dict[str, List[Dict]], experiment_name: str):
        """保存原始数据（文件+数据库）"""
        filepath = None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_raw_{timestamp}.json"
        filepath = os.path.join(self.base_dir, "raw", filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        if self.use_database and self._current_experiment_id:
            if 'vibration' in data:
                self.db.add_vibration_data(self._current_experiment_id, data['vibration'])
            if 'laser_detector' in data:
                self.db.add_laser_data(self._current_experiment_id, data['laser_detector'])
            if 'qcl' in data:
                self.db.add_qcl_data(self._current_experiment_id, data['qcl'])
            if 'temperature' in data:
                self.db.add_temperature_data(self._current_experiment_id, data['temperature'])
        
        return filepath
    
    def save_processed_data(self, data: Dict[str, np.ndarray], experiment_name: str):
        """保存处理后数据（文件+数据库）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_processed_{timestamp}.npz"
        filepath = os.path.join(self.base_dir, "processed", filename)
        
        np.savez(filepath, **data)
        
        if self.use_database and self._current_experiment_id:
            for data_type, data_array in data.items():
                self.db.add_processed_data(self._current_experiment_id, data_type, data_array)
        
        return filepath
    
    def save_analysis_results(self, results: Dict, experiment_name: str):
        """保存分析结果（文件+数据库）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_analysis_{timestamp}.json"
        filepath = os.path.join(self.base_dir, "analysis", filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if self.use_database and self._current_experiment_id:
            self.db.add_analysis_results(self._current_experiment_id, results)
        
        return filepath
    
    def save_figure(self, figure, filename: str):
        """保存图表"""
        filepath = os.path.join(self.base_dir, "figures", filename)
        figure.savefig(filepath, dpi=300, bbox_inches='tight')
        return filepath
    
    def load_raw_data(self, source: str, experiment_id: int = None) -> Dict:
        """加载原始数据（支持文件路径或数据库查询）"""
        if self.use_database and experiment_id is not None:
            return {
                'vibration': self.db.get_vibration_data(experiment_id),
                'laser_detector': self.db.get_laser_data(experiment_id),
                'qcl': self.db.get_qcl_data(experiment_id),
                'temperature': self.db.get_temperature_data(experiment_id)
            }
        
        with open(source, 'r') as f:
            return json.load(f)
    
    def load_processed_data(self, source: str, experiment_id: int = None) -> Dict[str, np.ndarray]:
        """加载处理后数据（支持文件路径或数据库查询）"""
        if self.use_database and experiment_id is not None:
            data_types = ['vibration', 'vibration_timestamps', 
                         'laser_detector', 'laser_detector_timestamps',
                         'qcl', 'qcl_timestamps',
                         'temperature', 'temperature_timestamps',
                         'filtered_state']
            
            result = {}
            for data_type in data_types:
                data = self.db.get_processed_data(experiment_id, data_type)
                if data is not None:
                    result[data_type] = data
            return result
        
        return dict(np.load(source))
    
    def load_analysis_results(self, source: str, experiment_id: int = None) -> Dict:
        """加载分析结果（支持文件路径或数据库查询）"""
        if self.use_database and experiment_id is not None:
            return self.db.get_analysis_results(experiment_id)
        
        with open(source, 'r') as f:
            return json.load(f)
    
    def create_experiment_metadata(self, config, sensor_info: Dict) -> Dict:
        """创建实验元数据"""
        metadata = {
            "experiment_name": config.experiment.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(config),
            "sensor_info": sensor_info,
            "data_format": {
                "vibration": ["timestamp", "x", "y", "z", "amplitude"],
                "laser_detector": ["timestamp", "intensity_nw", "intensity_w", "detector_temp_c"],
                "qcl": ["timestamp", "power_mw", "temperature_c", "current_ma"],
                "temperature": ["timestamp", "temperature_c"]
            }
        }
        return metadata
    
    def save_metadata(self, metadata: Dict, experiment_name: str):
        """保存实验元数据"""
        filename = f"{experiment_name}_metadata.json"
        filepath = os.path.join(self.base_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return filepath
    
    def query_experiments(self, name: str = None, status: str = None) -> List[Dict]:
        """查询实验列表"""
        if self.use_database:
            return self.db.get_experiments(name, status)
        return []
    
    def query_data_by_time(self, experiment_id: int, start_time: float, 
                           end_time: float) -> Dict[str, np.ndarray]:
        """按时间范围查询数据"""
        if self.use_database:
            return self.db.query_by_time_range(experiment_id, start_time, end_time)
        return {}
    
    def get_experiment_statistics(self, experiment_id: int) -> Dict:
        """获取实验统计信息"""
        if self.use_database:
            return self.db.get_statistics(experiment_id)
        return {}
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """获取实验信息"""
        if self.use_database:
            return self.db.get_experiment(experiment_id)
        return None
    
    def add_material(self, **kwargs):
        """添加材料信息"""
        if self.use_database:
            return self.db.add_material(**kwargs)
    
    def add_materials_batch(self, materials: List[Dict]):
        """批量添加材料信息"""
        if self.use_database:
            return self.db.add_materials_batch(materials)
    
    def get_materials(self, category: str = None) -> List[Dict]:
        """获取材料列表"""
        if self.use_database:
            return self.db.get_materials(category)
        return []
    
    def get_material_by_id(self, material_id: int) -> Optional[Dict]:
        """根据ID获取材料信息"""
        if self.use_database:
            return self.db.get_material_by_id(material_id)
        return None
    
    def get_material_categories(self) -> List[str]:
        """获取所有材料类别"""
        if self.use_database:
            return self.db.get_all_categories()
        return []
    
    def get_material_statistics(self, category: str = None) -> Dict:
        """获取材料统计信息"""
        if self.use_database:
            return self.db.get_material_statistics(category)
        return {}
    
    def add_inversion_result(self, **kwargs):
        """添加反演结果"""
        if self.use_database and self._current_experiment_id:
            return self.db.add_inversion_result(experiment_id=self._current_experiment_id, **kwargs)
    
    def add_inversion_results_batch(self, results: List[Dict]):
        """批量添加反演结果"""
        if self.use_database and self._current_experiment_id:
            return self.db.add_inversion_results_batch(experiment_id=self._current_experiment_id, results=results)
    
    def get_inversion_results(self, experiment_id: int = None, method: str = None) -> np.ndarray:
        """获取反演结果"""
        if self.use_database:
            exp_id = experiment_id if experiment_id else self._current_experiment_id
            if exp_id:
                return self.db.get_inversion_results(exp_id, method)
        return np.array([])
    
    def add_evaluation_metric(self, **kwargs):
        """添加评估指标"""
        if self.use_database and self._current_experiment_id:
            return self.db.add_evaluation_metric(experiment_id=self._current_experiment_id, **kwargs)
    
    def add_evaluation_metrics_batch(self, metrics: List[Dict]):
        """批量添加评估指标"""
        if self.use_database and self._current_experiment_id:
            return self.db.add_evaluation_metrics_batch(experiment_id=self._current_experiment_id, metrics=metrics)
    
    def get_evaluation_metrics(self, experiment_id: int = None, method: str = None,
                               metric_type: str = None) -> List[Dict]:
        """获取评估指标"""
        if self.use_database:
            exp_id = experiment_id if experiment_id else self._current_experiment_id
            if exp_id:
                return self.db.get_evaluation_metrics(exp_id, method, metric_type)
        return []
    
    def get_metrics_summary(self, experiment_id: int = None) -> Dict[str, Dict[str, float]]:
        """获取评估指标摘要"""
        if self.use_database:
            exp_id = experiment_id if experiment_id else self._current_experiment_id
            if exp_id:
                return self.db.get_metrics_summary(exp_id)
        return {}
    
    def add_noise_configuration(self, **kwargs):
        """添加噪声配置"""
        if self.use_database and self._current_experiment_id:
            return self.db.add_noise_configuration(experiment_id=self._current_experiment_id, **kwargs)
    
    def get_noise_configurations(self, experiment_id: int = None) -> List[Dict]:
        """获取噪声配置"""
        if self.use_database:
            exp_id = experiment_id if experiment_id else self._current_experiment_id
            if exp_id:
                return self.db.get_noise_configurations(exp_id)
        return []
    
    def add_measurement_condition(self, **kwargs):
        """添加测量条件"""
        if self.use_database and self._current_experiment_id:
            return self.db.add_measurement_condition(experiment_id=self._current_experiment_id, **kwargs)
    
    def get_measurement_conditions(self, experiment_id: int = None) -> List[Dict]:
        """获取测量条件"""
        if self.use_database:
            exp_id = experiment_id if experiment_id else self._current_experiment_id
            if exp_id:
                return self.db.get_measurement_conditions(exp_id)
        return []

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
    
    def compute_statistics(self, data: Dict[str, np.ndarray]) -> Dict:
        """计算数据统计"""
        stats = {}
        
        for sensor_name, sensor_data in data.items():
            if isinstance(sensor_data, np.ndarray):
                stats[sensor_name] = {
                    "mean": float(np.mean(sensor_data)),
                    "std": float(np.std(sensor_data)),
                    "min": float(np.min(sensor_data)),
                    "max": float(np.max(sensor_data)),
                    "median": float(np.median(sensor_data)),
                    "count": int(len(sensor_data))
                }
        
        return stats
    
    def compute_rmse(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """计算RMSE"""
        return float(np.sqrt(np.mean((predictions - ground_truth)**2)))
    
    def compute_mae(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """计算MAE"""
        return float(np.mean(np.abs(predictions - ground_truth)))
    
    def compute_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """计算相关系数"""
        return float(np.corrcoef(data1, data2)[0, 1])
    
    def detect_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """检测异常值"""
        mean = np.mean(data)
        std = np.std(data)
        outliers = np.abs(data - mean) > threshold * std
        return outliers
    
    def analyze_filter_performance(self, filtered_data: np.ndarray, 
                                   ground_truth: np.ndarray) -> Dict:
        """分析滤波性能"""
        return {
            "rmse": self.compute_rmse(filtered_data, ground_truth),
            "mae": self.compute_mae(filtered_data, ground_truth),
            "correlation": self.compute_correlation(filtered_data.flatten(), ground_truth.flatten()),
            "outlier_ratio": float(np.mean(self.detect_outliers(filtered_data.flatten())))
        }
    
    def analyze_inversion_performance(self, emissivity_pred: np.ndarray, 
                                      emissivity_true: np.ndarray,
                                      reflectivity_pred: np.ndarray = None,
                                      reflectivity_true: np.ndarray = None) -> Dict:
        """分析反演性能"""
        results = {
            "emissivity_rmse": self.compute_rmse(emissivity_pred, emissivity_true),
            "emissivity_mae": self.compute_mae(emissivity_pred, emissivity_true),
            "emissivity_correlation": self.compute_correlation(emissivity_pred, emissivity_true)
        }
        
        if reflectivity_pred is not None and reflectivity_true is not None:
            results.update({
                "reflectivity_rmse": self.compute_rmse(reflectivity_pred, reflectivity_true),
                "reflectivity_mae": self.compute_mae(reflectivity_pred, reflectivity_true),
                "reflectivity_correlation": self.compute_correlation(reflectivity_pred, reflectivity_true)
            })
        
        constraint_violations = np.sum(emissivity_pred + reflectivity_pred > 1.01) if reflectivity_pred is not None else 0
        results["constraint_violation_rate"] = float(constraint_violations / len(emissivity_pred))
        
        return results
    
    def generate_report(self, analysis_results: Dict, output_file: str = None) -> str:
        """生成分析报告"""
        report = f"""
========================================
        Data Analysis Report
========================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

----------------------------------------
1. Basic Statistics
----------------------------------------
"""
        
        if "statistics" in analysis_results:
            for sensor, stats in analysis_results["statistics"].items():
                report += f"\n{sensor.upper()}:\n"
                report += f"  Mean: {stats['mean']:.4f}\n"
                report += f"  Std: {stats['std']:.4f}\n"
                report += f"  Min: {stats['min']:.4f}\n"
                report += f"  Max: {stats['max']:.4f}\n"
                report += f"  Median: {stats['median']:.4f}\n"
        
        report += """
----------------------------------------
2. Filter Performance
----------------------------------------
"""
        
        if "filter_performance" in analysis_results:
            perf = analysis_results["filter_performance"]
            report += f"\nRMSE: {perf['rmse']:.4f}\n"
            report += f"MAE: {perf['mae']:.4f}\n"
            report += f"Correlation: {perf['correlation']:.4f}\n"
            report += f"Outlier Ratio: {perf['outlier_ratio']:.4f}\n"
        
        report += """
----------------------------------------
3. Inversion Performance
----------------------------------------
"""
        
        if "inversion_performance" in analysis_results:
            inv_perf = analysis_results["inversion_performance"]
            report += f"\nEmissivity RMSE: {inv_perf['emissivity_rmse']:.4f}\n"
            report += f"Emissivity MAE: {inv_perf['emissivity_mae']:.4f}\n"
            report += f"Emissivity Correlation: {inv_perf['emissivity_correlation']:.4f}\n"
            if "reflectivity_rmse" in inv_perf:
                report += f"\nReflectivity RMSE: {inv_perf['reflectivity_rmse']:.4f}\n"
                report += f"Reflectivity MAE: {inv_perf['reflectivity_mae']:.4f}\n"
                report += f"Reflectivity Correlation: {inv_perf['reflectivity_correlation']:.4f}\n"
            report += f"\nConstraint Violation Rate: {inv_perf['constraint_violation_rate']:.4f}\n"
        
        report += """
========================================
              End of Report
========================================
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report

def create_default_storage(base_dir: str = "./data", use_database: bool = True) -> DataStorage:
    """创建默认数据存储"""
    return DataStorage(base_dir, use_database)

def create_default_analyzer(storage: DataStorage = None) -> DataAnalyzer:
    """创建默认数据分析器"""
    if storage is None:
        storage = create_default_storage()
    return DataAnalyzer(storage)