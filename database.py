import sqlite3
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import os
import logging

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """数据库操作异常"""
    pass

class ExperimentDatabase:
    """实验数据库管理类
    
    负责实验数据的存储、查询、更新等操作，基于SQLite实现。
    支持材料管理、噪声配置、测量条件、反演结果、评估指标等数据的持久化。
    """
    
    def __init__(self, db_path: str = "./data/experiment.db"):
        self.db_path = db_path
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self):
        dir_path = os.path.dirname(self.db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration REAL,
                config_json TEXT,
                status TEXT DEFAULT 'running',
                output_dir TEXT,
                dataset_split TEXT DEFAULT 'train',
                noise_type TEXT DEFAULT 'none'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                material_name TEXT NOT NULL,
                emissivity_mean REAL NOT NULL,
                emissivity_std REAL NOT NULL,
                reflectivity_mean REAL NOT NULL,
                reflectivity_std REAL NOT NULL,
                thermal_conductivity REAL,
                specific_heat REAL,
                density REAL,
                roughness REAL,
                description TEXT,
                UNIQUE(category, material_name)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noise_configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                noise_type TEXT NOT NULL,
                param_1 REAL DEFAULT 0.0,
                param_2 REAL DEFAULT 0.0,
                param_3 REAL DEFAULT 0.0,
                param_4 REAL DEFAULT 0.0,
                description TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurement_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                distance_m REAL NOT NULL,
                angle_deg REAL NOT NULL,
                vibration_hz REAL DEFAULT 0.0,
                vibration_amp REAL DEFAULT 0.0,
                temperature_c REAL NOT NULL,
                laser_power_mw REAL NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                config_json TEXT,
                enabled INTEGER DEFAULT 1,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(experiment_id, type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_vibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                amplitude REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_laser_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                intensity_nw REAL NOT NULL,
                intensity_w REAL NOT NULL,
                detector_temp_c REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_qcl_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                power_mw REAL NOT NULL,
                temperature_c REAL NOT NULL,
                current_ma REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_temperature_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                temperature_c REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inversion_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                material_id INTEGER,
                timestamp REAL NOT NULL,
                emissivity_pred REAL NOT NULL,
                emissivity_true REAL,
                emissivity_std REAL DEFAULT 0.0,
                reflectivity_pred REAL NOT NULL,
                reflectivity_true REAL,
                reflectivity_std REAL DEFAULT 0.0,
                constraint_satisfied INTEGER DEFAULT 1,
                method TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                FOREIGN KEY (material_id) REFERENCES materials(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                confidence_interval TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                data_blob BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(experiment_id, data_type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                results_json TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_vibration_timestamp ON raw_vibration_data(experiment_id, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_laser_timestamp ON raw_laser_data(experiment_id, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_qcl_timestamp ON raw_qcl_data(experiment_id, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_temperature_timestamp ON raw_temperature_data(experiment_id, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_experiment_name ON experiments(name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_material_category ON materials(category)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_inversion_experiment ON inversion_results(experiment_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_experiment_method ON evaluation_metrics(experiment_id, method)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_conditions_experiment ON measurement_conditions(experiment_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def _connect(self) -> sqlite3.Connection:
        """建立数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA foreign_keys = ON')
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseError(f"Database connection failed: {e}")
    
    def add_experiment(self, name: str, config = None, duration: float = None, 
                       output_dir: str = None, dataset_split: str = 'train',
                       noise_type: str = 'none') -> int:
        """添加实验记录
        
        Args:
            name: 实验名称
            config: 实验配置对象
            duration: 实验持续时间（秒）
            output_dir: 输出目录
            dataset_split: 数据集划分（train/val/test）
            noise_type: 噪声类型
        
        Returns:
            实验ID
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            config_json = json.dumps(asdict(config)) if config else None
            
            cursor.execute('''
                INSERT OR IGNORE INTO experiments 
                (name, timestamp, duration, config_json, status, output_dir, dataset_split, noise_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, timestamp, duration, config_json, 'running', output_dir, dataset_split, noise_type))
            
            cursor.execute('''
                SELECT id FROM experiments WHERE name = ? AND timestamp = ?
            ''', (name, timestamp))
            
            result = cursor.fetchone()
            conn.commit()
            conn.close()
            
            if result:
                logger.info(f"Experiment '{name}' created with ID {result[0]}")
                return result[0]
            else:
                raise DatabaseError(f"Failed to create experiment '{name}'")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add experiment: {e}")
            raise DatabaseError(f"Failed to add experiment: {e}")
    
    def update_experiment_status(self, experiment_id: int, status: str):
        """更新实验状态
        
        Args:
            experiment_id: 实验ID
            status: 状态（running/completed/failed）
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute('UPDATE experiments SET status = ? WHERE id = ?', (status, experiment_id))
            conn.commit()
            conn.close()
            logger.info(f"Experiment {experiment_id} status updated to '{status}'")
        except sqlite3.Error as e:
            logger.error(f"Failed to update experiment status: {e}")
            raise DatabaseError(f"Failed to update experiment status: {e}")
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0], 'name': result[1], 'timestamp': result[2],
                'duration': result[3], 'config_json': result[4], 'status': result[5],
                'output_dir': result[6], 'dataset_split': result[7], 'noise_type': result[8]
            }
        return None
    
    def get_experiments(self, name: str = None, status: str = None, 
                       dataset_split: str = None, noise_type: str = None) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM experiments WHERE 1=1'
        params = []
        
        if name:
            query += ' AND name = ?'
            params.append(name)
        if status:
            query += ' AND status = ?'
            params.append(status)
        if dataset_split:
            query += ' AND dataset_split = ?'
            params.append(dataset_split)
        if noise_type:
            query += ' AND noise_type = ?'
            params.append(noise_type)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'name': r[1], 'timestamp': r[2],
            'duration': r[3], 'config_json': r[4], 'status': r[5],
            'output_dir': r[6], 'dataset_split': r[7], 'noise_type': r[8]
        } for r in results]
    
    def add_material(self, category: str, material_name: str, emissivity_mean: float,
                     emissivity_std: float, reflectivity_mean: float, reflectivity_std: float,
                     thermal_conductivity: float = None, specific_heat: float = None,
                     density: float = None, roughness: float = None, description: str = ""):
        """添加材料信息
        
        Args:
            category: 材料类别
            material_name: 材料名称
            emissivity_mean: 发射率均值
            emissivity_std: 发射率标准差
            reflectivity_mean: 反射率均值
            reflectivity_std: 反射率标准差
            thermal_conductivity: 热导率
            specific_heat: 比热容
            density: 密度
            roughness: 粗糙度
            description: 描述
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO materials 
                (category, material_name, emissivity_mean, emissivity_std, 
                 reflectivity_mean, reflectivity_std, thermal_conductivity, 
                 specific_heat, density, roughness, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (category, material_name, emissivity_mean, emissivity_std,
                  reflectivity_mean, reflectivity_std, thermal_conductivity,
                  specific_heat, density, roughness, description))
            
            conn.commit()
            conn.close()
            logger.info(f"Material '{material_name}' added")
        except sqlite3.Error as e:
            logger.error(f"Failed to add material: {e}")
            raise DatabaseError(f"Failed to add material: {e}")
    
    def add_materials_batch(self, materials: List[Dict]):
        """批量添加材料信息
        
        Args:
            materials: 材料列表，每个材料为字典
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            for mat in materials:
                cursor.execute('''
                    INSERT OR REPLACE INTO materials 
                    (category, material_name, emissivity_mean, emissivity_std, 
                     reflectivity_mean, reflectivity_std, thermal_conductivity, 
                     specific_heat, density, roughness, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (mat['category'], mat['material_name'], mat['emissivity_mean'],
                      mat['emissivity_std'], mat['reflectivity_mean'], mat['reflectivity_std'],
                      mat.get('thermal_conductivity'), mat.get('specific_heat'),
                      mat.get('density'), mat.get('roughness'), mat.get('description', '')))
            
            conn.commit()
            conn.close()
            logger.info(f"Batch added {len(materials)} materials")
        except sqlite3.Error as e:
            logger.error(f"Failed to add materials batch: {e}")
            raise DatabaseError(f"Failed to add materials batch: {e}")
    
    def get_materials(self, category: str = None) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM materials'
        params = []
        
        if category:
            query += ' WHERE category = ?'
            params.append(category)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'category': r[1], 'material_name': r[2],
            'emissivity_mean': r[3], 'emissivity_std': r[4],
            'reflectivity_mean': r[5], 'reflectivity_std': r[6],
            'thermal_conductivity': r[7], 'specific_heat': r[8],
            'density': r[9], 'roughness': r[10], 'description': r[11]
        } for r in results]
    
    def get_material_by_id(self, material_id: int) -> Optional[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM materials WHERE id = ?', (material_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0], 'category': result[1], 'material_name': result[2],
                'emissivity_mean': result[3], 'emissivity_std': result[4],
                'reflectivity_mean': result[5], 'reflectivity_std': result[6],
                'thermal_conductivity': result[7], 'specific_heat': result[8],
                'density': result[9], 'roughness': result[10], 'description': result[11]
            }
        return None
    
    def add_noise_configuration(self, experiment_id: int, noise_type: str,
                                param_1: float = 0.0, param_2: float = 0.0,
                                param_3: float = 0.0, param_4: float = 0.0,
                                description: str = ""):
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO noise_configurations 
            (experiment_id, noise_type, param_1, param_2, param_3, param_4, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (experiment_id, noise_type, param_1, param_2, param_3, param_4, description))
        
        conn.commit()
        conn.close()
    
    def get_noise_configurations(self, experiment_id: int = None) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM noise_configurations'
        params = []
        
        if experiment_id:
            query += ' WHERE experiment_id = ?'
            params.append(experiment_id)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'experiment_id': r[1], 'noise_type': r[2],
            'param_1': r[3], 'param_2': r[4], 'param_3': r[5],
            'param_4': r[6], 'description': r[7]
        } for r in results]
    
    def add_measurement_condition(self, experiment_id: int, distance_m: float,
                                   angle_deg: float, temperature_c: float,
                                   laser_power_mw: float, vibration_hz: float = 0.0,
                                   vibration_amp: float = 0.0, timestamp: float = None):
        conn = self._connect()
        cursor = conn.cursor()
        
        if timestamp is None:
            timestamp = time.time()
        
        cursor.execute('''
            INSERT INTO measurement_conditions 
            (experiment_id, distance_m, angle_deg, vibration_hz, vibration_amp,
             temperature_c, laser_power_mw, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (experiment_id, distance_m, angle_deg, vibration_hz, vibration_amp,
              temperature_c, laser_power_mw, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_measurement_conditions(self, experiment_id: int) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM measurement_conditions WHERE experiment_id = ?', (experiment_id,))
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'experiment_id': r[1], 'distance_m': r[2],
            'angle_deg': r[3], 'vibration_hz': r[4], 'vibration_amp': r[5],
            'temperature_c': r[6], 'laser_power_mw': r[7], 'timestamp': r[8]
        } for r in results]
    
    def add_sensors(self, experiment_id: int, sensors: Dict[str, Any]):
        conn = self._connect()
        cursor = conn.cursor()
        
        for sensor_type, sensor_info in sensors.items():
            if hasattr(sensor_info, '__dict__'):
                config_json = json.dumps(asdict(sensor_info))
                sensor_name = sensor_info.name
            elif isinstance(sensor_info, dict):
                config_json = json.dumps(sensor_info)
                sensor_name = sensor_info.get('name', 'Unknown')
            else:
                config_json = json.dumps(str(sensor_info))
                sensor_name = 'Unknown'
            
            cursor.execute('''
                INSERT OR REPLACE INTO sensors 
                (experiment_id, type, name, config_json, enabled)
                VALUES (?, ?, ?, ?, ?)
            ''', (experiment_id, sensor_type, sensor_name, config_json, 1))
        
        conn.commit()
        conn.close()
    
    def get_sensors(self, experiment_id: int) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM sensors WHERE experiment_id = ?', (experiment_id,))
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'experiment_id': r[1], 'type': r[2],
            'name': r[3], 'config_json': r[4], 'enabled': bool(r[5])
        } for r in results]
    
    def add_vibration_data(self, experiment_id: int, data: List[Dict]):
        conn = self._connect()
        cursor = conn.cursor()
        for item in data:
            cursor.execute('''
                INSERT INTO raw_vibration_data 
                (experiment_id, timestamp, x, y, z, amplitude)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (experiment_id, item['timestamp'], item['x'], item['y'], 
                  item['z'], item['amplitude']))
        conn.commit()
        conn.close()
    
    def add_laser_data(self, experiment_id: int, data: List[Dict]):
        conn = self._connect()
        cursor = conn.cursor()
        for item in data:
            cursor.execute('''
                INSERT INTO raw_laser_data 
                (experiment_id, timestamp, intensity_nw, intensity_w, detector_temp_c)
                VALUES (?, ?, ?, ?, ?)
            ''', (experiment_id, item['timestamp'], item['intensity_nw'], 
                  item['intensity_w'], item['detector_temp_c']))
        conn.commit()
        conn.close()
    
    def add_qcl_data(self, experiment_id: int, data: List[Dict]):
        conn = self._connect()
        cursor = conn.cursor()
        for item in data:
            cursor.execute('''
                INSERT INTO raw_qcl_data 
                (experiment_id, timestamp, power_mw, temperature_c, current_ma)
                VALUES (?, ?, ?, ?, ?)
            ''', (experiment_id, item['timestamp'], item['power_mw'], 
                  item['temperature_c'], item['current_ma']))
        conn.commit()
        conn.close()
    
    def add_temperature_data(self, experiment_id: int, data: List[Dict]):
        conn = self._connect()
        cursor = conn.cursor()
        for item in data:
            cursor.execute('''
                INSERT INTO raw_temperature_data 
                (experiment_id, timestamp, temperature_c)
                VALUES (?, ?, ?)
            ''', (experiment_id, item['timestamp'], item['temperature_c']))
        conn.commit()
        conn.close()
    
    def get_vibration_data(self, experiment_id: int, start_time: float = None, 
                           end_time: float = None) -> np.ndarray:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT timestamp, x, y, z, amplitude FROM raw_vibration_data WHERE experiment_id = ?'
        params = [experiment_id]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp ASC'
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return np.array(results) if results else np.array([])
    
    def get_laser_data(self, experiment_id: int, start_time: float = None, 
                       end_time: float = None) -> np.ndarray:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT timestamp, intensity_nw, intensity_w, detector_temp_c FROM raw_laser_data WHERE experiment_id = ?'
        params = [experiment_id]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp ASC'
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return np.array(results) if results else np.array([])
    
    def get_qcl_data(self, experiment_id: int, start_time: float = None, 
                     end_time: float = None) -> np.ndarray:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT timestamp, power_mw, temperature_c, current_ma FROM raw_qcl_data WHERE experiment_id = ?'
        params = [experiment_id]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp ASC'
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return np.array(results) if results else np.array([])
    
    def get_temperature_data(self, experiment_id: int, start_time: float = None, 
                             end_time: float = None) -> np.ndarray:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT timestamp, temperature_c FROM raw_temperature_data WHERE experiment_id = ?'
        params = [experiment_id]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp ASC'
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return np.array(results) if results else np.array([])
    
    def add_inversion_result(self, experiment_id: int, timestamp: float,
                             emissivity_pred: float, reflectivity_pred: float,
                             emissivity_true: float = None, reflectivity_true: float = None,
                             emissivity_std: float = 0.0, reflectivity_std: float = 0.0,
                             constraint_satisfied: bool = True, method: str = 'ssm_pinn',
                             material_id: int = None):
        """添加反演结果
        
        Args:
            experiment_id: 实验ID
            timestamp: 时间戳
            emissivity_pred: 预测发射率
            reflectivity_pred: 预测反射率
            emissivity_true: 真实发射率（可选）
            reflectivity_true: 真实反射率（可选）
            emissivity_std: 发射率标准差
            reflectivity_std: 反射率标准差
            constraint_satisfied: 约束是否满足
            method: 反演方法名称
            material_id: 材料ID（可选）
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inversion_results 
                (experiment_id, material_id, timestamp, emissivity_pred, emissivity_true, 
                 emissivity_std, reflectivity_pred, reflectivity_true, reflectivity_std, 
                 constraint_satisfied, method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (experiment_id, material_id, timestamp, emissivity_pred, emissivity_true,
                  emissivity_std, reflectivity_pred, reflectivity_true, reflectivity_std,
                  int(constraint_satisfied), method))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to add inversion result: {e}")
            raise DatabaseError(f"Failed to add inversion result: {e}")
    
    def add_inversion_results_batch(self, experiment_id: int, results: List[Dict]):
        """批量添加反演结果
        
        Args:
            experiment_id: 实验ID
            results: 反演结果列表
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            for res in results:
                cursor.execute('''
                    INSERT INTO inversion_results 
                    (experiment_id, material_id, timestamp, emissivity_pred, emissivity_true, 
                     emissivity_std, reflectivity_pred, reflectivity_true, reflectivity_std, 
                     constraint_satisfied, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (experiment_id, res.get('material_id'), res['timestamp'],
                      res['emissivity_pred'], res.get('emissivity_true'),
                      res.get('emissivity_std', 0.0), res['reflectivity_pred'],
                      res.get('reflectivity_true'), res.get('reflectivity_std', 0.0),
                      int(res.get('constraint_satisfied', True)), res.get('method', 'ssm_pinn')))
            
            conn.commit()
            conn.close()
            logger.info(f"Batch added {len(results)} inversion results for experiment {experiment_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to add inversion results batch: {e}")
            raise DatabaseError(f"Failed to add inversion results batch: {e}")
    
    def get_inversion_results(self, experiment_id: int, method: str = None) -> np.ndarray:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM inversion_results WHERE experiment_id = ?'
        params = [experiment_id]
        
        if method:
            query += ' AND method = ?'
            params.append(method)
        
        query += ' ORDER BY timestamp ASC'
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return np.array(results) if results else np.array([])
    
    def add_evaluation_metric(self, experiment_id: int, method: str,
                              metric_type: str, value: float,
                              confidence_interval: str = ""):
        """添加评估指标
        
        Args:
            experiment_id: 实验ID
            method: 方法名称
            metric_type: 指标类型（如rmse, mae, correlation等）
            value: 指标值
            confidence_interval: 置信区间（可选）
        
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO evaluation_metrics 
                (experiment_id, method, metric_type, value, confidence_interval, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (experiment_id, method, metric_type, value, confidence_interval, timestamp))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to add evaluation metric: {e}")
            raise DatabaseError(f"Failed to add evaluation metric: {e}")
    
    def add_evaluation_metrics_batch(self, experiment_id: int, metrics: List[Dict]):
        conn = self._connect()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for metric in metrics:
            cursor.execute('''
                INSERT INTO evaluation_metrics 
                (experiment_id, method, metric_type, value, confidence_interval, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (experiment_id, metric['method'], metric['metric_type'],
                  metric['value'], metric.get('confidence_interval', ''), timestamp))
        
        conn.commit()
        conn.close()
    
    def get_evaluation_metrics(self, experiment_id: int, method: str = None,
                               metric_type: str = None) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM evaluation_metrics WHERE experiment_id = ?'
        params = [experiment_id]
        
        if method:
            query += ' AND method = ?'
            params.append(method)
        if metric_type:
            query += ' AND metric_type = ?'
            params.append(metric_type)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'id': r[0], 'experiment_id': r[1], 'method': r[2],
            'metric_type': r[3], 'value': r[4], 'confidence_interval': r[5],
            'timestamp': r[6]
        } for r in results]
    
    def get_metrics_summary(self, experiment_id: int) -> Dict[str, Dict[str, float]]:
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT method, metric_type, AVG(value) as avg_value, 
                   MIN(value) as min_value, MAX(value) as max_value, COUNT(*) as count
            FROM evaluation_metrics WHERE experiment_id = ?
            GROUP BY method, metric_type
        ''', (experiment_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {}
        for r in results:
            method, metric_type, avg_val, min_val, max_val, count = r
            if method not in summary:
                summary[method] = {}
            summary[method][metric_type] = {
                'avg': avg_val, 'min': min_val, 'max': max_val, 'count': count
            }
        
        return summary
    
    def add_processed_data(self, experiment_id: int, data_type: str, data: np.ndarray):
        conn = self._connect()
        cursor = conn.cursor()
        
        data_blob = data.tobytes()
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO processed_data 
            (experiment_id, data_type, data_blob, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (experiment_id, data_type, data_blob, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_processed_data(self, experiment_id: int, data_type: str) -> Optional[np.ndarray]:
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT data_blob FROM processed_data WHERE experiment_id = ? AND data_type = ?',
                      (experiment_id, data_type))
        
        result = cursor.fetchone()
        conn.close()
        
        return np.frombuffer(result[0]) if result else None
    
    def add_analysis_results(self, experiment_id: int, results: Dict):
        conn = self._connect()
        cursor = conn.cursor()
        
        results_json = json.dumps(results)
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (experiment_id, results_json, timestamp)
            VALUES (?, ?, ?)
        ''', (experiment_id, results_json, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_analysis_results(self, experiment_id: int) -> Optional[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT results_json FROM analysis_results WHERE experiment_id = ? ORDER BY timestamp DESC LIMIT 1',
                      (experiment_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return json.loads(result[0]) if result else None
    
    def get_raw_data_count(self, experiment_id: int) -> Dict[str, int]:
        conn = self._connect()
        cursor = conn.cursor()
        
        counts = {}
        for table in ['raw_vibration_data', 'raw_laser_data', 'raw_qcl_data', 'raw_temperature_data']:
            cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE experiment_id = ?', (experiment_id,))
            counts[table] = cursor.fetchone()[0]
        
        conn.close()
        return counts
    
    def delete_experiment(self, experiment_id: int):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM experiments WHERE id = ?', (experiment_id,))
        conn.commit()
        conn.close()
    
    def get_statistics(self, experiment_id: int) -> Dict:
        conn = self._connect()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute('''
            SELECT COUNT(*), AVG(x), AVG(y), AVG(z), AVG(amplitude),
                   MIN(x), MAX(x), MIN(y), MAX(y), MIN(z), MAX(z)
            FROM raw_vibration_data WHERE experiment_id = ?
        ''', (experiment_id,))
        result = cursor.fetchone()
        if result and result[0] > 0:
            stats['vibration'] = {
                'count': result[0],
                'avg_x': result[1], 'avg_y': result[2], 'avg_z': result[3],
                'avg_amplitude': result[4],
                'min_x': result[5], 'max_x': result[6],
                'min_y': result[7], 'max_y': result[8],
                'min_z': result[9], 'max_z': result[10]
            }
        
        cursor.execute('''
            SELECT COUNT(*), AVG(intensity_nw), MIN(intensity_nw), MAX(intensity_nw)
            FROM raw_laser_data WHERE experiment_id = ?
        ''', (experiment_id,))
        result = cursor.fetchone()
        if result and result[0] > 0:
            stats['laser_detector'] = {
                'count': result[0],
                'avg_intensity_nw': result[1],
                'min_intensity_nw': result[2],
                'max_intensity_nw': result[3]
            }
        
        cursor.execute('''
            SELECT COUNT(*), AVG(power_mw), MIN(power_mw), MAX(power_mw)
            FROM raw_qcl_data WHERE experiment_id = ?
        ''', (experiment_id,))
        result = cursor.fetchone()
        if result and result[0] > 0:
            stats['qcl'] = {
                'count': result[0],
                'avg_power_mw': result[1],
                'min_power_mw': result[2],
                'max_power_mw': result[3]
            }
        
        cursor.execute('''
            SELECT COUNT(*), AVG(temperature_c), MIN(temperature_c), MAX(temperature_c)
            FROM raw_temperature_data WHERE experiment_id = ?
        ''', (experiment_id,))
        result = cursor.fetchone()
        if result and result[0] > 0:
            stats['temperature'] = {
                'count': result[0],
                'avg_temperature_c': result[1],
                'min_temperature_c': result[2],
                'max_temperature_c': result[3]
            }
        
        cursor.execute('''
            SELECT COUNT(*), AVG(emissivity_pred), AVG(reflectivity_pred)
            FROM inversion_results WHERE experiment_id = ?
        ''', (experiment_id,))
        result = cursor.fetchone()
        if result and result[0] > 0:
            stats['inversion'] = {
                'count': result[0],
                'avg_emissivity': result[1],
                'avg_reflectivity': result[2]
            }
        
        conn.close()
        return stats
    
    def query_by_time_range(self, experiment_id: int, start_time: float, 
                           end_time: float) -> Dict[str, np.ndarray]:
        return {
            'vibration': self.get_vibration_data(experiment_id, start_time, end_time),
            'laser_detector': self.get_laser_data(experiment_id, start_time, end_time),
            'qcl': self.get_qcl_data(experiment_id, start_time, end_time),
            'temperature': self.get_temperature_data(experiment_id, start_time, end_time)
        }
    
    def get_material_statistics(self, category: str = None) -> Dict:
        conn = self._connect()
        cursor = conn.cursor()
        
        query = 'SELECT COUNT(*), AVG(emissivity_mean), STDDEV(emissivity_mean), ' \
                'AVG(reflectivity_mean), STDDEV(reflectivity_mean) FROM materials'
        params = []
        
        if category:
            query += ' WHERE category = ?'
            params.append(category)
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            return {
                'count': result[0],
                'avg_emissivity': result[1],
                'std_emissivity': result[2],
                'avg_reflectivity': result[3],
                'std_reflectivity': result[4]
            }
        return {}
    
    def get_all_categories(self) -> List[str]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM materials')
        results = cursor.fetchall()
        conn.close()
        
        return [r[0] for r in results]

def create_default_database(db_path: str = "./data/experiment.db") -> ExperimentDatabase:
    return ExperimentDatabase(db_path)