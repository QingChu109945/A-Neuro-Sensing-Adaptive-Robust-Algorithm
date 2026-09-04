import numpy as np
import time
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class NoiseConfig:
    noise_type: str = 'gaussian'
    level: float = 0.01
    seed: int = 42
    param_1: float = 0.0
    param_2: float = 0.0

class NoiseInjector:
    """噪声注入器"""
    
    def __init__(self, config: NoiseConfig = None):
        self.config = config if config else NoiseConfig()
        self.rng = np.random.default_rng(self.config.seed)
    
    def inject_gaussian(self, data: np.ndarray, std: float) -> np.ndarray:
        """注入高斯噪声"""
        return data + self.rng.normal(0, std, data.shape)
    
    def inject_uniform(self, data: np.ndarray, amplitude: float) -> np.ndarray:
        """注入均匀噪声"""
        return data + self.rng.uniform(-amplitude, amplitude, data.shape)
    
    def inject_poisson(self, data: np.ndarray) -> np.ndarray:
        """注入泊松噪声"""
        data_positive = np.abs(data) + 1e-10
        return self.rng.poisson(data_positive).astype(np.float64) - data_positive + data
    
    def inject_salt_pepper(self, data: np.ndarray, prob: float) -> np.ndarray:
        """注入椒盐噪声"""
        noisy = data.copy()
        mask = self.rng.random(data.shape) < prob
        noisy[mask] = np.where(self.rng.random(data.shape)[mask] < 0.5, 
                               data.min() * 0.9, data.max() * 1.1)
        return noisy
    
    def inject_impulse(self, data: np.ndarray, prob: float, amplitude: float = 1.0) -> np.ndarray:
        """注入脉冲噪声"""
        noisy = data.copy()
        mask = self.rng.random(data.shape) < prob
        noisy[mask] += self.rng.choice([-amplitude, amplitude], np.sum(mask))
        return noisy
    
    def inject_gaussian_mixture(self, data: np.ndarray, std: float) -> np.ndarray:
        """注入高斯混合噪声 (论文Section 5.1.3 Noise Type 1)
        p(v) = 0.7·N(0,σ²) + 0.3·N(0,(5σ)²)
        """
        noisy = data.copy()
        mask = self.rng.random(data.shape) < 0.3  # 30%来自大方差分布
        noise_small = self.rng.normal(0, std, data.shape)
        noise_large = self.rng.normal(0, 5 * std, data.shape)
        noisy = data + np.where(mask, noise_large, noise_small)
        return noisy
    
    def inject_time_varying(self, data: np.ndarray, sigma_0: float, k_period: int = 100) -> np.ndarray:
        """注入时变噪声 (论文Section 5.1.3 Noise Type 3)
        σ_k = σ_0·(1 + 0.5·sin(2πk/K_period))
        """
        noisy = data.copy()
        n_samples = data.shape[0]
        for k in range(n_samples):
            sigma_k = sigma_0 * (1 + 0.5 * np.sin(2 * np.pi * k / k_period))
            if data.ndim == 1:
                noisy[k] += self.rng.normal(0, sigma_k)
            else:
                noisy[k] += self.rng.normal(0, sigma_k, data.shape[1:])
        return noisy
    
    def inject(self, data: np.ndarray) -> np.ndarray:
        """根据配置注入噪声"""
        if self.config.noise_type == 'gaussian':
            return self.inject_gaussian(data, self.config.level)
        elif self.config.noise_type == 'gaussian_mixture':
            # 论文Section 5.1.3: p(v) = 0.7·N(0,σ²) + 0.3·N(0,(5σ)²)
            return self.inject_gaussian_mixture(data, self.config.level)
        elif self.config.noise_type == 'uniform':
            return self.inject_uniform(data, self.config.level)
        elif self.config.noise_type == 'poisson':
            return self.inject_poisson(data)
        elif self.config.noise_type == 'salt_pepper':
            # 论文Section 5.1.3 Noise Type 2: 椒盐脉冲噪声
            return self.inject_salt_pepper(data, self.config.level)
        elif self.config.noise_type == 'impulse':
            return self.inject_impulse(data, self.config.level, self.config.param_1)
        elif self.config.noise_type == 'time_varying':
            # 论文Section 5.1.3 Noise Type 3: 时变噪声
            return self.inject_time_varying(data, self.config.level, k_period=100)
        elif self.config.noise_type == 'mixed':
            noisy = self.inject_gaussian(data, self.config.level * 0.5)
            noisy = self.inject_impulse(noisy, self.config.param_1, self.config.param_2)
            return noisy
        else:
            return data

@dataclass
class SimulationConfig:
    duration_seconds: int = 30
    sample_rate_hz: int = 100
    distance_m: float = 1000.0
    angle_deg: float = 30.0
    temperature_c: float = 25.0
    vibration_amp_mm: float = 10.0
    vibration_freq_hz: float = 5.0
    laser_power_mw: float = 20.0
    noise_config: NoiseConfig = None

class DataGenerator:
    """实验数据生成器"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config if config else SimulationConfig()
        self.noise_injector = NoiseInjector(config.noise_config) if config and config.noise_config else NoiseInjector()
    
    def generate_time_vector(self) -> np.ndarray:
        """生成时间向量"""
        num_samples = self.config.duration_seconds * self.config.sample_rate_hz
        return np.linspace(0, self.config.duration_seconds, num_samples)
    
    def generate_vibration_data(self, t: np.ndarray) -> np.ndarray:
        """生成振动数据"""
        amp = self.config.vibration_amp_mm
        freq = self.config.vibration_freq_hz
        
        x = amp * np.sin(2 * np.pi * freq * t) + 0.5 * amp * np.sin(2 * np.pi * 2 * freq * t)
        y = amp * np.cos(2 * np.pi * freq * t) + 0.5 * amp * np.cos(2 * np.pi * 2 * freq * t)
        z = 0.3 * amp * np.sin(2 * np.pi * 3 * freq * t)
        
        data = np.column_stack([x, y, z])
        return self.noise_injector.inject(data)
    
    def generate_laser_signal(self, t: np.ndarray) -> np.ndarray:
        """生成激光信号数据"""
        base_intensity = 1000.0
        distance_factor = 1.0 / (1 + self.config.distance_m / 5000.0)
        angle_factor = np.cos(np.deg2rad(self.config.angle_deg))
        
        intensity_nw = base_intensity * distance_factor * angle_factor * (
            1 + 0.1 * np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.randn(len(t))
        )
        intensity_w = intensity_nw * 0.95
        
        detector_temp = self.config.temperature_c + 5.0 * np.sin(2 * np.pi * 0.1 * t)
        
        data = np.column_stack([intensity_nw, intensity_w, detector_temp])
        return self.noise_injector.inject(data)
    
    def generate_qcl_data(self, t: np.ndarray) -> np.ndarray:
        """生成QCL控制器数据"""
        power_mw = self.config.laser_power_mw * (1 + 0.02 * np.sin(2 * np.pi * 0.5 * t))
        temperature_c = 15.0 + 2.0 * np.sin(2 * np.pi * 0.2 * t)
        wavelength_nm = 4500.0 + 50.0 * np.sin(2 * np.pi * 0.1 * t)
        current_ma = 150.0 + 10.0 * np.sin(2 * np.pi * 0.5 * t)
        
        data = np.column_stack([power_mw, temperature_c, wavelength_nm, current_ma])
        return self.noise_injector.inject(data)
    
    def generate_temperature_data(self, t: np.ndarray) -> np.ndarray:
        """生成温度传感器数据"""
        temp_c = self.config.temperature_c + 3.0 * np.sin(2 * np.pi * 0.05 * t) + 1.0 * np.random.randn(len(t))
        humidity_rh = 40.0 + 10.0 * np.sin(2 * np.pi * 0.02 * t)
        
        data = np.column_stack([temp_c, humidity_rh])
        return self.noise_injector.inject(data)
    
    def generate_all(self) -> Dict[str, np.ndarray]:
        """生成所有传感器数据"""
        t = self.generate_time_vector()
        
        return {
            'vibration': self.generate_vibration_data(t),
            'laser_detector': self.generate_laser_signal(t),
            'qcl': self.generate_qcl_data(t),
            'temperature': self.generate_temperature_data(t),
            'timestamps': t
        }
    
    def generate_inversion_dataset(self, material_id: int, num_samples: int = 1000) -> List[Dict]:
        """生成反演数据集"""
        dataset = []
        
        for i in range(num_samples):
            distance = np.random.uniform(100, 5000)
            angle = np.random.uniform(0, 75)
            temperature = np.random.uniform(-40, 60)
            vibration_amp = np.random.uniform(0, 50)
            
            sim_config = SimulationConfig(
                duration_seconds=1,
                sample_rate_hz=10,
                distance_m=distance,
                angle_deg=angle,
                temperature_c=temperature,
                vibration_amp_mm=vibration_amp,
                noise_config=NoiseConfig(
                    noise_type=np.random.choice(['gaussian', 'uniform', 'mixed']),
                    level=np.random.uniform(0.001, 0.05)
                )
            )
            
            generator = DataGenerator(sim_config)
            data = generator.generate_all()
            
            dataset.append({
                'material_id': material_id,
                'distance_m': distance,
                'angle_deg': angle,
                'temperature_c': temperature,
                'vibration_amp_mm': vibration_amp,
                'vibration_data': data['vibration'].tolist(),
                'laser_data': data['laser_detector'].tolist(),
                'qcl_data': data['qcl'].tolist(),
                'temperature_data': data['temperature'].tolist()
            })
        
        return dataset

def generate_dataset_for_materials(materials: List[Dict], samples_per_material: int = 500,
                                   output_path: str = None) -> List[Dict]:
    """为多种材料生成数据集"""
    all_data = []
    
    for material in materials:
        generator = DataGenerator()
        dataset = generator.generate_inversion_dataset(
            material_id=material['id'],
            num_samples=samples_per_material
        )
        
        for item in dataset:
            item['material_category'] = material['category']
            item['material_name'] = material['material_name']
            item['emissivity_true'] = material['emissivity_mean']
            item['reflectivity_true'] = material['reflectivity_mean']
        
        all_data.extend(dataset)
        print(f"Generated {len(dataset)} samples for {material['material_name']}")
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"Dataset saved to {output_path}")
    
    return all_data


# ============================================================================
# 论文Section 5.1.2: 完整数据集生成 (125,000 samples, 70/15/15 split)
# 论文Table 4 (tab:material_database): 12 material categories, 35 subtypes
# ============================================================================

@dataclass
class DatasetConfig:
    """完整数据集配置 (论文Section 5.1.2 Dataset Statistics)
    
    论文要求:
    - Total samples: 125,000
    - Training set: 87,500 (70%)
    - Validation set: 18,750 (15%)
    - Test set: 18,750 (15%)
    
    测量参数 (论文Section 5.1.2 Measurement Parameters):
    - Distance: 100-5000 m
    - Reflection angle: 0°-75°
    - Vibration amplitude: 0-50 μm
    - Laser wavelength: 1.064 μm (Nd:YAG)
    - Temperature measurement accuracy: ±0.5 K
    """
    total_samples: int = 125000
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    
    # 测量参数范围 (论文Section 5.1.2)
    distance_range: Tuple[float, float] = (100.0, 5000.0)
    angle_range: Tuple[float, float] = (0.0, 75.0)
    vibration_range: Tuple[float, float] = (0.0, 50.0)
    temperature_range: Tuple[float, float] = (273.0, 1273.0)  # K, 覆盖所有材料
    laser_wavelength_um: float = 1.064  # Nd:YAG (论文要求)
    temp_accuracy_k: float = 0.5  # ±0.5 K


class FullDatasetGenerator:
    """完整论文数据集生成器 (论文Section 5.1.2)
    
    生成125,000个样本,覆盖12个材料类别,35个子类型
    每个样本包含: 测量值 (T, V, D, θ, I_echo) 和标签 (ε, ρ, M_type)
    按 70/15/15 划分训练/验证/测试集
    """
    
    # 12个材料类别 (论文Table 4)
    MATERIAL_CATEGORIES = [
        'Carbon Fiber Composite', 'High-Hardness Steel', 'Carburized Aluminum',
        'Aluminum Alloy', 'Ni-Mo-W Alloy', 'Corroded Steel',
        'Anti-Optical Coating', 'Anti-Infrared Coating', 'Polyurethane Coating',
        'Polyimide Film', 'Ceramic Coating', 'Titanium Alloy'
    ]
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config if config else DatasetConfig()
        self.rng = np.random.default_rng(self.config.seed)
    
    def _generate_single_sample(self, material: Dict) -> Dict:
        """生成单个样本 (论文Section 5.1.2 测量参数)
        
        输入测量向量 z = [T, V, D, θ, I_echo]^T
        输出标签 y = [ε, ρ, M_type]^T
        """
        # 测量参数 (论文Section 5.1.2 Measurement Parameters)
        distance = self.rng.uniform(*self.config.distance_range)
        angle = self.rng.uniform(*self.config.angle_range)
        # 按材料类别采样温度 (论文 B-1/Table 1): 每类材料有其有效温度范围,
        # 不再用全局 (273, 1273) 统一采样, 避免生成越界样本 (如 1273 K 下的
        # Carbon Fiber). 若材料未指定 temp_range 则回退到全局配置.
        temperature_k = self.rng.uniform(*material.get('temp_range', self.config.temperature_range))
        vibration_um = self.rng.uniform(*self.config.vibration_range)
        
        # 真实材料属性 (从数据库获取,带物理波动)
        eps_true = material['emissivity_mean'] + self.rng.normal(0, material['emissivity_std'])
        eps_true = np.clip(eps_true, 0.01, 0.99)
        
        rho_true = material['reflectivity_mean'] + self.rng.normal(0, material['reflectivity_std'])
        rho_true = np.clip(rho_true, 0.01, 1.0 - eps_true)  # 保证 ε + ρ ≤ 1
        
        # 激光回波强度模型 (论文Section 4.5: I_echo = I_thermal + I_reflection + η)
        sigma_sb = 5.67e-8  # Stefan-Boltzmann常数
        # 热辐射分量: I_thermal = ε·σ_SB·T^4·f(D,θ)
        distance_factor = 1.0 / (1.0 + distance / 1000.0)
        angle_factor = np.cos(np.deg2rad(angle))
        i_thermal = eps_true * sigma_sb * (temperature_k ** 4) * distance_factor * angle_factor
        
        # 直接反射分量: I_reflection = ρ·I_laser·g(D,θ,α_rough)
        i_laser = 1000.0 * distance_factor  # 激光功率归一化
        roughness_factor = 1.0 - material.get('roughness', 0.3) * 0.3
        i_reflection = rho_true * i_laser * angle_factor * roughness_factor
        
        # 测量噪声 η
        i_echo = i_thermal + i_reflection + self.rng.normal(0, 0.01 * (i_thermal + i_reflection))
        
        # 温度测量精度 ±0.5 K (论文要求)
        temp_measured = temperature_k + self.rng.normal(0, self.config.temp_accuracy_k)
        
        # 振动特征 (频域特征)
        vib_freq = self.rng.uniform(1, 20)  # Hz
        vib_amp = vibration_um * 1e-6  # 转为米
        
        return {
            # 输入测量向量 z = [T, V, D, θ, I_echo]
            'temperature': float(temp_measured),
            'vibration': float(vib_amp),
            'distance': float(distance),
            'angle': float(angle),
            'laser_echo': float(i_echo),
            # 辅助特征
            'vibration_freq': float(vib_freq),
            'laser_wavelength': self.config.laser_wavelength_um,
            # 标签 y = [ε, ρ, M_type]
            'emissivity_true': float(eps_true),
            'reflectivity_true': float(rho_true),
            'material_id': material.get('id', 0),
            'material_category': material['category'],
            'material_name': material['material_name'],
            # 物理参数 (用于软约束)
            'thermal_conductivity': material.get('thermal_conductivity', 10.0),
            'roughness': material.get('roughness', 0.3),
        }
    
    def generate_full_dataset(self, materials: List[Dict]) -> List[Dict]:
        """生成完整125,000样本数据集
        
        Args:
            materials: 材料数据库列表 (来自 init_database.MATERIAL_DATA)
        
        Returns:
            完整数据集列表
        """
        n_total = self.config.total_samples
        n_materials = len(materials)
        samples_per_material = n_total // n_materials
        remainder = n_total - samples_per_material * n_materials
        
        print(f"Generating full dataset: {n_total} samples")
        print(f"  Materials: {n_materials}")
        print(f"  Samples per material: ~{samples_per_material}")
        
        all_samples = []
        for i, material in enumerate(materials):
            n_samples = samples_per_material + (1 if i < remainder else 0)
            # 添加 material id
            if 'id' not in material:
                material = {**material, 'id': i + 1}
            
            for _ in range(n_samples):
                sample = self._generate_single_sample(material)
                all_samples.append(sample)
            
            if (i + 1) % 10 == 0 or i == n_materials - 1:
                print(f"  [{i+1}/{n_materials}] {material['material_name']}: {n_samples} samples")
        
        # 打乱数据集
        self.rng.shuffle(all_samples)
        print(f"Total generated: {len(all_samples)} samples")
        
        return all_samples
    
    def split_dataset(self, all_samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """按 70/15/15 划分训练/验证/测试集 (论文Section 5.1.2)
        
        Returns:
            (train_set, val_set, test_set)
        """
        n = len(all_samples)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        
        train_set = all_samples[:n_train]
        val_set = all_samples[n_train:n_train + n_val]
        test_set = all_samples[n_train + n_val:]
        
        print(f"Dataset split (70/15/15):")
        print(f"  Train: {len(train_set)} ({len(train_set)/n*100:.1f}%)")
        print(f"  Validation: {len(val_set)} ({len(val_set)/n*100:.1f}%)")
        print(f"  Test: {len(test_set)} ({len(test_set)/n*100:.1f}%)")
        
        return train_set, val_set, test_set


class DatasetLoader:
    """数据集加载与预处理 (论文Section 5.1.2)
    
    支持从文件加载已生成的数据集,并提供预处理功能:
    - 特征标准化 (z-score归一化)
    - 标签提取
    - 批量加载
    """
    
    # 输入特征列 (z = [T, V, D, θ, I_echo])
    FEATURE_COLUMNS = ['temperature', 'vibration', 'distance', 'angle', 'laser_echo']
    # 标签列 (y = [ε, ρ])
    LABEL_COLUMNS = ['emissivity_true', 'reflectivity_true']
    
    def __init__(self, data_dir: str = "./data/dataset"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._scaler = None
    
    def save_dataset(self, train_set, val_set, test_set):
        """保存数据集到磁盘"""
        splits = {'train': train_set, 'val': val_set, 'test': test_set}
        for name, data in splits.items():
            path = os.path.join(self.data_dir, f"{name}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {name} set: {len(data)} samples -> {path}")
    
    def load_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """从磁盘加载数据集"""
        splits = {}
        for name in ['train', 'val', 'test']:
            path = os.path.join(self.data_dir, f"{name}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    splits[name] = pickle.load(f)
                print(f"Loaded {name} set: {len(splits[name])} samples")
            else:
                print(f"Warning: {path} not found")
                splits[name] = []
        return splits.get('train', []), splits.get('val', []), splits.get('test', [])
    
    def extract_features_labels(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取特征矩阵和标签向量
        
        Returns:
            X: 特征矩阵 (N, 5) - [T, V, D, θ, I_echo]
            y: 标签矩阵 (N, 2) - [ε, ρ]
            material_ids: 材料类别ID (N,)
        """
        n = len(dataset)
        X = np.zeros((n, len(self.FEATURE_COLUMNS)))
        y = np.zeros((n, len(self.LABEL_COLUMNS)))
        material_ids = np.zeros(n, dtype=int)
        
        for i, sample in enumerate(dataset):
            for j, col in enumerate(self.FEATURE_COLUMNS):
                X[i, j] = sample[col]
            for j, col in enumerate(self.LABEL_COLUMNS):
                y[i, j] = sample[col]
            material_ids[i] = sample.get('material_id', 0)
        
        return X, y, material_ids
    
    def fit_scaler(self, X_train: np.ndarray):
        """拟合标准化器 (z-score归一化)"""
        self._scaler = {
            'mean': np.mean(X_train, axis=0),
            'std': np.std(X_train, axis=0) + 1e-8
        }
        return self._scaler
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用标准化"""
        if self._scaler is None:
            return X
        return (X - self._scaler['mean']) / self._scaler['std']
    
    def get_dataloader(self, split: str = 'train', batch_size: int = 64):
        """获取批量数据生成器
        
        Args:
            split: 'train', 'val', or 'test'
            batch_size: 批量大小
        
        Yields:
            (X_batch, y_batch, material_ids_batch)
        """
        splits = {'train': 0, 'val': 1, 'test': 2}
        train, val, test = self.load_dataset()
        datasets = [train, val, test]
        dataset = datasets[splits[split]]
        
        if not dataset:
            return
        
        X, y, mids = self.extract_features_labels(dataset)
        if split == 'train' and self._scaler is None:
            self.fit_scaler(X)
        X = self.transform(X)
        
        n = len(X)
        indices = np.arange(n)
        if split == 'train':
            self.rng = np.random.default_rng(42)
            self.rng.shuffle(indices)
        
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield X[idx], y[idx], mids[idx]


def generate_and_save_full_dataset(data_dir: str = "./data/dataset",
                                     total_samples: int = 125000,
                                     seed: int = 42):
    """生成并保存完整论文数据集 (论文Section 5.1.2)
    
    生成125,000样本,按70/15/15划分,保存到磁盘
    
    Args:
        data_dir: 数据保存目录
        total_samples: 总样本数 (论文: 125,000)
        seed: 随机种子
    """
    # 导入材料数据库
    try:
        from .init_database import MATERIAL_DATA
    except ImportError:
        from init_database import MATERIAL_DATA
    
    config = DatasetConfig(total_samples=total_samples, seed=seed)
    generator = FullDatasetGenerator(config)
    
    print("=" * 60)
    print("Generating Full Paper Dataset")
    print(f"  Total samples: {total_samples}")
    print(f"  Split: 70% train / 15% val / 15% test")
    print("=" * 60)
    
    # 生成完整数据集
    all_samples = generator.generate_full_dataset(MATERIAL_DATA)
    
    # 划分数据集
    train_set, val_set, test_set = generator.split_dataset(all_samples)
    
    # 保存
    loader = DatasetLoader(data_dir)
    loader.save_dataset(train_set, val_set, test_set)
    
    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    config = SimulationConfig(
        duration_seconds=10,
        sample_rate_hz=100,
        noise_config=NoiseConfig(
            noise_type='gaussian',
            level=0.02,
            seed=42
        )
    )
    
    generator = DataGenerator(config)
    data = generator.generate_all()
    
    print("Generated data shapes:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")
    
    print("\nSample vibration data:")
    print(data['vibration'][:5])
    
    print("\nSample laser data:")
    print(data['laser_detector'][:5])