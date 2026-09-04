import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FilteringMetrics:
    distance_rmse: float = 0.0
    angle_rmse: float = 0.0
    velocity_rmse: float = 0.0
    overall_rmse: float = 0.0
    mae: float = 0.0
    nees_mean: float = 0.0
    nees_std: float = 0.0
    consistency_passed: bool = False

@dataclass
class InversionMetrics:
    emissivity_rmse: float = 0.0
    reflectivity_rmse: float = 0.0
    classification_acc: float = 0.0
    f1_score: float = 0.0
    constraint_violation_rate: float = 0.0
    picp: float = 0.0
    mpiw: float = 0.0
    calibration_error: float = 0.0

@dataclass
class SystemMetrics:
    position_rmse: float = 0.0
    angle_rmse: float = 0.0
    emissivity_rmse: float = 0.0
    overall_score: float = 0.0

@dataclass
class EfficiencyMetrics:
    inference_time_ms: float = 0.0
    memory_mb: float = 0.0
    training_time_h: float = 0.0
    fps: float = 0.0

def compute_rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """计算均方根误差"""
    return np.sqrt(np.mean((true - pred) ** 2))

def compute_mae(true: np.ndarray, pred: np.ndarray) -> float:
    """计算平均绝对误差"""
    return np.mean(np.abs(true - pred))

def compute_nees(true: np.ndarray, pred: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """计算归一化估计误差平方"""
    nees_values = []
    for i in range(len(true)):
        error = true[i] - pred[i]
        cov_i = cov[i] if cov.ndim == 3 else cov
        try:
            cov_inv = np.linalg.inv(cov_i)
            nees = error @ cov_inv @ error.T
            nees_values.append(nees)
        except:
            nees_values.append(np.nan)
    return np.array(nees_values)

def compute_nees_stats(nees_values: np.ndarray) -> Tuple[float, float, bool]:
    """计算NEES统计量和一致性检验"""
    valid_nees = nees_values[~np.isnan(nees_values)]
    if len(valid_nees) == 0:
        return 0.0, 0.0, False
    
    mean_nees = np.mean(valid_nees)
    std_nees = np.std(valid_nees)
    
    dim = len(valid_nees) if len(valid_nees) > 0 else 1
    chi2_lower = 0.1 * dim
    chi2_upper = 2.0 * dim
    
    consistent = chi2_lower <= mean_nees <= chi2_upper
    
    return mean_nees, std_nees, consistent

def compute_filtering_metrics(true_states: np.ndarray, pred_states: np.ndarray, 
                              cov_matrices: Optional[np.ndarray] = None) -> FilteringMetrics:
    """计算滤波性能指标"""
    distance_rmse = compute_rmse(true_states[:, 0], pred_states[:, 0])
    angle_rmse = compute_rmse(true_states[:, 1], pred_states[:, 1])
    velocity_rmse = compute_rmse(true_states[:, 2], pred_states[:, 2])
    
    overall_rmse = compute_rmse(true_states, pred_states)
    mae = compute_mae(true_states, pred_states)
    
    nees_mean = 0.0
    nees_std = 0.0
    consistency_passed = False
    
    if cov_matrices is not None:
        nees_values = compute_nees(true_states, pred_states, cov_matrices)
        nees_mean, nees_std, consistency_passed = compute_nees_stats(nees_values)
    
    return FilteringMetrics(
        distance_rmse=distance_rmse,
        angle_rmse=angle_rmse,
        velocity_rmse=velocity_rmse,
        overall_rmse=overall_rmse,
        mae=mae,
        nees_mean=nees_mean,
        nees_std=nees_std,
        consistency_passed=consistency_passed
    )

def compute_inversion_metrics(true_emissivity: np.ndarray, true_reflectivity: np.ndarray,
                              pred_emissivity: np.ndarray, pred_reflectivity: np.ndarray,
                              true_material_ids: Optional[np.ndarray] = None,
                              pred_material_ids: Optional[np.ndarray] = None,
                              uncertainty_intervals: Optional[np.ndarray] = None,
                              confidence_level: float = 0.95) -> InversionMetrics:
    """计算反演性能指标"""
    emissivity_rmse = compute_rmse(true_emissivity, pred_emissivity)
    reflectivity_rmse = compute_rmse(true_reflectivity, pred_reflectivity)
    
    total = pred_emissivity + pred_reflectivity
    violations = np.sum(total > 1.0 + 1e-6)
    constraint_violation_rate = violations / len(total) if len(total) > 0 else 0.0
    
    classification_acc = 0.0
    f1_score = 0.0
    
    if true_material_ids is not None and pred_material_ids is not None:
        classification_acc = compute_classification_accuracy(true_material_ids, pred_material_ids)
        f1_score = compute_f1_score(true_material_ids, pred_material_ids)
    
    picp = 0.0
    mpiw = 0.0
    calibration_error = 0.0
    
    if uncertainty_intervals is not None:
        picp, mpiw = compute_prediction_interval_stats(
            true_emissivity, 
            uncertainty_intervals[:, 0], 
            uncertainty_intervals[:, 1]
        )
        calibration_error = abs(picp - confidence_level)
    
    return InversionMetrics(
        emissivity_rmse=emissivity_rmse,
        reflectivity_rmse=reflectivity_rmse,
        classification_acc=classification_acc,
        f1_score=f1_score,
        constraint_violation_rate=constraint_violation_rate,
        picp=picp,
        mpiw=mpiw,
        calibration_error=calibration_error
    )

def compute_classification_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """计算分类准确率"""
    if len(true_labels) == 0:
        return 0.0
    return np.mean(true_labels == pred_labels) * 100.0

def compute_f1_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """计算F1分数（多类别）"""
    unique_labels = np.unique(true_labels)
    
    f1_scores = []
    for label in unique_labels:
        tp = np.sum((true_labels == label) & (pred_labels == label))
        fp = np.sum((true_labels != label) & (pred_labels == label))
        fn = np.sum((true_labels == label) & (pred_labels != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if len(f1_scores) > 0 else 0.0

def compute_prediction_interval_stats(true_values: np.ndarray, 
                                      lower_bounds: np.ndarray, 
                                      upper_bounds: np.ndarray) -> Tuple[float, float]:
    """计算预测区间统计量"""
    in_interval = (true_values >= lower_bounds) & (true_values <= upper_bounds)
    picp = np.mean(in_interval)
    mpiw = np.mean(upper_bounds - lower_bounds)
    
    return picp, mpiw

# Overall Score 参考尺度与权重 (与 CAL0827.tex Section~\ref{subsec:eval} 公式一致)
#   Overall = lambda_f*(1 - Pos_RMSE/Pos_ref)
#           + lambda_i*(1 - Emis_RMSE/Emis_ref)
#           + lambda_c*Acc
OVERALL_LAMBDA_F = 0.4   # 滤波项权重
OVERALL_LAMBDA_I = 0.4   # 反演项权重
OVERALL_LAMBDA_C = 0.2   # 分类项权重
OVERALL_POS_REF = 2.0    # 位置参考尺度 (m)
OVERALL_EMIS_REF = 0.1   # 发射率参考尺度


def compute_system_metrics(position_rmse: float, angle_rmse: float,
                           emissivity_rmse: float,
                           classification_acc: float = 0.0) -> SystemMetrics:
    """计算系统级综合指标 (Overall Score).

    公式与论文 CAL0827.tex 评价指标一节完全一致:
        Overall = lambda_f*(1 - Pos_RMSE/Pos_ref)
                + lambda_i*(1 - Emis_RMSE/Emis_ref)
                + lambda_c*Acc
    其中 lambda_f=0.4, lambda_i=0.4, lambda_c=0.2, Pos_ref=2.0 m, Emis_ref=0.1.
    Acc 为材料分类准确率, 取值范围 [0,1]. 分数越大越好.

    参数:
        classification_acc: 分类准确率, 传入 [0,1] 的小数; 若误传百分比 (>1)
            则自动折算为小数, 以保持公式量纲一致.
    """
    if classification_acc > 1.0:
        classification_acc = classification_acc / 100.0

    filtering_term = 1.0 - position_rmse / OVERALL_POS_REF
    inversion_term = 1.0 - emissivity_rmse / OVERALL_EMIS_REF

    filtering_term = max(0.0, min(1.0, filtering_term))
    inversion_term = max(0.0, min(1.0, inversion_term))
    acc_term = max(0.0, min(1.0, classification_acc))

    overall_score = (OVERALL_LAMBDA_F * filtering_term
                     + OVERALL_LAMBDA_I * inversion_term
                     + OVERALL_LAMBDA_C * acc_term)

    return SystemMetrics(
        position_rmse=position_rmse,
        angle_rmse=angle_rmse,
        emissivity_rmse=emissivity_rmse,
        overall_score=overall_score
    )

def compute_efficiency_metrics(inference_time_ms: float, memory_mb: float,
                                training_time_s: float = 0.0) -> EfficiencyMetrics:
    """计算计算效率指标"""
    fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0
    training_time_h = training_time_s / 3600.0 if training_time_s > 0 else 0.0
    
    return EfficiencyMetrics(
        inference_time_ms=inference_time_ms,
        memory_mb=memory_mb,
        training_time_h=training_time_h,
        fps=fps
    )

def compute_improvement(baseline_value: float, current_value: float) -> float:
    """计算相较于基线的提升幅度百分比 (针对"越小越好"的误差指标, 如 RMSE)

    improvement = (baseline - current) / baseline * 100
    """
    if baseline_value == 0:
        return 0.0
    improvement = (baseline_value - current_value) / baseline_value * 100.0
    return improvement


def compute_gain(baseline_value: float, current_value: float) -> float:
    """计算相较于基线的增益百分比 (针对"越大越好"的指标, 如 Overall Score / Accuracy)

    gain = (current - baseline) / baseline * 100
    与 compute_improvement 方向相反, 避免对 score 类指标出现负号错误 (论文 B-6).
    """
    if baseline_value == 0:
        return 0.0
    gain = (current_value - baseline_value) / baseline_value * 100.0
    return gain

def generate_filtering_comparison_table(results: Dict[str, FilteringMetrics],
                                         noise_types: List[str]) -> str:
    """生成滤波算法性能对比表格"""
    table = "=" * 80 + "\n"
    table += f"{'Filtering Algorithm Comparison':^80}\n"
    table += "=" * 80 + "\n"
    
    header = f"{'Method':<20} {'Distance RMSE':>12} {'Angle RMSE':>12} {'Velocity RMSE':>14} {'Overall RMSE':>14}\n"
    table += header
    table += "-" * 80 + "\n"
    
    for method, metrics in results.items():
        table += f"{method:<20} {metrics.distance_rmse:>12.3f} {metrics.angle_rmse:>12.3f} "
        table += f"{metrics.velocity_rmse:>14.3f} {metrics.overall_rmse:>14.3f}\n"
    
    table += "=" * 80 + "\n"
    
    best_overall = min(results.items(), key=lambda x: x[1].overall_rmse)
    table += f"Best Overall: {best_overall[0]} (RMSE: {best_overall[1].overall_rmse:.3f})\n"
    table += "=" * 80 + "\n"
    
    return table

def generate_inversion_comparison_table(results: Dict[str, InversionMetrics]) -> str:
    """生成反演算法性能对比表格"""
    table = "=" * 90 + "\n"
    table += f"{'Inversion Algorithm Comparison':^90}\n"
    table += "=" * 90 + "\n"
    
    header = f"{'Method':<18} {'Emissivity RMSE':>16} {'Reflectivity RMSE':>18} {'Classification Acc %':>20} {'F1 Score':>10}\n"
    table += header
    table += "-" * 90 + "\n"
    
    for method, metrics in results.items():
        table += f"{method:<18} {metrics.emissivity_rmse:>16.4f} {metrics.reflectivity_rmse:>18.4f} "
        table += f"{metrics.classification_acc:>20.1f} {metrics.f1_score:>10.3f}\n"
    
    table += "=" * 90 + "\n"
    
    best_eps = min(results.items(), key=lambda x: x[1].emissivity_rmse)
    table += f"Best Emissivity: {best_eps[0]} (RMSE: {best_eps[1].emissivity_rmse:.4f})\n"
    
    best_acc = max(results.items(), key=lambda x: x[1].classification_acc)
    table += f"Best Classification: {best_acc[0]} (Acc: {best_acc[1].classification_acc:.1f}%)\n"
    
    table += "=" * 90 + "\n"
    
    return table

def generate_system_comparison_table(results: Dict[str, SystemMetrics]) -> str:
    """生成系统级性能对比表格"""
    table = "=" * 85 + "\n"
    table += f"{'End-to-End System Comparison':^85}\n"
    table += "=" * 85 + "\n"
    
    header = f"{'System Combination':<28} {'Position RMSE (m)':>20} {'Angle RMSE (deg)':>20} {'Overall Score':>15}\n"
    table += header
    table += "-" * 85 + "\n"
    
    for system, metrics in results.items():
        table += f"{system:<28} {metrics.position_rmse:>20.3f} {metrics.angle_rmse:>20.3f} "
        table += f"{metrics.overall_score:>15.3f}\n"
    
    table += "=" * 85 + "\n"
    
    best_system = max(results.items(), key=lambda x: x[1].overall_score)
    table += f"Best Overall: {best_system[0]} (Score: {best_system[1].overall_score:.3f})\n"
    table += "=" * 85 + "\n"
    
    return table

def generate_ablation_table(results: Dict[str, Dict[str, float]], baseline_method: str = "EKF") -> str:
    """生成消融实验对比表格"""
    table = "=" * 80 + "\n"
    table += f"{'Component Ablation Analysis':^80}\n"
    table += "=" * 80 + "\n"
    
    header = f"{'Configuration':<35} {'Position RMSE':>15} {'Emissivity RMSE':>18} {'Improvement %':>10}\n"
    table += header
    table += "-" * 80 + "\n"
    
    baseline_pos_rmse = results[baseline_method]['position_rmse']
    baseline_eps_rmse = results[baseline_method]['emissivity_rmse']
    
    for config, metrics in results.items():
        combined_rmse = 0.7 * metrics['position_rmse'] + 0.3 * metrics['emissivity_rmse']
        baseline_combined = 0.7 * baseline_pos_rmse + 0.3 * baseline_eps_rmse
        improvement = compute_improvement(baseline_combined, combined_rmse)
        
        table += f"{config:<35} {metrics['position_rmse']:>15.3f} {metrics['emissivity_rmse']:>18.4f} "
        table += f"{improvement:>10.1f}\n"
    
    table += "=" * 80 + "\n"
    
    best_config = max(results.items(), key=lambda x: compute_improvement(
        0.7 * baseline_pos_rmse + 0.3 * baseline_eps_rmse,
        0.7 * x[1]['position_rmse'] + 0.3 * x[1]['emissivity_rmse']
    ))
    table += f"Best Configuration: {best_config[0]}\n"
    table += "=" * 80 + "\n"
    
    return table

def generate_efficiency_comparison_table(results: Dict[str, EfficiencyMetrics]) -> str:
    """生成计算效率对比表格"""
    table = "=" * 85 + "\n"
    table += f"{'Computational Efficiency Comparison':^85}\n"
    table += "=" * 85 + "\n"
    
    header = f"{'Method':<22} {'Inference Time (ms)':>22} {'Memory (MB)':>15} {'Training Time (h)':>20} {'FPS':>6}\n"
    table += header
    table += "-" * 85 + "\n"
    
    for method, metrics in results.items():
        table += f"{method:<22} {metrics.inference_time_ms:>22.2f} {metrics.memory_mb:>15.1f} "
        if metrics.training_time_h > 0:
            table += f"{metrics.training_time_h:>20.1f} "
        else:
            table += f"{'N/A':>20} "
        table += f"{metrics.fps:>6.1f}\n"
    
    table += "=" * 85 + "\n"
    
    fastest_inference = min(results.items(), key=lambda x: x[1].inference_time_ms)
    table += f"Fastest Inference: {fastest_inference[0]} ({fastest_inference[1].inference_time_ms:.2f} ms)\n"
    
    most_efficient = max(results.items(), key=lambda x: x[1].fps)
    table += f"Highest FPS: {most_efficient[0]} ({most_efficient[1].fps:.1f} FPS)\n"
    
    table += "=" * 85 + "\n"
    
    return table

def calculate_composite_improvement(filter_improvement: float, inversion_improvement: float) -> float:
    """计算综合提升幅度"""
    return 0.5 * filter_improvement + 0.5 * inversion_improvement


def compute_picp_at_confidence_levels(true_values: np.ndarray, pred_mean: np.ndarray,
                                        pred_std: np.ndarray,
                                        confidence_levels: List[float] = None) -> List[Dict]:
    """计算多个置信水平下的预测区间覆盖概率 (PICP) - 论文Table 8 (tab:uncertainty_calibration)
    
    论文Section 5.4.4: 不确定性校准实验
    基于 贝叶斯变分推断 的高斯后验 q(y|z) = N(μ_φ, σ_φ²),
    在不同置信水平 (1-α) 下计算:
      - PICP (Prediction Interval Coverage Probability): 真值落入区间的比例
      - MPIW (Mean Prediction Interval Width): 平均区间宽度
      - Calibration Error: |PICP - (1-α)|
    
    置信区间由高斯分位数确定: [μ - z_{α/2}·σ, μ + z_{α/2}·σ]
    
    Args:
        true_values: 真实值 (N,)
        pred_mean: 预测均值 (N,)
        pred_std: 预测标准差 (N,)
        confidence_levels: 置信水平列表, 默认 [0.68, 0.90, 0.95, 0.99]
    
    Returns:
        各置信水平的校准结果列表
    """
    from scipy.stats import norm
    
    if confidence_levels is None:
        # 论文Table 8: 68%/90%/95%/99%
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
    
    results = []
    for cl in confidence_levels:
        # 高斯分位数 z_{α/2}, 使 P(|Y-μ| <= z·σ) = cl
        z_value = norm.ppf(0.5 + cl / 2.0)
        
        # 构建预测区间 [μ - z·σ, μ + z·σ]
        lower = pred_mean - z_value * pred_std
        upper = pred_mean + z_value * pred_std
        
        # 计算 PICP
        in_interval = (true_values >= lower) & (true_values <= upper)
        picp = np.mean(in_interval)
        
        # 计算 MPIW
        mpiw = np.mean(upper - lower)
        
        # 校准误差
        calibration_error = abs(picp - cl)
        
        results.append({
            'confidence_level': cl,
            'target_picp': cl,
            'actual_picp': picp,
            'calibration_error': calibration_error,
            'mpiw': mpiw,
            'z_value': z_value
        })
    
    return results


def generate_uncertainty_calibration_table(true_values: np.ndarray, pred_mean: np.ndarray,
                                            pred_std: np.ndarray,
                                            confidence_levels: List[float] = None) -> str:
    """生成不确定性校准对比表格 - 论文Table 8 (tab:uncertainty_calibration)
    
    Returns:
        格式化的表格字符串
    """
    if confidence_levels is None:
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
    
    results = compute_picp_at_confidence_levels(
        true_values, pred_mean, pred_std, confidence_levels
    )
    
    table = "=" * 86 + "\n"
    table += f"{'Uncertainty Calibration Performance':^86}\n"
    table += f"{'(Paper Table 8: tab:uncertainty_calibration)':^86}\n"
    table += "=" * 86 + "\n"

    header = f"{'Confidence':>12} {'PICP (Target)':>15} {'PICP (Actual)':>15} {'MPIW':>10} {'Cal. Error':>12}\n"
    table += header
    table += "-" * 86 + "\n"

    for r in results:
        cl_pct = int(r['confidence_level'] * 100)
        table += f"{cl_pct:>11}% {r['target_picp']:>15.3f} {r['actual_picp']:>15.3f} "
        table += f"{r['mpiw']:>10.4f} {r['calibration_error']:>12.3f}\n"

    table += "=" * 86 + "\n"

    avg_cal_error = np.mean([r['calibration_error'] for r in results])
    avg_mpiw = np.mean([r['mpiw'] for r in results])
    table += f"Average Calibration Error: {avg_cal_error:.4f}    Average MPIW: {avg_mpiw:.4f}\n"
    table += "=" * 86 + "\n"

    return table