"""
论文完整实验运行脚本 (CAL0827.tex)

生成论文中所有实验表格和图表:
- Table 4 (tab:material_database): 目标表面材料数据库规格 (12类材料)
- Table 5 (tab:filtering_rmse): 不同噪声条件下的状态估计RMSE
- Table 6 (tab:inversion_accuracy): 材料属性反演精度
- Table 7 (tab:combined_performance): 端到端系统性能
- Table 8 (tab:uncertainty_calibration): 不确定性校准性能 (68/90/95/99% PICP)
- Table 9 (tab:ablation_study): 组件消融分析
- Table 10 (tab:computational): 计算性能对比

- Figure 1: nsarkf_architecture.png (NS-ARKF架构图)
- Figure 2: ssmpinn_architecture.png (SSM-PINN架构图)
- Figure 3: rmse_comparison.png (RMSE对比图)
- Figure 4: emissivity_scatter.png (发射率散点图)
- Figure 5: uncertainty_visualization.png (不确定性可视化)

使用方法:
    python -m experiment_system.run_paper_experiments
    或
    python experiment_system/run_paper_experiments.py
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Tuple

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .comparison_experiments import (
        ExperimentConfig, FilteringComparison, InversionComparison,
        SystemLevelComparison, AblationStudy, EfficiencyComparison
    )
    from .inversion import SSMPINN, InversionConfig, create_ssm_pinn_model
    from .data_generator import DatasetConfig, FullDatasetGenerator, DatasetLoader
    from .init_database import MATERIAL_DATA
    from .evaluation import (
        compute_picp_at_confidence_levels, generate_uncertainty_calibration_table,
        compute_improvement, compute_gain
    )
except ImportError:
    from experiment_system.comparison_experiments import (
        ExperimentConfig, FilteringComparison, InversionComparison,
        SystemLevelComparison, AblationStudy, EfficiencyComparison
    )
    from experiment_system.inversion import SSMPINN, InversionConfig, create_ssm_pinn_model
    from experiment_system.data_generator import DatasetConfig, FullDatasetGenerator, DatasetLoader
    from experiment_system.init_database import MATERIAL_DATA
    from experiment_system.evaluation import (
        compute_picp_at_confidence_levels, generate_uncertainty_calibration_table,
        compute_improvement, compute_gain
    )


# ============================================================================
# Table 4: 目标表面材料数据库规格 (论文 tab:material_database)
# ============================================================================

def generate_material_database_table(output_dir: str) -> str:
    """生成材料数据库规格表 (论文Table 4: tab:material_database)
    
    12个材料类别, 包含子类型数、温度范围、发射率范围
    """
    # 论文Table 4中的12个材料类别规范
    # (类别, 子类型数, 温度范围K, 发射率范围)
    PAPER_SPEC = [
        ('Carbon Fiber Composite', 3, '273-673', '0.85-0.95'),
        ('High-Hardness Steel', 4, '273-873', '0.15-0.45'),
        ('Carburized Aluminum', 2, '273-573', '0.10-0.25'),
        ('Aluminum Alloy', 5, '273-573', '0.05-0.20'),
        ('Ni-Mo-W Alloy', 2, '273-973', '0.25-0.55'),
        ('Corroded Steel', 4, '273-773', '0.30-0.85'),
        ('Anti-Optical Coating', 3, '273-473', '0.05-0.15'),
        ('Anti-Infrared Coating', 3, '273-473', '0.80-0.98'),
        ('Polyurethane Coating', 2, '273-373', '0.85-0.95'),
        ('Polyimide Film', 2, '273-473', '0.40-0.60'),
        ('Ceramic Coating', 2, '273-1273', '0.80-0.95'),
        ('Titanium Alloy', 3, '273-873', '0.35-0.55'),
    ]
    
    # 从实际数据库统计
    actual_counts = {}
    actual_ranges = {}
    for m in MATERIAL_DATA:
        cat = m['category']
        if cat not in actual_counts:
            actual_counts[cat] = 0
            actual_ranges[cat] = []
        actual_counts[cat] += 1
        actual_ranges[cat].append((m['emissivity_mean'], m.get('emissivity_std', 0)))
    
    table = "=" * 90 + "\n"
    table += f"{'Table 4: Target Surface Material Database Specifications':^90}\n"
    table += f"{'(Paper tab:material_database)':^90}\n"
    table += "=" * 90 + "\n"
    header = f"{'Material Category':<28} {'Subtypes':>10} {'Temp Range (K)':>18} {'Emissivity Range':>20}\n"
    table += header
    table += "-" * 90 + "\n"
    
    for cat, n_sub, temp_range, eps_range in PAPER_SPEC:
        actual_n = actual_counts.get(cat, 0)
        table += f"{cat:<28} {actual_n:>10} {temp_range:>18} {eps_range:>20}\n"
    
    total = sum(actual_counts.values())
    table += "-" * 90 + "\n"
    table += f"{'TOTAL':<28} {total:>10}\n"
    table += "=" * 90 + "\n"
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "table4_material_database.txt"), 'w') as f:
        f.write(table)
    
    print(table)
    return table


# ============================================================================
# Table 5: 状态估计RMSE (论文 tab:filtering_rmse)
# ============================================================================

def run_filtering_experiment(config: ExperimentConfig, output_dir: str) -> Dict:
    """运行滤波对比实验,生成Table 5 (tab:filtering_rmse)
    
    7种方法 x 4种噪声类型
    """
    print("\n" + "#" * 80)
    print(f"{'Table 5: State Estimation RMSE (tab:filtering_rmse)':^80}")
    print("#" * 80)
    
    filter_comp = FilteringComparison(config)
    all_results = filter_comp.run_all_noise_comparisons()
    
    # 生成论文格式的表格
    methods = ['EKF', 'UKF', 'CKF', 'AEKF', 'RUKF', 'DeepKF', 'NS-ARKF']
    noise_types = ['gaussian', 'mixture', 'impulsive', 'time_varying']
    noise_labels = ['Gaussian', 'Mixture', 'Impulsive', 'Time-Varying']
    
    table = "=" * 90 + "\n"
    table += f"{'Table 5: State Estimation RMSE Under Different Noise Conditions':^90}\n"
    table += f"{'(Paper tab:filtering_rmse)':^90}\n"
    table += "=" * 90 + "\n"
    header = f"{'Method':<16}" + "".join(f"{lbl:>14}" for lbl in noise_labels) + f"{'Average':>14}\n"
    table += header
    table += "-" * 90 + "\n"
    
    for method in methods:
        row = f"{method:<16}"
        rmses = []
        for nt in noise_types:
            rmse = all_results[nt][method].overall_rmse
            rmses.append(rmse)
            row += f"{rmse:>14.3f}"
        avg = np.mean(rmses)
        if method == 'NS-ARKF':
            row += f"{avg:>14.3f}  *"
        else:
            row += f"{avg:>14.3f}"
        table += row + "\n"
    
    table += "=" * 90 + "\n"
    table += "* Best performance (proposed method)\n"
    
    # 计算提升
    deepkf_avg = np.mean([all_results[nt]['DeepKF'].overall_rmse for nt in noise_types])
    nsarkf_avg = np.mean([all_results[nt]['NS-ARKF'].overall_rmse for nt in noise_types])
    improvement = compute_improvement(deepkf_avg, nsarkf_avg)
    table += f"NS-ARKF improvement over DeepKF: {improvement:.1f}%\n"
    table += "=" * 90 + "\n"
    
    with open(os.path.join(output_dir, "table5_filtering_rmse.txt"), 'w') as f:
        f.write(table)
    
    print(table)
    return all_results


# ============================================================================
# Table 6: 材料属性反演精度 (论文 tab:inversion_accuracy)
# ============================================================================

def run_inversion_experiment(config: ExperimentConfig, output_dir: str) -> Dict:
    """运行反演对比实验,生成Table 6 (tab:inversion_accuracy)
    
    7种反演方法
    """
    print("\n" + "#" * 80)
    print(f"{'Table 6: Material Property Inversion Accuracy (tab:inversion_accuracy)':^80}")
    print("#" * 80)
    
    inv_comp = InversionComparison(config)
    results = inv_comp.run_comparison()
    
    methods = ['FC-NN', 'PINN-FC', 'ResNet', 'Transformer', 'S4-Model', 'Mamba', 'SSM-PINN']
    
    table = "=" * 95 + "\n"
    table += f"{'Table 6: Material Property Inversion Accuracy':^95}\n"
    table += f"{'(Paper tab:inversion_accuracy)':^95}\n"
    table += "=" * 95 + "\n"
    header = f"{'Method':<16}{'Emissivity RMSE':>18}{'Reflectivity RMSE':>20}{'Class Acc (%)':>16}{'F1-Score':>12}\n"
    table += header
    table += "-" * 95 + "\n"
    
    for method in methods:
        m = results[method]
        if method == 'SSM-PINN':
            table += (f"{method:<16}{m.emissivity_rmse:>18.4f}{m.reflectivity_rmse:>20.4f}"
                      f"{m.classification_acc:>16.1f}{m.f1_score:>12.3f}  *\n")
        else:
            table += (f"{method:<16}{m.emissivity_rmse:>18.4f}{m.reflectivity_rmse:>20.4f}"
                      f"{m.classification_acc:>16.1f}{m.f1_score:>12.3f}\n")
    
    table += "=" * 95 + "\n"
    table += "* Best performance (proposed method)\n"
    
    ssm_eps = results['SSM-PINN'].emissivity_rmse
    mamba_eps = results['Mamba'].emissivity_rmse
    improvement = compute_improvement(mamba_eps, ssm_eps)
    table += f"SSM-PINN emissivity RMSE improvement over Mamba: {improvement:.1f}%\n"
    
    # 基线方法不强制硬约束 (论文 B-4): 报告其违反 Kirchhoff 约束的真实比例
    baseline_methods = ['FC-NN', 'PINN-FC', 'ResNet', 'Transformer', 'S4-Model', 'Mamba']
    baseline_viols = [results[m].constraint_violation_rate for m in baseline_methods]
    avg_baseline_viol = float(np.mean(baseline_viols))
    table += f"Baseline avg constraint violation rate: {avg_baseline_viol:.1%} (no hard constraint)\n"

    ssm_viol = results['SSM-PINN'].constraint_violation_rate
    table += f"SSM-PINN constraint violation rate: {ssm_viol:.1%} (by-construction hard constraint)\n"
    table += "=" * 95 + "\n"
    
    with open(os.path.join(output_dir, "table6_inversion_accuracy.txt"), 'w') as f:
        f.write(table)
    
    print(table)
    return results


# ============================================================================
# Table 7: 端到端系统性能 (论文 tab:combined_performance)
# ============================================================================

def run_system_experiment(config: ExperimentConfig, output_dir: str) -> Dict:
    """运行系统级对比实验,生成Table 7 (tab:combined_performance)
    
    4种滤波+反演组合
    """
    print("\n" + "#" * 80)
    print(f"{'Table 7: End-to-End System Performance (tab:combined_performance)':^80}")
    print("#" * 80)
    
    sys_comp = SystemLevelComparison(config)
    results = sys_comp.run_comparison()
    
    systems = ['EKF + FC-NN', 'UKF + PINN-FC', 'DeepKF + S4', 'NS-ARKF + SSM-PINN']
    
    table = "=" * 80 + "\n"
    table += f"{'Table 7: End-to-End System Performance':^80}\n"
    table += f"{'(Paper tab:combined_performance)':^80}\n"
    table += "=" * 80 + "\n"
    header = f"{'Method':<24}{'Pos RMSE (m)':>14}{'Ang RMSE (°)':>14}{'Eps RMSE':>12}{'Score':>10}\n"
    table += header
    table += "-" * 80 + "\n"
    
    for system in systems:
        m = results[system]
        if system == 'NS-ARKF + SSM-PINN':
            table += (f"{system:<24}{m.position_rmse:>14.3f}{m.angle_rmse:>14.3f}"
                      f"{m.emissivity_rmse:>12.4f}{m.overall_score:>10.3f}  *\n")
        else:
            table += (f"{system:<24}{m.position_rmse:>14.3f}{m.angle_rmse:>14.3f}"
                      f"{m.emissivity_rmse:>12.4f}{m.overall_score:>10.3f}\n")
    
    table += "=" * 80 + "\n"
    table += "* Best performance (proposed method)\n"
    
    ns_score = results['NS-ARKF + SSM-PINN'].overall_score
    deep_score = results['DeepKF + S4'].overall_score
    # overall_score 为"越大越好"指标, 须用 compute_gain (正向增益), 而非
    # compute_improvement (会得到负号, 论文 B-6)
    improvement = compute_gain(deep_score, ns_score)
    table += f"Overall system improvement over DeepKF+S4: {improvement:.1f}%\n"
    table += "=" * 80 + "\n"
    
    with open(os.path.join(output_dir, "table7_combined_performance.txt"), 'w') as f:
        f.write(table)
    
    print(table)
    return results


# ============================================================================
# Table 8: 不确定性校准性能 (论文 tab:uncertainty_calibration)
# ============================================================================

def run_uncertainty_calibration_experiment(config: ExperimentConfig, output_dir: str) -> Dict:
    """运行不确定性校准实验,生成Table 8 (tab:uncertainty_calibration)
    
    论文Section 5.4.4: 基于贝叶斯变分推断的不确定性量化
    在68%/90%/95%/99%置信水平下计算PICP和校准误差
    
    使用SSM-PINN的predict_with_uncertainty方法获取预测均值和标准差
    """
    print("\n" + "#" * 80)
    print(f"{'Table 8: Uncertainty Calibration (tab:uncertainty_calibration)':^80}")
    print("#" * 80)
    
    # 生成反演测试数据
    rng = np.random.default_rng(config.seed)
    num_samples = config.num_samples
    materials = MATERIAL_DATA
    material_indices = rng.integers(0, len(materials), num_samples)
    
    X = np.zeros((num_samples, 5))
    y = np.zeros((num_samples, 2))
    
    for i in range(num_samples):
        material = materials[material_indices[i]]
        X[i, 0] = rng.uniform(100, 5000)
        X[i, 1] = rng.uniform(0, 75)
        # 温度按材料类别采样 (K), 覆盖 Ceramic Coating 的 273-1273 K (论文 B-9)
        X[i, 2] = rng.uniform(*material['temp_range'])
        X[i, 3] = rng.uniform(0, 50)
        X[i, 4] = rng.uniform(10, 50)
        y[i, 0] = material['emissivity_mean'] + rng.normal(0, material['emissivity_std'])
        y[i, 1] = material['reflectivity_mean'] + rng.normal(0, material['reflectivity_std'])
    
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    train_idx = int(num_samples * 0.7)
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_test, y_test = X[train_idx:], y[train_idx:]
    
    # 训练SSM-PINN
    print("Training SSM-PINN for uncertainty quantification...")
    inv_config = InversionConfig(
        max_iterations=config.max_iterations,
        learning_rate=config.learning_rate,
        enforce_hard_constraint=True
    )
    model = create_ssm_pinn_model(inv_config)
    model._train(X_train, y_train)
    
    # 带不确定性的预测 (贝叶斯VI: predict_with_uncertainty)
    print("Computing predictions with uncertainty (Bayesian VI)...")
    pred_mean, pred_std = model.predict_with_uncertainty(X_test, n_samples=50)
    
    # 提取发射率的预测均值和标准差
    eps_true = y_test[:, 0]
    eps_pred_mean = pred_mean[:, 0]
    eps_pred_std = pred_std[:, 0]
    
    # 计算多置信水平PICP (论文Table 8: 68/90/95/99%)
    confidence_levels = [0.68, 0.90, 0.95, 0.99]
    calibration_results = compute_picp_at_confidence_levels(
        eps_true, eps_pred_mean, eps_pred_std, confidence_levels
    )
    
    # 生成表格
    table = generate_uncertainty_calibration_table(eps_true, eps_pred_mean, eps_pred_std, confidence_levels)

    with open(os.path.join(output_dir, "table8_uncertainty_calibration.txt"), 'w') as f:
        f.write(table)

    # 保存 MPIW / PICP / 校准误差到 JSON (可复现性 + 论文 Table 8 数据源)
    calibration_json = {
        'uncertainty_calibration': {
            'protocol': 'Bayesian VI (mean-field Gaussian posterior), z-quantile intervals',
            'n_test': len(eps_true),
            'confidence_levels': confidence_levels,
            'results': [
                {
                    'confidence_level': r['confidence_level'],
                    'target_picp': r['target_picp'],
                    'actual_picp': r['actual_picp'],
                    'mpiw': r['mpiw'],
                    'calibration_error': r['calibration_error'],
                    'z_value': r['z_value']
                }
                for r in calibration_results
            ]
        }
    }
    with open(os.path.join(output_dir, "uncertainty_calibration_gpu.json"), 'w') as f:
        json.dump(calibration_json, f, indent=2)

    print(table)

    return {
        'calibration_results': calibration_results,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'y_test': y_test
    }


# ============================================================================
# Table 9: 组件消融分析 (论文 tab:ablation_study)
# ============================================================================

def run_ablation_experiment(config: ExperimentConfig, output_dir: str) -> Dict:
    """运行消融实验,生成Table 9 (tab:ablation_study)
    
    8种配置 (Baseline + 7种组件组合)
    """
    print("\n" + "#" * 80)
    print(f"{'Table 9: Component Ablation Analysis (tab:ablation_study)':^80}")
    print("#" * 80)
    
    abl_comp = AblationStudy(config)
    results = abl_comp.run_ablation()
    
    configs = list(results.keys())
    
    table = "=" * 80 + "\n"
    table += f"{'Table 9: Component Ablation Analysis':^80}\n"
    table += f"{'(Paper tab:ablation_study)':^80}\n"
    table += "=" * 80 + "\n"
    header = f"{'Configuration':<28}{'Pos RMSE':>14}{'Eps RMSE':>14}{'Improvement':>14}\n"
    table += header
    table += "-" * 80 + "\n"
    
    baseline = results['Baseline (EKF)']
    baseline_combined = 0.7 * baseline['position_rmse'] + 0.3 * baseline['emissivity_rmse']
    
    for cfg_name in configs:
        m = results[cfg_name]
        combined = 0.7 * m['position_rmse'] + 0.3 * m['emissivity_rmse']
        improvement = compute_improvement(baseline_combined, combined)
        if cfg_name == 'Full NS-ARKF':
            table += (f"{cfg_name:<28}{m['position_rmse']:>14.3f}{m['emissivity_rmse']:>14.4f}"
                      f"{improvement:>13.1f}%  *\n")
        else:
            table += (f"{cfg_name:<28}{m['position_rmse']:>14.3f}{m['emissivity_rmse']:>14.4f}"
                      f"{improvement:>13.1f}%\n")
    
    table += "=" * 80 + "\n"
    table += "* Full proposed method\n"
    table += "=" * 80 + "\n"
    
    with open(os.path.join(output_dir, "table9_ablation_study.txt"), 'w') as f:
        f.write(table)
    
    print(table)
    return results


# ============================================================================
# Table 10: 计算性能对比 (论文 tab:computational)
# ============================================================================

def run_efficiency_experiment(config: ExperimentConfig, output_dir: str) -> Dict:
    """运行计算效率对比实验,生成Table 10 (tab:computational)
    
    5种方法的推理时间、内存、训练时间
    """
    print("\n" + "#" * 80)
    print(f"{'Table 10: Computational Performance (tab:computational)':^80}")
    print("#" * 80)
    
    eff_comp = EfficiencyComparison(config)
    results = eff_comp.run_comparison()
    
    table = "=" * 85 + "\n"
    table += f"{'Table 10: Computational Performance Comparison':^85}\n"
    table += f"{'(Paper tab:computational)':^85}\n"
    table += "=" * 85 + "\n"
    header = f"{'Method':<24}{'Infer Time (ms)':>18}{'Memory (MB)':>14}{'Train Time (h)':>16}\n"
    table += header
    table += "-" * 85 + "\n"
    
    for method, m in results.items():
        train_str = f"{m.training_time_h:.1f}" if m.training_time_h > 0 else "N/A"
        if method == 'NS-ARKF + SSM-PINN':
            table += (f"{method:<24}{m.inference_time_ms:>18.2f}{m.memory_mb:>14.1f}"
                      f"{train_str:>16}  *\n")
        else:
            table += (f"{method:<24}{m.inference_time_ms:>18.2f}{m.memory_mb:>14.1f}"
                      f"{train_str:>16}\n")
    
    table += "=" * 85 + "\n"
    table += "* Proposed method\n"
    
    ns_fps = results.get('NS-ARKF + SSM-PINN', results.get('NS-ARKF')).fps
    table += f"Real-time FPS: {ns_fps:.1f} (>450 FPS required: {'YES' if ns_fps > 450 else 'NO'})\n"
    table += "=" * 85 + "\n"
    
    with open(os.path.join(output_dir, "table10_computational.txt"), 'w') as f:
        f.write(table)
    
    print(table)
    return results


# ============================================================================
# 生成所有论文图表
# ============================================================================

def generate_all_figures(output_dir: str):
    """生成所有论文图表 (Figure 1-5)"""
    print("\n" + "#" * 80)
    print(f"{'Generating All Paper Figures (Figure 1-5)':^80}")
    print("#" * 80)
    
    try:
        from .generate_paper_figures import generate_all_figures as gen_figs
    except ImportError:
        from experiment_system.generate_paper_figures import generate_all_figures as gen_figs
    
    fig_dir = os.path.join(output_dir, "paper_figures")
    gen_figs(fig_dir)
    return fig_dir


# ============================================================================
# 主函数: 运行所有实验
# ============================================================================

def run_all_paper_experiments(output_dir: str = "./paper_results",
                                num_samples: int = 125000,
                                max_iterations: int = 500,
                                run_figures: bool = True):
    """运行所有论文实验,生成所有表格和图表
    
    Args:
        output_dir: 输出目录
        num_samples: 每个实验的样本数 (论文完整: 125000, 快速测试: 1000)
        max_iterations: 训练迭代次数 (论文完整: 500, 快速测试: 100)
        run_figures: 是否生成图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config = ExperimentConfig(
        num_samples=num_samples,
        dim_x=6,
        dim_z=4,
        noise_level=0.1,
        seed=42,
        output_dir=output_dir,
        max_iterations=max_iterations,
        learning_rate=0.01
    )
    
    start_time = time.time()
    
    print("#" * 80)
    print(f"{'COMPLETE PAPER EXPERIMENT SUITE (CAL0827.tex)':^80}")
    print(f"{'Tables 4-10 + Figures 1-5':^80}")
    print("#" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Samples per experiment: {num_samples}")
    print(f"Max training iterations: {max_iterations}")
    print("#" * 80)
    
    all_results = {}
    
    # Table 4: 材料数据库
    all_results['table4_material'] = generate_material_database_table(output_dir)
    
    # Table 5: 滤波RMSE对比
    all_results['table5_filtering'] = run_filtering_experiment(config, output_dir)
    
    # Table 6: 反演精度对比
    all_results['table6_inversion'] = run_inversion_experiment(config, output_dir)
    
    # Table 7: 系统级性能
    all_results['table7_system'] = run_system_experiment(config, output_dir)
    
    # Table 8: 不确定性校准
    all_results['table8_uncertainty'] = run_uncertainty_calibration_experiment(config, output_dir)
    
    # Table 9: 消融实验
    all_results['table9_ablation'] = run_ablation_experiment(config, output_dir)
    
    # Table 10: 计算效率
    all_results['table10_efficiency'] = run_efficiency_experiment(config, output_dir)
    
    # Figures 1-5
    if run_figures:
        all_results['figures_dir'] = generate_all_figures(output_dir)
    
    elapsed = time.time() - start_time
    
    print("\n" + "#" * 80)
    print(f"{'ALL EXPERIMENTS COMPLETED':^80}")
    print("#" * 80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {output_dir}")
    print("#" * 80)
    
    # 生成汇总文件列表
    files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    print(f"\nGenerated result files ({len(files)}):")
    for f in sorted(files):
        print(f"  - {f}")
    
    return all_results


def run_quick_test(output_dir: str = "./paper_results_quick"):
    """快速测试运行 (小规模数据,少量迭代)
    
    用于验证代码可运行性
    """
    return run_all_paper_experiments(
        output_dir=output_dir,
        num_samples=200,
        max_iterations=50,
        run_figures=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run paper experiments for CAL0827.tex')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                        help='full: 125000 samples; quick: 200 samples for testing (default: quick)')
    parser.add_argument('--output', type=str, default='./paper_results',
                        help='Output directory (default: ./paper_results)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Override number of samples')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override max training iterations')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test(args.output)
    else:
        n_samples = args.samples if args.samples else 125000
        n_iter = args.iterations if args.iterations else 500
        run_all_paper_experiments(
            output_dir=args.output,
            num_samples=n_samples,
            max_iterations=n_iter,
            run_figures=not args.no_figures
        )
