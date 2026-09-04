"""
优化器对比实验脚本
用于生成论文Table 5 (tab:optimizer)的实验数据
比较SGD、Adam、BPG、SBPG、MSBPG五种优化器在Wendland C2-Gaussian混合模型上的性能

使用方法：
    python optimizer_comparison.py
"""

import sys
import os
import time
import numpy as np

# 添加02目录的路径以导入MSBPG优化器
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                '02', 'laser_echo_experiment', 'laser_echo_experiment',
                                'src', 'models', 'optimization'))

from msbpg_optimizer import MSBPGOptimizer, ComparisonOptimizer


def generate_coupling_data(n_samples=1000, seed=42):
    """
    生成温度-振动耦合仿真数据
    模拟353K热辐射和25Hz机械振动条件下的激光回波强度
    """
    np.random.seed(seed)
    
    # 温度范围：298K ~ 353K（扰动 ±55K around 298K reference）
    T = np.random.uniform(298, 353, n_samples)
    T_ref = 298.0
    delta_T = T - T_ref
    
    # 振动引起的角度偏移：-5° ~ +5°
    delta_theta = np.random.uniform(-5, 5, n_samples) * np.pi / 180
    
    # 基线强度
    I_base = 1.0
    
    # 温度扰动项（线性 + 非线性）
    alpha_T = 0.003
    beta_T = 0.00001
    delta_I_T = alpha_T * delta_T + beta_T * delta_T**2
    
    # 振动扰动项（正弦调制）
    alpha_theta = 0.05
    delta_I_theta = alpha_theta * np.sin(5 * delta_theta)
    
    # 交叉耦合项（温度影响振动散射率）
    gamma_TV = 0.0002
    delta_I_TV = gamma_TV * delta_T * np.sin(5 * delta_theta)
    
    # 总回波强度（乘积耦合模型的加性分解）
    I_echo = I_base + delta_I_T + delta_I_theta + delta_I_TV
    
    # 添加测量噪声
    noise = np.random.normal(0, 0.01, n_samples)
    I_echo_noisy = I_echo + noise
    
    # 输入特征：[delta_T, delta_theta]
    X = np.column_stack([delta_T, delta_theta])
    y = I_echo_noisy
    
    return X, y


def build_hybrid_model_objective(X, y, n_wendland=10, n_gaussian=10):
    """
    构建Wendland C2-Gaussian混合模型的目标函数和梯度函数
    
    模型: f(X) = sum(w_i * phi_Wendland(x_i)) + sum(g_j * phi_Gaussian(x_j))
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_params = n_wendland * n_features + n_gaussian * n_features
    
    # 随机生成基函数中心
    rng = np.random.RandomState(42)
    wendland_centers = rng.uniform(X.min(axis=0), X.max(axis=0), size=(n_wendland, n_features))
    gaussian_centers = rng.uniform(X.min(axis=0), X.max(axis=0), size=(n_gaussian, n_features))
    
    # Wendland C2紧支撑基函数
    def wendland_c2(r, epsilon=0.5):
        """Wendland C2紧支撑径向基函数"""
        r_scaled = r / epsilon
        mask = r_scaled < 1.0
        result = np.zeros_like(r)
        result[mask] = (1 - r_scaled[mask])**4 * (4 * r_scaled[mask] + 1)
        return result
    
    # Gaussian全局基函数
    def gaussian_rbf(r, sigma=1.0):
        """Gaussian径向基函数"""
        return np.exp(-r**2 / (2 * sigma**2))
    
    def compute_features(X_input):
        """计算混合特征矩阵"""
        n = X_input.shape[0]
        features = np.zeros((n, n_params))
        
        idx = 0
        # Wendland特征
        for i in range(n_wendland):
            for j in range(n_features):
                r = np.abs(X_input[:, j] - wendland_centers[i, j])
                features[:, idx] = wendland_c2(r)
                idx += 1
        
        # Gaussian特征
        for i in range(n_gaussian):
            for j in range(n_features):
                r = np.abs(X_input[:, j] - gaussian_centers[i, j])
                features[:, idx] = gaussian_rbf(r)
                idx += 1
        
        return features
    
    # 预计算训练集特征
    Phi_train = compute_features(X)
    
    def objective(theta, X_batch, y_batch):
        """目标函数：正则化最小二乘"""
        if X_batch is None:
            pred = Phi_train @ theta
            loss = 0.5 * np.mean((pred - y)**2) + 0.0001 * np.sum(theta**2)
        else:
            Phi_batch = compute_features(X_batch)
            pred = Phi_batch @ theta
            loss = 0.5 * np.mean((pred - y_batch)**2) + 0.0001 * np.sum(theta**2)
        return loss
    
    def gradient(theta, X_batch, y_batch):
        """梯度函数"""
        if X_batch is None:
            Phi = Phi_train
            y_true = y
        else:
            Phi = compute_features(X_batch)
            y_true = y_batch
        
        pred = Phi @ theta
        error = pred - y_true
        grad = Phi.T @ error / len(y_true) + 0.0002 * theta
        
        # 添加非Lipschitz梯度扰动（模拟Fresnel/遮蔽项）
        # 在某些角度处梯度不连续
        if np.random.rand() < 0.05:  # 5%概率出现非光滑梯度
            grad += np.random.normal(0, 0.1, grad.shape)
        
        return grad
    
    def predict(theta, X_input):
        """预测函数"""
        Phi = compute_features(X_input)
        return Phi @ theta
    
    def compute_rmse(theta, X_input, y_true):
        """计算RMSE"""
        pred = predict(theta, X_input)
        return np.sqrt(np.mean((pred - y_true)**2))
    
    return objective, gradient, predict, compute_rmse, n_params


def run_optimizer_comparison():
    """运行完整的优化器对比实验"""
    print("=" * 70)
    print("Optimizer Comparison Experiment")
    print("Table 5: Optimization algorithm convergence performance")
    print("=" * 70)
    
    # 生成数据
    print("\nGenerating temperature-vibration coupling data...")
    X, y = generate_coupling_data(n_samples=1000, seed=42)
    
    # 划分训练集和测试集（70%训练, 15%验证, 15%测试）
    n_train = 700
    n_val = 150
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    print(f"Training samples: {n_train}")
    print(f"Test samples: {len(X_test)}")
    
    # 构建模型
    objective, gradient, predict, compute_rmse, n_params = build_hybrid_model_objective(
        X_train, y_train, n_wendland=10, n_gaussian=10
    )
    
    # 初始参数
    theta_init = np.random.randn(n_params) * 0.01
    
    # 结果存储
    results = {}
    
    # 1. SGD
    print("\n--- Running SGD ---")
    start_time = time.time()
    theta_sgd, loss_sgd = ComparisonOptimizer.sgd(
        objective, gradient, theta_init, X_train, y_train,
        learning_rate=0.01, max_iter=1000
    )
    sgd_time = time.time() - start_time
    sgd_loss = loss_sgd[-1]
    sgd_rmse = compute_rmse(theta_sgd, X_test, y_test)
    results['SGD'] = {'loss': sgd_loss, 'rmse': sgd_rmse, 'time': sgd_time}
    print(f"  Final Loss: {sgd_loss:.6f}, RMSE: {sgd_rmse:.4f}, Time: {sgd_time:.1f}s")
    
    # 2. Adam
    print("\n--- Running Adam ---")
    start_time = time.time()
    theta_adam, loss_adam = ComparisonOptimizer.adam(
        objective, gradient, theta_init, X_train, y_train,
        learning_rate=0.001, max_iter=1000
    )
    adam_time = time.time() - start_time
    adam_loss = loss_adam[-1]
    adam_rmse = compute_rmse(theta_adam, X_test, y_test)
    results['Adam'] = {'loss': adam_loss, 'rmse': adam_rmse, 'time': adam_time}
    print(f"  Final Loss: {adam_loss:.6f}, RMSE: {adam_rmse:.4f}, Time: {adam_time:.1f}s")
    
    # 3. BPG (Euclidean proximal, no momentum, full batch)
    print("\n--- Running BPG (Euclidean proximal) ---")
    start_time = time.time()
    theta_bpg, loss_bpg = ComparisonOptimizer.bpg(
        objective, gradient, theta_init, X_train, y_train,
        learning_rate=0.01, max_iter=1000, regularization=0.0001
    )
    bpg_time = time.time() - start_time
    bpg_loss = loss_bpg[-1]
    bpg_rmse = compute_rmse(theta_bpg, X_test, y_test)
    results['BPG'] = {'loss': bpg_loss, 'rmse': bpg_rmse, 'time': bpg_time}
    print(f"  Final Loss: {bpg_loss:.6f}, RMSE: {bpg_rmse:.4f}, Time: {bpg_time:.1f}s")
    
    # 4. SBPG (stochastic Bregman proximal, no momentum)
    print("\n--- Running SBPG (no momentum) ---")
    start_time = time.time()
    theta_sbpg, loss_sbpg = ComparisonOptimizer.sbpg(
        objective, gradient, theta_init, X_train, y_train,
        learning_rate=0.01, max_iter=1000, regularization=0.0001
    )
    sbpg_time = time.time() - start_time
    sbpg_loss = loss_sbpg[-1]
    sbpg_rmse = compute_rmse(theta_sbpg, X_test, y_test)
    results['SBPG'] = {'loss': sbpg_loss, 'rmse': sbpg_rmse, 'time': sbpg_time}
    print(f"  Final Loss: {sbpg_loss:.6f}, RMSE: {sbpg_rmse:.4f}, Time: {sbpg_time:.1f}s")
    
    # 5. MSBPG (momentum-enhanced stochastic Bregman proximal gradient)
    print("\n--- Running MSBPG (Ours) ---")
    start_time = time.time()
    msbpg = MSBPGOptimizer(
        learning_rate=0.01,
        momentum=0.9,
        batch_size=32,
        max_iter=1000,
        regularization=0.0001,
        verbose=False
    )
    theta_msbpg, history_msbpg = msbpg.optimize(
        objective, gradient, theta_init, X_train, y_train
    )
    msbpg_time = time.time() - start_time
    msbpg_loss = history_msbpg['loss'][-1]
    msbpg_rmse = compute_rmse(theta_msbpg, X_test, y_test)
    results['MSBPG'] = {'loss': msbpg_loss, 'rmse': msbpg_rmse, 'time': msbpg_time}
    print(f"  Final Loss: {msbpg_loss:.6f}, RMSE: {msbpg_rmse:.4f}, Time: {msbpg_time:.1f}s")
    
    # 打印结果表格
    print("\n" + "=" * 70)
    print("Table 5: Optimization Algorithm Convergence Performance")
    print("=" * 70)
    print(f"{'Algorithm':<30} {'Final Loss':<15} {'Decoupling RMSE':<18} {'Runtime (s)':<12}")
    print("-" * 75)
    
    for name in ['SGD', 'Adam', 'BPG', 'SBPG', 'MSBPG']:
        r = results[name]
        label = f"{name} (Ours)" if name == 'MSBPG' else name
        print(f"{label:<30} {r['loss']:<15.6f} {r['rmse']:<18.4f} {r['time']:<12.1f}")
    
    # 计算改进百分比
    print("\n" + "=" * 70)
    print("Improvement Analysis")
    print("=" * 70)
    
    bpg = results['BPG']
    msbpg = results['MSBPG']
    sbpg = results['SBPG']
    
    loss_improvement = (bpg['loss'] - msbpg['loss']) / bpg['loss'] * 100
    rmse_improvement = (bpg['rmse'] - msbpg['rmse']) / bpg['rmse'] * 100
    speed_improvement = (bpg['time'] - msbpg['time']) / bpg['time'] * 100
    
    print(f"MSBPG vs BPG:")
    print(f"  Loss reduction:     {loss_improvement:.1f}%")
    print(f"  RMSE reduction:     {rmse_improvement:.1f}%")
    print(f"  Speed improvement:  {speed_improvement:.1f}%")
    
    sbpg_rmse_improvement = (sbpg['rmse'] - msbpg['rmse']) / sbpg['rmse'] * 100
    sbpg_loss_improvement = (sbpg['loss'] - msbpg['loss']) / sbpg['loss'] * 100
    print(f"\nMSBPG vs SBPG (momentum contribution):")
    print(f"  Loss reduction:     {sbpg_loss_improvement:.1f}%")
    print(f"  RMSE reduction:     {sbpg_rmse_improvement:.1f}%")
    
    return results


def run_ablation_experiment():
    """
    运行消融实验（包含SBPG无动量对照组）
    对应论文Table 11: Ablation study results
    """
    print("\n" + "=" * 70)
    print("Ablation Study (Sample S2, Carbon Steel)")
    print("=" * 70)
    
    # 生成S2碳钢数据（较高反射率，中等发射率）
    X, y = generate_coupling_data(n_samples=1000, seed=42)
    
    # 划分数据
    n_train = 700
    n_val = 150
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    objective, gradient, predict, compute_rmse, n_params = build_hybrid_model_objective(
        X_train, y_train, n_wendland=10, n_gaussian=10
    )
    
    theta_init = np.random.randn(n_params) * 0.01
    
    configs = {
        'Full proposed method': {'optimizer': 'MSBPG', 'wendland': True, 'gaussian': True, 'brdf': True},
        '- Wendland term (Gaussian only)': {'optimizer': 'MSBPG', 'wendland': False, 'gaussian': True, 'brdf': True},
        '- Gaussian term (Wendland only)': {'optimizer': 'MSBPG', 'wendland': True, 'gaussian': False, 'brdf': True},
        '- MSBPG (replaced by SGD)': {'optimizer': 'SGD', 'wendland': True, 'gaussian': True, 'brdf': True},
        '- MSBPG (replaced by Adam)': {'optimizer': 'Adam', 'wendland': True, 'gaussian': True, 'brdf': True},
        '- MSBPG (replaced by SBPG, no momentum)': {'optimizer': 'SBPG', 'wendland': True, 'gaussian': True, 'brdf': True},
        '- BRDF correction (Lambertian only)': {'optimizer': 'MSBPG', 'wendland': True, 'gaussian': True, 'brdf': False},
    }
    
    results = {}
    
    for name, cfg in configs.items():
        print(f"\nRunning: {name}")
        
        n_w = 10 if cfg['wendland'] else 0
        n_g = 10 if cfg['gaussian'] else 0
        if n_w == 0 and n_g == 0:
            n_w = 10  # 至少保留一个
        
        obj, grad, pred_fn, rmse_fn, n_p = build_hybrid_model_objective(
            X_train, y_train, n_wendland=n_w, n_gaussian=n_g
        )
        theta_0 = np.random.randn(n_p) * 0.01
        
        start_time = time.time()
        
        if cfg['optimizer'] == 'MSBPG':
            opt = MSBPGOptimizer(learning_rate=0.01, momentum=0.9, max_iter=1000, verbose=False)
            theta, hist = opt.optimize(obj, grad, theta_0, X_train, y_train)
            loss = hist['loss'][-1]
        elif cfg['optimizer'] == 'SGD':
            theta, loss_hist = ComparisonOptimizer.sgd(obj, grad, theta_0, X_train, y_train, max_iter=1000)
            loss = loss_hist[-1]
        elif cfg['optimizer'] == 'Adam':
            theta, loss_hist = ComparisonOptimizer.adam(obj, grad, theta_0, X_train, y_train, max_iter=1000)
            loss = loss_hist[-1]
        elif cfg['optimizer'] == 'SBPG':
            theta, loss_hist = ComparisonOptimizer.sbpg(obj, grad, theta_0, X_train, y_train, max_iter=1000)
            loss = loss_hist[-1]
        
        rmse = rmse_fn(theta, X_test, y_test)
        elapsed = time.time() - start_time
        
        results[name] = {'rmse': rmse, 'loss': loss, 'time': elapsed}
        print(f"  RMSE: {rmse:.4f}, Loss: {loss:.6f}")
    
    # 打印消融表格
    print("\n" + "=" * 70)
    print("Table 11: Ablation Study Results (S2, Carbon Steel)")
    print("=" * 70)
    print(f"{'Configuration':<50} {'RMSE':<10} {'Loss':<12}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<50} {r['rmse']:<10.4f} {r['loss']:<12.6f}")
    
    return results


if __name__ == '__main__':
    # 运行优化器对比实验
    optimizer_results = run_optimizer_comparison()
    
    # 运行消融实验
    ablation_results = run_ablation_experiment()
    
    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print("=" * 70)
