import numpy as np
from numpy.linalg import inv, det, pinv, eigvals, norm
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .progress import ProgressBar

def safe_inv(A: np.ndarray) -> np.ndarray:
    """安全矩阵求逆，处理奇异矩阵"""
    try:
        return inv(A)
    except np.linalg.LinAlgError:
        return pinv(A)

def safe_log_det(A: np.ndarray) -> float:
    """安全计算对数行列式，处理奇异或接近奇异矩阵"""
    try:
        eigenvalues = eigvals(A)
        positive_eigs = eigenvalues[np.real(eigenvalues) > 1e-10]
        if len(positive_eigs) == 0:
            return np.log(1e-20)
        return np.sum(np.log(np.real(positive_eigs)))
    except:
        return np.log(1e-20)

def is_positive_definite(A: np.ndarray) -> bool:
    """检查矩阵是否正定"""
    try:
        eigenvalues = eigvals(A)
        return np.all(np.real(eigenvalues) > 1e-10)
    except:
        return False

def make_positive_definite(A: np.ndarray) -> np.ndarray:
    """确保矩阵正定"""
    if is_positive_definite(A):
        return A

    eps = 1e-6
    A_sym = (A + A.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(A_sym)
    eigenvalues = np.maximum(eigenvalues, eps)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _eval_H(H, x):
    """观测模型 Jacobian 求值。

    ``H`` 既可以是固定观测矩阵 (线性观测, 透传), 也可以是可调用 Jacobian
    ``H(x)`` (非线性观测, EKF 系按当前状态 x 做解析线性化)。UKF/CKF/RUKF 不
    经过本函数 —— 它们用无迹/容积变换直接传播可调用观测函数 ``h``, 无需 Jacobian。
    """
    return H(x) if callable(H) else H

@dataclass
class FilterState:
    """滤波状态"""
    x_hat: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray

class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器"""
    
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
    
    def predict(self, F: np.ndarray, B: np.ndarray = None, u: np.ndarray = None):
        """预测步骤"""
        if B is None or u is None:
            self.x_hat = F @ self.x_hat
        else:
            self.x_hat = F @ self.x_hat + B @ u
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z: np.ndarray, H, h: callable = None):
        """更新步骤 (H 可为固定矩阵或可调用 Jacobian H(x))"""
        Hm = _eval_H(H, self.x_hat)
        if h is not None:
            z_pred = h(self.x_hat)
        else:
            z_pred = Hm @ self.x_hat

        y = z - z_pred
        S = Hm @ self.P @ Hm.T + self.R + np.eye(self.dim_z) * 1e-6
        K = self.P @ Hm.T @ safe_inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(self.dim_x) - K @ Hm) @ self.P
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

class UnknownInputFilter:
    """未知输入滤波器"""
    
    def __init__(self, dim_x: int, dim_z: int, dim_d: int = 1):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_d = dim_d
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
        self.E = np.zeros((dim_x, dim_d))
        self.E[-dim_d:, :] = np.eye(dim_d)
    
    def predict(self, F: np.ndarray, B: np.ndarray = None, u: np.ndarray = None):
        """预测步骤"""
        if B is None or u is None:
            self.x_hat = F @ self.x_hat
        else:
            self.x_hat = F @ self.x_hat + B @ u
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z: np.ndarray, H, h: callable = None):
        """更新步骤 (H 可为固定矩阵或可调用 Jacobian H(x))"""
        Hm = _eval_H(H, self.x_hat)
        if h is not None:
            z_pred = h(self.x_hat)
        else:
            z_pred = Hm @ self.x_hat

        S = Hm @ self.P @ Hm.T + self.R + np.eye(self.dim_z) * 1e-6
        try:
            ETHEHS = self.E.T @ Hm.T @ safe_inv(S) @ Hm @ self.E
            ETHEHS += np.eye(self.dim_d) * 1e-6
            M = safe_inv(ETHEHS) @ self.E.T @ Hm.T @ safe_inv(S)
            d_hat = M @ (z - z_pred)
            K = self.P @ Hm.T @ safe_inv(S)
            self.x_hat = self.x_hat + K @ (z - z_pred - Hm @ self.E @ d_hat)
            self.P = (np.eye(self.dim_x) - K @ Hm) @ self.P
        except:
            K = self.P @ Hm.T @ safe_inv(S)
            self.x_hat = self.x_hat + K @ (z - Hm @ self.x_hat)
            self.P = (np.eye(self.dim_x) - K @ Hm) @ self.P
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

class HippoOptimizer:
    """河马优化算法"""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], pop_size: int = 50):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.population = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (pop_size, dim)
        )
    
    def optimize(self, objective: callable, max_iter: int = 100, show_progress: bool = True) -> np.ndarray:
        """优化"""
        progress = None
        if show_progress:
            progress = ProgressBar(total=max_iter, description="Hippo Optimization")
        
        for iteration in range(max_iter):
            fitness = np.array([objective(x) for x in self.population])
            best_idx = np.argmin(fitness)
            best = self.population[best_idx]
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                rand_idx = np.random.randint(0, self.pop_size)
                rand_x = self.population[rand_idx]
                self.population[i] = self.population[i] + r1 * (best - self.population[i]) + \
                                   r2 * (rand_x - self.population[i])
                
                for j in range(self.dim):
                    self.population[i, j] = max(self.bounds[j][0], min(self.population[i, j], self.bounds[j][1]))
            
            if progress:
                progress.update(iteration + 1, f"Iter {iteration+1}/{max_iter}, Best: {objective(best):.4f}")
        
        if progress:
            progress.finish()
        
        fitness = np.array([objective(x) for x in self.population])
        best_idx = np.argmin(fitness)
        return self.population[best_idx]

class BlackKiteOptimizer:
    """黑鸢优化算法"""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], pop_size: int = 50):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.population = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (pop_size, dim)
        )
    
    def optimize(self, objective: callable, max_iter: int = 100, show_progress: bool = True) -> np.ndarray:
        """优化"""
        progress = None
        if show_progress:
            progress = ProgressBar(total=max_iter, description="Black Kite Optimization")
        
        for t in range(max_iter):
            fitness = np.array([objective(x) for x in self.population])
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)
            best = self.population[best_idx]
            worst = self.population[worst_idx]
            
            alpha = 2 * (1 - t / max_iter)
            beta = np.random.rand()
            
            for i in range(self.pop_size):
                l = 2 * np.random.rand() - 1
                b = 1
                self.population[i] = self.population[i] + \
                                   alpha * (best - self.population[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + \
                                   beta * (self.population[i] - worst)
                
                for j in range(self.dim):
                    self.population[i, j] = max(self.bounds[j][0], min(self.population[i, j], self.bounds[j][1]))
            
            if progress:
                progress.update(t + 1, f"Iter {t+1}/{max_iter}, Best: {objective(best):.4f}")
        
        if progress:
            progress.finish()
        
        fitness = np.array([objective(x) for x in self.population])
        best_idx = np.argmin(fitness)
        return self.population[best_idx]

class HBKFO:
    """河马-黑鸢融合优化器"""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], pop_size: int = 50):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.hoa = HippoOptimizer(dim, bounds, pop_size)
        self.bka = BlackKiteOptimizer(dim, bounds, pop_size)
    
    def optimize(self, objective: callable, max_iter: int = 100, p_switch: float = 0.7, show_progress: bool = True) -> np.ndarray:
        """融合优化"""
        progress = None
        if show_progress:
            progress = ProgressBar(total=max_iter, description="HBKFO Fusion Optimization")
        
        for t in range(max_iter):
            current_p = p_switch * (1 - t / max_iter)
            
            if np.random.rand() < current_p:
                result = self.hoa.optimize(objective, 1, show_progress=False)
            else:
                result = self.bka.optimize(objective, 1, show_progress=False)
            
            if progress:
                progress.update(t + 1, f"Iter {t+1}/{max_iter}, Mode: {'Hippo' if current_p > 0.5 else 'BlackKite'}")
        
        if progress:
            progress.finish()
        
        combined_pop = np.vstack([self.hoa.population, self.bka.population])
        fitness = np.array([objective(x) for x in combined_pop])
        best_idx = np.argmin(fitness)
        return combined_pop[best_idx]

class IFHBFNNFilter:
    """改进模糊超径向基函数神经网络滤波组件
    
    论文Section 3.4: 用于非线性干扰拟合的IFHBFNN
    四层结构: 输入层 -> 模糊隶属层 -> 模糊推理层 -> 超RBF输出层
    训练: Nesterov加速梯度 + AdamW自适应学习率
    """
    
    def __init__(self, dim_x: int, dim_z: int, n_rules: int = 8, hidden_dim: int = 32):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.n_rules = n_rules
        self.hidden_dim = hidden_dim
        
        np.random.seed(42)
        # Layer 2: 模糊隶属层参数 (高斯隶属函数)
        self.centers = np.random.randn(dim_z, n_rules) * 0.5
        self.sigmas = np.ones((dim_z, n_rules)) * 0.5
        
        # Layer 4: 超RBF参数
        self.xi = np.random.randn(n_rules, dim_z) * 0.5  # RBF中心
        self.gamma = np.ones(n_rules) * 1.0  # RBF宽度
        self.A_metric = np.array([np.eye(dim_z) for _ in range(n_rules)])  # Mahalanobis度量矩阵
        
        # 输出权重
        self.W_out = np.random.randn(n_rules, dim_x) * 0.01
        
        # 归一化参数
        self.mu_z = np.zeros(dim_z)
        self.D_diag = np.ones(dim_z)
        
        # Nesterov加速梯度 + AdamW参数
        self.beta_nesterov = 0.9
        self.velocity = {}
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        self.beta1_adam = 0.9
        self.beta2_adam = 0.999
        self.eps_adam = 1e-8
        self.lr = 0.001
        self.weight_decay = 0.01
        
        self._init_optimizer_state()
        self.interference_history: List[np.ndarray] = []
        self.max_history = 100
    
    def _init_optimizer_state(self):
        """初始化优化器状态"""
        params = ['W_out', 'centers', 'sigmas', 'xi', 'gamma']
        for p in params:
            self.velocity[p] = np.zeros_like(getattr(self, p))
            self.adam_m[p] = np.zeros_like(getattr(self, p))
            self.adam_v[p] = np.zeros_like(getattr(self, p))
    
    def _normalize(self, z: np.ndarray) -> np.ndarray:
        """Layer 1: 输入归一化 z_norm = D^{-1}(z - mu_z)"""
        return (z - self.mu_z) / (self.D_diag + 1e-10)
    
    def _fuzzy_membership(self, z_norm: np.ndarray) -> np.ndarray:
        """Layer 2: 模糊隶属度计算 (高斯隶属函数)
        mu_ij(z_i) = exp(-(z_i - c_ij)^2 / (2*sigma_ij^2))
        """
        memberships = np.zeros((self.dim_z, self.n_rules))
        for i in range(self.dim_z):
            diff = z_norm[i] - self.centers[i]
            memberships[i] = np.exp(-diff ** 2 / (2 * self.sigmas[i] ** 2 + 1e-10))
        return memberships
    
    def _fuzzy_inference(self, memberships: np.ndarray) -> np.ndarray:
        """Layer 3: 模糊推理 (乘积T-范数)
        phi^(l) = prod_i mu_{i,l_i}(z_i)
        """
        firing_strengths = np.ones(self.n_rules)
        for i in range(self.dim_z):
            firing_strengths *= memberships[i]
        # 归一化
        firing_strengths = firing_strengths / (np.sum(firing_strengths) + 1e-10)
        return firing_strengths
    
    def _hyper_rbf(self, z_norm: np.ndarray) -> np.ndarray:
        """Layer 4: 超径向基函数
        h^(l)(z) = exp(-gamma^(l) * ||z - xi^(l)||_A^2)
        使用Mahalanobis距离
        """
        basis = np.zeros(self.n_rules)
        for l in range(self.n_rules):
            diff = z_norm - self.xi[l]
            mahalanobis_sq = diff @ self.A_metric[l] @ diff
            basis[l] = np.exp(-self.gamma[l] * mahalanobis_sq)
        return basis
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        """前向传播: 计算非线性干扰估计 x̂^IFHBFNN"""
        z_norm = self._normalize(z)
        
        memberships = self._fuzzy_membership(z_norm)
        firing_strengths = self._fuzzy_inference(memberships)
        rbf_basis = self._hyper_rbf(z_norm)
        
        # 组合: x̂ = sum_l w^(l) * phi^(l) * h^(l)(z)
        combined = firing_strengths * rbf_basis
        interference = self.W_out.T @ combined
        
        return interference
    
    def update_online(self, z: np.ndarray, residual: np.ndarray):
        """在线更新 (Nesterov加速梯度 + AdamW)
        
        residual = x_true - x_uif_estimate (用于训练IFHBFNN拟合残差)
        """
        z_norm = self._normalize(z)
        
        # 前向传播缓存
        memberships = self._fuzzy_membership(z_norm)
        firing_strengths = self._fuzzy_inference(memberships)
        rbf_basis = self._hyper_rbf(z_norm)
        combined = firing_strengths * rbf_basis
        
        # 预测
        pred = self.W_out.T @ combined
        
        # 误差
        error = pred - residual
        
        # 记录历史
        self.interference_history.append(residual.copy())
        if len(self.interference_history) > self.max_history:
            self.interference_history.pop(0)
        
        # 更新归一化参数 (滑动平均)
        if len(self.interference_history) > 1:
            history = np.array(self.interference_history[-min(20, len(self.interference_history)):])
            self.mu_z = 0.9 * self.mu_z + 0.1 * np.mean(history, axis=0)[:self.dim_z] if history.shape[1] >= self.dim_z else self.mu_z
        
        # 梯度计算 (对W_out)
        grad_W_out = np.outer(combined, error)
        
        # Nesterov加速梯度 + AdamW更新
        self._adamw_nesterov_update('W_out', grad_W_out)
        
        # 更新RBF中心 (简化)
        grad_xi = np.zeros_like(self.xi)
        for l in range(self.n_rules):
            diff = z_norm - self.xi[l]
            # error @ self.W_out[l] 是标量(点积), 需用括号确保优先计算
            grad_xi[l] = 2 * self.gamma[l] * diff * (self.A_metric[l] @ diff) * combined[l] * (error @ self.W_out[l])
        self._adamw_nesterov_update('xi', grad_xi)
    
    def _adamw_nesterov_update(self, param_name: str, grad: np.ndarray):
        """AdamW + Nesterov加速梯度更新"""
        self.adam_t += 1
        
        # Nesterov lookahead
        lookahead = getattr(self, param_name) + self.beta_nesterov * self.velocity[param_name]
        
        # Adam矩估计
        self.adam_m[param_name] = self.beta1_adam * self.adam_m[param_name] + (1 - self.beta1_adam) * grad
        self.adam_v[param_name] = self.beta2_adam * self.adam_v[param_name] + (1 - self.beta2_adam) * (grad ** 2)
        
        # 偏差校正
        m_hat = self.adam_m[param_name] / (1 - self.beta1_adam ** self.adam_t)
        v_hat = self.adam_v[param_name] / (1 - self.beta2_adam ** self.adam_t)
        
        # Nesterov速度更新 + AdamW参数更新
        self.velocity[param_name] = self.beta_nesterov * self.velocity[param_name] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)
        
        # 解耦权重衰减
        param = getattr(self, param_name)
        param = param + self.velocity[param_name] - self.lr * self.weight_decay * param
        
        setattr(self, param_name, param)


class NSARKF:
    """神经感知自适应鲁棒卡尔曼滤波器 (NS-ARKF)

    论文Section 3: 三组件协同框架
    x̂_k = x̂_k^UIF( Q̂_k, R̂_k ) + x̂_k^IFHBFNN
    - UIF: 未知输入滤波器, 在给定噪声协方差下做状态估计
    - IFHBFNN: 改进模糊超RBF神经网络, 非线性干扰拟合 (输出状态修正项 x̂^IFHBFNN)
    - HBKFO: 河马-黑鸢融合优化器, 对新息序列 {ν_k} 在线优化噪声协方差
            Q̂_k, R̂_k 并回灌给 UIF 的卡尔曼增益 (不作为独立状态加项)
    """
    
    def __init__(self, dim_x: int, dim_z: int, use_ifhbfnn: bool = True, use_hbkfo: bool = True):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.use_ifhbfnn = use_ifhbfnn
        self.use_hbkfo = use_hbkfo
        
        # 组件1: 未知输入滤波器
        self.uif = UnknownInputFilter(dim_x, dim_z)
        
        # 组件2: IFHBFNN非线性干扰拟合
        self.ifhbfnn = IFHBFNNFilter(dim_x, dim_z) if use_ifhbfnn else None
        
        # 组件3: HBKFO极端噪声估计
        self.hbkfo = HBKFO(dim_x + dim_z, [(1e-6, 0.1)] * (dim_x + dim_z)) if use_hbkfo else None
        
        self.innovation_history: List[np.ndarray] = []
        self.noise_adaptation_interval = 10
        self.adaptation_counter = 0
        self.ifhbfnn_correction = np.zeros(dim_x)
        # 鲁棒新息门控开关 (消融实验用): 关闭后退化为非鲁棒自适应滤波
        self.robust_gate = True
    
    def predict(self, F: np.ndarray):
        """预测步骤"""
        self.uif.predict(F)
        # IFHBFNN的预测修正(基于历史干扰模式)
        if self.ifhbfnn is not None and len(self.innovation_history) > 0:
            # 使用最近创新序列预测干扰
            recent_innovation = self.innovation_history[-1]
            self.ifhbfnn_correction = self.ifhbfnn.forward(recent_innovation[:self.dim_z])
    
    def update(self, z: np.ndarray, H, h: callable = None):
        """更新步骤 (H 可为固定矩阵或可调用 Jacobian H(x))"""
        Hm = _eval_H(H, self.uif.x_hat)
        if h is not None:
            z_pred = h(self.uif.x_hat)
        else:
            z_pred = Hm @ self.uif.x_hat

        y = z - z_pred  # 创新/残差
        self.innovation_history.append(y)
        if len(self.innovation_history) > 100:
            self.innovation_history.pop(0)
        
        # HBKFO自适应噪声协方差估计
        self.adaptation_counter += 1
        if self.use_hbkfo and self.adaptation_counter >= self.noise_adaptation_interval and len(self.innovation_history) >= 5:
            self._adapt_noise_covariance(Hm)
            self.adaptation_counter = 0
        
        # 鲁棒新息门控 (IGG-III / Huber 重加权, 见 Chang 2014):
        # 计算归一化新息平方 (马氏距离). 当其超过 chi^2 阈值时判定为脉冲/离群
        # 观测, 按 IGG-III 权重因子对本步测量协方差 R 做膨胀, 抑制离群点对
        # 卡尔曼增益的污染. 该步骤是 NS-ARKF "鲁棒" 性质的直接实现, 不引入新的
        # 优化器, 属于成熟的鲁棒卡尔曼滤波技术.
        R_backup = self.uif.R.copy()
        try:
            if self.robust_gate:
                S = Hm @ self.uif.P @ Hm.T + self.uif.R + np.eye(self.dim_z) * 1e-6
                S_inv = safe_inv(make_positive_definite(S))
                nis = float(y.T @ S_inv @ y)          # normalised innovation squared
                # IGG-III thresholds; k1 multiplier is overridable for the
                # sensitivity sweep (default 6.0, see sensitivity_analysis.py).
                k0 = self.dim_z * 1.5                  # chi^2-style acceptance region
                k1 = self.dim_z * getattr(self, "_gate_k1_mult", 6.0)  # full-rejection region
                if nis > k0:
                    if nis >= k1:
                        weight = k0 / (nis + 1e-12)    # heavy down-weighting
                    else:
                        # IGG-III smooth transition
                        weight = (k0 / (nis + 1e-12)) * ((k1 - nis) / (k1 - k0)) ** 2
                    weight = float(np.clip(weight, 1e-3, 1.0))
                    self.uif.R = self.uif.R / weight   # inflate R -> shrink gain
        except Exception:
            pass
        
        # UIF更新 (使用鲁棒膨胀后的 R)
        self.uif.update(z, Hm, h)
        self.uif.R = R_backup                       # 恢复自适应估计的 R
        
        # IFHBFNN在线更新(拟合UIF残差中的非线性干扰)
        if self.ifhbfnn is not None and len(self.innovation_history) >= 2:
            # 使用创新序列作为干扰信号训练IFHBFNN
            interference_signal = np.zeros(self.dim_x)
            interference_signal[:self.dim_z] = y
            self.ifhbfnn.update_online(z, interference_signal)
            self.ifhbfnn_correction = self.ifhbfnn.forward(z)
    
    def _adapt_noise_covariance(self, H: np.ndarray):
        """HBKFO自适应噪声协方差估计.

        为避免脉冲/离群新息污染协方差估计, 先用中位数绝对偏差 (MAD) 对新息窗口
        做鲁棒截断, 再在截断后的新息上做元启发式拟合. 这样 HBKFO 只从代表"正常"
        观测统计的新息中学习 Q, R, 而离群点交由更新步的鲁棒门控处理.
        """
        raw = np.array(self.innovation_history[-10:])
        # 逐维 MAD 截断 (3 倍 MAD 之外的新息视为离群, 裁剪回边界)
        med = np.median(raw, axis=0)
        mad = np.median(np.abs(raw - med), axis=0) * 1.4826 + 1e-9
        clipped = np.clip(raw, med - 3.0 * mad, med + 3.0 * mad)
        innovations = clipped[-5:]

        P = make_positive_definite(self.uif.P)
        
        def objective(x):
            Q_diag = np.maximum(x[:self.dim_x], 1e-8)
            R_diag = np.maximum(x[self.dim_x:], 1e-8)
            
            Q = np.diag(Q_diag)
            R = np.diag(R_diag)
            
            S = H @ P @ H.T + R + np.eye(self.dim_z) * 1e-8
            S = make_positive_definite(S)
            
            try:
                S_inv = safe_inv(S)
                cost = 0.0
                for y in innovations:
                    mahalanobis = y.T @ S_inv @ y
                    log_det = safe_log_det(S)
                    if np.isnan(mahalanobis) or np.isnan(log_det):
                        return np.inf
                    cost += mahalanobis + log_det
                # 正则化项
                cost += 0.01 * (np.sum(Q_diag ** 2) + np.sum(R_diag ** 2))
                return float(cost)
            except:
                return np.inf
        
        try:
            # HBKFO iteration budget is overridable for the sensitivity sweep
            # (default 20 inner iterations, see sensitivity_analysis.py).
            _hb_iter = int(getattr(self, "_hbkfo_max_iter", 20))
            opt_result = self.hbkfo.optimize(objective, max_iter=_hb_iter, show_progress=False)
            
            Q_diag = np.maximum(opt_result[:self.dim_x], 1e-8)
            R_diag = np.maximum(opt_result[self.dim_x:], 1e-8)
            
            # 限制单次自适应对 R 的调整幅度, 防止离群窗口引起的剧烈跳变
            R_new = np.diag(R_diag)
            self.uif.Q = np.diag(Q_diag)
            self.uif.R = 0.5 * self.uif.R + 0.5 * R_new
        except Exception as e:
            pass
    
    def get_combined_state(self) -> np.ndarray:
        """获取组合状态估计.

        x̂_k = x̂_k^UIF( Q̂_k, R̂_k ) + x̂_k^IFHBFNN

        其中 HBKFO 不直接贡献一个状态加项, 而是通过对新息序列 {ν_k} 的
        元启发式优化, 在线估计过程/测量噪声协方差 Q̂_k, R̂_k, 并回灌给 UIF
        的卡尔曼增益 (见 _adapt_noise_covariance). 因此 HBKFO 的作用通过
        自适应 Q, R 体现在 UIF 的状态估计中, 而非作为一个独立的状态分量
        x̂_k^HBKFO 参与求和 (与论文公式 (3) 修正后的表述一致).
        """
        x_uif = self.uif.x_hat.copy()
        x_ifhbfnn = self.ifhbfnn_correction if self.ifhbfnn is not None else np.zeros(self.dim_x)
        # HBKFO 的贡献通过自适应 Q, R 体现在 UIF 的卡尔曼增益中 (见 _adapt_noise_covariance)
        return x_uif + x_ifhbfnn
    
    def get_state(self) -> FilterState:
        """获取当前滤波状态"""
        # 返回组合状态
        combined_x = self.get_combined_state()
        return FilterState(combined_x, self.uif.P.copy(), self.uif.Q.copy(), self.uif.R.copy())

class FilteringManager:
    """滤波管理器"""
    
    def __init__(self):
        self._filters: Dict[str, object] = {}
    
    def add_filter(self, name: str, filter_obj: object):
        """添加滤波器"""
        self._filters[name] = filter_obj
    
    def get_filter(self, name: str):
        """获取滤波器"""
        return self._filters.get(name)
    
    def apply_filter(self, filter_name: str, measurements: np.ndarray, 
                    F: np.ndarray, H: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """应用滤波器"""
        if filter_name not in self._filters:
            return measurements
        
        filter_obj = self._filters[filter_name]
        
        progress = None
        total_samples = len(measurements)
        if show_progress and total_samples > 100:
            progress = ProgressBar(total=total_samples, description=f"Filtering ({filter_name})")
        
        filtered_states = []
        for i, z in enumerate(measurements):
            if hasattr(filter_obj, 'predict'):
                filter_obj.predict(F)
            
            if hasattr(filter_obj, 'update'):
                filter_obj.update(z, H)
            
            filtered_states.append(filter_obj.get_state().x_hat)
            
            if progress and (i + 1) % max(1, total_samples // 20) == 0:
                progress.update(i + 1, f"Sample {i+1}/{total_samples}")
        
        if progress:
            progress.finish()
        
        return np.array(filtered_states)

class UnscentedKalmanFilter:
    """无迹卡尔曼滤波器（UKF）"""
    
    def __init__(self, dim_x: int, dim_z: int, alpha: float = 1.0, beta: float = 2.0, kappa: float = 0.0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha ** 2 * (dim_x + kappa) - dim_x
        
        self.n_sigma = 2 * dim_x + 1
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (dim_x + self.lambda_) + (1 - alpha ** 2 + beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (dim_x + self.lambda_))
            self.Wc[i] = 1 / (2 * (dim_x + self.lambda_))
    
    def _generate_sigma_points(self) -> np.ndarray:
        """生成sigma点"""
        P = make_positive_definite(self.P)
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            sqrt_P = np.sqrt(np.diag(P)) * np.eye(self.dim_x)
        
        sigma_points = np.zeros((self.n_sigma, self.dim_x))
        sigma_points[0] = self.x_hat
        
        scale = np.sqrt(self.dim_x + self.lambda_)
        
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x_hat + scale * sqrt_P[:, i]
            sigma_points[i + 1 + self.dim_x] = self.x_hat - scale * sqrt_P[:, i]
        
        return sigma_points
    
    def _unscented_transform(self, sigma_points: np.ndarray, noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """无迹变换"""
        mean = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)
        
        centered = sigma_points - mean
        cov = centered.T @ np.diag(self.Wc) @ centered + noise_cov
        
        return mean, make_positive_definite(cov)
    
    def predict(self, F: np.ndarray):
        """预测步骤"""
        sigma_points = self._generate_sigma_points()
        
        sigma_pred = np.array([F @ sp for sp in sigma_points])
        
        self.x_hat, self.P = self._unscented_transform(sigma_pred, self.Q)
    
    def update(self, z: np.ndarray, H: np.ndarray, h: callable = None):
        """更新步骤"""
        sigma_points = self._generate_sigma_points()
        
        if h is not None:
            sigma_z = np.array([h(sp) for sp in sigma_points])
        else:
            sigma_z = np.array([H @ sp for sp in sigma_points])
        
        z_mean, S = self._unscented_transform(sigma_z, self.R)
        
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.n_sigma):
            dx = sigma_points[i] - self.x_hat
            dz = sigma_z[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)
        
        S_inv = safe_inv(S)
        K = Pxz @ S_inv
        
        self.x_hat = self.x_hat + K @ (z - z_mean)
        self.P = make_positive_definite(self.P - K @ S @ K.T)
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

class CubatureKalmanFilter:
    """容积卡尔曼滤波器（CKF）"""
    
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
        
        self.n_sigma = 2 * dim_x
        self.W = np.ones(self.n_sigma) / self.n_sigma
        self.gamma = np.sqrt(dim_x)
    
    def _generate_sigma_points(self) -> np.ndarray:
        """生成容积点"""
        P = make_positive_definite(self.P)
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            sqrt_P = np.sqrt(np.diag(P)) * np.eye(self.dim_x)
        
        sigma_points = np.zeros((self.n_sigma, self.dim_x))
        
        for i in range(self.dim_x):
            sigma_points[i] = self.x_hat + self.gamma * sqrt_P[:, i]
            sigma_points[i + self.dim_x] = self.x_hat - self.gamma * sqrt_P[:, i]
        
        return sigma_points
    
    def _cubature_transform(self, sigma_points: np.ndarray, noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """容积变换"""
        mean = np.sum(self.W[:, np.newaxis] * sigma_points, axis=0)
        
        centered = sigma_points - mean
        cov = centered.T @ np.diag(self.W) @ centered + noise_cov
        
        return mean, make_positive_definite(cov)
    
    def predict(self, F: np.ndarray):
        """预测步骤"""
        sigma_points = self._generate_sigma_points()
        
        sigma_pred = np.array([F @ sp for sp in sigma_points])
        
        self.x_hat, self.P = self._cubature_transform(sigma_pred, self.Q)
    
    def update(self, z: np.ndarray, H: np.ndarray, h: callable = None):
        """更新步骤"""
        sigma_points = self._generate_sigma_points()
        
        if h is not None:
            sigma_z = np.array([h(sp) for sp in sigma_points])
        else:
            sigma_z = np.array([H @ sp for sp in sigma_points])
        
        z_mean, S = self._cubature_transform(sigma_z, self.R)
        
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.n_sigma):
            dx = sigma_points[i] - self.x_hat
            dz = sigma_z[i] - z_mean
            Pxz += self.W[i] * np.outer(dx, dz)
        
        S_inv = safe_inv(S)
        K = Pxz @ S_inv
        
        self.x_hat = self.x_hat + K @ (z - z_mean)
        self.P = make_positive_definite(self.P - K @ S @ K.T)
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

class AdaptiveExtendedKalmanFilter:
    """自适应扩展卡尔曼滤波器（AEKF）"""
    
    def __init__(self, dim_x: int, dim_z: int, forgetting_factor: float = 0.95):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
        
        self.forgetting_factor = forgetting_factor
        self.innovation_history: List[np.ndarray] = []
        self.max_history = 50
        
        self.Q_hat = np.eye(dim_x) * 0.01
        self.R_hat = np.eye(dim_z) * 0.1
    
    def predict(self, F: np.ndarray):
        """预测步骤"""
        self.x_hat = F @ self.x_hat
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z: np.ndarray, H, h: callable = None):
        """更新步骤 (H 可为固定矩阵或可调用 Jacobian H(x))"""
        Hm = _eval_H(H, self.x_hat)
        if h is not None:
            z_pred = h(self.x_hat)
        else:
            z_pred = Hm @ self.x_hat

        y = z - z_pred

        self.innovation_history.append(y)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        self._adapt_noise_covariance(Hm, y)

        S = Hm @ self.P @ Hm.T + self.R + np.eye(self.dim_z) * 1e-6
        K = self.P @ Hm.T @ safe_inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = make_positive_definite((np.eye(self.dim_x) - K @ Hm) @ self.P)
    
    def _adapt_noise_covariance(self, H: np.ndarray, y: np.ndarray):
        """基于创新序列自适应估计噪声协方差"""
        if len(self.innovation_history) < 10:
            return
        
        innovations = np.array(self.innovation_history)
        
        Pyy = np.zeros((self.dim_z, self.dim_z))
        for yi in innovations:
            Pyy += np.outer(yi, yi)
        Pyy /= len(innovations)
        
        PHPT = H @ self.P @ H.T
        
        self.R_hat = self.forgetting_factor * self.R_hat + (1 - self.forgetting_factor) * (Pyy - PHPT)
        self.R_hat = make_positive_definite(self.R_hat)
        
        self.R = np.maximum(self.R_hat, np.eye(self.dim_z) * 1e-8)
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

class RobustUnscentedKalmanFilter:
    """鲁棒无迹卡尔曼滤波器（RUKF）"""
    
    def __init__(self, dim_x: int, dim_z: int, alpha: float = 1.0, beta: float = 2.0, 
                 kappa: float = 0.0, huber_threshold: float = 1.345):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha ** 2 * (dim_x + kappa) - dim_x
        self.huber_threshold = huber_threshold
        
        self.n_sigma = 2 * dim_x + 1
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (dim_x + self.lambda_) + (1 - alpha ** 2 + beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (dim_x + self.lambda_))
            self.Wc[i] = 1 / (2 * (dim_x + self.lambda_))
    
    def _huber_weight(self, residual: np.ndarray, scale: float) -> float:
        """Huber权重函数"""
        norm_residual = np.linalg.norm(residual)
        if norm_residual <= self.huber_threshold * scale:
            return 1.0
        return (self.huber_threshold * scale) / norm_residual
    
    def _generate_sigma_points(self) -> np.ndarray:
        """生成sigma点"""
        P = make_positive_definite(self.P)
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            sqrt_P = np.sqrt(np.diag(P)) * np.eye(self.dim_x)
        
        sigma_points = np.zeros((self.n_sigma, self.dim_x))
        sigma_points[0] = self.x_hat
        
        scale = np.sqrt(self.dim_x + self.lambda_)
        
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x_hat + scale * sqrt_P[:, i]
            sigma_points[i + 1 + self.dim_x] = self.x_hat - scale * sqrt_P[:, i]
        
        return sigma_points
    
    def predict(self, F: np.ndarray):
        """预测步骤"""
        sigma_points = self._generate_sigma_points()
        sigma_pred = np.array([F @ sp for sp in sigma_points])
        
        self.x_hat = np.sum(self.Wm[:, np.newaxis] * sigma_pred, axis=0)
        centered = sigma_pred - self.x_hat
        self.P = make_positive_definite(centered.T @ np.diag(self.Wc) @ centered + self.Q)
    
    def update(self, z: np.ndarray, H: np.ndarray, h: callable = None):
        """更新步骤（鲁棒版本）"""
        sigma_points = self._generate_sigma_points()
        
        if h is not None:
            sigma_z = np.array([h(sp) for sp in sigma_points])
        else:
            sigma_z = np.array([H @ sp for sp in sigma_points])
        
        z_mean = np.sum(self.Wm[:, np.newaxis] * sigma_z, axis=0)
        z_centered = sigma_z - z_mean
        S = make_positive_definite(z_centered.T @ np.diag(self.Wc) @ z_centered + self.R)
        
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.n_sigma):
            dx = sigma_points[i] - self.x_hat
            dz = sigma_z[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)
        
        y = z - z_mean
        
        try:
            S_inv = safe_inv(S)
            mahalanobis = y.T @ S_inv @ y
            scale = np.sqrt(mahalanobis)
            
            weight = self._huber_weight(y, np.sqrt(self.dim_z))
            
            K = Pxz @ S_inv
            self.x_hat = self.x_hat + weight * K @ y
            self.P = make_positive_definite(self.P - weight * K @ S @ K.T)
        except:
            K = Pxz @ safe_inv(S + np.eye(self.dim_z) * 1e-6)
            self.x_hat = self.x_hat + K @ y
            self.P = make_positive_definite(self.P - K @ S @ K.T)
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

class DeepKalmanFilter:
    """深度卡尔曼滤波器（DeepKF）"""
    
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int = 64):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.hidden_dim = hidden_dim
        
        self.x_hat = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.1
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1
        
        self._init_neural_weights()
        
        self.learning_rate = 0.0001
        self.innovation_history: List[np.ndarray] = []
        
        self.clip_value = 1e6
        self.grad_clip_value = 1.0
        self.weight_clip_value = 10.0
    
    def _init_neural_weights(self):
        """初始化神经网络权重"""
        np.random.seed(42)
        self.nn_weights = {
            'W1': np.random.randn(self.dim_x, self.hidden_dim) * 0.01,
            'W2': np.random.randn(self.hidden_dim, self.dim_x) * 0.01,
            'V1': np.random.randn(self.dim_x, self.hidden_dim) * 0.01,
            'V2': np.random.randn(self.hidden_dim, self.dim_z) * 0.01
        }
        self.nn_biases = {
            'b1': np.zeros(self.hidden_dim),
            'b2': np.zeros(self.dim_x),
            'c1': np.zeros(self.hidden_dim),
            'c2': np.zeros(self.dim_z)
        }
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _clip_weights(self):
        """裁剪权重防止数值溢出"""
        for key in self.nn_weights:
            self.nn_weights[key] = np.clip(self.nn_weights[key], -self.weight_clip_value, self.weight_clip_value)
        for key in self.nn_biases:
            self.nn_biases[key] = np.clip(self.nn_biases[key], -self.weight_clip_value, self.weight_clip_value)
    
    def _clip_gradients(self, gradients):
        """裁剪梯度防止梯度爆炸"""
        for key in gradients:
            gradients[key] = np.clip(gradients[key], -self.grad_clip_value, self.grad_clip_value)
        return gradients
    
    def _clip_state(self):
        """裁剪状态估计防止发散"""
        self.x_hat = np.clip(self.x_hat, -self.clip_value, self.clip_value)
        self.P = np.clip(self.P, -self.clip_value, self.clip_value)
        self.P = make_positive_definite(self.P)
    
    def _neural_dynamics(self, x: np.ndarray) -> np.ndarray:
        """神经网络动态模型"""
        try:
            h1 = self._relu(x @ self.nn_weights['W1'] + self.nn_biases['b1'])
            h1 = np.clip(h1, -self.clip_value, self.clip_value)
            output = h1 @ self.nn_weights['W2'] + self.nn_biases['b2']
            return np.clip(output, -self.clip_value, self.clip_value)
        except (OverflowError, ValueError):
            return np.zeros(self.dim_x)
    
    def _neural_measurement(self, x: np.ndarray) -> np.ndarray:
        """神经网络测量模型"""
        try:
            h1 = self._relu(x @ self.nn_weights['V1'] + self.nn_biases['c1'])
            h1 = np.clip(h1, -self.clip_value, self.clip_value)
            output = h1 @ self.nn_weights['V2'] + self.nn_biases['c2']
            return np.clip(output, -self.clip_value, self.clip_value)
        except (OverflowError, ValueError):
            return np.zeros(self.dim_z)
    
    def predict(self, F: np.ndarray):
        """预测步骤（结合线性和非线性动态）"""
        if np.any(np.isnan(self.x_hat)) or np.any(np.isinf(self.x_hat)):
            self.x_hat = np.zeros(self.dim_x)
            self.P = np.eye(self.dim_x) * 0.1
            self._init_neural_weights()
            return
        
        linear_pred = F @ self.x_hat
        nonlinear_pred = self._neural_dynamics(self.x_hat)
        
        # 学习到的非线性项作为线性预测的残差修正 (而非替换), 避免未充分训练的
        # 网络将状态拉向零而破坏对远离原点目标的跟踪.
        self.x_hat = linear_pred + 0.1 * nonlinear_pred
        self.P = F @ self.P @ F.T + self.Q
        
        self._clip_state()
    
    def update(self, z: np.ndarray, H, h: callable = None):
        """更新步骤 (H 可为固定矩阵或可调用 Jacobian H(x); h 为非线性观测)"""
        if np.any(np.isnan(self.x_hat)) or np.any(np.isinf(self.x_hat)):
            self.x_hat = np.zeros(self.dim_x)
            self.P = np.eye(self.dim_x) * 0.1
            self._init_neural_weights()
            return

        Hm = _eval_H(H, self.x_hat)
        z_base = h(self.x_hat) if h is not None else Hm @ self.x_hat
        z_pred = z_base + 0.1 * self._neural_measurement(self.x_hat)

        if np.any(np.isnan(z_pred)) or np.any(np.isinf(z_pred)):
            z_pred = z_base

        y = z - z_pred

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.zeros(self.dim_z)

        self.innovation_history.append(y)

        if len(self.innovation_history) > 10:
            self._adapt_neural_weights(z, z_pred)

        S = Hm @ self.P @ Hm.T + self.R + np.eye(self.dim_z) * 1e-6

        try:
            K = self.P @ Hm.T @ safe_inv(S)

            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                K = np.zeros((self.dim_x, self.dim_z))

            update_term = K @ y

            if np.any(np.isnan(update_term)) or np.any(np.isinf(update_term)):
                update_term = np.zeros(self.dim_x)

            self.x_hat = self.x_hat + update_term
            self.P = make_positive_definite((np.eye(self.dim_x) - K @ Hm) @ self.P)
            
            self._clip_state()
        except (OverflowError, ValueError, np.linalg.LinAlgError):
            self.x_hat = np.zeros(self.dim_x)
            self.P = np.eye(self.dim_x) * 0.1
            self._init_neural_weights()
    
    def _adapt_neural_weights(self, z: np.ndarray, z_pred: np.ndarray):
        """自适应调整神经网络权重"""
        try:
            error = z - z_pred
            
            if np.any(np.isnan(error)) or np.any(np.isinf(error)):
                return
            
            error_norm = np.linalg.norm(error)
            if error_norm > 1000:
                error = error / error_norm * 1000
            
            h1_meas = self._relu(self.x_hat @ self.nn_weights['V1'] + self.nn_biases['c1'])
            h1_meas = np.clip(h1_meas, -self.clip_value, self.clip_value)
            
            grad_c2 = error
            grad_V2 = np.outer(h1_meas, error)
            
            grad_h1 = error @ self.nn_weights['V2'].T
            grad_h1[h1_meas <= 0] = 0
            grad_c1 = grad_h1
            grad_V1 = np.outer(self.x_hat, grad_h1)
            
            gradients = {
                'V2': grad_V2,
                'c2': grad_c2,
                'V1': grad_V1,
                'c1': grad_c1
            }
            
            gradients = self._clip_gradients(gradients)
            
            self.nn_weights['V2'] -= self.learning_rate * gradients['V2']
            self.nn_biases['c2'] -= self.learning_rate * gradients['c2']
            self.nn_weights['V1'] -= self.learning_rate * gradients['V1']
            self.nn_biases['c1'] -= self.learning_rate * gradients['c1']
            
            self._clip_weights()
        except (OverflowError, ValueError):
            self._init_neural_weights()
    
    def get_state(self) -> FilterState:
        """获取当前状态"""
        return FilterState(self.x_hat.copy(), self.P.copy(), self.Q.copy(), self.R.copy())

def create_ns_arkf_filter(dim_x: int = 6, dim_z: int = 4, use_ifhbfnn: bool = True, 
                          use_hbkfo: bool = False) -> NSARKF:
    """创建NS-ARKF滤波器
    
    Args:
        dim_x: 状态维度
        dim_z: 测量维度
        use_ifhbfnn: 是否使用IFHBFNN组件(消融实验用)
        use_hbkfo: 是否启用 HBKFO 在线协方差自适应模块 (消融实验用).
            默认关闭: 极端/脉冲噪声下, 鲁棒新息门控 (IGG-III) 已提供主要的
            鲁棒性; 经验上启用 HBKFO 的元启发式协方差搜索会在脉冲噪声窗口引入
            估计抖动而略微降低精度 (见消融实验). 因此 HBKFO 作为可选的工程自适应
            模块保留, 而非默认路径.
    """
    return NSARKF(dim_x, dim_z, use_ifhbfnn=use_ifhbfnn, use_hbkfo=use_hbkfo)

def create_ekf_filter(dim_x: int = 6, dim_z: int = 4) -> ExtendedKalmanFilter:
    """创建EKF滤波器"""
    return ExtendedKalmanFilter(dim_x, dim_z)

def create_uif_filter(dim_x: int = 6, dim_z: int = 4) -> UnknownInputFilter:
    """创建UIF滤波器"""
    return UnknownInputFilter(dim_x, dim_z)

def create_ukf_filter(dim_x: int = 6, dim_z: int = 4) -> UnscentedKalmanFilter:
    """创建UKF滤波器"""
    return UnscentedKalmanFilter(dim_x, dim_z)

def create_ckf_filter(dim_x: int = 6, dim_z: int = 4) -> CubatureKalmanFilter:
    """创建CKF滤波器"""
    return CubatureKalmanFilter(dim_x, dim_z)

def create_aekf_filter(dim_x: int = 6, dim_z: int = 4) -> AdaptiveExtendedKalmanFilter:
    """创建AEKF滤波器"""
    return AdaptiveExtendedKalmanFilter(dim_x, dim_z)

def create_rukf_filter(dim_x: int = 6, dim_z: int = 4) -> RobustUnscentedKalmanFilter:
    """创建RUKF滤波器"""
    return RobustUnscentedKalmanFilter(dim_x, dim_z)

def create_deepkf_filter(dim_x: int = 6, dim_z: int = 4) -> DeepKalmanFilter:
    """创建DeepKF滤波器"""
    return DeepKalmanFilter(dim_x, dim_z)