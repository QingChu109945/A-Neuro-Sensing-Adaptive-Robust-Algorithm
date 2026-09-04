import numpy as np
from numpy.linalg import inv, norm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .progress import ProgressBar

class AdamOptimizer:
    """Adam优化器"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """执行一步优化"""
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def compute_vectorized_gradients(model, X: np.ndarray, y: np.ndarray, pred: np.ndarray, 
                                eps: float = 1e-4) -> Dict[str, np.ndarray]:
    """向量化数值梯度计算"""
    grads = {}
    batch_size = X.shape[0]
    
    for key in model.weights:
        w = model.weights[key].copy()
        shape = w.shape
        grad = np.zeros(shape)
        
        flat_w = w.flatten()
        flat_grad = np.zeros_like(flat_w)
        
        for i in range(len(flat_w)):
            flat_w[i] += eps
            model.weights[key] = flat_w.reshape(shape)
            pred_plus = model._forward(X)
            
            flat_w[i] -= 2 * eps
            model.weights[key] = flat_w.reshape(shape)
            pred_minus = model._forward(X)
            
            flat_w[i] += eps
            model.weights[key] = flat_w.reshape(shape)
            
            diff = np.mean((pred_plus - pred_minus) * (pred - y)) / (2 * eps)
            flat_grad[i] = diff
        
        grads[key] = flat_grad.reshape(shape)
    
    return grads

@dataclass
class InversionConfig:
    """反演配置"""
    method: str = 'ssm_pinn'
    enforce_hard_constraint: bool = True
    constraint_tolerance: float = 1e-6
    learning_rate: float = 0.01
    max_iterations: int = 500
    regularization_weight: float = 0.1
    noise_level: float = 0.01

@dataclass
class InversionResult:
    """反演结果"""
    emissivity_pred: float
    emissivity_true: Optional[float] = None
    emissivity_std: float = 0.0
    reflectivity_pred: float = 0.0
    reflectivity_true: Optional[float] = None
    reflectivity_std: float = 0.0
    constraint_satisfied: bool = True
    loss_value: float = 0.0
    method: str = 'ssm_pinn'

class SelectiveSSM:
    """选择性状态空间模型 (S6/Mamba核心单元)
    
    论文Section 4.3: 输入依赖的状态转移矩阵
    h'(t) = A(x(t))h(t) + B(x(t))x(t)
    y(t) = C(x(t))h(t) + D(x(t))x(t)
    """
    
    def __init__(self, input_dim: int, state_dim: int = 32):
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        np.random.seed(42)
        # 输入依赖参数的投影矩阵
        self.W_A = np.random.randn(input_dim, state_dim) * np.sqrt(2 / input_dim)
        self.W_B = np.random.randn(input_dim, state_dim) * np.sqrt(2 / input_dim)
        self.W_C = np.random.randn(input_dim, state_dim) * np.sqrt(2 / input_dim)
        self.W_D = np.random.randn(input_dim, input_dim) * np.sqrt(2 / input_dim)
        
        # SSM参数
        self.dt = 0.1  # 离散化步长
        
        # 可训练参数的梯度缓存
        self._cache = {}
    
    def _input_dependent_params(self, x: np.ndarray):
        """计算输入依赖的状态矩阵
        A(x) = diag(W_A · Linear(x))  -- 使用sigmoid保证稳定
        B(x) = W_B · x
        C(x) = W_C · x  
        D(x) = W_D (常数)
        """
        batch_size = x.shape[0]
        
        # A: 使用对角矩阵,元素通过sigmoid限制在(0,1)保证稳定
        A_diag = 1.0 / (1.0 + np.exp(-(x @ self.W_A)))  # (batch, state_dim)
        
        # B, C: 线性投影
        B = x @ self.W_B  # (batch, state_dim)
        C = x @ self.W_C  # (batch, state_dim)
        D = self.W_D  # (input_dim, input_dim)
        
        return A_diag, B, C, D
    
    def _discretize(self, A_diag: np.ndarray, B: np.ndarray):
        """零阶保持离散化
        Ā = exp(Δ·A) ≈ I + Δ·A (一阶近似)
        B̄ = (Δ·A)^{-1}(exp(Δ·A) - I)·Δ·B ≈ Δ·B
        """
        batch_size = A_diag.shape[0]
        
        # 离散化A (对角矩阵)
        A_bar_diag = np.exp(self.dt * A_diag)  # (batch, state_dim)
        
        # 离散化B
        B_bar = self.dt * B  # (batch, state_dim)
        
        return A_bar_diag, B_bar
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: 并行扫描计算SSM
        h_k = Ā_k ⊙ h_{k-1} + B̄_k
        y_k = C_k ⊙ h_k + D·x_k
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        batch_size, input_dim = x.shape
        
        A_diag, B, C, D = self._input_dependent_params(x)
        A_bar_diag, B_bar = self._discretize(A_diag, B)
        
        # 状态演化 (将输入序列视为时间步)
        h = np.zeros((batch_size, self.state_dim))
        outputs = []
        
        seq_len = max(1, input_dim // self.input_dim) if input_dim >= self.input_dim else 1
        
        # 对每个样本独立演化
        for t in range(seq_len):
            idx = min(t, input_dim - 1)
            x_t = x[:, idx:idx+1] if input_dim > 1 else x
            
            # 状态更新: h_k = Ā ⊙ h_{k-1} + B̄
            h = A_bar_diag * h + B_bar
            
            # 输出: y_k = C ⊙ h + D·x_k
            y_t = np.sum(C * h, axis=1, keepdims=True)
            if input_dim >= self.input_dim:
                y_t = y_t + x_t @ D.T if D.shape[0] == x_t.shape[1] else y_t
            outputs.append(y_t)
        
        # 聚合输出 (平均池化)
        output = np.mean(np.array(outputs), axis=0)  # (batch, 1)
        
        # 扩展到state_dim维度
        output_full = np.hstack([output] * self.state_dim)  # (batch, state_dim)
        
        # 缓存用于反向传播
        self._cache = {
            'x': x, 'A_diag': A_diag, 'B': B, 'C': C,
            'A_bar_diag': A_bar_diag, 'B_bar': B_bar,
            'h': h, 'output': output_full
        }
        
        return output_full
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {'W_A': self.W_A, 'W_B': self.W_B, 'W_C': self.W_C, 'W_D': self.W_D}
    
    def set_params(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            setattr(self, k, v)


class SSMPINN:
    """结构化状态空间模型增强的物理信息神经网络 (SSM-PINN)
    
    论文Section 4: Encoder-State Evolver-Decoder (E-S-D) 架构
    - Encoder: 输入测量 -> 初始隐状态
    - StateEvolver: Selective SSM (S6) 演化隐状态
    - HardConstraintDecoder: 硬约束输出层 (Kirchhoff能量守恒)
    - Bayesian VI: 不确定性量化
    - Gumbel-softmax: 材料类型分类
    """
    
    def __init__(self, config: InversionConfig = None, n_material_classes: int = 12):
        self.config = config if config else InversionConfig()
        self.n_material_classes = n_material_classes
        self.state_dim = 32
        self.hidden_dim = 64
        
        # 损失权重 (论文§4.5.6 / CAL0827.tex L686: λ1..λ5 经验证集网格搜索确定)
        self.lambda_pred = 1.0     # λ1 预测损失 (主导数据保真项)
        self.lambda_phy = 0.5      # λ2 物理残差
        self.lambda_interf = 0.3   # λ3 干扰残差
        self.lambda_elbo = 0.1     # λ4 ELBO/KL (原代码 0.2, 修正为论文 0.1)
        self.lambda_reg = 1e-4    # λ5 权重正则化 (原代码 0.01, 修正为论文 1e-4, 避免过度收缩)
        
        # 训练参数
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps_adam = 1e-8
        
        # Gumbel-softmax温度退火
        self.tau = 1.0
        self.tau_min = 0.1
        
        self._initialized = False
        self._cache = {}
    
    def _init_params(self, input_dim: int):
        """初始化所有参数"""
        np.random.seed(42)
        self.input_dim = input_dim
        
        # Encoder: 输入 -> 隐状态
        self.enc_W1 = np.random.randn(input_dim, self.hidden_dim) * np.sqrt(2 / input_dim)
        self.enc_b1 = np.zeros(self.hidden_dim)
        self.enc_W2 = np.random.randn(self.hidden_dim, self.state_dim) * np.sqrt(2 / self.hidden_dim)
        self.enc_b2 = np.zeros(self.state_dim)
        
        # StateEvolver: Selective SSM
        self.ssm = SelectiveSSM(self.state_dim, self.state_dim)
        
        # Decoder: 硬约束输出层
        # ε = σ(g_ε(h)), ρ = (1-ε)·σ(g_ρ(h))
        self.dec_W_eps = np.random.randn(self.state_dim, 1) * np.sqrt(2 / self.state_dim)
        self.dec_b_eps = np.zeros(1)
        self.dec_W_rho = np.random.randn(self.state_dim, 1) * np.sqrt(2 / self.state_dim)
        self.dec_b_rho = np.zeros(1)
        
        # 材料类型分类头: Gumbel-softmax
        self.dec_W_mat = np.random.randn(self.state_dim, self.n_material_classes) * np.sqrt(2 / self.state_dim)
        self.dec_b_mat = np.zeros(self.n_material_classes)
        
        # 贝叶斯VI: 变分参数 (μ_φ, σ_φ)
        self.vi_W_mu = np.random.randn(self.state_dim, 2) * np.sqrt(2 / self.state_dim)
        self.vi_b_mu = np.zeros(2)
        self.vi_W_logvar = np.random.randn(self.state_dim, 2) * np.sqrt(2 / self.state_dim)
        self.vi_b_logvar = np.zeros(2)
        
        self._initialized = True
        self._init_adam_state()
    
    def _init_adam_state(self):
        """初始化Adam优化器状态"""
        param_names = self._get_param_names()
        for name in param_names:
            self.adam_m[name] = np.zeros_like(self._get_param(name))
            self.adam_v[name] = np.zeros_like(self._get_param(name))
    
    def _get_param_names(self) -> List[str]:
        return ['enc_W1', 'enc_b1', 'enc_W2', 'enc_b2',
                'dec_W_eps', 'dec_b_eps', 'dec_W_rho', 'dec_b_rho',
                'dec_W_mat', 'dec_b_mat',
                'vi_W_mu', 'vi_b_mu', 'vi_W_logvar', 'vi_b_logvar',
                'ssm_W_A', 'ssm_W_B', 'ssm_W_C', 'ssm_W_D']
    
    def _get_param(self, name: str) -> np.ndarray:
        if name.startswith('ssm_'):
            ssm_name = name[4:]
            return getattr(self.ssm, ssm_name)
        return getattr(self, name)
    
    def _set_param(self, name: str, value: np.ndarray):
        if name.startswith('ssm_'):
            ssm_name = name[4:]
            setattr(self.ssm, ssm_name, value)
        else:
            setattr(self, name, value)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(np.clip(x, -500, 500)))
    
    def _gumbel_softmax(self, logits: np.ndarray, tau: float = 1.0, hard: bool = False) -> np.ndarray:
        """Gumbel-softmax: 可微的分类采样
        M_type = GumbelSoftmax(g_M(h), τ)
        """
        gumbel_noise = -np.log(-np.log(np.random.uniform(1e-10, 1.0, logits.shape) + 1e-10) + 1e-10)
        y = (logits + gumbel_noise) / tau
        # Softmax
        y_max = np.max(y, axis=-1, keepdims=True)
        exp_y = np.exp(y - y_max)
        softmax = exp_y / np.sum(exp_y, axis=-1, keepdims=True)
        
        if hard:
            # 直通估计器: 前向one-hot, 反向softmax梯度
            idx = np.argmax(softmax, axis=-1)
            onehot = np.zeros_like(softmax)
            onehot[np.arange(len(idx)), idx] = 1.0
            return onehot + softmax - softmax.detach() if hasattr(softmax, 'detach') else onehot
        return softmax
    
    def _encoder(self, x: np.ndarray) -> np.ndarray:
        """Encoder: 输入测量 -> 初始隐状态
        h_enc = ReLU(W2·ReLU(W1·x + b1) + b2)
        """
        h1 = self._relu(x @ self.enc_W1 + self.enc_b1)
        h_enc = h1 @ self.enc_W2 + self.enc_b2
        self._cache['enc_h1'] = h1
        self._cache['enc_h_enc'] = h_enc
        return h_enc
    
    def _state_evolver(self, h_enc: np.ndarray) -> np.ndarray:
        """StateEvolver: Selective SSM 演化隐状态
        h_evolved = SSM(h_enc)
        """
        h_evolved = self.ssm.forward(h_enc)
        self._cache['h_evolved'] = h_evolved
        return h_evolved
    
    def _hard_constraint_decoder(self, h_evolved: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """HardConstraintDecoder: 硬约束输出层
        
        ε = σ(g_ε(h_evolved)) ∈ (0, 1)
        ρ = (1-ε)·σ(g_ρ(h_evolved)) ∈ (0, 1-ε)
        
        保证: ε + ρ = ε + (1-ε)·σ(g_ρ) ≤ ε + (1-ε) = 1
        """
        # 发射率输出
        eps_logit = h_evolved @ self.dec_W_eps + self.dec_b_eps  # (batch, 1)
        eps = self._sigmoid(eps_logit)  # ∈ (0, 1)
        
        # 反射率输出 (硬约束: ρ = (1-ε)·σ(g_ρ))
        rho_logit = h_evolved @ self.dec_W_rho + self.dec_b_rho  # (batch, 1)
        rho = (1 - eps) * self._sigmoid(rho_logit)  # ∈ (0, 1-ε)
        
        # 材料类型输出
        mat_logits = h_evolved @ self.dec_W_mat + self.dec_b_mat  # (batch, n_classes)
        
        self._cache['eps_logit'] = eps_logit
        self._cache['rho_logit'] = rho_logit
        self._cache['mat_logits'] = mat_logits
        
        return eps, rho, mat_logits
    
    def _variational_posterior(self, h_evolved: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """贝叶斯变分推断: q_φ(y|z) = N(μ_φ(z), diag(σ_φ²(z)))
        
        重参数化: y = μ + σ ⊙ ε, ε~N(0,I)
        """
        mu = h_evolved @ self.vi_W_mu + self.vi_b_mu  # (batch, 2)
        logvar = h_evolved @ self.vi_W_logvar + self.vi_b_logvar  # (batch, 2)
        logvar = np.clip(logvar, -10, 10)  # 数值稳定
        sigma = np.exp(0.5 * logvar)
        
        self._cache['vi_mu'] = mu
        self._cache['vi_logvar'] = logvar
        self._cache['vi_sigma'] = sigma
        
        return mu, sigma
    
    def _physics_residual_loss(self, X: np.ndarray, eps: np.ndarray, rho: np.ndarray) -> float:
        """物理残差损失 (软约束) — 论文§4.5.5 / CAL0827.tex L605-L622

        激光回波强度 ε/ρ 派生方法:
            I_echo = ε·I_thermal^model + ρ·I_reflection^model + η
        其中 I_thermal^model, I_reflection^model 为不含 ε/ρ 的物理基底,
        预测的 ε̂, ρ̂ 作为线性系数派生出回波强度, 与实测 I_echo 求残差:
            L_phy = (1/N)·‖I_echo - (ε̂·I_thermal^model + ρ̂·I_reflection^model)‖²

        关键修正(ε/ρ 派生): 反射基底必须用入射激光功率 I_laser
        (data_generator.py L342: i_laser = 1000·distance_factor), 而非回波
        laser_echo 本身 —— 后者会让 ρ̂ 作用于已含 ρ 贡献的量, 形成循环定义.
        本方法的线性结构亦使闭式最小二乘 (np.linalg.lstsq) 可解, 对应论文§5.1.1
        (L734-L737) 声明的 NumPy 闭式拟合; torch_training.py 提供 Algorithm 3
        的完整 PyTorch 训练实现.
        """
        # X 约定: [distance, angle, temperature, vibration, laser_echo] (归一化)
        if X.shape[1] >= 5:
            distance = X[:, 0]; angle = X[:, 1]
            temperature = X[:, 2]; laser_echo = X[:, 4]
        else:
            distance = X[:, 0]
            angle = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
            temperature = X[:, 2] if X.shape[1] > 2 else np.ones(len(X))
            laser_echo = X[:, 0]

        sigma_SB = 5.67e-8  # Stefan-Boltzmann 常数

        # 距离衰减因子 f(D) 与入射激光功率 I_laser, 与 data_generator 严格一致
        distance_real = np.abs(distance) * 5000.0 + 100.0          # 反归一化到 [100, 5000] m
        distance_factor = 1.0 / (1.0 + distance_real / 1000.0)   # data_generator L337
        angle_real = angle * 75.0                                 # 反归一化到 [0, 75]°
        angle_factor = np.cos(np.deg2rad(angle_real))             # data_generator L338

        # 热辐射基底 (不含 ε): I_thermal^model = σ_SB·T⁴·f(D,θ)
        I_thermal_model = sigma_SB * (temperature + 273.15) ** 4 * distance_factor * angle_factor

        # 入射激光功率 I_laser (data_generator L342)
        I_laser_model = 1000.0 * distance_factor
        roughness_factor = 0.91  # 1 - 0.3·0.3, 与 data_generator 默认粗糙度一致

        # 直接反射基底 (不含 ρ): I_reflection^model = I_laser·g(D,θ,α_rough)
        I_reflection_model = I_laser_model * angle_factor * roughness_factor

        # ε/ρ 派生: 模型预测回波 = ε̂·I_thermal^model + ρ̂·I_reflection^model
        I_model = eps[:, 0] * I_thermal_model + rho[:, 0] * I_reflection_model

        # 物理残差 (论文 L620-L622)
        residual = laser_echo - I_model
        return np.mean(residual ** 2)
    
    def _interference_residual_loss(self, X: np.ndarray, eps: np.ndarray, rho: np.ndarray) -> float:
        """多源干扰残差损失
        L_interf = ||V - M_vib(M_type, T, p_env)||²
        """
        if X.shape[1] >= 4:
            vibration = X[:, 3]
        else:
            vibration = np.zeros(len(X))
        
        # 简化的振动模型: V = α·ε·T + β·ρ
        alpha_vib = 0.1
        beta_vib = 0.05
        v_model = alpha_vib * eps[:, 0] * np.abs(X[:, 2] if X.shape[1] > 2 else 1) + \
                  beta_vib * rho[:, 0]
        
        return np.mean((vibration - v_model) ** 2)
    
    def _elbo_loss(self, mu: np.ndarray, sigma: np.ndarray, y_true: np.ndarray) -> float:
        """ELBO损失 (证据下界)
        L_ELBO = -E_q[log p(z|y)] + KL(q||p)
        = reconstruction_loss + KL_divergence
        """
        # 重参数化采样
        epsilon = np.random.randn(*mu.shape)
        y_sample = mu + sigma * epsilon
        
        # 重建损失 (负对数似然)
        recon_loss = 0.5 * np.mean(np.sum((y_sample - y_true) ** 2, axis=1))
        
        # KL散度: KL(N(μ,σ²) || N(0,1)) = 0.5·Σ(μ² + σ² - log(σ²) - 1)
        kl_div = 0.5 * np.mean(np.sum(mu ** 2 + sigma ** 2 - 2 * np.log(sigma + 1e-10) - 1, axis=1))
        
        return recon_loss + kl_div
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """完整前向传播: E-S-D架构
        
        返回: [ε, ρ] (发射率, 反射率)
        """
        if not self._initialized:
            self._init_params(x.shape[1])
        
        # Encoder
        h_enc = self._encoder(x)
        
        # StateEvolver (Selective SSM)
        h_evolved = self._state_evolver(h_enc)
        
        # HardConstraintDecoder
        eps, rho, mat_logits = self._hard_constraint_decoder(h_evolved)
        
        # Bayesian VI
        mu, sigma = self._variational_posterior(h_evolved)
        
        # 存储缓存
        self._cache['eps'] = eps
        self._cache['rho'] = rho
        self._cache['mat_logits'] = mat_logits
        
        return np.hstack([eps, rho])
    
    def _compute_total_loss(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """计算多目标损失
        L_total = λ1·L_pred + λ2·L_phy + λ3·L_interf + λ4·L_ELBO + λ5·L_reg
        """
        eps = self._cache['eps']
        rho = self._cache['rho']
        mu = self._cache['vi_mu']
        sigma = self._cache['vi_sigma']
        
        # L_pred: 预测损失
        loss_pred = np.mean((pred - y) ** 2)
        
        # L_phy: 物理残差损失
        loss_phy = self._physics_residual_loss(X, eps, rho)
        
        # L_interf: 干扰残差损失
        loss_interf = self._interference_residual_loss(X, eps, rho)
        
        # L_ELBO: 变分推断损失
        loss_elbo = self._elbo_loss(mu, sigma, y)
        
        # L_reg: 权重正则化
        loss_reg = 0.0
        for name in self._get_param_names():
            p = self._get_param(name)
            loss_reg += np.sum(p ** 2)
        loss_reg *= 0.001
        
        # 总损失
        total = (self.lambda_pred * loss_pred + 
                 self.lambda_phy * loss_phy + 
                 self.lambda_interf * loss_interf + 
                 self.lambda_elbo * loss_elbo + 
                 self.lambda_reg * loss_reg)
        
        return {
            'total': total, 'pred': loss_pred, 'phy': loss_phy,
            'interf': loss_interf, 'elbo': loss_elbo, 'reg': loss_reg
        }
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """计算所有参数的梯度 (数值梯度 + 解析梯度混合)"""
        grads = {}
        batch_size = X.shape[0]
        
        eps = self._cache['eps']
        rho = self._cache['rho']
        h_evolved = self._cache['h_evolved']
        
        # 解码器梯度 (解析)
        # dL/dε = 2*(ε - y_0)/N * λ_pred
        d_eps = 2 * (eps[:, 0] - y[:, 0]) / batch_size * self.lambda_pred
        # dL/dρ = 2*(ρ - y_1)/N * λ_pred
        d_rho = 2 * (rho[:, 0] - y[:, 1]) / batch_size * self.lambda_pred
        
        # ε = σ(g_ε), dε/dg_ε = ε(1-ε)
        d_eps_logit = d_eps * eps[:, 0] * (1 - eps[:, 0])
        
        # ρ = (1-ε)·σ(g_ρ), dρ/dg_ρ = (1-ε)·σ(g_ρ)·(1-σ(g_ρ))
        rho_sigmoid = self._sigmoid(self._cache['rho_logit'][:, 0])
        d_rho_logit = d_rho * (1 - eps[:, 0]) * rho_sigmoid * (1 - rho_sigmoid)
        
        # 解码器权重梯度
        grads['dec_W_eps'] = h_evolved.T @ d_eps_logit[:, np.newaxis]
        grads['dec_b_eps'] = np.sum(d_eps_logit)
        grads['dec_W_rho'] = h_evolved.T @ d_rho_logit[:, np.newaxis]
        grads['dec_b_rho'] = np.sum(d_rho_logit)
        
        # 对h_evolved的梯度
        d_h_evolved = (d_eps_logit[:, np.newaxis] @ self.dec_W_eps.T + 
                       d_rho_logit[:, np.newaxis] @ self.dec_W_rho.T)
        
        # Encoder和SSM使用数值梯度 (简化)
        eps_num = 1e-4
        for name in ['enc_W1', 'enc_b1', 'enc_W2', 'enc_b2',
                      'ssm_W_A', 'ssm_W_B', 'ssm_W_C', 'ssm_W_D',
                      'vi_W_mu', 'vi_b_mu', 'vi_W_logvar', 'vi_b_logvar']:
            param = self._get_param(name)
            grad = np.zeros_like(param)
            flat_param = param.flatten()
            flat_grad = np.zeros_like(flat_param)
            
            # 采样部分参数计算梯度 (加速)
            n_sample = min(len(flat_param), 20)
            indices = np.random.choice(len(flat_param), n_sample, replace=False)
            
            for idx in indices:
                orig = flat_param[idx]
                flat_param[idx] = orig + eps_num
                self._set_param(name, flat_param.reshape(param.shape))
                pred_plus = self._forward(X)
                loss_plus = np.mean((pred_plus - y) ** 2)
                
                flat_param[idx] = orig - eps_num
                self._set_param(name, flat_param.reshape(param.shape))
                pred_minus = self._forward(X)
                loss_minus = np.mean((pred_minus - y) ** 2)
                
                flat_param[idx] = orig
                self._set_param(name, flat_param.reshape(param.shape))
                
                flat_grad[idx] = (loss_plus - loss_minus) / (2 * eps_num)
            
            # 对未采样位置使用0梯度
            grads[name] = flat_grad.reshape(param.shape)
        
        # 材料类型分类头梯度 (简化: 使用交叉熵)
        mat_logits = self._cache['mat_logits']
        # 假设y中有材料类别信息(这里简化处理)
        if y.shape[1] > 2:
            mat_target = np.zeros_like(mat_logits)
            mat_target[np.arange(batch_size), (y[:, 2] * self.n_material_classes).astype(int) % self.n_material_classes] = 1
        else:
            mat_target = np.ones_like(mat_logits) / self.n_material_classes
        
        softmax = np.exp(mat_logits - np.max(mat_logits, axis=1, keepdims=True))
        softmax = softmax / np.sum(softmax, axis=1, keepdims=True)
        d_mat = (softmax - mat_target) / batch_size * 0.1  # 小权重
        
        grads['dec_W_mat'] = h_evolved.T @ d_mat
        grads['dec_b_mat'] = np.sum(d_mat, axis=0)
        
        return grads
    
    def _adamw_update(self, grads: Dict[str, np.ndarray], lr: float):
        """AdamW优化器更新"""
        self.adam_t += 1
        
        for name, grad in grads.items():
            if name not in self.adam_m:
                self.adam_m[name] = np.zeros_like(grad)
                self.adam_v[name] = np.zeros_like(grad)
            
            self.adam_m[name] = self.beta1 * self.adam_m[name] + (1 - self.beta1) * grad
            self.adam_v[name] = self.beta2 * self.adam_v[name] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.adam_m[name] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.adam_v[name] / (1 - self.beta2 ** self.adam_t)
            
            param = self._get_param(name)
            # AdamW: 解耦权重衰减
            param = param - lr * m_hat / (np.sqrt(v_hat) + self.eps_adam) - lr * 0.01 * param
            self._set_param(name, param)
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练SSM-PINN模型
        
        Algorithm 3: 多损失联合训练
        """
        if not self._initialized:
            self._init_params(X.shape[1])
        
        progress = ProgressBar(total=self.config.max_iterations, description="SSM-PINN Training")
        
        for iteration in range(self.config.max_iterations):
            # 前向传播
            pred = self._forward(X)
            
            # 计算多目标损失
            losses = self._compute_total_loss(X, y, pred)
            
            # 计算梯度
            grads = self._compute_gradients(X, y)
            
            # AdamW更新
            self._adamw_update(grads, self.config.learning_rate)
            
            # Gumbel-softmax温度退火
            self.tau = max(self.tau_min, 1.0 * (1 - iteration / self.config.max_iterations))
            
            if (iteration + 1) % 50 == 0:
                progress.update(iteration + 1, f"Loss: {losses['total']:.6f}")
        
        progress.finish()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性 [ε, ρ]"""
        if not self._initialized:
            self._init_params(X.shape[1])
        return self._forward(X)
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """带不确定性的预测 (贝叶斯VI)
        
        ŷ = (1/S) Σ y^(s), y^(s) ~ q_φ(y|z)
        Var(ŷ) = (1/S) Σ (y^(s) - ŷ)² + (1/S) Σ σ_φ²(z)
        """
        if not self._initialized:
            self._init_params(X.shape[1])
        
        # 前向传播获取μ, σ
        h_enc = self._encoder(X)
        h_evolved = self._state_evolver(h_enc)
        mu, sigma = self._variational_posterior(h_evolved)
        
        # 蒙特卡洛采样
        samples = np.zeros((n_samples, len(X), 2))
        for s in range(n_samples):
            epsilon = np.random.randn(*mu.shape)
            samples[s] = mu + sigma * epsilon
        
        # 均值和方差
        mean = np.mean(samples, axis=0)
        var = np.var(samples, axis=0) + sigma ** 2
        
        return mean, np.sqrt(var)
    
    def predict_material_type(self, X: np.ndarray) -> np.ndarray:
        """预测材料类型 (Gumbel-softmax)"""
        if not self._initialized:
            self._init_params(X.shape[1])
        
        h_enc = self._encoder(X)
        h_evolved = self._state_evolver(h_enc)
        _, _, mat_logits = self._hard_constraint_decoder(h_evolved)
        
        # Gumbel-softmax采样
        mat_probs = self._gumbel_softmax(mat_logits, tau=self.tau, hard=False)
        
        return mat_probs

class BayesianInversion:
    """贝叶斯变分推断反演"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
    
    def _prior(self, params: np.ndarray) -> float:
        """先验分布"""
        return -0.5 * np.sum(params ** 2)
    
    def _likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """似然函数"""
        eps = params[0]
        rho = params[1]
        
        if eps < 0 or eps > 1 or rho < 0 or rho > 1:
            return -np.inf
        
        if eps + rho > 1 + self.config.constraint_tolerance:
            return -np.inf
        
        laser_intensity = X[:, 0]
        distance = X[:, 1]
        angle = X[:, 2]
        
        expected = laser_intensity * (eps * np.cos(np.deg2rad(angle)) + rho) / (distance ** 2 + 1e-10)
        noise_var = self.config.noise_level ** 2
        
        return -0.5 * np.sum((y[:, 0] - expected) ** 2) / noise_var
    
    def _evidence_lower_bound(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """证据下界"""
        return self._likelihood(params, X, y) + self._prior(params)
    
    def _metropolis_hastings(self, X: np.ndarray, y: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Metropolis-Hastings采样"""
        samples = []
        current = np.array([0.5, 0.4])
        
        for _ in range(n_samples):
            proposal = current + np.random.randn(2) * 0.05
            
            current_energy = -self._evidence_lower_bound(current, X, y)
            proposal_energy = -self._evidence_lower_bound(proposal, X, y)
            
            if np.log(np.random.rand()) < current_energy - proposal_energy:
                current = proposal
            
            samples.append(current.copy())
        
        return np.array(samples)
    
    def predict(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测材料属性（返回均值和标准差）"""
        samples = self._metropolis_hastings(X, y)
        
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        
        if self.config.enforce_hard_constraint:
            if mean[0] + mean[1] > 1:
                scale = 1.0 / (mean[0] + mean[1])
                mean[0] *= scale
                mean[1] *= scale
        
        return mean, std

class IFHBFNN:
    """改进模糊超径向基函数神经网络"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.centers = None
        self.weights = None
        self.sigmas = None
    
    def _gaussian_basis(self, x: np.ndarray, center: np.ndarray, sigma: float) -> float:
        """高斯基函数"""
        return np.exp(-norm(x - center) ** 2 / (2 * sigma ** 2))
    
    def _fuzzy_layer(self, x: np.ndarray) -> np.ndarray:
        """模糊层"""
        n_centers = len(self.centers)
        activations = np.zeros(n_centers)
        
        for i, (center, sigma) in enumerate(zip(self.centers, self.sigmas)):
            activations[i] = self._gaussian_basis(x, center, sigma)
        
        return activations / (np.sum(activations) + 1e-10)
    
    def _inference(self, x: np.ndarray) -> np.ndarray:
        """推理层"""
        fuzzy_output = self._fuzzy_layer(x)
        return fuzzy_output @ self.weights
    
    def _train(self, X: np.ndarray, y: np.ndarray, n_centers: int = 10):
        """训练IFHBFNN"""
        np.random.seed(42)
        
        idx = np.random.choice(len(X), n_centers, replace=False)
        self.centers = X[idx]
        self.sigmas = np.array([np.mean(norm(X - c, axis=1)) for c in self.centers]) * 0.5
        self.weights = np.random.randn(n_centers, 2) * 0.1
        
        progress = ProgressBar(total=self.config.max_iterations, description="IFHBFNN Training")
        
        for iteration in range(self.config.max_iterations):
            pred = np.array([self._inference(x) for x in X])
            
            eps = pred[:, 0]
            rho = pred[:, 1]
            
            if self.config.enforce_hard_constraint:
                total = eps + rho
                mask = total > 1
                eps[mask] = eps[mask] / total[mask]
                rho[mask] = rho[mask] / total[mask]
                pred[:, 0] = eps
                pred[:, 1] = rho
            
            loss = np.mean((pred - y) ** 2)
            
            for i in range(n_centers):
                for j in range(2):
                    grad = 0
                    for k in range(len(X)):
                        fuzzy_out = self._fuzzy_layer(X[k])
                        grad += (pred[k, j] - y[k, j]) * fuzzy_out[i]
                    self.weights[i, j] -= self.config.learning_rate * grad / len(X)
            
            if (iteration + 1) % 50 == 0:
                progress.update(iteration + 1, f"Loss: {loss:.6f}")
        
        progress.finish()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.centers is None:
            self._train(X, np.zeros((len(X), 2)))
        
        pred = np.array([self._inference(x) for x in X])
        
        if self.config.enforce_hard_constraint:
            eps = pred[:, 0]
            rho = pred[:, 1]
            total = eps + rho
            mask = total > 1
            eps[mask] = eps[mask] / total[mask]
            rho[mask] = rho[mask] / total[mask]
            pred[:, 0] = eps
            pred[:, 1] = rho
        
        return np.clip(pred, 0, 1)

class InversionManager:
    """反演管理器"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self._models = {}
    
    def add_model(self, name: str, model):
        """添加反演模型"""
        self._models[name] = model
    
    def get_model(self, name: str):
        """获取反演模型"""
        return self._models.get(name)
    
    def perform_inversion(self, X: np.ndarray, y: np.ndarray = None, 
                          method: str = None) -> InversionResult:
        """执行反演"""
        method = method if method else self.config.method
        
        if method == 'ssm_pinn':
            model = SSMPINN(self.config)
            if y is not None:
                model._train(X, y)
            pred = model.predict(X)
            eps_pred = float(np.mean(pred[:, 0]))
            rho_pred = float(np.mean(pred[:, 1]))
            constraint_satisfied = eps_pred + rho_pred <= 1.0 + self.config.constraint_tolerance
            
            # 贝叶斯VI不确定性量化
            eps_std = 0.0
            rho_std = 0.0
            try:
                mean_pred, std_pred = model.predict_with_uncertainty(X, n_samples=30)
                eps_std = float(np.mean(std_pred[:, 0]))
                rho_std = float(np.mean(std_pred[:, 1]))
            except:
                pass
            
            return InversionResult(
                emissivity_pred=eps_pred,
                emissivity_std=eps_std,
                reflectivity_pred=rho_pred,
                reflectivity_std=rho_std,
                constraint_satisfied=constraint_satisfied,
                method='ssm_pinn'
            )
        
        elif method == 'bayesian':
            model = BayesianInversion(self.config)
            mean, std = model.predict(X, y)
            constraint_satisfied = mean[0] + mean[1] <= 1.0 + self.config.constraint_tolerance
            
            return InversionResult(
                emissivity_pred=float(mean[0]),
                emissivity_std=float(std[0]),
                reflectivity_pred=float(mean[1]),
                reflectivity_std=float(std[1]),
                constraint_satisfied=constraint_satisfied,
                method='bayesian'
            )
        
        elif method == 'ifhbfnn':
            model = IFHBFNN(self.config)
            if y is not None:
                model._train(X, y)
            pred = model.predict(X)
            eps_pred = float(np.mean(pred[:, 0]))
            rho_pred = float(np.mean(pred[:, 1]))
            constraint_satisfied = eps_pred + rho_pred <= 1.0 + self.config.constraint_tolerance
            
            return InversionResult(
                emissivity_pred=eps_pred,
                reflectivity_pred=rho_pred,
                constraint_satisfied=constraint_satisfied,
                method='ifhbfnn'
            )
        
        else:
            raise ValueError(f"Unknown inversion method: {method}")

class FullyConnectedNN:
    """全连接神经网络基线（FC-NN）"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.weights = None
        self.biases = None
    
    def _init_weights(self, input_dim: int, hidden_dims: List[int] = [64, 64, 32]):
        """初始化网络权重"""
        np.random.seed(42)
        dims = [input_dim] + hidden_dims + [2]
        self.weights = {}
        self.biases = {}
        for i in range(len(dims) - 1):
            self.weights[f'W{i+1}'] = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2 / dims[i])
            self.biases[f'b{i+1}'] = np.zeros(dims[i+1])
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        h = x
        for i in range(len(self.weights)):
            h = self._relu(h @ self.weights[f'W{i+1}'] + self.biases[f'b{i+1}'])
        
        output = self._sigmoid(h)
        return output
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练FC-NN"""
        self._init_weights(X.shape[1])
        
        progress = ProgressBar(total=self.config.max_iterations, description="FC-NN Training")
        
        for iteration in range(self.config.max_iterations):
            pred = self._forward(X)
            
            loss = np.mean((pred - y) ** 2)
            
            grads = self._compute_gradients(X, y, pred)
            
            for key in self.weights:
                self.weights[key] -= self.config.learning_rate * grads['dW_' + key]
            for key in self.biases:
                self.biases[key] -= self.config.learning_rate * grads['db_' + key]
            
            if (iteration + 1) % 50 == 0:
                progress.update(iteration + 1, f"Loss: {loss:.6f}")
        
        progress.finish()
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Dict:
        """计算梯度"""
        batch_size = X.shape[0]
        grads = {}
        
        h = [X]
        for i in range(len(self.weights)):
            h.append(self._relu(h[-1] @ self.weights[f'W{i+1}'] + self.biases[f'b{i+1}']))
        
        h[-1] = pred
        grad = (pred - y) * pred * (1 - pred) / batch_size
        
        for i in range(len(self.weights), 0, -1):
            grads[f'dW_W{i}'] = h[i-1].T @ grad
            grads[f'db_b{i}'] = np.sum(grad, axis=0)
            
            if i > 1:
                grad = grad @ self.weights[f'W{i}'].T
                grad[h[i-1] <= 0] = 0
        
        return grads
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.weights is None:
            self._init_weights(X.shape[1])
        
        pred = self._forward(X)
        
        if self.config.enforce_hard_constraint:
            eps = pred[:, 0]
            rho = pred[:, 1]
            total = eps + rho
            mask = total > 1
            eps[mask] = eps[mask] / total[mask]
            rho[mask] = rho[mask] / total[mask]
            pred[:, 0] = eps
            pred[:, 1] = rho
        
        return np.clip(pred, 0, 1)

class PINNFC:
    """物理信息全连接神经网络（PINN-FC）"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.weights = None
        self.biases = None
    
    def _init_weights(self, input_dim: int, hidden_dims: List[int] = [64, 64, 32]):
        """初始化网络权重"""
        np.random.seed(42)
        dims = [input_dim] + hidden_dims + [2]
        self.weights = {}
        self.biases = {}
        for i in range(len(dims) - 1):
            self.weights[f'W{i+1}'] = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2 / dims[i])
            self.biases[f'b{i+1}'] = np.zeros(dims[i+1])
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        h = x
        for i in range(len(self.weights)):
            h = self._relu(h @ self.weights[f'W{i+1}'] + self.biases[f'b{i+1}'])
        
        eps_logit = h[:, 0:1]
        rho_logit = h[:, 1:2]
        
        eps = self._softplus(eps_logit) / (1 + self._softplus(eps_logit))
        rho = self._softplus(rho_logit) / (1 + self._softplus(rho_logit))
        
        if self.config.enforce_hard_constraint:
            total = eps + rho
            scale = np.minimum(1.0, 1.0 / (total + 1e-10))
            eps = eps * scale
            rho = rho * scale
        
        return np.hstack([eps, rho])
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus激活函数"""
        return np.log(1 + np.exp(x))
    
    def _physics_residual(self, X: np.ndarray, pred: np.ndarray) -> float:
        """物理残差损失"""
        eps = pred[:, 0]
        rho = pred[:, 1]
        
        laser_intensity = X[:, 0]
        distance = X[:, 1]
        angle = X[:, 2]
        
        expected = laser_intensity * (eps * np.cos(np.deg2rad(angle)) + rho) / (distance ** 2 + 1e-10)
        actual = X[:, 3] if X.shape[1] > 3 else laser_intensity
        
        return np.mean((actual - expected) ** 2)
    
    def _kirchhoff_loss(self, pred: np.ndarray) -> float:
        """Kirchhoff约束损失"""
        eps = pred[:, 0]
        rho = pred[:, 1]
        violation = np.maximum(0, eps + rho - 1)
        return np.mean(violation ** 2)
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练PINN-FC"""
        self._init_weights(X.shape[1])
        
        progress = ProgressBar(total=self.config.max_iterations, description="PINN-FC Training")
        
        for iteration in range(self.config.max_iterations):
            pred = self._forward(X)
            
            mse_loss = np.mean((pred - y) ** 2)
            physics_loss = self._physics_residual(X, pred)
            constraint_loss = self._kirchhoff_loss(pred)
            
            loss = mse_loss + self.config.regularization_weight * physics_loss + constraint_loss
            
            grads = self._compute_gradients(X, y, pred)
            
            for key in self.weights:
                self.weights[key] -= self.config.learning_rate * grads['dW_' + key]
            for key in self.biases:
                self.biases[key] -= self.config.learning_rate * grads['db_' + key]
            
            if (iteration + 1) % 50 == 0:
                progress.update(iteration + 1, f"Loss: {loss:.6f}")
        
        progress.finish()
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Dict:
        """计算梯度"""
        batch_size = X.shape[0]
        grads = {}
        
        h = [X]
        for i in range(len(self.weights)):
            h.append(self._relu(h[-1] @ self.weights[f'W{i+1}'] + self.biases[f'b{i+1}']))
        
        eps_logit = h[-1][:, 0:1]
        rho_logit = h[-1][:, 1:2]
        
        eps_softplus = self._softplus(eps_logit)
        rho_softplus = self._softplus(rho_logit)
        
        d_eps_d_logit = eps_softplus / ((1 + eps_softplus) ** 2)
        d_rho_d_logit = rho_softplus / ((1 + rho_softplus) ** 2)
        
        grad_h = np.zeros_like(h[-1])
        grad_h[:, 0:1] = (pred[:, 0:1] - y[:, 0:1]) * d_eps_d_logit / batch_size
        grad_h[:, 1:2] = (pred[:, 1:2] - y[:, 1:2]) * d_rho_d_logit / batch_size
        
        for i in range(len(self.weights), 0, -1):
            grads[f'dW_W{i}'] = h[i-1].T @ grad_h
            grads[f'db_b{i}'] = np.sum(grad_h, axis=0)
            
            if i > 1:
                grad_h = grad_h @ self.weights[f'W{i}'].T
                grad_h[h[i-1] <= 0] = 0
        
        return grads
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.weights is None:
            self._init_weights(X.shape[1])
        
        return self._forward(X)

class ResNetModel:
    """残差网络模型（ResNet）"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.weights = None
        self.biases = None
    
    def _init_weights(self, input_dim: int):
        """初始化网络权重"""
        np.random.seed(42)
        hidden_dim = 16
        
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'W2': np.random.randn(hidden_dim, hidden_dim) * 0.1,
            'W3': np.random.randn(hidden_dim, 2) * 0.1
        }
        self.biases = {
            'b1': np.zeros(hidden_dim),
            'b2': np.zeros(hidden_dim),
            'b3': np.zeros(2)
        }
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        h1 = self._relu(x @ self.weights['W1'] + self.biases['b1'])
        h2 = self._relu(h1 @ self.weights['W2'] + self.biases['b2']) + h1
        output = h2 @ self.weights['W3'] + self.biases['b3']
        
        eps = np.clip(output[:, 0:1], 0, 1)
        rho = np.clip(output[:, 1:2], 0, 1)
        
        if self.config.enforce_hard_constraint:
            total = eps + rho
            mask = total > 1
            eps[mask] = eps[mask] / total[mask]
            rho[mask] = rho[mask] / total[mask]
        
        return np.hstack([eps, rho])
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练ResNet（使用伪逆一步训练）"""
        self._init_weights(X.shape[1])
        
        h1 = self._relu(X @ self.weights['W1'] + self.biases['b1'])
        h2 = self._relu(h1 @ self.weights['W2'] + self.biases['b2']) + h1
        
        h2_bias = np.hstack([h2, np.ones((h2.shape[0], 1))])
        
        try:
            W_final = np.linalg.lstsq(h2_bias, y, rcond=None)[0]
            self.weights['W3'] = W_final[:-1, :]
            self.biases['b3'] = W_final[-1, :]
        except:
            pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.weights is None:
            self._init_weights(X.shape[1])
        
        return self._forward(X)

class TransformerModel:
    """Transformer模型"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.weights = None
        self.biases = None
    
    def _init_weights(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        """初始化网络权重"""
        np.random.seed(42)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.weights = {
            'W_proj': np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim),
            'W_q': np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim),
            'W_k': np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim),
            'W_v': np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim),
            'W_o': np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim),
            'W_mlp1': np.random.randn(hidden_dim, hidden_dim * 4) * np.sqrt(2 / hidden_dim),
            'W_mlp2': np.random.randn(hidden_dim * 4, hidden_dim) * np.sqrt(2 / (hidden_dim * 4)),
            'W_final': np.random.randn(hidden_dim, 2) * np.sqrt(2 / hidden_dim)
        }
        self.biases = {
            'b_proj': np.zeros(hidden_dim),
            'b_q': np.zeros(hidden_dim),
            'b_k': np.zeros(hidden_dim),
            'b_v': np.zeros(hidden_dim),
            'b_o': np.zeros(hidden_dim),
            'b_mlp1': np.zeros(hidden_dim * 4),
            'b_mlp2': np.zeros(hidden_dim),
            'b_final': np.zeros(2)
        }
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _multihead_attention(self, x: np.ndarray) -> np.ndarray:
        """多头注意力机制"""
        batch_size, seq_len, _ = x.shape
        
        Q = (x @ self.weights['W_q'] + self.biases['b_q']).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.weights['W_k'] + self.biases['b_k']).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.weights['W_v'] + self.biases['b_v']).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        attn = self._softmax(scores)
        
        output = attn @ V
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return output @ self.weights['W_o'] + self.biases['b_o']
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        
        h = X @ self.weights['W_proj'] + self.biases['b_proj']
        h = h[:, np.newaxis, :] if h.ndim == 2 else h
        
        h = h + self._multihead_attention(h)
        
        mlp_out = self._relu(h @ self.weights['W_mlp1'] + self.biases['b_mlp1'])
        mlp_out = mlp_out @ self.weights['W_mlp2'] + self.biases['b_mlp2']
        h = h + mlp_out
        
        features = np.mean(h, axis=1)
        
        output = features @ self.weights['W_final'] + self.biases['b_final']
        
        eps = self._softplus(output[:, 0:1]) / (1 + self._softplus(output[:, 0:1]))
        rho = self._softplus(output[:, 1:2]) / (1 + self._softplus(output[:, 1:2]))
        
        if self.config.enforce_hard_constraint:
            total = eps + rho
            scale = np.minimum(1.0, 1.0 / (total + 1e-10))
            eps = eps * scale
            rho = rho * scale
        
        return np.hstack([eps, rho])
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus激活函数"""
        return np.log(1 + np.exp(x))
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练Transformer（使用伪逆一步训练）"""
        self._init_weights(X.shape[1])
        
        if X.ndim == 2:
            X_proj = X @ self.weights['W_proj'] + self.biases['b_proj']
            X_proj = X_proj[:, np.newaxis, :]
        else:
            X_proj = X @ self.weights['W_proj'] + self.biases['b_proj']
        
        h = X_proj + self._multihead_attention(X_proj)
        
        mlp_out = self._relu(h @ self.weights['W_mlp1'] + self.biases['b_mlp1'])
        mlp_out = mlp_out @ self.weights['W_mlp2'] + self.biases['b_mlp2']
        h = h + mlp_out
        
        features = np.mean(h, axis=1)
        features_bias = np.hstack([features, np.ones((features.shape[0], 1))])
        
        try:
            W_final = np.linalg.lstsq(features_bias, y, rcond=None)[0]
            self.weights['W_final'] = W_final[:-1, :]
            self.biases['b_final'] = W_final[-1, :]
        except:
            pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.weights is None:
            self._init_weights(X.shape[1])
        
        return self._forward(X)

class S4Model:
    """结构化状态空间模型（S4-Model）"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.weights = None
        self.biases = None
    
    def _init_weights(self, input_dim: int, state_dim: int = 64):
        """初始化网络权重"""
        np.random.seed(42)
        self.state_dim = state_dim
        
        dt = 1.0 / 100
        
        A = -np.random.rand(state_dim)
        B = np.random.randn(state_dim, input_dim) * np.sqrt(2 / input_dim)
        C = np.random.randn(input_dim, state_dim) * np.sqrt(2 / state_dim)
        D = np.random.randn(input_dim) * 0.1
        
        self.A = np.diag(A)
        self.B = B
        self.C = C
        self.D = D
        
        self.A_bar = np.eye(state_dim) + dt * self.A
        self.B_bar = dt * self.B
        
        self.weights = {
            'W_out': np.random.randn(input_dim, 2) * np.sqrt(2 / input_dim)
        }
        self.biases = {
            'b_out': np.zeros(2)
        }
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        
        batch_size, seq_len, input_dim = X.shape
        
        h = np.zeros((batch_size, self.state_dim))
        outputs = []
        
        for t in range(seq_len):
            h = self.A_bar @ h.T + self.B_bar @ X[:, t, :].T
            h = h.T
            
            out_t = np.sum(X[:, t, :] @ self.C * h, axis=1, keepdims=True)
            outputs.append(out_t)
        
        features = np.mean(np.array(outputs), axis=0)
        
        output = features @ self.weights['W_out'] + self.biases['b_out']
        
        eps = self._softplus(output[:, 0:1]) / (1 + self._softplus(output[:, 0:1]))
        rho = self._softplus(output[:, 1:2]) / (1 + self._softplus(output[:, 1:2]))
        
        if self.config.enforce_hard_constraint:
            total = eps + rho
            scale = np.minimum(1.0, 1.0 / (total + 1e-10))
            eps = eps * scale
            rho = rho * scale
        
        return np.hstack([eps, rho])
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus激活函数"""
        return np.log(1 + np.exp(x))
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练S4-Model（使用伪逆一步训练）"""
        self._init_weights(X.shape[1])
        
        if X.ndim == 2:
            X_seq = X[:, np.newaxis, :]
        else:
            X_seq = X
        
        batch_size, seq_len, input_dim = X_seq.shape
        
        h = np.zeros((batch_size, self.state_dim))
        outputs = []
        
        for t in range(seq_len):
            h = self.A_bar @ h.T + self.B_bar @ X_seq[:, t, :].T
            h = h.T
            out_t = np.sum(X_seq[:, t, :] @ self.C * h, axis=1, keepdims=True)
            outputs.append(out_t)
        
        features = np.mean(np.array(outputs), axis=0)
        features_bias = np.hstack([features, np.ones((features.shape[0], 1))])
        
        try:
            W_out = np.linalg.lstsq(features_bias, y, rcond=None)[0]
            self.weights['W_out'] = W_out[:-1, :]
            self.biases['b_out'] = W_out[-1, :]
        except:
            pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.weights is None:
            self._init_weights(X.shape[1])
        
        return self._forward(X)

class MambaModel:
    """选择性状态空间模型（Mamba）"""
    
    def __init__(self, config: InversionConfig = None):
        self.config = config if config else InversionConfig()
        self.weights = None
        self.biases = None
    
    def _init_weights(self, input_dim: int, state_dim: int = 64, expand_dim: int = 128):
        """初始化网络权重"""
        np.random.seed(42)
        self.state_dim = state_dim
        self.expand_dim = expand_dim
        
        self.weights = {
            'W_in': np.random.randn(input_dim, expand_dim * 2) * np.sqrt(2 / input_dim),
            'W_conv': np.random.randn(expand_dim, 3) * np.sqrt(2 / expand_dim),
            'W_state': np.random.randn(expand_dim, state_dim) * np.sqrt(2 / expand_dim),
            'W_out': np.random.randn(expand_dim, 2) * np.sqrt(2 / expand_dim)
        }
        self.biases = {
            'b_in': np.zeros(expand_dim * 2),
            'b_conv': np.zeros(expand_dim),
            'b_out': np.zeros(2)
        }
        
        self.A_log = np.random.randn(state_dim)
        self.D = np.random.randn(state_dim)
    
    def _silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU激活函数"""
        return x * self._sigmoid(x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def _conv1d(self, x: np.ndarray) -> np.ndarray:
        """一维卷积"""
        batch_size, seq_len, dim = x.shape
        
        padded = np.zeros((batch_size, seq_len + 2, dim))
        padded[:, 1:-1, :] = x
        
        output = np.zeros((batch_size, seq_len, dim))
        for t in range(seq_len):
            for k in range(3):
                output[:, t, :] += padded[:, t + k, :] @ self.weights['W_conv'][:, k:k+1]
        
        return output + self.biases['b_conv']
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        h1 = self._silu(X @ self.weights['W_in'][:, :self.expand_dim] + self.biases['b_in'][:self.expand_dim])
        output = h1 @ self.weights['W_out'] + self.biases['b_out']
        
        eps = np.clip(output[:, 0:1], 0, 1)
        rho = np.clip(output[:, 1:2], 0, 1)
        
        if self.config.enforce_hard_constraint:
            total = eps + rho
            mask = total > 1
            eps[mask] = eps[mask] / total[mask]
            rho[mask] = rho[mask] / total[mask]
        
        return np.hstack([eps, rho])
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus激活函数"""
        return np.log(1 + np.exp(x))
    
    def _train(self, X: np.ndarray, y: np.ndarray):
        """训练Mamba（使用伪逆一步训练）"""
        self._init_weights(X.shape[1])
        
        features = X @ self.weights['W_in'][:, :self.expand_dim] + self.biases['b_in'][:self.expand_dim]
        features_bias = np.hstack([features, np.ones((features.shape[0], 1))])
        
        try:
            W_out = np.linalg.lstsq(features_bias, y, rcond=None)[0]
            self.weights['W_out'] = W_out[:-1, :]
            self.biases['b_out'] = W_out[-1, :]
        except:
            pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测材料属性"""
        if self.weights is None:
            self._init_weights(X.shape[1])
        
        return self._forward(X)

def create_inversion_manager(config: InversionConfig = None) -> InversionManager:
    """创建反演管理器"""
    manager = InversionManager(config)
    manager.add_model('ssm_pinn', SSMPINN(config))
    manager.add_model('bayesian', BayesianInversion(config))
    manager.add_model('ifhbfnn', IFHBFNN(config))
    manager.add_model('fc_nn', FullyConnectedNN(config))
    manager.add_model('pinn_fc', PINNFC(config))
    manager.add_model('resnet', ResNetModel(config))
    manager.add_model('transformer', TransformerModel(config))
    manager.add_model('s4', S4Model(config))
    manager.add_model('mamba', MambaModel(config))
    return manager

def create_ssm_pinn_model(config: InversionConfig = None) -> SSMPINN:
    """创建SSM-PINN模型"""
    return SSMPINN(config)

def create_bayesian_model(config: InversionConfig = None) -> BayesianInversion:
    """创建贝叶斯反演模型"""
    return BayesianInversion(config)

def create_ifhbfnn_model(config: InversionConfig = None) -> IFHBFNN:
    """创建IFHBFNN模型"""
    return IFHBFNN(config)

def create_fc_nn_model(config: InversionConfig = None) -> FullyConnectedNN:
    """创建FC-NN模型"""
    return FullyConnectedNN(config)

def create_pinn_fc_model(config: InversionConfig = None) -> PINNFC:
    """创建PINN-FC模型"""
    return PINNFC(config)

def create_resnet_model(config: InversionConfig = None) -> ResNetModel:
    """创建ResNet模型"""
    return ResNetModel(config)

def create_transformer_model(config: InversionConfig = None) -> TransformerModel:
    """创建Transformer模型"""
    return TransformerModel(config)

def create_s4_model(config: InversionConfig = None) -> S4Model:
    """创建S4-Model模型"""
    return S4Model(config)

def create_mamba_model(config: InversionConfig = None) -> MambaModel:
    """创建Mamba模型"""
    return MambaModel(config)
