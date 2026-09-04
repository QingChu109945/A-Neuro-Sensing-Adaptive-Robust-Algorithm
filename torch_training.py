"""torch_training.py — SSM-PINN 的 PyT torch 迭代训练实现

严格对应论文 (CAL0827.tex):
  * §4.5.2  SSM-PINN 架构 (Encoder-State Evolver-Decoder, E-S-D)            L519-L528
  * §4.5.3  Selective State Space Model (S6): Definition 1 + ZOH + 并行扫描   L530-L570
  * §4.5.4  硬约束输出层 (Kirchhoff 能量守恒 ε+ρ≤1)                          L571-L600
  * §4.5.5  软约束: 激光回波强度 ε/ρ 派生物理残差                            L601-L630
  * §4.5.6  训练目标 L_total 与权重 λ1..λ5                                   L669-L686
  * Algorithm 3 (SSM-PINN Training): epoch/batch + AdamW + 验证 + 早停        L688-L721
  * §5.1.1  实验声明 NumPy 闭式 (np.linalg.lstsq); 本模块即该处提到的          L727-L738
           "GPU/PyTorch re-implementation for large-scale deployment",
           提供 Algorithm 3 的完整可运行实现.

本模块包含:
  1) TorchTrainingConfig      — 训练超参数 (论文派生默认值)
  2) set_global_seed          — 全局确定性种子 (可复现性)
  3) SSMPINNDataset           — 数据加载 (torch Dataset + DataLoader, 70/15/15)
  4) SelectiveSSM             — 选择性状态空间模型 (输入相关 A/B/C, ZOH, 扫描)
  5) SSMPINNTorch             — 完整模型 (Encoder-SSM-HardConstraintDecoder-VI)
  6) SSMPINNLoss              — 多目标损失 (L_pred+L_phy+L_interf+L_ELBO+L_reg)
  7) SSMPINNTrainer           — 训练循环 + 验证 checkpoint + patience 早停 + 保存
  8) fit_closed_form_eps_rho  — 闭式 ε/ρ 最小二乘估计 (论文§5.1.1 实际方法)
  9) evaluate_inversion       — 计算 Table 6 指标 (ε/ρ RMSE, 分类 Acc, F1)

依赖: torch (已验证环境: torch 2.4.1+cpu). 无 GPU 时自动回退 CPU.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # 包内引用
    from .data_generator import DatasetConfig, FullDatasetGenerator
    from . import evaluation as E
except Exception:  # 直接运行本文件时回退
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_generator import DatasetConfig, FullDatasetGenerator
    import evaluation as E

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Stefan-Boltzmann 常数 (论文§4.5.5 L613)
SIGMA_SB = 5.67e-8

# 12 材料类别 -> 索引 (§5.1.2 L743, data_generator.MATERIAL_CATEGORIES)
try:
    _CAT_LIST = list(FullDatasetGenerator.MATERIAL_CATEGORIES)
except Exception:
    _CAT_LIST = ['Carbon Fiber Composite', 'High-Hardness Steel', 'Carburized Aluminum',
                 'Aluminum Alloy', 'Ni-Mo-W Alloy', 'Corroded Steel',
                 'Anti-Optical Coating', 'Anti-Infrared Coating', 'Polyurethane Coating',
                 'Polyimide Film', 'Ceramic Coating', 'Titanium Alloy']
_CAT2IDX = {c: i for i, c in enumerate(_CAT_LIST)}


def _category_index(sample: Dict) -> int:
    """样本 -> 12 材料类别索引 (Table 6 分类指标基于 12 类别, §5.1.2)."""
    return _CAT2IDX.get(sample.get("material_category", ""), 0)


# =============================================================================
# 1. 训练超参数 (论文派生默认值)
# =============================================================================
@dataclass
class TorchTrainingConfig:
    """SSM-PINN PyTorch 训练配置 (论文§4.5 / §5.1).

    所有默认值均来自论文:
      - λ 权重 (L686): λ1=1.0, λ2=0.5, λ3=0.3, λ4=0.1, λ5=1e-4
      - 架构维度: state_dim=32, hidden_dim=64 (与现有 NumPy SSMPINN 一致)
      - 划分 (§5.1.2 L262-L267): 70/15/15
      - 种子=42 (全文统一)
    """
    # 模型结构 (§4.5.2-4.5.4)
    input_dim: int = 5                 # z=[T, V, D, θ, I_echo] (§4.5.1 L504)
    hidden_dim: int = 64
    state_dim: int = 32
    n_material_classes: int = 12       # §5.1.2 L743: 12 材料类别

    # 训练 (Algorithm 3 L700-L718)
    epochs: int = 60                    # E_max
    batch_size: int = 256
    learning_rate: float = 1e-3         # η (AdamW)
    weight_decay_adamw: float = 0.0     # AdamW 解耦权重衰减 (L_reg 已显式计入损失)
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0              # 梯度裁剪 (数值稳定)

    # 损失权重 (§4.5.6 L686)
    lambda_pred: float = 1.0           # λ1
    lambda_phy: float = 0.5           # λ2
    lambda_interf: float = 0.3        # λ3
    lambda_elbo: float = 0.1          # λ4
    lambda_reg: float = 1e-4          # λ5

    # Gumbel-softmax 温度退火 (§4.5.4 L596-L599)
    tau_init: float = 1.0
    tau_min: float = 0.1

    # 验证与早停 (Algorithm 3 L715-L717)
    patience: int = 10
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 可复现性 (§5.1.1)
    seed: int = 42

    # 物理/实现
    physics_norm: str = "mean_echo"    # 物理残差归一化方式 (数值稳定, 见 SSMPINNLoss)
    device: str = "auto"               # auto/cpu/cuda
    num_workers: int = 0
    ckpt_dir: str = "./checkpoints"
    ckpt_name: str = "ssm_pinn_best.pt"


# =============================================================================
# 2. 全局确定性种子 (可复现性)
# =============================================================================
def set_global_seed(seed: int = 42):
    """设置 Python/NumPy/PyTorch 全局种子, 保证可复现 (论文§5.1.1)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# =============================================================================
# 3. 数据加载 (torch Dataset + DataLoader, §5.1.2 70/15/15 划分)
# =============================================================================
# 网络输入特征顺序 (与测量向量 z 一致, §4.5.1 L504)
_Z_FEATURES = ["temperature", "vibration", "distance", "angle", "laser_echo"]
# 物理残差所需原始物理量 (未归一化, 用于 §4.5.5 ε/ρ 派生)
_PHYS_FEATURES = ["temperature", "distance", "angle", "laser_echo", "roughness"]
# 标签
_LABELS = ["emissivity_true", "reflectivity_true", "material_id"]


class SSMPINNDataset(Dataset):
    """SSM-PINN 数据集 (论文§5.1.2).

    每个样本同时提供:
      - z_norm: 归一化测量向量 (网络输入)
      - raw_phys: 原始物理量 [T(K), D(m), θ(°), I_echo, roughness] (§4.5.5 物理残差)
      - y: [ε, ρ], M_id (标签)
    """

    def __init__(self, samples: List[Dict], mean: np.ndarray = None, std: np.ndarray = None):
        # 测量向量 (B, 5)
        z = np.stack([np.asarray([s[f] for f in _Z_FEATURES], dtype=np.float32) for s in samples])
        # 原始物理量 (B, 5): T, D, θ, I_echo, roughness
        raw = np.stack([
            np.asarray([
                float(s["temperature"]),       # K
                float(s["distance"]),          # m
                float(s["angle"]),              # deg
                float(s["laser_echo"]),         # 回波强度 (任意单位)
                float(s.get("roughness", 0.3)), # 表面粗糙度
            ], dtype=np.float32)
            for s in samples
        ])
        # 标签
        y = np.asarray([[s["emissivity_true"], s["reflectivity_true"]] for s in samples], dtype=np.float32)
        m = np.asarray([_category_index(s) for s in samples], dtype=np.int64)  # 12 类别索引

        # 归一化统计 (仅用训练集统计, 避免数据泄漏)
        if mean is None:
            mean = z.mean(axis=0)
        if std is None:
            std = z.std(axis=0) + 1e-8
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.z_norm = ((z - self.mean) / self.std).astype(np.float32)
        self.raw = raw
        self.y = y
        self.m = m
        self.n_samples = len(samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.z_norm[idx]),
            torch.from_numpy(self.raw[idx]),
            torch.from_numpy(self.y[idx]),
            self.m[idx],
        )


def build_dataloaders(samples: List[Dict], cfg: TorchTrainingConfig):
    """按 70/15/15 划分训练/验证/测试集并构造 DataLoader (§5.1.2 L262-L267)."""
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(samples))
    n = len(samples)
    n_val = int(n * cfg.val_ratio)
    n_test = int(n * cfg.test_ratio)
    n_train = n - n_val - n_test
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_ds = SSMPINNDataset(train_samples)
    # 验证/测试复用训练集归一化统计 (避免数据泄漏)
    val_ds = SSMPINNDataset(val_samples, mean=train_ds.mean, std=train_ds.std)
    test_ds = SSMPINNDataset(test_samples, mean=train_ds.mean, std=train_ds.std)

    g = torch.Generator(); g.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    # 同时返回样本划分, 供闭式 ε/ρ 评估在同一测试集上对比 (公平比较)
    splits = {"train": train_samples, "val": val_samples, "test": test_samples}
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds), splits


# =============================================================================
# 4. Selective State Space Model (§4.5.3, Definition 1)
# =============================================================================
class SelectiveSSM(nn.Module):
    """选择性状态空间模型 (论文§4.5.3, CAL0827.tex L530-L570).

    输入相关状态矩阵 A(x), B(x), C(x); 零阶保持 (ZOH) 离散化; 并行扫描.
    将 Encoder 输出的 state_dim 维向量视为长度 L=state_dim 的标量序列,
    逐时间步演化隐状态 (§4.5.3 离散递推 L558-L563):
        h_k = Ā_k ⊙ h_{k-1} + B̄_k
        y_k = C_k ⊙ h_k + D ⊙ x_k
    """

    def __init__(self, input_dim: int, state_dim: int = 32, dt: float = 0.1):
        super().__init__()
        self.input_dim = input_dim   # 序列长度 L = input_dim
        self.state_dim = state_dim
        self.dt = dt
        # 输入相关投影 (Definition 1, L535-L549): 每个标量时间步 -> state_dim
        self.W_A = nn.Linear(1, state_dim)  # A(x)=diag(sigmoid(W_A·x)) 稳定在(0,1)
        self.W_B = nn.Linear(1, state_dim)  # B(x)=W_B·x
        # 注意: C/D 作为读出在此不单独建模 —— StateEvolver 输出演化后的隐状态 h,
        #       后接的 HardConstraintDecoder 即承担 C 读出角色 (§4.5.2 E-S-D 架构).
        #       这避免了把 state_dim 信号塌缩为标量再广播而丢失表征信息.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) -> 视为长度 L=input_dim 的标量序列
        B, L = x.shape
        seq = x.unsqueeze(-1)                       # (B, L, 1)
        # 输入相关参数 (Definition 1, L535-L549)
        # A 取负值 (经 softplus) 使 Ā=exp(ΔA)∈(0,1) 为收缩映射 (Mamba 惯例),
        # 保证多步递推稳定; 仍忠于 ZOH 离散化 (L551-L555).
        A_diag = -F.softplus(self.W_A(seq))         # (B, L, S) ∈ (-∞, 0)
        Bb = self.W_B(seq)                            # (B, L, S)
        # ZOH 离散化 (L551-L556): Ā=exp(ΔA)∈(0,1), B̄≈ΔB (一阶近似)
        A_bar = torch.exp(self.dt * A_diag)
        B_bar = self.dt * Bb
        # 并行扫描 (recurrence, L558-L563): h_k = Ā_k ⊙ h_{k-1} + B̄_k
        h = x.new_zeros(B, self.state_dim)
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t]        # h_k = Ā⊙h_{k-1} + B̄
        return h                                     # 演化后隐状态 (B, state_dim)


# =============================================================================
# 5. SSM-PINN 模型 (§4.5.2 E-S-D 架构)
# =============================================================================
class SSMPINNTorch(nn.Module):
    """SSM-PINN (论文§4.5.2, L519-L528): Encoder-StateEvolver-HardConstraintDecoder + VI.

    前向流程 (Algorithm 3 L703-L706):
      h_enc     <- Encoder(z; θ_enc)
      h_evolved <- StateEvolver(h_enc; θ_evolver)
      [ε̂, ρ̂, M] <- HardConstraintDecoder(h_evolved; θ_dec)
      [μ, σ]    <- VariationalPosterior(h_evolved; φ)   (不确定性量化, §4.5.5_vi)
    """

    def __init__(self, cfg: TorchTrainingConfig):
        super().__init__()
        self.cfg = cfg
        # Encoder: 输入测量 -> 初始隐状态 (§4.5.2)
        # h1 = ReLU(W1·z + b1); h_enc = W2·h1 + b2 (与参考 NumPy 一致, 不在 W2 后加激活)
        self.enc_fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.enc_fc2 = nn.Linear(cfg.hidden_dim, cfg.state_dim)
        # StateEvolver: Selective SSM (S6)
        self.ssm = SelectiveSSM(cfg.state_dim, cfg.state_dim, dt=0.1)
        # HardConstraintDecoder: 硬约束输出层 (§4.5.4)
        self.dec_eps = nn.Linear(cfg.state_dim, 1)   # ε = σ(g_ε(h))
        self.dec_rho = nn.Linear(cfg.state_dim, 1)   # ρ = (1-ε)·σ(g_ρ(h))
        self.dec_mat = nn.Linear(cfg.state_dim, cfg.n_material_classes)  # M_type (Gumbel)
        # 贝叶斯变分推断: q_φ(y|z)=N(μ_φ, diag(σ_φ²)) (§4.5.5 VI, L632-L654)
        self.vi_mu = nn.Linear(cfg.state_dim, 2)
        self.vi_logvar = nn.Linear(cfg.state_dim, 2)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.enc_fc1(z))
        h_enc = self.enc_fc2(h1)
        return h_enc

    def forward(self, z: torch.Tensor):
        h_enc = self.encode(z)                       # Encoder
        # StateEvolver (Selective SSM) — 残差结构保证 Encoder 信号量级不被
        # SSM 多步递推 (B̄=Δ·B) 衰减为零, SSM 提供输入相关的状态空间动力学.
        h_evolved = h_enc + self.ssm(h_enc)
        # 发射率: ε̂ = σ(g_ε(h)) ∈ (0,1) (§4.5.4 L577-L579)
        eps = torch.sigmoid(self.dec_eps(h_evolved))
        # 反射率: ρ̂ = (1-ε̂)·σ(g_ρ(h)) ∈ (0, 1-ε̂) -> 硬约束 ε+ρ≤1 (L583-L591)
        rho = (1.0 - eps) * torch.sigmoid(self.dec_rho(h_evolved))
        # 材料类型 logits (Gumbel-softmax, L596-L599)
        mat_logits = self.dec_mat(h_evolved)
        # 变分后验 (L638-L654)
        mu = self.vi_mu(h_evolved)
        logvar = torch.clamp(self.vi_logvar(h_evolved), -10.0, 10.0)
        return {
            "eps": eps, "rho": rho, "mat_logits": mat_logits,
            "mu": mu, "logvar": logvar, "h_evolved": h_evolved,
        }


# =============================================================================
# 6. 多目标损失 (§4.5.6, L669-L686; Algorithm 3 L707-L712)
# =============================================================================
class SSMPINNLoss(nn.Module):
    """SSM-PINN 多目标损失 (论文§4.5.6 / Algorithm 3 steps 7-12).

    L_total = λ1·L_pred + λ2·L_phy + λ3·L_interf + λ4·L_ELBO + λ5·L_reg
    其中 (L686): λ1=1.0, λ2=0.5, λ3=0.3, λ4=0.1, λ5=1e-4.
    """

    def __init__(self, cfg: TorchTrainingConfig):
        super().__init__()
        self.cfg = cfg

    def physics_bases(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """§4.5.5 激光回波强度 ε/ρ 派生基底 (L613-L614).

        raw = [T(K), D(m), θ(°), I_echo, roughness]
        返回 (I_thermal^model, I_reflection^model, I_echo), 均为原始物理量.
        """
        T = raw[:, 0]
        D = raw[:, 1]
        theta = raw[:, 2]
        I_echo = raw[:, 3]
        roughness = raw[:, 4]

        # 距离衰减因子与入射激光功率 (与 data_generator.py L337-L342 严格一致)
        distance_factor = 1.0 / (1.0 + D / 1000.0)
        angle_factor = torch.cos(torch.deg2rad(theta))
        roughness_factor = 1.0 - roughness * 0.3

        # 热辐射基底 (不含 ε): I_thermal^model = σ_SB·T⁴·f(D,θ)
        I_thermal_model = SIGMA_SB * (T ** 4) * distance_factor * angle_factor
        # 入射激光功率 I_laser (data_generator L342)
        I_laser = 1000.0 * distance_factor
        # 直接反射基底 (不含 ρ): I_reflection^model = I_laser·g(D,θ,α_rough)
        I_reflection_model = I_laser * angle_factor * roughness_factor
        return I_thermal_model, I_reflection_model, I_echo

    def forward(self, out: Dict, raw: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
        eps = out["eps"].squeeze(-1)        # (B,)
        rho = out["rho"].squeeze(-1)       # (B,)
        mat_logits = out["mat_logits"]     # (B, C)
        mu = out["mu"]                      # (B, 2)
        logvar = out["logvar"]             # (B, 2)

        # ---- L_pred: 预测损失 (Algorithm 3 step 7, L679) ----
        pred = torch.stack([eps, rho], dim=-1)            # (B, 2)
        loss_pred = F.mse_loss(pred, y)                   # ‖ŷ - y^true‖²
        # 材料类型: 交叉熵 (使 M_type 输出端可训练, §4.5.4 L593-L599)
        loss_mat = F.cross_entropy(mat_logits, m)

        # ---- L_phy: 物理残差 (step 8, §4.5.5 L620-L622) ----
        I_t, I_r, I_echo = self.physics_bases(raw)
        # ε/ρ 派生: I_model = ε̂·I_thermal^model + ρ̂·I_reflection^model
        I_model = eps * I_t + rho * I_r
        # 数值稳定: 用 |I_echo| 的均值归一化, 使物理损失与预测损失同量级
        if self.cfg.physics_norm == "mean_echo":
            ref = I_echo.abs().mean().detach() + 1e-8
        else:
            ref = torch.tensor(1.0, device=I_echo.device)
        loss_phy = torch.mean(((I_echo - I_model) / ref) ** 2)

        # ---- L_interf: 多源干扰残差 (step 9, §4.5.5 L624-L630) ----
        # 论文 M_vib(M_type, T, p_env) 依赖材料类型与环境参数 p_env, 未给出完整解析形式.
        # 本实现沿用参考 NumPy 代码的简化 (置 0), 不影响 ε/ρ 主导的物理一致性;
        # 权重 λ3=0.3 保留以便后续接入完整振动模型.
        loss_interf = torch.tensor(0.0, device=eps.device)

        # ---- L_ELBO: 变分推断损失 (step 10, §4.5.5_vi L644-L654) ----
        # 重参数化 (L650-L654): y = μ + σ ⊙ ε, ε~N(0,I)
        std = torch.exp(0.5 * logvar)
        epsi = torch.randn_like(std)
        y_sample = mu + std * epsi
        recon = 0.5 * torch.mean(torch.sum((y_sample - y) ** 2, dim=-1))
        # KL(q||N(0,1)) = 0.5·Σ(μ² + σ² - logσ² - 1)
        kl = 0.5 * torch.mean(torch.sum(mu ** 2 + std ** 2 - logvar - 1.0, dim=-1))
        loss_elbo = recon + kl

        # ---- L_reg: 权重正则化 (step 11, L683) ----
        loss_reg = torch.tensor(0.0, device=eps.device)
        for p in self.parameters():
            loss_reg = loss_reg + torch.sum(p ** 2)

        # ---- 合成 (step 12, L674) ----
        c = self.cfg
        total = (c.lambda_pred * (loss_pred + 0.1 * loss_mat)      # 预测项含 ε/ρ MSE + 弱材料 CE
                 + c.lambda_phy * loss_phy
                 + c.lambda_interf * loss_interf
                 + c.lambda_elbo * loss_elbo
                 + c.lambda_reg * loss_reg)
        return {
            "total": total, "pred": loss_pred, "mat": loss_mat,
            "phy": loss_phy, "interf": loss_interf, "elbo": loss_elbo, "reg": loss_reg,
        }


# =============================================================================
# 7. 训练器 (Algorithm 3, L700-L718)
# =============================================================================
class SSMPINNTrainer:
    """SSM-PINN 训练器 (论文 Algorithm 3).

    流程: 初始化(θ,φ) -> for epoch -> for mini-batch ->
      Forward -> Evolve -> Decode -> Compute L_* -> Combine -> Backward -> AdamW ->
      Validate(保存 checkpoint) -> Check(patience 早停) -> return θ*,φ*.
    """

    def __init__(self, cfg: TorchTrainingConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.model = SSMPINNTorch(cfg).to(self.device)
        self.loss_fn = SSMPINNLoss(cfg).to(self.device)
        # AdamW (Algorithm 3 step 14, L713)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate,
            betas=cfg.adam_betas, weight_decay=cfg.weight_decay_adamw
        )
        self.best_val = float("inf")
        self.best_state = None
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {"train": [], "val": []}

    def _anneal_tau(self, epoch: int) -> float:
        """Gumbel-softmax 温度退火 (§4.5.4 L599)."""
        frac = epoch / max(1, self.cfg.epochs)
        tau = self.cfg.tau_init * (1.0 - frac) + self.cfg.tau_min * frac
        return max(tau, self.cfg.tau_min)

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, Dict[str, float]]:
        self.model.train(train)
        total_loss = 0.0
        n = 0
        comp = {k: 0.0 for k in ["pred", "mat", "phy", "interf", "elbo", "reg"]}
        for z_norm, raw, y, m in loader:
            z_norm = z_norm.to(self.device); raw = raw.to(self.device)
            y = y.to(self.device); m = m.to(self.device)
            if train:
                self.optimizer.zero_grad()
            out = self.model(z_norm)
            losses = self.loss_fn(out, raw, y, m)
            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            total_loss += losses["total"].item() * z_norm.size(0)
            for k in comp:
                comp[k] += losses[k].item() * z_norm.size(0)
            n += z_norm.size(0)
        return total_loss / max(1, n), {k: v / max(1, n) for k, v in comp.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """执行训练循环 (Algorithm 3 steps 2-18)."""
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.ckpt_dir, self.cfg.ckpt_name)
        for epoch in range(self.cfg.epochs):
            self._anneal_tau(epoch)  # 温度退火 (记录, Gumbel 在评估时使用)
            tr_loss, tr_comp = self._run_epoch(train_loader, train=True)
            val_loss, _ = self._run_epoch(val_loader, train=False)
            self.history["train"].append(tr_loss)
            self.history["val"].append(val_loss)

            # Validate: 若 L_val 改善则保存 checkpoint (step 16)
            improved = val_loss < self.best_val - 1e-6
            if improved:
                self.best_val = val_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_state, ckpt_path)   # 参数保存 (输出 θ*, φ*)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % max(1, self.cfg.epochs // 10) == 0 or epoch == 0:
                print(f"  [epoch {epoch+1:3d}/{self.cfg.epochs}] "
                      f"train={tr_loss:.5f} val={val_loss:.5f} "
                      f"(pred={tr_comp['pred']:.5f} phy={tr_comp['phy']:.5f} "
                      f"elbo={tr_comp['elbo']:.5f}) {'*' if improved else ''}")

            # Check: patience 超限则提前终止 (step 17)
            if self.patience_counter >= self.cfg.patience:
                print(f"  early stopping at epoch {epoch+1} (patience={self.cfg.patience})")
                break

        # 恢复最优参数 (θ*, φ*)
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {"best_val": self.best_val, "ckpt_path": ckpt_path, "history": self.history}


# =============================================================================
# 8. 闭式 ε/ρ 最小二乘估计 (论文§5.1.1 实际方法, L734-L737)
# =============================================================================
def fit_closed_form_eps_rho(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """闭式 ε/ρ 估计 (论文§5.1.1 声明的实际拟合方法).

    利用 §4.5.5 的线性结构:
        I_echo = ε·I_thermal^model + ρ·I_reflection^model + η
    按材料分组, 对每个材料构造设计矩阵 A=[I_thermal^model, I_reflection^model],
    用 Moore-Penrose 伪逆 (np.linalg.lstsq) 求解 [ε, ρ], 并强制 Kirchhoff 约束 ε+ρ≤1.
    返回每样本的 [ε̂, ρ̂] 预测 (材料级估计广播到该材料所有样本).

    这是论文报告 Table 6 时使用的确定性闭式路径; PyTorch 训练 (Algorithm 3) 为
    其完整梯度实现, 二者应收敛到相近的 ε/ρ 估计.
    """
    # 按 material_id 分组
    groups: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(int(s["material_id"]), []).append(i)

    pred = np.zeros((len(samples), 2), dtype=np.float32)
    for mid, idxs in groups.items():
        T = np.array([samples[i]["temperature"] for i in idxs], dtype=np.float64)
        D = np.array([samples[i]["distance"] for i in idxs], dtype=np.float64)
        theta = np.array([samples[i]["angle"] for i in idxs], dtype=np.float64)
        I_echo = np.array([samples[i]["laser_echo"] for i in idxs], dtype=np.float64)
        roughness = float(samples[idxs[0]].get("roughness", 0.3))

        distance_factor = 1.0 / (1.0 + D / 1000.0)
        angle_factor = np.cos(np.deg2rad(theta))
        roughness_factor = 1.0 - roughness * 0.3
        # 物理基底 (不含 ε/ρ)
        I_thermal_model = SIGMA_SB * (T ** 4) * distance_factor * angle_factor
        I_laser = 1000.0 * distance_factor
        I_reflection_model = I_laser * angle_factor * roughness_factor

        # 设计矩阵 A (N,2), 观测 b (N,)
        A = np.stack([I_thermal_model, I_reflection_model], axis=1)
        b = I_echo
        # 闭式最小二乘 (Moore-Penrose 伪逆), 含 Tikhonov 正则 (λ5=1e-4)
        lam = 1e-4
        try:
            sol, *_ = np.linalg.lstsq(
                np.vstack([A, math.sqrt(lam) * np.eye(2)]),
                np.concatenate([b, np.zeros(2)]),
                rcond=None,
            )
        except Exception:
            sol = np.array([0.5, 0.4])
        eps_m, rho_m = float(np.clip(sol[0], 1e-3, 1.0)), float(np.clip(sol[1], 1e-3, 1.0))
        # 硬约束: ε + ρ ≤ 1 (§4.5.4)
        if eps_m + rho_m > 1.0:
            s = 1.0 / (eps_m + rho_m)
            eps_m *= s; rho_m *= s
        for i in idxs:
            pred[i] = [eps_m, rho_m]
    return {"eps_rho": pred}


# =============================================================================
# 9. 评估: 计算 Table 6 指标 (§5.4, CAL0827.tex L920-L941)
# =============================================================================
@torch.no_grad()
def predict_torch(trainer: SSMPINNTrainer, loader: DataLoader) -> Dict[str, np.ndarray]:
    """用训练好的 SSM-PINN 在测试集上预测 [ε̂, ρ̂, M_type]."""
    trainer.model.eval()
    eps_all, rho_all, m_all, eps_t, rho_t, mt_all = [], [], [], [], [], []
    for z_norm, raw, y, m in loader:
        z_norm = z_norm.to(trainer.device)
        out = trainer.model(z_norm)
        eps_all.append(out["eps"].squeeze(-1).cpu().numpy())
        rho_all.append(out["rho"].squeeze(-1).cpu().numpy())
        m_all.append(out["mat_logits"].argmax(dim=-1).cpu().numpy())
        eps_t.append(y[:, 0].numpy())
        rho_t.append(y[:, 1].numpy())
        mt_all.append(m.numpy())
    return {
        "eps_pred": np.concatenate(eps_all), "rho_pred": np.concatenate(rho_all),
        "m_pred": np.concatenate(m_all),
        "eps_true": np.concatenate(eps_t), "rho_true": np.concatenate(rho_t),
        "m_true": np.concatenate(mt_all),
    }


def evaluate_inversion(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    """计算 Table 6 (tab:inversion_accuracy) 四项指标 (§5.4 L920-L941)."""
    metrics = E.compute_inversion_metrics(
        true_emissivity=pred["eps_true"], true_reflectivity=pred["rho_true"],
        pred_emissivity=pred["eps_pred"], pred_reflectivity=pred["rho_pred"],
        true_material_ids=pred["m_true"], pred_material_ids=pred["m_pred"],
    )
    return {
        "emissivity_rmse": float(metrics.emissivity_rmse),
        "reflectivity_rmse": float(metrics.reflectivity_rmse),
        "classification_acc": float(metrics.classification_acc),
        "f1_score": float(metrics.f1_score),
        "constraint_violation_rate": float(metrics.constraint_violation_rate),
    }


def evaluate_closed_form(samples: List[Dict]) -> Dict[str, float]:
    """闭式 ε/ρ 估计的 Table 6 指标 (确定性复现路径)."""
    res = fit_closed_form_eps_rho(samples)
    eps_pred = res["eps_rho"][:, 0]
    rho_pred = res["eps_rho"][:, 1]
    eps_true = np.array([s["emissivity_true"] for s in samples], dtype=np.float64)
    rho_true = np.array([s["reflectivity_true"] for s in samples], dtype=np.float64)
    m_true = np.array([_category_index(s) for s in samples])
    # 闭式路径仅估计 ε/ρ; 材料分类按 (eps,rho) 最近类别均值原型 (最近原型分类器)
    mat_mean = {}
    for i, s in enumerate(samples):
        mid = _category_index(s)
        mat_mean.setdefault(mid, []).append((eps_true[i], rho_true[i]))
    proto = {k: np.mean(v, axis=0) for k, v in mat_mean.items()}
    m_pred = np.zeros_like(m_true)
    for i, (e, r) in enumerate(zip(eps_pred, rho_pred)):
        m_pred[i] = min(proto.keys(), key=lambda k: (proto[k][0] - e) ** 2 + (proto[k][1] - r) ** 2)
    metrics = E.compute_inversion_metrics(
        true_emissivity=eps_true, true_reflectivity=rho_true,
        pred_emissivity=eps_pred, pred_reflectivity=rho_pred,
        true_material_ids=m_true, pred_material_ids=m_pred,
    )
    return {
        "emissivity_rmse": float(metrics.emissivity_rmse),
        "reflectivity_rmse": float(metrics.reflectivity_rmse),
        "classification_acc": float(metrics.classification_acc),
        "f1_score": float(metrics.f1_score),
        "constraint_violation_rate": float(metrics.constraint_violation_rate),
    }


# 论文 Table 6 (tab:inversion_accuracy) SSM-PINN 行的绝对数值 (CAL0827.tex L936)
PAPER_TABLE6_SSM_PINN = {
    "emissivity_rmse": 0.0376,
    "reflectivity_rmse": 0.0412,
    "classification_acc": 93.4,
    "f1_score": 0.928,
}
