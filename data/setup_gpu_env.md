# GPU 加速运行环境配置说明 (RTX 5070 Ti / Blackwell)

本说明记录如何为 `public_dataset_validation.py` 的 GPU 加速路径配置运行环境，
并给出实测性能与复现命令。

## 一、硬件与现状

- **GPU**：NVIDIA GeForce RTX 5070 Ti（16 GB），架构 Blackwell（`sm_120`）。
- **原环境（base）问题**：
  - 安装的是 `torch==2.4.1+cpu`（纯 CPU 版），`torch.cuda.is_available()` 返回 `False`。
  - 解释器为 **Python 3.8.8**。PyTorch 自 2.5 起不再发布 3.8 的 wheel，而支持
    Blackwell `sm_120` 需要 **PyTorch ≥ 2.7 + CUDA 12.8**，只提供 Python ≥ 3.9 的 wheel。
  - 因此在原 base 环境**无法**安装可驱动 RTX 5070 Ti 的 CUDA PyTorch。

## 二、使用的 GPU 环境

实测采用已具备 CUDA PyTorch 的 `dream_code_v2` 环境（避免重复下载大体积 wheel）：

```
解释器：C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe
PyTorch 版本：2.8.0.dev20250316+cu128
CUDA 版本：12.8
CUDA 是否可用：True
设备：NVIDIA GeForce RTX 5070 Ti
```

如需从零新建，可参考：

```bash
conda create -n m2gpu python=3.10 -y
"C:\ProgramData\Anaconda3\envs\m2gpu\python.exe" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 torch numpy
"C:\ProgramData\Anaconda3\envs\m2gpu\python.exe" -m pip install psutil
```

> `--index-url .../whl/cu128` 指定 CUDA 12.8 wheel 源；这是当前支持 RTX 5070 Ti 的最低 CUDA 版本。
> `psutil` 为 `experiment_system` 导入所需。

## 三、验证 GPU 可用

```bash
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
期望输出形如：`2.8.0.dev...+cu128 True NVIDIA GeForce RTX 5070 Ti`

## 四、用 GPU 后端运行验证实验

```bash
# 在项目根目录 e:\Document\Code\2026\08\M2 下
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -m experiment_system.data.public_dataset_validation --backend torch
```

- `--backend torch`：SSM-PINN 使用 `inversion_torch.py` 的 GPU autograd 实现（CUDA 优先，否则 CPU）。
- 默认写入 `public_validation_results_gpu.json`（不覆盖 CPU 版 `public_validation_results.json`）。
- `--backend numpy`（默认）：仍走原 NumPy 有限差分实现，结果与手稿逐位一致，可复现。

整个验证（滤波 + 反演 + 交叉验证）GPU 路径 **约 35 s** 完成；原纯 CPU 路径约 **31 分钟**。

## 五、性能基准（CPU vs GPU，同数据集/同迭代/同学习率）

```bash
# 全量对比 + 写 JSON（含 CPU 有限差分基线，较慢）
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -m experiment_system.data.benchmark_gpu --iters 50 --json gpu_benchmark_iters50.json
# 仅 GPU（快速）
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -m experiment_system.data.benchmark_gpu --iters 300 --skip-numpy --json gpu_benchmark_gpu_only.json
```

实测（MODIS UCSB + SLUM，n_train=4757 / n_test=1020，dim=6）：

| 迭代 | 后端 | 设备 | 训练耗时 | 发射率 RMSE | R² | 显存 |
|------|------|------|---------|-------------|-----|------|
| 50   | GPU  | RTX 5070 Ti | 2.32 s  | 0.086 | 0.714 | 24.4 MB |
| 50   | CPU  | 有限差分     | 160.98 s | 0.207 | -0.66 | — |
| 300  | GPU  | RTX 5070 Ti | 4.52 s  | 0.047 | 0.914 | 24.4 MB |

- **同迭代加速比：约 60–70×**（50 迭代：160.98 s → 2.32 s ≈ 69×）。
- GPU 在更短时间内达到**更低 RMSE**（autograd 精确梯度 vs CPU 采样有限差分），
  300 迭代下 SSM-PINN 发射率 RMSE 降至 0.047、R²=0.914，且 Kirchhoff 违反率为 0。

## 六、加速原理与数据传输优化

原 SSM-PINN 训练用**有限差分数值梯度**（每次迭代对参数逐个 ±ε 两次前向），
占整个验证运行时的约 **98%**（~30 分钟）。GPU 路径改为 `torch.autograd` **真实反向传播**：
一次前向 + 一次反向即得全部梯度，且矩阵运算在 GPU 上并行执行。

数据传输优化：训练数据（X/y）在 fit **开始时一次性上传** GPU，训练全程驻留显存，
仅在预测结束时把结果**一次性下载**回主机（`.cpu().numpy()`），
CPU↔GPU 交互降到每次拟合两次，传输开销可忽略。

## 七、OpenMP 运行时冲突（重要）

NumPy(MKL) 与 PyTorch 各自捆绑一份 Intel OpenMP 运行时（`libiomp5md.dll`），
在**同进程同时运行** NumPy 反演与 torch GPU 训练（如 CPU-vs-GPU 基准）时会报
`OMP: Error #15 ... libiomp5md.dll already initialized` 并以退出码 3 崩溃。

已在代码中修复：`inversion_torch.py` 与 `benchmark_gpu.py` 在导入 torch **之前**
设置 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`。
纯 GPU 路径（`--backend torch` 不训练 NumPy SSM-PINN）不受影响。

## 八、用 GPU 结果对齐手稿与重绘图（复现步骤）

GPU 反演结果写入 `public_validation_results_gpu.json`（SSM-PINN 由 CUDA autograd 训练）。
手稿 `CAL0828.tex` 的表 `tab:public_inversion` 与 5.2 节讨论、两张公开验证图均已改用该结果。

```bash
# 1) 生成 GPU 结果 JSON（若尚未生成）
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -m experiment_system.data.public_dataset_validation --backend torch

# 2) 逐格校验手稿数值（默认读 GPU JSON，全部应 PASS）
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -m experiment_system.data.check_manuscript_alignment
#   如需回退核对 CPU 版：加 --json public_validation_results.json

# 3) 用 GPU JSON 重绘公开验证图（须以 -m 方式运行以满足包内相对导入）
"C:\ProgramData\Anaconda3\envs\dream_code_v2\python.exe" -m experiment_system.generate_validation_figures --public public_validation_results_gpu.json
```

- `check_manuscript_alignment.py` 与 `generate_validation_figures.py` 均默认读取
  `public_validation_results_gpu.json`，`--json` / `--public` 可切回 CPU 版结果。
- 手稿 GPU 数值：SSM-PINN 发射率 RMSE **0.047**、$R^2$ **0.914**、Kirchhoff 违反率 **0.0%**、PICP$_{95}$ **1.000**。
- 图 `public_inversion_bars.png` 中 SSM-PINN（蓝）RMSE 约为 Mamba/ResNet 的一半。

> 注：CUDA 驱动初始化会写 `C:\ProgramData\NVIDIA Corporation\Drs\nvAppTimestamps`，
> 在受限沙箱下该写入被拦截，可能使进程返回非零退出码，但**发生在图片保存之后**，
> 不影响输出；如需干净退出，可在沙箱外运行绘图命令。

## 九、一键两阶段实验 (run_experiment.py)

`run_experiment.py` 默认以**两阶段模式**自动执行完整实验，无需人工干预：

- **Phase 1 (simulator)**：合成仿真数据集 → 温度补偿 → NS-ARKF 滤波 → SSM-PINN 反演
  → 统计分析 → 可视化 → 自适应阈值验证。对应手稿合成研究（表 `tab:filtering_rmse`、
  `tab:inversion_accuracy`）。
- **Phase 2 (public)**：公开数据集（MODIS UCSB + SLUM）→ 同一 `predict/update` 与
  `fit/predict` 接口 → 标准化测试结果。对应手稿公开验证（表 `tab:public_filtering`、
  `tab:public_inversion`）。

每个阶段使用**独立工作目录**（`run_<时间戳>/phase1_simulator/work_data/` 与
`phase2_public/work_data/`），避免历史数据累积导致归档卡顿。归档阶段不再 `copytree`
整个 `./data`，而是就地保留阶段目录，60 s 采集的总端到端用时 < 80 s。

```bash
# 两阶段自动执行（默认）
python run_experiment.py --no-interactive --duration 60

# 仅单阶段（向后兼容）
python run_experiment.py --no-interactive --duration 60 --data-source simulator
python run_experiment.py --no-interactive --duration 60 --data-source public
```

两阶段滤波 RMSE 对比（实测）：

| 阶段 | 数据源 | 滤波 RMSE | 说明 |
|------|--------|----------|------|
| Phase 1 | simulator | 0.330 | 含注入极端噪声 |
| Phase 2 | public | 0.053 | 确定性公开数据，RMSE 低一个数量级 |

> Phase 2 的 SSM-PINN 反演数值仍以 `public_dataset_validation.py --backend torch`
> 的 GPU 结果为准（`public_validation_results_gpu.json`），手稿已对齐。
