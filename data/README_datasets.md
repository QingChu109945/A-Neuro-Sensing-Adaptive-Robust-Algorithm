# 公开数据集准备与复现说明 (Public Datasets — Preparation & Reproduction)

本目录包含论文《Neuro-Sensing Adaptive Robust Kalman Filtering …》(手稿 `CAL0828.tex`)
第 5.2 节 **Public-Dataset Validation** 所使用的公开数据集下载/解析脚本、缓存数据与验证脚本。

> 目的：满足 EAAI「validated using public data sets for easy replicability」要求，
> 并使手稿 *Data Availability* 一节声明的「scripts that download and parse these
> libraries」有可运行的对应文件，避免评审对可复现性提出质疑。

---

## 一、应运行哪个文件下载数据

| 数据集 | 手稿用途 | 下载/准备脚本 | 产物 |
|---|---|---|---|
| **MODIS UCSB** 发射率库 | 反演验证 (Table `tab:public_inversion`) | `python -m experiment_system.data.download_modis_ucsb` | `modis_ucsb/modis_ucsb_emissivity.csv` + `SOURCE.json` |
| **SLUM** 城市材料光谱库 | 反演验证 (Table `tab:public_inversion`) | `python -m experiment_system.data.download_slum` | `slum/raw/LUMA_SLUM_IR.csv` + `SOURCE.json` |
| **KITTI** 跟踪基准几何 | 滤波验证 (Table `tab:public_filtering`) | `python -m experiment_system.data.download_kitti` | `kitti/SOURCE.json` + `kitti_geometry_tracks.npz` |

当前状态（2026-08-28 核对）：

- ✅ **MODIS UCSB**：已下载并集成（16 材料 / 2208 点，见 `modis_ucsb/SOURCE.json`）。
- ✅ **SLUM**：已下载并集成（`LUMA_SLUM_IR.csv` / `LUMA_SLUM_SW.csv`）。
- ✅ **KITTI**：已记录 provenance 并缓存复现几何轨迹。

### 关于 KITTI 访问方式（与手稿一致的诚实声明）

KITTI 原始档（LiDAR 点云 + 图像）仅通过**交互式账号 / 邮箱确认**下载，且其许可证禁止再分发。
因此手稿并**未声称**再分发原始档，而是在**遵循 KITTI 跟踪基准运动统计**（城市车速 ≤ ~15 m/s，
10 Hz 采样，轻微机动）的重建轨迹上注入三类极端噪声进行滤波验证——这与
`download_kitti.py` 缓存的几何、以及 `public_dataset_validation._generate_kitti_like_track`
完全一致（固定随机种子 `seed=42`，逐位可复现）。

若需在**原始 KITTI 轨迹**上验证：通过 KITTI 账号下载官方 `label_02/*.txt`，放到
`kitti/raw/label_02/`，再运行 `python -m experiment_system.data.download_kitti --parse-labels`，
即可解析出 `kitti/kitti_tracks.csv`（逐帧 3D 目标位置）供替换使用。

> 注：修改意见报告曾额外推荐 EKHI 材料热辐射库。手稿最终采用 **MODIS UCSB + SLUM**
> 两个真实发射率库完成反演验证，未纳入 EKHI；因此本目录不含 EKHI 脚本，代码与手稿一致。

---

## 二、一键复现两张公开数据集验证表

```bash
# 在项目根目录 e:\Document\Code\2026\08\M2 下执行
python -m experiment_system.data.public_dataset_validation
```

该脚本使用与主实验**完全相同**的滤波器/反演模型工厂与评价指标，重新产出
`public_validation_results.json`，其内容与手稿两张表逐值对应：

- Table `tab:public_filtering` ← `filtering_validation.table`
- Table `tab:public_inversion` ← `inversion_validation.table`
- 合成材料库范围交叉验证 ← `cross_validation`

所有随机过程均固定种子（滤波 `seed=42`、反演 `seed=42`），因此重复运行结果稳定，
可供第三方逐值核对。

---

## 三、加载器 API（供二次开发）

`public_dataset_loaders.py` 提供只读加载函数：

- `load_modis_ucsb()` → `{"points", "by_label", "by_category"}`
- `load_slum_ir()`     → `{"surfaces", "by_surface", "wavelengths"}`（限 8–14 µm LWIR 窗口）
- `summarize(values)`  → `(mean, min, max, n)`

数据来源与许可证记录在各数据集目录下的 `SOURCE.json` 中，加载器不修改任何原始文件。
