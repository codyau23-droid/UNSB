# 白片 → H&E 虚拟染色：基于 Flow Matching / Schrödinger Bridge 的弱对齐多器官算法设计

> 本文件给出一份针对“同一张切片先白片扫描、再化学 H&E 染色后扫描”这一**弱对齐配对数据**场景下，跨多器官（肺、甲状腺、卵巢、乳腺 …）联合训练的虚拟染色算法方案。基线代码仓库为 UNSB（Unpaired Neural Schrödinger Bridge, ICLR 2024），原 README 见 `README.md`。

---

## 1. 问题定义

- **输入 X**：同一张切片的**白片（未染色）WSI**，经扫描后得到的 RGB / 多通道图像（如自发荧光 / 相位 / 亮场未染色）。
- **输出 Y**：**同一张切片**经化学 H&E 染色后再扫描得到的 RGB 图像。
- **配对关系**：切片级别是配对的，但：
  - 覆盖玻片去除、再染色、再封片、再扫描会带来**非刚性形变**、轻微破损、褶皱、气泡；
  - WSI 级 affine 配准后，patch（例如 512×512 @ 40×）内**约 70% 区域语义一致**，其余 30% 存在偏移 / 形变 / 缺失。
- **训练集构成**：肺 / 甲状腺 / 卵巢 / 乳腺 等**多种器官**的弱对齐 (X, Y) 对，**同器官同病种**但跨器官之间差异显著（形态学、染色强度分布、背景比例都不同），需要**一个模型联合训练、按器官条件生成**。
- **推理**：给定任意器官的白片 patch（或经滑窗的 WSI），输出其虚拟 H&E 染色结果。

### 关键挑战

1. **弱对齐**：像素级 L1/L2、paired diffusion、pix2pix 类监督会被 30% 的错位像素带偏。
2. **多器官异构**：各器官组织结构、细胞密度、核质比、背景占比差异大，单一无条件模型容易“平均化”风格。
3. **保真度 vs 真实感**：必须保留诊断相关形态（核、腺体、纤维结构），同时 H&E 颜色分布要逼真。
4. **评估困难**：缺少像素级 GT，只能用分布级 + 下游任务 + 病理医生主观评估。
5. **WSI 尺度**：训练在 patch 级，推理需要无缝拼回 WSI，避免 tile 边界伪影。

---

## 2. 相关工作综述（用于支撑本方案）

### 2.1 非配对 / 弱配对图像翻译
- **CUT / FastCUT (ECCV 2020)** — 用 **PatchNCE** 对比损失在特征空间保持局部一致性，天然对错位鲁棒。
- **Adaptive Supervised PatchNCE, ASP (MICCAI 2023, Li et al., H&E→IHC)** — 对弱配对 H&E↔IHC，用一个**自适应权重**降低错位 patch 的监督强度，在 MIST 数据集上显著超越 CycleGAN / CUT。这正是我们任务的最近邻工作之一。
- **UNSB (ICLR 2024, 本仓库基线)** — 无配对 Schrödinger Bridge + 对抗 + 正则，多步细化，在 AFHQ/Horse2Zebra 等上优于单步 GAN。

### 2.2 Diffusion / Flow 在虚拟染色
- **BBDM (Brownian Bridge Diffusion Model, CVPR 2023)** — 用布朗桥把源图像作为起点、目标作为终点显式约束扩散路径，已被用于病理 stain transfer（Lou et al., 2024），结构保真优于普通 DDPM。
- **I2SB: Image-to-Image Schrödinger Bridge (ICML 2023)** — 用 SB 代替 DDPM 的高斯先验，起点直接是源图像，训练目标是条件 flow matching，广泛用于逆问题与 I2I。
- **Pixel Super-Resolved Virtual Staining (Nat. Commun. 2025)** — 用布朗桥扩散做标签-free → H&E，兼顾超分 + 染色，证明桥式扩散在病理虚拟染色上的可行性。
- **StainDiffuser (arXiv 2403.11340)** — 针对小数据的 multi-task dual diffusion（stain + segmentation），验证了多任务辅助对稳定 stain 生成的价值。
- **Conditional Diffusion for H&E→IHC (ICPR 2024 workshop)** — 条件扩散超越 CycleGAN / pix2pix。

### 2.3 Flow Matching / Rectified Flow
- **Flow Matching (Lipman et al., ICLR 2023)** 与 **Rectified Flow (Liu et al., ICLR 2023)** — 直接回归两分布之间的速度场，训练简单、采样步数少。
- **OT-CFM (Tong et al., 2023)** — 用 mini-batch 最优传输耦合源目标样本，显著减小路径方差，提高样本效率，**天然适合弱配对**：可以用我们的（X_i, Y_i）弱配对作为耦合，而不是完全随机耦合。
- **InstaFlow / Rectified Flow 2** — 通过 reflow 可蒸馏为 1–2 步生成，便于 WSI 推理加速。

### 2.4 病理基础模型作为感知 / 对比特征
- **UNI (Nature Medicine 2024)**、**CONCH**、**Virchow/Virchow2**、**Phikon** — 在 1M+ WSI 上自监督预训练的 ViT，可作为 frozen encoder 为弱对齐损失提供**组织学语义鲁棒特征**，远胜 ImageNet-VGG 的 LPIPS / PatchNCE。

---

## 3. 方案总览：**WP-CFSB**（Weakly-Paired Conditional Flow-matching Schrödinger Bridge）

一句话：**以源图（白片）为起点、目标图（H&E）为终点的条件流匹配桥**，用**器官 embedding** 做多域条件，用 **ASP-PatchNCE（在病理基础模型特征上）+ 软形变重加权**处理弱对齐，最后以 **rectified flow reflow** 蒸馏到少步采样。

### 3.1 生成过程（核心方程）

采用 **I2SB / Brownian-Bridge 风格**的条件流匹配：

- 对弱配对样本 (x, y, c)，c 为器官条件：
  - 线性插值路径（Rectified Flow / CFM）
    `x_t = (1 - t) · x + t · y + σ(t) · ε`,  t∈[0,1], ε~N(0,I)
  - 速度场回归目标：`v*(x_t, t, c) = y - x`（或含噪布朗桥对应的条件均值速度）
- 训练目标（Conditional Flow Matching loss）：
  `L_CFM = E_{t, (x,y,c), ε} [ w(t) · || v_θ(x_t, t, c) - (y - x) ||² ]`

推理：从 `x_0 = x`（白片）出发，用 ODE/SDE 解 `dz/dt = v_θ(z, t, c)`，N 步（N=4–10）得到 `z_1 ≈ ŷ`。

此框架的优点：
1. **起点固定为源图**：与 I2SB/BBDM 一致，天然保留源图的低频结构；
2. **无需完全对齐的 y**：y 只出现在训练目标 `y-x` 里，我们用**软加权**和**特征级损失**弱化其像素级错位的影响（见 3.3）；
3. **器官条件 c**：把所有器官数据合并训练，模型自动学习共享的“白→H&E”变换先验 + 器官特异风格。

### 3.2 条件化与架构

- **主干**：U-Net（ADM / DiT-ish 皆可），输入为 `x_t` 与 `x`（白片条件，沿通道拼接），时间 t 通过正弦 embedding，器官 c 通过可学习 embedding。
- **条件注入**：
  - `t`、`c` → MLP → 每个 ResBlock 的 **FiLM / AdaGN**（scale, shift）。
  - 器官 embedding 维度 ~ 128，支持后续新增器官的 few-shot 微调（只扩字典、冻主干）。
- **源图条件的两种并行接法**（均启用，互补）：
  1. **通道拼接**：`[x_t, x]` 输入 U-Net 起始层；
  2. **Cross-attention on x features**：用轻量编码器（或基础模型 UNI 的浅层 token）提取 `x` 的 token，在解码器中做 cross-attn，便于模型“回看”源图局部细节。
- **染色先验注入（可选增益）**：解码头末端加一个**色彩域解耦头**（参考 stain separation），输出 hematoxylin / eosin 两个单通道浓度图，再用 Beer–Lambert 合成 RGB。能稳定 H&E 颜色分布，避免偏色。

### 3.3 弱对齐监督：多级损失组合

记网络一步预测的 `ŷ = x + v_θ(x_t=x, t=1, c)·1`（或用少步 ODE 展开得到的伪样本 `ŷ_N`）。

#### (a) CFM 主损失（分布级）
`L_CFM` 如上，其中 `w(t)` 推荐用 **min-SNR weighting** 或 I2SB 默认。

#### (b) ASP-PatchNCE on pathology foundation features（弱对齐主角）
- 特征提取器 **F**：frozen **UNI / CONCH / Virchow**（或 `vgg_sb` 作为 fallback）。
- 在同一空间位置从 `F(x)`、`F(ŷ)` 抽取 N 个 patch token，做 **InfoNCE**，但每个正样本对的权重 `α_{ij}` 由以下两项决定：
  1. **自适应 alignment mask** `α^align`：用一个**轻量形变估计网络** φ（见 3.4）估计 (x, y) 之间的局部位移场 d，`α^align = exp(-||d|| / τ)`；位移大的位置权重小。
  2. **内容置信度** `α^conf`：用 `F` 特征的相似度 `cos(F(x), F(y))` 作为 soft mask，语义差异大的 patch 降权。
- 最终 `L_ASP = -Σ α_{ij} · log [exp(sim+)/Σexp(sim·)]`。
- 另加一路**无监督 PatchNCE**（x ↔ ŷ 的同位 patch），保持源→译自一致性（CUT 风格），避免 collapse。

#### (c) 形变-容忍的像素损失（Deformable L1）
- 用 φ 估计 x→y 的稠密位移 d，将 y warped 到 x 坐标得 ỹ。
- 只在 `α^align > τ₀` 的像素上计算 `L_pix = || mask · (ŷ - ỹ) ||_1`。
- 同时对 φ 加平滑正则 `L_smooth(d)`，防止 φ “硬性对齐”到任意目标导致 cheat。
- **注意**：φ 与 v_θ 联合训练但**梯度 detach**回 y（只用 y 提供监督信号，不让 φ 把 y 搬去拟合 ŷ）。

#### (d) 分布级对抗 / 判别器（可选但推荐）
- 条件判别器 D(·, c) 在器官条件下判别真假 H&E，提供**分布级**的染色真实感监督，与 UNSB 的 adversarial 思路一致，补偿弱对齐下像素损失不足。
- 损失：非饱和 GAN + R1 正则。

#### (e) 结构 / 病理一致性正则
- **Nuclei / tissue mask 一致性**：用现成 HoverNet / CellViT 等在 x 与 ŷ 上分别产生核分割或组织 mask，用 Dice/BCE 约束其**空间一致性**（核在 x 有则 ŷ 也要有）。这是对抗 hallucination 的关键。
- **对比语义一致性**：`cos(F_UNI(x), F_UNI(ŷ))` 的 margin loss，在 slide / patch 级保持语义不漂移。

#### (f) 器官分类辅助（multi-task）
- 小头 MLP 从 U-Net 瓶颈特征预测器官 c。鼓励条件分支真正利用 c，防止器官 embedding 失效。

**总损失**：
`L = λ_cfm L_CFM + λ_asp L_ASP + λ_cut L_PatchNCE(x,ŷ) + λ_pix L_pix + λ_adv L_adv + λ_nuc L_nuclei + λ_sem L_sem + λ_org L_org`

推荐起始权重（需扫）：`λ_cfm=1, λ_asp=1, λ_cut=1, λ_pix=0.5, λ_adv=0.2, λ_nuc=0.5, λ_sem=0.2, λ_org=0.1`。

### 3.4 软形变估计网络 φ（关键组件）

- 结构：轻量 U-Net（VoxelMorph 风格），输入 `[x, y]`，输出 2 通道位移场 d。
- 训练信号：
  - 自监督 NCC / MIND loss：`NCC(warp(y, d), x)`（仅做软对齐，不追求完美）；
  - 平滑项 `||∇d||²`；
  - **重要**：不对 `L_CFM` 反向传播，避免“把 y 挪去等于 x”的捷径。
- 输出被主网络使用：产生 `α^align` 与 warped target `ỹ`。
- 可选：冷启动阶段只训 φ（1–2 epoch），稳定后再联合训练 v_θ。

### 3.5 多器官条件训练策略

- **数据打平**：把所有器官的弱配对 (x, y, c) 放同一个 DataLoader，每个 batch 内做**器官均衡采样**（WeightedRandomSampler），避免样本多的器官主导。
- **OT-CFM 小批量耦合**：batch 内在**同一器官内**用 Sinkhorn 做 (x_i, y_j) 的最优传输耦合（默认使用天然配对；OT 作为可选的额外耦合增强，用于扩充等效配对并降低路径方差）。
- **Stain augmentation**：对 y 做 stain jitter（Macenko / Vahadane 扰动）增强颜色鲁棒性；对 x 做亮度/对比度/伪影扰动。
- **跨器官泛化**：留一器官做 zero-/few-shot 测试。

### 3.6 推理与加速

- **多步 ODE 采样**：默认 Heun / DPM-Solver++，N=4–8 步，已足够高质量。
- **Rectified Flow Reflow**：训练稳定后，用学好的 v_θ 生成 (x, ŷ_N) 伪配对，再训练一个 straighter 的 v'_θ，可蒸馏到 1–2 步，WSI 级推理大幅提速。
- **WSI 推理**：
  - Tile 512（overlap 64），按器官条件执行；
  - 用 Hanning 窗加权融合避免 seam；
  - 前景 mask（otsu on white-slide）跳过空白区。

### 3.7 评估协议

1. **分布级**：Patch-FID、KID、sFID（按器官分别报告 + 合并）。
2. **结构保真**：
   - 在弱对齐 GT 上：SSIM / PSNR on φ-warped `ỹ`（只报告 high-confidence 区域）；
   - **Foundation feature distance**：`1 - cos(F_UNI(ŷ), F_UNI(y))`（比 LPIPS 更契合病理语义）。
3. **下游任务保真**：
   - 用公开预训练的 nuclei / gland segmentation、mitosis 检测模型，在真实 H&E 与 ŷ 上比较指标一致性（**TCR – Task-Consistency Ratio**）。
4. **病理医生主观评估**：双盲 Turing-style 对比，打分 1–5（染色自然度、结构保真、诊断可用性）。
5. **Ablation**：
   - w/o ASP、w/o φ、w/o 器官 embedding、w/o stain-separation head、w/o nuclei 一致性；
   - 单器官 vs 多器官联合。

---

## 4. 与现有基线 UNSB 的关系

- UNSB 是**无配对** SB + 对抗 + 多步细化。本方案可以看作是 UNSB 的以下扩展：
  1. **弱配对**：把 y 作为 CFM 目标（而不是完全随机目标分布），显著降低方差；
  2. **器官条件**：UNSB 默认单域；我们加 FiLM 条件与器官分类头；
  3. **结构一致性**：引入 φ 软形变 + 病理基础模型特征 + 核一致性，解决病理专属的 hallucination；
  4. **可蒸馏**：通过 reflow 一键蒸馏，解决 UNSB 多步慢的问题。
- 代码层面可复用：`models/sb_model.py`、`vgg_sb/`（替换为 UNI/CONCH）、训练循环 `train.py`、数据管线 `data/`。

---

## 5. 数据准备 Pipeline

1. **WSI 级粗配准**：对每一对 (white-slide, H&E) WSI 做 thumbnail 级 SIFT + affine，得到 `T_affine`。
2. **前景 mask**：Otsu on 白片灰度，过滤纯背景。
3. **同坐标 patch 裁剪**：在 20× 或 40× 下按固定 stride 切 512×512，记 (x_i, y_i, c_i)。
4. **质控**：
   - 丢弃前景率 < 10% 的 patch；
   - 丢弃两图前景 IoU < 0.4 的 patch（严重错位，对训练有害）；
   - 丢弃明显伪影/气泡（用简单分类器或阈值）。
5. **划分**：按 **slide-level** 划分 train/val/test（避免同切片穿越），每器官 70/10/20。
6. **存储**：HDF5 / WebDataset，key 含器官 id、slide id、坐标，便于按器官采样。

---

## 6. 实施计划（分阶段，可落地）

### Phase 0 — 数据与基线（1–2 周）
- 数据 pipeline 打通（§5）；
- 直接跑 UNSB（本仓库）作为 baseline，记录各器官 FID / FFD / 医生打分。

### Phase 1 — WP-CFSB v1：CFM + ASP + 器官条件（2–3 周）
- 实现 `models/wp_cfsb_model.py`（基于 I2SB/CFM 训练循环）；
- 加器官 embedding + FiLM；
- 加 ASP-PatchNCE（以 `vgg_sb` 先行，再切 UNI）；
- 验证在 1–2 个器官上超越 UNSB。

### Phase 2 — 弱对齐强化：φ + 可形变像素损失 + 核一致性（2 周）
- 接入 VoxelMorph 风格 φ；
- 接 HoverNet 推理得 nuclei mask 做一致性损失；
- 全器官联合训练。

### Phase 3 — 真实感：对抗 + 染色分离头 + stain augmentation（1–2 周）
- 条件判别器；
- Hematoxylin/Eosin 双通道解耦头；
- 全面 ablation。

### Phase 4 — 加速：Rectified-flow reflow + WSI 推理（1 周）
- 生成 (x, ŷ_N) 伪配对做 reflow；
- 蒸馏到 2 步；
- WSI tiled inference + seamless blending。

### Phase 5 — 评估与泛化（持续）
- Leave-one-organ-out 泛化；
- 病理医生双盲评估；
- 下游任务保真（TCR）。

---

## 7. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 弱对齐导致 hallucination（生成不存在的核） | 核一致性损失 + 基础模型语义损失 + 条件判别器温度控制 |
| 多器官风格互相污染 | 器官条件 FiLM + 器官均衡采样 + 辅助分类头 |
| 像素损失被 30% 错位误导 | φ 估计置信 + ASP 自适应权重 + 主力用特征损失 |
| UNI/CONCH 许可限制 | 提供 Phikon / DINOv2-pathology 作为替代；fallback 到 VGG |
| WSI 推理慢 | Rectified flow reflow 到 2 步 + half precision + tile 并行 |
| y 颜色分布跨批次漂移 | stain augmentation + 染色分离头 + batch 内器官内归一化 |

---

## 8. 关键参考文献

1. Liu, X. et al. **I2SB: Image-to-Image Schrödinger Bridge.** ICML 2023. `arXiv:2302.05872`
2. Lipman, Y. et al. **Flow Matching for Generative Modeling.** ICLR 2023. `arXiv:2210.02747`
3. Liu, X. et al. **Flow Straight and Fast (Rectified Flow).** ICLR 2023. `arXiv:2209.03003`
4. Tong, A. et al. **Improving and Generalizing Flow-Based Generative Models with Minibatch OT (OT-CFM).** 2023. `arXiv:2302.00482`
5. Li, B. et al. **BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models.** CVPR 2023.
6. Kim, B. et al. **Unpaired Image-to-Image Translation via Neural Schrödinger Bridge (UNSB).** ICLR 2024. (本仓库)
7. Park, T. et al. **Contrastive Learning for Unpaired Image-to-Image Translation (CUT).** ECCV 2020.
8. Li, F. et al. **Adaptive Supervised PatchNCE Loss for H&E-to-IHC with Inconsistent Pairs.** MICCAI 2023.
9. Pillar, N. et al. **Pixel Super-Resolved Virtual Staining of Label-Free Tissue via Brownian-Bridge Diffusion.** Nature Communications, 2025.
10. Dubey, S. et al. **StainDiffuser: MultiTask Dual Diffusion Model for Virtual Staining.** arXiv:2403.11340, 2024.
11. Chen, R. J. et al. **Towards a General-Purpose Foundation Model for Pathology (UNI).** Nature Medicine, 2024.
12. Lu, M. Y. et al. **A Visual-Language Foundation Model for Pathology (CONCH).** Nature Medicine, 2024.
13. Vorontsov, E. et al. **Virchow: A Million-Slide Digital Pathology Foundation Model.** 2024.
14. Balakrishnan, G. et al. **VoxelMorph: A Learning Framework for Deformable Medical Image Registration.** TMI 2019.
15. Graham, S. et al. **HoverNet: Simultaneous Segmentation and Classification of Nuclei.** MedIA 2019.

---

## 9. 目录组织建议（落地时）

```
models/
  wp_cfsb_model.py          # 主模型（CFM + SB + 条件）
  deform_net.py             # φ 软形变估计
  stain_head.py             # Hematoxylin/Eosin 分离头
  losses/
    asp_patchnce.py         # ASP on foundation features
    nuclei_consistency.py
    cfm.py
data/
  weakpaired_dataset.py     # (x, y, organ) + 质控
util/
  foundation_encoder.py     # UNI/CONCH/Virchow wrappers
options/
  train_wpcfsb.py
scripts/
  reflow_distill.py
  wsi_inference.py
```

---

## 10. 一页总结

> **WP-CFSB** = **Schrödinger-bridge 式条件流匹配（起点=白片、终点=H&E）** × **器官条件 FiLM** × **病理基础模型特征上的 ASP-PatchNCE** × **软形变加权的可形变像素损失** × **核/语义一致性正则** × **（可选）条件对抗 + 染色分离头** × **Rectified-flow reflow 两步蒸馏**。
> 它把 UNSB 的无配对 SB 升级为**弱配对条件 SB**，并用病理专属的对齐鲁棒与结构保真机制，稳定地处理“同片白→H&E、70% 对齐、多器官联合”这一场景。

---

原始 UNSB 仓库说明见 [`README.md`](README.md)。
