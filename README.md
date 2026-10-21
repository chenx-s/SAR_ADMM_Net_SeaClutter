# ComplexADMM — SAR 复值图像海杂波抑制

MATLAB 版 NC / CSK 标量与 15×15 滑窗特征图计算见 [`matlab/README.md`](matlab/README.md)。

远程顺序运行 ref 四组消融实验（自动训练、整图评估和汇总报告）：
见 [EXPERIMENTS.md](EXPERIMENTS.md)。入口为 `python run_ref_ablation.py`。

完整复值 ADMM 的独立五组筛选实验见 [ADMM_EXPERIMENTS.md](ADMM_EXPERIMENTS.md)。
入口为 `python run_admm_screening.py`；ref 四组结束后使用新目录运行。

复值 **MCA-ADMM 深度展开**网络：把论文 *Modified MCA for SAR Image Clutter
Suppression*（Xiao et al., Remote Sens. 2025, 17, 1727）的迭代求解过程展开成
固定层数的网络，用可学习的复值卷积字典把 SLC 图像分解为

```
X = Xt + Xc        Xt = 舰船目标分量（抑制结果）   Xc = 海杂波分量
```

---

## 1. 快速开始

```bash
# 环境：D:\tool\miniconda\envs\pytorch\python.exe  (torch 2.5.1 + cu124)
cd D:\project\ComplexADMMWORK

python selftest.py                       # 全部自检（约 3 分钟，必须先全绿）
python main.py --config config.yaml      # 完整训练
python evaluate.py --ckpt runs/complex_mca/best.pth --out runs/complex_mca/eval
python inference.py --ckpt runs/complex_mca/best.pth --input <某.mat 或数据集目录> --out outputs/
```

冒烟测试（几十秒，验证链路通畅）：

```bash
python main.py --epochs 2 --batch-size 2 --save-dir runs/smoke \
  --set data.max_train_images=40 --set data.max_val_images=8 --set model.num_layers=4
```

命令行覆盖任意配置项都走 `--set a.b.c=value` 的点号路径，不用改 yaml：

```bash
python main.py --set model.num_layers=12 --set training.lr=1e-4 \
               --set training.loss_weights.bg_suppression=8.0
```

---

## 2. 文件索引

| 文件 | 职责 | 关键点 |
|---|---|---|
| `models/complex_ops.py` | 复值基础算子 | 双通道实数表示、复卷积块矩阵、复软阈值、厄米内积、NC/CSK 统计 |
| `models/complex_dict.py` | 复值卷积字典 | 复值 Gabor / DCT 初始化、**帧界校准**、编码解码互为伴随 |
| `models/blocks.py` | 功能模块 | **CRIA 复值激活**、**复值卷积层**、L0 梯度最小化、图像域非相干、阈值调制、CNR 引导 |
| `models/complex_mca.py` | 深度展开网络 | N 层共享参数 LADMM、层内 5 步迭代、局部统计量调制 |
| `data.py` | SLC 数据管线 | 保相位 dB 归一化、DOTA 栅格化、按 acquisition 切分、相位旋转增强 |
| `losses.py` | 损失函数 | 19 项，每项对应一条物理先验（见 §5） |
| `utils/metrics.py` | 评价指标 | TCR / BSF / CSR / NMSE / Coh / SEP / CFAR-Pd |
| `utils/viz.py` | 可视化 | 幅度+相位双行面板、展开层演化、训练曲线 |
| `utils/ckpt.py` | 检查点 | 存档带 config 快照；加载允许 `strict=False` 但会打印缺/多键 |
| `config.py` | 配置 | 代码内默认值 + yaml 覆盖 + 一致性校验 |
| `main.py` / `evaluate.py` / `inference.py` | 入口 | 训练 / 评估 / 推理 |
| `bench.py` | 耗时分解 | 逐模块量前向与训练步耗时，用来选配置 |

---

## 3. 为什么必须全程用复值（而不是把 SLC 当两幅实图）

SLC 每个像素是复散射系数 `z = |z|·e^{jθ}`：

- `|z|` 是后向散射强度 —— 舰船强、海面弱
- `θ` 是相位。**海杂波相位近似均匀随机，舰船在相干积累后相位是结构化的**

把 SLC 拆成 (实部, 虚部) 两幅独立的实图送进网络，等于断言两者互不相关 ——
恰好丢掉了唯一能把舰船和海杂波区分开的信息。本项目的三处关键设计都是
围绕这一点：

### 3.1 双通道实数表示（不用 `torch.complex64`）

PyTorch 2.5 在 CUDA 上复数卷积**前向可跑、反向抛 `ERROR_UNSUPPORTED_CAST`**
（aten 的复数 cast kernel 在 CUDA 上没实现）。所以内部一律用

```
z = re + j·im    <->    x[B, 2, H, W]，通道 0=实部、1=虚部
```

这是精确重编码而非近似。复卷积写成块矩阵后用一次 `conv2d` 完成：

```
[y_re]   [ C(wr)  −C(wi) ] [x_re]
[y_im] = [ C(wi)   C(wr) ] [x_im]
```

`selftest.py` 里验证了 ⟨D·α, x⟩ = ⟨α, D^H·x⟩ 到 1e-18 量级。

### 3.2 复软阈值（不是实部虚部各自软阈值）

```
S_th(z) = z · max(0, 1 − th/|z|)
```

这是 prox_{th·|·|}(z) 的闭式解：只收缩**模**、完全不改变**相位**。
若拆成两个实通道各自做软阈值（即 prox_{th·‖·‖₁}），相位会向各象限塌缩 ——
物理意义完全不同。

### 3.3 CRIA 实虚交融激活（王浩添论文式 4-6）

```
CRIA(z) = σ₁(Re z)·σ₂(Im z) + j·σ₂(Re z)·σ₁(Im z)
```

对照的旧做法是「实虚分离激活」RISA：`σ(Re z) + j·σ(Im z)`。RISA 把复平面按
象限切死（ReLU 各自清负半轴），相位分辨率退化到象限级。CRIA 用两个**不同**
激活函数的乘积做交叉，输出幅度依赖实虚部的**联合**状态。

采用 σ₁=Sigmoid、σ₂=Tanh 的两个理由：
1. `CRIA(0) = 0` —— 保零点，残差支路零初始化时不会注入直流偏置
2. 有界 → 深展开数值更稳

落在 `CNRGate`（输入并联 `[X, CRIA(X)]`）与 `ComplexRefine`（真·复值隐层）里。

---

## 4. 相对 deepADMM 的三个实质改动

| 改动 | 位置 | 为什么 |
|---|---|---|
| **帧界校准 = 往返增益→1** | `complex_dict._calibrate` | 校准目标是 `decode(encode(x)) ≈ x`，**不是**把 `‖DD^H‖` 归一到 1（本项目踩过的大坑）。过完备卷积字典的谱极度不均（实测 dict_t 的 B/A≈675），按 B 归一化后往返增益只剩 0.19 —— 网络初值 `α₀=D^H(X·gate)` 直接塌掉、残差卡在 0.5。代价是 B 涨到 ≈3.8，故 ISTA 步长须满足 μ<2/B |
| **μ 上界跟着字典刷新** | `complex_mca.refresh_frame_bounds` | 字典是可学习的，原子形状漂移会让 B 变，而 μ<2/B 是关于 B 的条件、不是一次拍定的超参。每个 epoch 末重测 B 并重算 μ_max（反解 sigmoid 参数，**保持 μ 当前取值不变**）。思想来自 dlADMM 的 ρ>2H：惩罚/步长被问题的 Lipschitz 常数**决定** |
| **相对阈值** | `complex_mca._coef_rms` | 阈值 = 系数 RMS 的比例。绝对阈值要同时适配「32 原子 9×9」与「16 原子 5×5」两个尺度完全不同的系数空间，定大了系数被抹零、定小了软阈值形同虚设 |
| **归一化用 `target_atom_norm`** | `complex_dict.normalize_` | 不是拉回 1.0，而是拉回校准得到的 gain。硬拉回 1.0 等于把上一步的帧界校准撤销掉 |

---

## 5. 损失函数：每一项对应一条物理先验

分离问题是严重欠定的 —— `Xt=0, Xc=X` 也能完美重建。所以重建项只负责
"不许丢东西"，真正决定分离质量的是下面的约束。**每项都要能回答："去掉它，
网络会用哪种偷懒方式绕过任务？"**

| 项 | 物理含义 | 去掉会怎样 |
|---|---|---|
| `reconstruction` | 不许丢信息 | Xt=Xc=0 |
| `target_sparsity` / `clutter_sparsity` | 字典域稀疏 | — |
| `cross_target` / `cross_clutter` | **MCA 核心**：D_c 表示不了舰船、D_t 表示不了海杂波 | 两字典分工消失，退化成"把图切两半" |
| `dict_incoherence` | 两字典原子互不相干 | D_t ≈ D_c |
| `incoherence` | 图像域 Xt⊥Xc（论文式 17 的 η 项） | Xt 与 Xc 高度相关 |
| `clutter_tv` | 海杂波空间平滑 | Xc 里留下舰船轮廓 |
| `target_l0_grad` | 目标分段平滑（论文式 28） | 强杂波尖峰留在 Xt |
| `clutter_circular` | **海杂波 ≈ 圆性复高斯**（论文式 2-15/2-18） | 相位结构信息没被利用 |
| `clutter_cv` | 散斑对比度符合海况 | Xc 退化成低频平滑图 |
| `ship_preservation` | 船体区 Xt → X | 舰船被当成杂波压掉 |
| `ship_clutter_suppression` | 船体区 Xc → 0 | **见 §5.1，这是最容易漏的一面** |
| `clutter_fidelity` | 海面区 Xc → X | — |
| `bg_suppression` | 海面区 Xt → 0 | 抑制失效 |
| `bg_p99` / `bg_tv` | 背景 99 分位 / 梯度惩罚 | 稀疏强点虚警（均值惩罚压不住） |
| `annulus_suppression` | 保护带内：亮像素（十字光芒）留 Xt，暗像素归 Xc | 一刀切会把舰船光芒当杂波删掉 |
| `deep_sup` | 中间层应**逐步满足重建**（不是 Xt→X） | — |

### 5.1 踩过的坑：四个区域约束必须成套

这是本项目最隐蔽的一个失败模式，值得单独写出来。

mask 引导的四个项构成一个**闭合系统**：

```
船体区：Xt → X   (ship_preservation)      Xc → 0  (ship_clutter_suppression)
海面区：Xt → 0   (bg_suppression)         Xc → X  (clutter_fidelity)
```

**只写三面会留下一个退化解。** 如果漏掉「船体区 Xc→0」，那么：

1. `Xc` 白白拿走船区能量，没有任何一项惩罚它
2. 于是 `Xt → 0`，残差 `R = X − Xt − Xc → 0`
3. ISTA 的更新量 `Tt = Xt + λ·R` 也 `→ 0`，软阈值只收不放 → `Xt` 再也爬不起来

**重建损失完全拦不住。** 在退化点上 `Xt + Xc − X = 0`，而
`d/dXc ‖Xt+Xc−X‖² = 2(Xt+Xc−X) = 0` —— 重建项对 `Xc` 的梯度**恒等于零**。
`selftest.py` 里有一条回归测试专门验这一点：

```
[region grad] 船区 Xc 推离强度：开启 ship_clutter_suppression = 1.94724 | 关闭 = 0.00367
```

实测症状（第一轮 12 epoch 的方向验证）：

| epoch | SEP | NE (dB) | NMSE (dB) |
|---|---|---|---|
| 1 | 0.005 | −5.94 | −2.16 |
| 3 | 0.001 | −9.76 | −2.19 |
| 5 | 0.002 | −9.29 | **−4.18** |

**重建指标一路变好，分离指标一路变坏** —— 这就是退化解的签名。
（`Xt=0, Xc=X` 确实能完美重建，重建损失完全没有理由阻止它。）

同时还有第二个错误：权重配比写反了。deepADMM 验证过的配置是
`ship_preservation 5.0 / bg_suppression 1.0`，我写成了 `2.0 / 5.0`。
由于背景占 99% 像素，反过来的配比会通过**共享层的单一阈值**把目标区一起压死
（抬高阈值能让背景项大赚，代价是船区小亏）。

两个修法都做了：
1. 补齐 `ship_clutter_suppression`
2. 权重改用 deepADMM 验证过的比例
3. 模型选择默认改为 **SEP** —— 它是这个塌缩模式最敏感的探测器
   （NE 也能反映，但数值受场景自身对比度影响，不如 SEP 直观）
4. `main.py` 里加了塌缩预警：`SEP < 0.05` 时显式打印警告并指出可能原因

### 5.2 另外两个容易写错的地方

**`incoherence` 必须归一化。** 论文式 17 的 `η|XcᵀXt|²` 是未归一化的，数值随
幅度平方增长；而我们的输入经过逐 patch 分位归一化，各样本幅度不同，等价于
给不同样本随机分配了不同的正则强度。归一化成夹角余弦后与幅度解耦。

**`deep_sup` 监督的是重建而不是 Xt。** deepADMM 原版把中间层的 `Xt` 直接往
原图 `X` 上回归 —— 那是在鼓励"Xt 把整幅图都重建出来"，与分离目标直接冲突。
而且 `detach_intermediates` 必须为 false，否则这一项恒为 0 梯度、静默失效
（`config.validate_config` 里加了互斥校验，会直接报错而不是静默放过）。

---

## 6. 评价指标口径

**比值类指标必须反归一化回物理幅度再算。** 数据管线做了保相位的 dB 压缩
（非线性），归一化域的功率比与物理域的功率比可以差好几个 dB，直接对比论文
数值（BSF 89.07、TCR 25.28 dB 那种量级）完全对不上。

记 `P(·)` 为区域平均功率 `mean|·|²`，`bg` = 保护带外的纯海面：

| 指标 | 定义 | 说明 |
|---|---|---|
| `TCR` (dB) | `10log10(P(Xt,ship)/P(Xt,bg))` | 越大越好 |
| `TCR_in` (dB) | 同式作用于原图 X | 未处理基准 |
| **`NE`** (dB) | `TCR − TCR_in` | **净增强量，跨口径可比** |
| `BSF` (dB) | `10log10(P(X,bg)/P(Xt,bg))` | 论文式 30 |
| `CSR` (dB) | `−TCR` | 杂信比 |
| `NMSE` (dB) | 重建残差能量占比 | 只看模值 |
| `Coh` | `\|⟨X,Xt+Xc⟩\|/(‖X‖‖Xt+Xc‖)` | **相位敏感**，复值链路必须配 |
| `SEP` | 船区能量落在 Xt 的比例 | |
| `Pd` / `Pfa` | CFAR 口径：先用分位定阈值使背景虚警率 = Pfa，再看船体越阈比例 | 比固定阈值可跨场景比较 |

**模型选择按 `NE`**，不按验证损失 —— 最偷懒的模型 `Xt=X, Xc=0` 重建损失
几乎完美，但它一点抑制都没做。

---

## 7. 数据事实（已核对）

```
D:\download\FSI-TRAINVAL-ShipOnly-Complex   4918 张（纯舰船子集）
D:\download\FSI-TEST-Ship-Complex           1325 张
  ├─ SLCMats/     1024² complex64，.mat 键名 'data'   ← 唯一可用的输入
  ├─ Annotations/ DOTA 4 点多边形 txt
  ├─ PNGImages/   1024² uint8 可视化
  └─ KeyPoints/ METAXmls/
```

- **不能用 PNG**：PNG 是拉伸裁剪后的可视化，与 `|SLC|` 的相关性只有 0.39
- `|SLC|` 跨 6 个数量级（0.0053 ~ 8356），dB 分位 −3.66 / 18.23 / 28.78 (p1/p50/p99)
- 单文件 8 MB × 4918 ≈ 39 GB → 必须在线裁 patch，不能全量缓存
- 切分按 **acquisition 分组**（文件名前 10 段），防止同景相邻切片同时进训练和
  验证（相邻 patch 高度相关，随机切分会把验证集"泄漏"掉）

---

## 8. 已知环境坑

| 现象 | 原因 / 对策 |
|---|---|
| Bash 工具里 `dirname`/`ls`/`tail` 全部 command not found | 本机 shell 环境 PATH 有问题。用 `Read` 读文件、用完整路径调 python，输出重定向到文件再读 |
| PowerShell 工具 stdout 不回传 | 同上，把输出写文件 |
| `DataLoader` 报 spawn 递归错误 | Windows 下 worker 子进程会重新 import `__main__`，所有入口脚本都有 `if __name__ == '__main__':` 守卫 |
| 多进程下 `collate_fn` 报 `expected Tensor, but got float` | `norm_lo`/`norm_hi` 是 python float，`torch.stack` 不接受。只在 worker 进程暴露，`num_workers=0` 反而正常 |
| 首次跑 CUDA 复数卷积反向报 `ERROR_UNSUPPORTED_CAST` | PyTorch 2.5 的限制，全程改用双通道实数表示即可规避 |
| matplotlib 中文变成方框 | 已在 `utils/viz.py` 顶部显式指定中文字体 |

---

## 9. 参考材料

- `mca.pdf` — Xiao et al., *Modified MCA Method for SAR Image Clutter Suppression*,
  Remote Sens. 2025, 17, 1727（图像域非相干约束 + L0 梯度最小化）
- 王浩添《基于复数域深度特征融合的 SAR 图像舰船目标识别方法研究》，哈工大 2025
  （CRIA 式 4-6、NC 式 2-15、CSK 式 2-18、CCMaxPool）
- `refs/mca_text.txt`、`refs/wang_text.txt` — 上述两篇的 PDF 抽取全文，便于核对公式
- `D:\project\deepADMM` — 实数版前身，很多工程经验（RESULTS.md）直接沿用
#   S A R _ A D M M _ N e t _ S e a C l u t t e r  
 