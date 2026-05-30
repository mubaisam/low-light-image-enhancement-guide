# 弱光图像增强：这份就够了

> 面向初学者和研究生——概念、方法、论文、数据集、指标、评测代码，一站式备齐。

本项目旨在帮助刚接触弱光图像增强方向的学习者快速建立领域框架。内容覆盖问题定义、主流技术路线、代表性论文、常用数据集、评价指标和评测脚本。不是面面俱到的文献数据库，而是一个结构化的学习和科研入门指南。

欢迎通过 [issue](../../issues) 或 [pull request](../../pulls) 补充资源、修正链接或完善内容。

---

## 目录

- [基础概念速览](#基础概念速览)
- [入门推荐论文](#入门推荐论文)
- [数据集](#数据集)
- [方法总览](#方法总览)
  - [基于直方图均衡的方法 (HE-based)](#基于直方图均衡的方法-he-based)
  - [基于 Retinex 理论的方法 (Retinex-based)](#基于-retinex-理论的方法-retinex-based)
  - [基于深度学习的方法 (Learning-based)](#基于深度学习的方法-learning-based)
  - [其他方法](#其他方法)
- [评价指标](#评价指标)
- [评测代码](#评测代码)
- [综述与基准测试](#综述与基准测试)
- [相关研究方向](#相关研究方向)
- [更多参考资源](#更多参考资源)

## 基础概念速览

### 什么是弱光图像增强？

弱光图像增强（Low-Light Image Enhancement）是指通过算法改善在低光照条件下拍摄的图像质量，使其亮度提升、细节可见、噪声可控，同时尽量保持自然的外观。这项技术在**夜间监控、自动驾驶、医学荧光显微成像、手机夜景拍摄**等场景中有广泛应用。

### 为什么这是个难题？

弱光图像面临的核心挑战：
- **光子数少** → 信号弱，信噪比低
- **噪声复杂** → 暗光下噪声呈现复杂的分布（泊松+高斯混合）
- **动态范围大** → 场景中同时存在极暗和极亮区域
- **颜色失真** → 低光照下颜色信息严重退化
- **缺乏真值** → 现实中很难获取同一场景的"亮-暗"完美配对图像

### 三大主流方法范式

#### 1. 直方图均衡化 (Histogram Equalization, HE)

**核心思想**：通过重新分布图像的像素亮度值，让暗部"拉伸"、亮部"压缩"，从而提升整体对比度。

- 经典方法：全局直方图均衡化、自适应直方图均衡化 (AHE/CLAHE)
- 优点：简单、快速、无需训练
- 缺点：容易过度增强、放大噪声、产生不自然效果
- **入门推荐**：先理解 BPDHE（亮度保持动态直方图均衡化），它解决了经典 HE 亮度漂移的问题。

#### 2. Retinex 理论

**核心思想**：基于 Edwin Land 提出的 Retinex 理论，认为一幅图像可以分解为**光照分量 (Illumination)** 和 **反射分量 (Reflectance)**：

```
图像 = 光照 × 反射
```

其中反射分量反映物体本身的颜色/纹理（不受光照影响），是我们真正想恢复的。方法的核心就是：从弱光图像中估计出光照分量，然后校正它，或者直接从原图中剥离光照得到反射分量。

- 经典方法：单尺度 Retinex (SSR)、多尺度 Retinex (MSR)、带色彩恢复的多尺度 Retinex (MSRCR)
- 深度学习版本：RetinexNet、KinD、URetinex-Net、Retinexformer
- 优点：有物理模型支撑，可解释性强
- 缺点：光照/反射分解本身是一个欠定问题（ill-posed），传统方法容易产生光晕或伪影
- **入门推荐**：先看 LIME（仅估计光照图来做增强，思路直观），再读 RetinexNet（第一个用深度学习做 Retinex 分解的工作）。

#### 3. 基于深度学习的方法

**核心思想**：不再手工设计规则，而是让神经网络从大量数据中自动学习"暗图→亮图"的映射。

几个典型子方向：
- **端到端增强**：直接学习暗图到亮图的映射（如 SID、MBLLEN）
- **Retinex + 深度学习**：用神经网络来做光照/反射分解（如 RetinexNet、KinD）
- **生成对抗网络 (GAN)**：用对抗训练提升增强的真实感（如 EnlightenGAN）
- **无监督/自监督**：不依赖配对数据训练（如 EnlightenGAN、SSEN）
- **扩散模型**：利用扩散模型的生成能力做增强（如 Diff-Retinex、ExposureDiffusion）
- **Transformer**：用自注意力机制捕捉全局光照关系（如 Retinexformer）

> **对初学者的建议**：不要被各种花哨的名字吓到。大部分方法的内核可以归结为两类——要么"学一个映射函数"，要么"分解+调整+重组"。从 RetinexNet 和 EnlightenGAN 两篇开始读，基本就能理解深度学习做弱光增强的核心思路。

### 评价指标简介

分为两大类：
- **全参考 (Full-Reference)**：需要有"真值"（正常光照的参考图），对比增强结果和真值之间的差异。常用 PSNR、SSIM、LPIPS。
- **无参考 (No-Reference)**：不需要真值，直接评价图像质量。常用 NIQE、LOE、MUSIQ。

> 实际应用中，PSNR 和 SSIM 高不代表"好看"。建议阅读 [LPIPS](https://github.com/richzhang/PerceptualSimilarity) 论文理解"感知相似度"与传统指标的差异。

---

## 入门推荐论文

如果你只有时间读 5-8 篇论文，以下是按学习顺序推荐的核心读物：

| 序号 | 论文 | 年份/会议 | 为什么推荐 | 难度 |
|:--:|------|:--:|------|:--:|
| 1 | **LIME** — Low-Light Image Enhancement via Illumination Map Estimation | 2017 TIP | 思路极其清晰：只估计光照图，然后做 gamma 校正。代码短、数学优雅，传统方法的最佳入门读物。 | ⭐ |
| 2 | **RetinexNet** — Deep Retinex Decomposition for Low-Light Enhancement | 2018 BMVC | 第一个将 Retinex 与深度学习结合的工作。网络结构简单（分解+增强两步），容易理解。 | ⭐⭐ |
| 3 | **KinD** — Kindling the Darkness: A Practical Low-light Image Enhancer | 2019 ACM MM | Retinex + 深度学习的经典之作，提出分解-调节-重建的完整流程，代码完善，实验扎实。 | ⭐⭐ |
| 4 | **EnlightenGAN** — Deep Light Enhancement without Paired Supervision | 2019 TIP | 无需配对数据即可训练的 GAN 方法，打破了"必须有暗-亮图像对"的限制，思想影响深远。 | ⭐⭐ |
| 5 | **SID** — Learning to See in the Dark | 2018 CVPR | 处理极端暗光（~0.1 lux）的里程碑工作，首次展示深度学习在极暗 Raw 数据上的惊人效果。 | ⭐⭐⭐ |
| 6 | **Zero-DCE** — Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement | 2020 CVPR | 无需参考图像、无需配对数据，通过估计高阶曲线来做增强。思路新颖，训练简单。 | ⭐⭐ |
| 7 | **URetinex-Net** — Retinex-Based Deep Unfolding Network | 2022 CVPR | 将传统优化展开 (unrolling) 与深度学习结合，展示了如何给黑盒网络赋予物理意义。 | ⭐⭐⭐ |
| 8 | **Retinexformer** — One-stage Retinex-based Transformer | 2023 ICCV | Transformer 做弱光增强的代表作，了解最新架构如何应用于该领域。 | ⭐⭐⭐ |

---

## 数据集

弱光增强常用数据集可以按是否有配对参考图像分为两类：**有监督配对数据集**和**非配对/无参考测试集**。

### 有监督配对数据集

这类数据集提供低光图像和正常光参考图像，适合训练或评测需要 PSNR、SSIM、LPIPS 等全参考指标的方法。

| 数据集 | 子集/版本 | 简介 | 链接 |
|:--:|:--:|------|:--:|
| LOLv1 | - | 经典配对弱光增强基准，常作为入门和方法对比的第一组数据 | [paper](https://arxiv.org/abs/1808.04560) / [project](https://daooshee.github.io/BMVC2018website) |
| LOLv2 | Real | 真实采集的低光/正常光配对数据，难度更接近真实场景 | [github](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) |
| LOLv2 | Synthetic | 合成退化得到的低光/正常光配对数据，常用于受控训练和对比 | [github](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) |
| MIT-Adobe FiveK | - | 5,000 张 RAW 照片及多位专家修图结果，常用于照片增强/曝光校正相关研究 | [dataset](https://data.csail.mit.edu/graphics/fivek/) |
| LSRW | Huawei | 使用华为设备采集的大规模真实低光/正常光配对数据 | [paper](https://arxiv.org/abs/2106.14501) / [github](https://github.com/JianghaiSCU/R2RNet) |
| LSRW | Nikon | 使用 Nikon 相机采集的大规模真实低光/正常光配对数据 | [paper](https://arxiv.org/abs/2106.14501) / [github](https://github.com/JianghaiSCU/R2RNet) |

### 非配对/无参考测试集

这类数据集通常只有低光图像或缺少严格配对参考图像，常用于无监督方法、零参考方法和跨数据集泛化测试。

| 数据集 | 简介 | 链接 |
|:--:|------|:--:|
| VV | 包含具有挑战性的低光/曝光问题图像，常用于视觉质量对比 | [github](https://github.com/baidut/BIMEF) |
| NPE | Naturalness Preserved Enhancement 常用测试集 | [github](https://github.com/baidut/BIMEF) |
| LIME | LIME 论文使用的低光测试图像集合 | [github](https://github.com/baidut/BIMEF) |
| MEF | 多曝光融合相关低光测试图像集合 | [github](https://github.com/baidut/BIMEF) |
| DICM | 商用数码相机拍摄的低光图像集合 | [github](https://github.com/baidut/BIMEF) |

---

## 方法总览

> 每个分类前有一段导读，帮助理解该类方法的核心思想。每篇论文标注了**难度等级**：⭐ 入门 / ⭐⭐ 进阶 / ⭐⭐⭐ 研究。

### 基于直方图均衡的方法 (HE-based)

**核心思想**：通过调整像素亮度分布来增强对比度。可以理解为把"挤在一起"的亮度值"拉开"。这类方法计算简单、不需要训练数据，适合嵌入式设备和对实时性要求高的场景。缺点是对局部光照不均匀的图像效果不佳，容易过增强。

| 年份 | 发表 | 论文 | 链接 | 方法名 | 难度 |
|:--:|------|------|:--:|------|:--:|
| 2007 | IEEE TCE | Brightness Preserving Dynamic Histogram Equalization for Image Contrast Enhancement | [pdf](https://ieeexplore.ieee.org/document/4429280) | BPDHE | ⭐ |
| 2013 | SITIS | Adaptive Multiscale Retinex for Image Contrast Enhancement | [pdf](https://doi.ieeecomputersociety.org/10.1109/SITIS.2013.19) | AMSR | ⭐⭐ |

### 基于 Retinex 理论的方法 (Retinex-based)

**核心思想**：把图像分解为光照 (Illumination) 和反射 (Reflectance) 两个分量，然后调整光照分量来增强图像。这是一种"物理模型驱动"的思路，可解释性强。传统方法用各种先验约束来做分解（如平滑性假设），深度学习方法则让网络自动学习如何分解。

> **新手提示**：先读 LIME（2017）理解"估计光照图→校正光照图"的思路，再读 RetinexNet（2018）看深度学习如何做这个事，最后看 Retinexformer（2023）了解最新的 Transformer 方案。

| 年份 | 发表 | 论文 | 链接 | 方法名 | 难度 |
|------|------|------|:--:|------|:--:|
| 2013 | SITIS | Adaptive Multiscale Retinex for Image Contrast Enhancement | [pdf](https://doi.ieeecomputersociety.org/10.1109/SITIS.2013.19) | AMSR | ⭐⭐ |
| 2016 | Signal Processing | A fusion-based enhancing method for weakly illuminated images | [pdf](https://doi.org/10.1016/j.sigpro.2016.05.031) | MF | ⭐ |
| 2017 | IEEE TIP | LIME: Low-Light Image Enhancement via Illumination Map Estimation | [pdf](http://ieeexplore.ieee.org/document/7782813/) [code](https://github.com/Sy-Zhang/LIME) | LIME | ⭐ |
| 2017 | ICCV | A Joint Intrinsic-Extrinsic Prior Model for Retinex | [pdf](http://caibolun.github.io/papers/JieP.pdf) [web](http://caibolun.github.io/JieP/) [code](https://github.com/caibolun/JieP) | JieP | ⭐⭐ |
| 2018 | BMVC | Deep Retinex Decomposition for Low-Light Enhancement | [pdf](https://arxiv.org/abs/1808.04560) [web](https://daooshee.github.io/BMVC2018website/) [code](https://github.com/yzhouas/Retinex-Net) | RetinexNet | ⭐⭐ |
| 2019 | Comput. Graphics Forum | Dual illumination estimation for robust exposure correction | [pdf](https://arxiv.org/pdf/1910.13688.pdf) [code](https://github.com/pvnieo/Low-light-Image-Enhancement) | Dual-Illumination | ⭐⭐ |
| 2019 | ICME | RDGAN: Retinex Decomposition Based Adversarial Learning for Low-Light Enhancement | [code](https://github.com/WangJY06/RDGAN/) [pdf](https://ieeexplore.ieee.org/document/8919666) | RDGAN | ⭐⭐ |
| 2020 | IEEE TIP | LR3M: Robust Low-Light Enhancement via Low-Rank Regularized Retinex Model | [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9056796) [code](https://github.com/ayu-bu/LR3M) | LR3M | ⭐⭐⭐ |
| 2021 | CVPR | Retinex-Inspired Unrolling with Cooperative Prior Architecture Search for Low-Light Image Enhancement | [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Retinex-Inspired_Unrolling_with_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf) [code](https://github.com/csjcai/RUAS) | RUAS | ⭐⭐⭐ |
| 2022 | CVPR | URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement | [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [code](https://github.com/ywmc/URetinex-Net) | URetinex-Net | ⭐⭐⭐ |
| 2023 | ICCV | Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model | [pdf](https://arxiv.org/pdf/2308.13164.pdf) [code](https://github.com/cschenxiang/Diff-Retinex) | Diff-Retinex | ⭐⭐⭐ |
| 2023 | ICCV | Retinexformer: One-stage retinex-based transformer for low-light image enhancement | [pdf](https://arxiv.org/abs/2303.06705) [code](https://github.com/caiyuanhao1998/Retinexformer) | Retinexformer | ⭐⭐⭐ |
| 2025 | ICCV | GT-Mean Loss: A Simple Yet Effective Solution for Brightness Mismatch in Low-Light Image Enhancement | [pdf](https://arxiv.org/abs/2507.20148) [code](https://github.com/jingxiLiao/GT-Mean-Loss) | GT-Mean Loss | ⭐⭐ |
| 2025 | ICLR (Spotlight) | Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model | [pdf](https://arxiv.org/pdf/2311.11638) [code](https://github.com/Chunming-Li/Reti-Diff) | Reti-Diff | ⭐⭐⭐ |
| 2025 | ICIP | RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement | [pdf](https://arxiv.org/abs/2505.24705) [code](https://github.com/jhakrraman/rt-xnet) [web](https://sites.google.com/view/rtxnet) | RT-X Net | ⭐⭐⭐ |
| 2025 | IJCV | Nonlocal Retinex-Based Variational Model and its Deep Unfolding Twin for Low-Light Image Enhancement | [pdf](https://link.springer.com/article/10.1007/s11263-025-02551-y) [code](https://github.com/RuiZhang97/NRVN) | NRVN | ⭐⭐⭐ |

### 基于深度学习的方法 (Learning-based)

**核心思想**：利用深度神经网络从大量数据中自动学习暗光到正常光的映射。这是当前**效果最好、发展最快**的方向，顶会论文层出不穷。缺点是通常需要大量数据、计算资源，且可解释性不如模型驱动方法。

> **新手提示**：这个列表比较长，初学者可以先关注 RetinexNet → KinD → EnlightenGAN → Zero-DCE 这几篇，理解"有监督 vs 无监督 vs 零参考"三种训练范式的区别。

| 年份 | 发表 | 论文 | 链接 | 方法名 | 难度 |
|------|------|------|:--:|------|:--:|
| 2017 | ACM Trans. Graph. | Deep bilateral learning for real-time image enhancement | [pdf](https://arxiv.org/abs/1707.02880) [web](https://groups.csail.mit.edu/graphics/hdrnet/) [code](https://github.com/google/hdrnet) | HDRnet | ⭐⭐ |
| 2017 | ICCV | DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks | [pdf](https://arxiv.org/abs/1704.02470) [code](https://github.com/cchen156/DPED) | DPED | ⭐⭐ |
| 2018 | BMVC | Deep Retinex Decomposition for Low-Light Enhancement | [pdf](https://arxiv.org/abs/1808.04560) [web](https://daooshee.github.io/BMVC2018website/) [code](https://github.com/yzhouas/Retinex-Net) | RetinexNet | ⭐⭐ |
| 2018 | BMVC | MBLLEN: Low-light Image/Video Enhancement Using CNNs | [pdf](http://bmvc2018.org/contents/papers/0700.pdf) [web](http://phi-ai.org/project/MBLLEN/default.html) [code](https://github.com/Lvenfang/MBLLEN) | MBLLEN | ⭐⭐ |
| 2018 | CVPR | Learning to See in the Dark | [pdf](https://cchen156.github.io/paper/18CVPR_SID.pdf) [web](https://cchen156.github.io/SID.html) [code](https://github.com/cchen156/Learning-to-See-in-the-Dark) | SID | ⭐⭐⭐ |
| 2018 | IEEE TIP | Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images | [pdf](https://doi.org/10.1109/TIP.2018.2794218) [code](https://github.com/csjcai/SICE) | SICE | ⭐⭐ |
| 2018 | ACM TOG | Exposure: A White-Box Photo Post-Processing Framework | [pdf](https://doi.org/10.1145/3181974) [code](https://github.com/yuanming-hu/exposure) | Exposure | ⭐⭐ |
| 2019 | IEEE TIP | EnlightenGAN: Deep Light Enhancement without Paired Supervision | [code](https://github.com/TAMU-VITA/EnlightenGAN) [pdf](https://arxiv.org/abs/1906.06972) | EnlightenGAN | ⭐⭐ |
| 2019 | ACM MM | Kindling the Darkness: A Practical Low-light Image Enhancer | [pdf](http://arxiv.org/abs/1905.04161) [code](https://github.com/zhangyhuaee/KinD) [code+](https://github.com/zhangyhuaee/KinD_plus) | KinD | ⭐⭐ |
| 2019 | CVPR | Underexposed Photo Enhancement Using Deep Illumination Estimation | [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953588) [code](https://github.com/wangbc2014/DeepUPE) | DeepUPE | ⭐⭐ |
| 2019 | ICME | RDGAN: Retinex Decomposition Based Adversarial Learning for Low-Light Enhancement | [code](https://github.com/WangJY06/RDGAN/) [pdf](https://ieeexplore.ieee.org/document/8919666) | RDGAN | ⭐⭐ |
| 2020 | CVPR | Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement | [pdf](https://arxiv.org/abs/2001.06826) [code](https://github.com/Li-Chongyi/Zero-DCE) | Zero-DCE | ⭐⭐ |
| 2020 | CVPR | DeepLPF: Deep Local Parametric Filters for Image Enhancement | [pdf](https://arxiv.org/abs/2003.13985) [code](https://github.com/sjmoran/DeepLPF) | DeepLPF | ⭐⭐ |
| 2020 | IEEE PAMI | Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time | [pdf](https://ieeexplore.ieee.org/document/9206076) [code](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | 3DLUT | ⭐⭐⭐ |
| 2020 | IET Image Proc. | Learning an Adaptive Model for Extreme Low-Light Raw Image Processing | [pdf](https://arxiv.org/pdf/2004.10447.pdf) [code](https://github.com/505030475/ExtremeLowLight) | ExtremeLowLight | ⭐⭐⭐ |
| 2020 | ArXiv | Visual Perception Model for Rapid and Adaptive Low-light Image Enhancement | [pdf](http://arxiv.org/abs/2005.07343) [code](https://github.com/MDLW/Low-Light-Image-Enhancement) | RAPID | ⭐⭐ |
| 2020 | ArXiv | Self-supervised Image Enhancement Network: Training with Low Light Images Only | [pdf](https://arxiv.org/abs/2002.11300) [code](https://github.com/hitzhangyu/Self-supervised-LLE) | SSEN | ⭐⭐ |
| 2021 | IJCV | Attention Guided Low-Light Image Enhancement with a Large Scale Low-Light Simulation Dataset | [pdf](https://link.springer.com/10.1007/s11263-021-01466-8) [code](https://github.com/yytang2012/AGLLIE) | AGLLIE | ⭐⭐ |
| 2021 | CVPR | Retinex-Inspired Unrolling with Cooperative Prior Architecture Search for Low-Light Image Enhancement | [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Retinex-Inspired_Unrolling_with_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf) [code](https://github.com/csjcai/RUAS) | RUAS | ⭐⭐⭐ |
| 2021 | ICCV | Seeing Dynamic Scene in the Dark: A High-Quality Video Dataset with Mechatronic Alignment | [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Seeing_Dynamic_Scene_in_the_Dark_A_High-Quality_Video_Dataset_with_Mechatronic_Alignment_ICCV_2021_paper.pdf) [code](https://github.com/dvlab-research/SDSD) | SDSD | ⭐⭐⭐ |
| 2021 | ICCV | Matching in the Dark: A Dataset for Matching Image Pairs of Low-Light Scenes | [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_Matching_in_the_Dark_A_Dataset_for_Matching_Image_Pairs_of_Low-Light_Scenes_ICCV_2021_paper.pdf) [code](https://github.com/wenzhengjiang/MID) | MID | ⭐⭐ |
| 2021 | JVCIR | R2RNet: Low-Light Image Enhancement via Real-Low to Real-Normal Network | [pdf](http://arxiv.org/abs/2106.14501) [code](https://github.com/abcdef2000/R2RNet) | R2RNet | ⭐⭐ |
| 2022 | CVPR | Toward Fast, Flexible, and Robust Low-Light Image Enhancement | [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [code](https://github.com/simmonsm/LE-UNet) | LE-UNet | ⭐⭐ |
| 2022 | CVPR | Deep Color Consistent Network for Low-Light Image Enhancement | [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [code](https://github.com/chosj95/DCNCCN) | DCNCCN | ⭐⭐ |
| 2022 | CVPR | URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement | [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [code](https://github.com/ywmc/URetinex-Net) | URetinex-Net | ⭐⭐⭐ |
| 2022 | ECCV | Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression | [pdf](https://arxiv.org/pdf/2207.10564.pdf) [code](https://github.com/jinyeye/Night-Enhancement) | UNIE | ⭐⭐ |
| 2022 | ECCV | LEDNet: Joint Low-Light Enhancement and Deblurring in the Dark | [pdf](https://link.springer.com/chapter/10.1007/978-3-031-20068-7_33) [code](https://github.com/sczhang/LEDNet) | LEDNet | ⭐⭐⭐ |
| 2022 | AAAI | Low-Light Image Enhancement with Normalizing Flow | [pdf](https://arxiv.org/pdf/2109.05923.pdf) [code](https://github.com/wyf0912/LLFlow) [web](https://wyf0912.github.io/LLFlow/) | LLFlow | ⭐⭐⭐ |
| 2022 | AAAI | Semantically contrastive learning for low-light image enhancement | [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/20046) [code](https://github.com/LingLIx/SCL) | SCL | ⭐⭐⭐ |
| 2022 | ACM MM | ChebyLighter: Optimal Curve Estimation for Low-Light Image Enhancement | [pdf](https://dl.acm.org/doi/10.1145/3503161.3548135) [code](https://github.com/eeerpjw/ChebyLighter) | ChebyLighter | ⭐⭐ |
| 2022 | IJCV | Low-Light Image Enhancement via Breaking down the Darkness | [pdf](http://dx.doi.org/10.1007/s11263-022-01667-9) [code](https://github.com/mingcv/Bread?utm_source=catalysis&utm_campaign=socialshare) | Bread | ⭐⭐⭐ |
| 2022 | IEEE TCSVT | EFINet: Restoration for Low-Light Images via Enhancement-Fusion Iterative Network | [pdf](https://ieeexplore.ieee.org/document/9849123/) [code](https://github.com/kyrie111/EFINet) | EFINet | ⭐⭐ |
| 2023 | Information Fusion | A Mutually Boosting Dual Sensor Computational Camera for High Quality Dark Videography | [pdf](https://doi.org/10.1016/j.inffus.2023.01.013) [code](https://github.com/xychen-zhu/MBDSC) | MBDSC | ⭐⭐⭐ |
| 2023 | Pattern Recognit. | TreEnhance: A tree search method for low-light image enhancement | [pdf](https://www.sciencedirect.com/science/article/pii/S0031320322007282?via%3Dihub) [code](https://github.com/VivianSZF/TreEnhance) | TreEnhance | ⭐⭐ |
| 2023 | AAAI | Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method | [pdf](https://arxiv.org/abs/2212.11548) [code](https://github.com/TaoWangzj/UHD-LoL) | UHD-LoL | ⭐⭐⭐ |
| 2023 | AAAI | Polarization-Aware Low-Light Image Enhancement | [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/25486) [code](https://github.com/fourson/Polarization-Aware-Low-light-Image-Enhancement) | PALE | ⭐⭐⭐ |
| 2023 | CVPR | DNF: Decouple and feedback network for seeing in the dark | [pdf](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.html) [code](https://github.com/jinyeye/DNF) | DNF | ⭐⭐⭐ |
| 2023 | CVPR | Learning semantic-aware knowledge guidance for low-light image enhancement | [pdf](http://openaccess.thecvf.com/content/CVPR2023/html/Wu_Learning_Semantic-Aware_Knowledge_Guidance_for_Low-Light_Image_Enhancement_CVPR_2023_paper.html) [code](https://github.com/eezing/SKGNet) | SKGNet | ⭐⭐⭐ |
| 2023 | IEEE TMM | Glow in the Dark: Low-Light Image Enhancement with External Memory | [pdf](https://ieeexplore.ieee.org/document/10177254/) [code](https://github.com/Lineves7/EMNet) | EMNet | ⭐⭐⭐ |
| 2023 | IEEE TPAMI | Learning With Nested Scene Modeling and Cooperative Architecture Search for Low-Light Vision | [pdf](https://ieeexplore.ieee.org/document/9914672/) [code](https://github.com/vis-opt-team/S2-RUAS) | S2-RUAS | ⭐⭐⭐ |
| 2023 | IEEE TIP | Unsupervised Low-Light Video Enhancement with Spatial-Temporal Co-attention Transformer | [pdf](https://ieeexplore.ieee.org/document/10210621/) [code](https://github.com/redknight990/LightenFormer) | LightenFormer | ⭐⭐⭐ |
| 2023 | SIGGRAPH ASIA | Low-light Image Enhancement with Wavelet-based Diffusion Models | [pdf](https://arxiv.org/pdf/2306.00306.pdf) [code](https://github.com/JianghaiSCU/Diffusion-Low-Light) | Diffusion-LL | ⭐⭐⭐ |
| 2023 | ACM MM | CLE Diffusion: Controllable Light Enhancement Diffusion Model | [pdf](https://arxiv.org/abs/2308.06725) [code](https://github.com/YuyangYin/CLEDiffusion) [web](https://yuyangyin.github.io/CLE-Diffusion/) | CLE-Diffusion | ⭐⭐⭐ |
| 2023 | ACM MM | FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information | [pdf](https://arxiv.org/abs/2308.03033) [code](https://github.com/wangchx67/FourLLIE) | FourLLIE | ⭐⭐⭐ |
| 2023 | Pattern Recognit. | SurroundNet: Towards effective low-light image enhancement | [pdf](https://linkinghub.elsevier.com/retrieve/pii/S0031320323003035) [code](https://github.com/ouc-ocean-group/SurroundNet) | SurroundNet | ⭐⭐ |
| 2023 | ICCV | Empowering low-light image enhancer through customized learnable priors | [pdf](http://export.arxiv.org/pdf/2309.01958) [code](https://github.com/zheng980629/CUE) | CUE | ⭐⭐⭐ |
| 2023 | ICCV | ExposureDiffusion: Learning to expose for low-light image enhancement | [pdf](https://arxiv.org/pdf/2307.07710.pdf) [code](https://github.com/wyf0912/ExposureDiffusion) | ExposureDiffusion | ⭐⭐⭐ |
| 2023 | ICCV | Retinexformer: One-stage retinex-based transformer for low-light image enhancement | [pdf](https://arxiv.org/abs/2303.06705) [code](https://github.com/caiyuanhao1998/Retinexformer) | Retinexformer | ⭐⭐⭐ |
| 2023 | ICCV | Lighting up NeRF via unsupervised decomposition and enhancement | [pdf](https://arxiv.org/abs/2307.10664) [code](https://github.com/onpix/LLNeRF) | LLNeRF | ⭐⭐⭐ |
| 2024 | IEEE Sens. Lett. | Integrating Graph Convolution Into a Deep Multilayer Framework for Low-Light Image Enhancement | [pdf](https://ieeexplore.ieee.org/document/10478172/) [code](https://github.com/yingqichao/GC-Net) | GC-Net | ⭐⭐⭐ |
| 2024 | IEEE TIP | AnlightenDiff: Anchoring Diffusion Probabilistic Model on Low Light Image Enhancement | [pdf](https://ieeexplore.ieee.org/document/10740586?source=authoralert) [code](https://github.com/ninglab/AnlightenDiff) | AnlightenDiff | ⭐⭐⭐ |
| 2025 | ICLR (Spotlight) | Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model | [pdf](https://arxiv.org/pdf/2311.11638) [code](https://github.com/Chunming-Li/Reti-Diff) | Reti-Diff | ⭐⭐⭐ |
| 2025 | CVPR | HVI: A New Color Space for Low-light Image Enhancement | [pdf](https://arxiv.org/abs/2502.20272) [code](https://github.com/Fediory/HVI-CIDNet) | HVI-CIDNet | ⭐⭐⭐ |
| 2025 | ICIP | RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement | [pdf](https://arxiv.org/abs/2505.24705) [code](https://github.com/jhakrraman/rt-xnet) [web](https://sites.google.com/view/rtxnet) | RT-X Net | ⭐⭐⭐ |
| 2025 | IJCV | Nonlocal Retinex-Based Variational Model and its Deep Unfolding Twin for Low-Light Image Enhancement | [pdf](https://link.springer.com/article/10.1007/s11263-025-02551-y) [code](https://github.com/RuiZhang97/NRVN) | NRVN | ⭐⭐⭐ |
| 2026 | ESWA | Towards lightest low-light image enhancement architecture for mobile devices | [pdf](https://arxiv.org/abs/2507.04277) [code](https://github.com/mubaisam/LiteIE) | LiteIE | ⭐⭐⭐ |
| 2026 | ArXiv | Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective | [pdf](https://arxiv.org/abs/2605.02627) [code](https://github.com/mubaisam/ICD) | ICD | ⭐⭐⭐ |

### 其他方法

**核心思想**：不严格属于 HE、Retinex 或深度学习的方法，通常结合相机响应模型、多曝光融合、图像滤波等技术。这类方法往往计算量小，适合资源受限的场景。

| 年份 | 发表 | 论文 | 链接 | 方法名 | 难度 |
|:--:|------|------|:--:|------|:--:|
| 2011 | ICME | Fast efficient algorithm for enhancement of low lighting video | [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6012107) | Xuan-Dong | ⭐ |
| 2017 | ICCVW | A New Low-Light Image Enhancement Algorithm Using Camera Response Model | [pdf](http://ieeexplore.ieee.org/document/8265567/) [code](https://github.com/baidut/OpenCE/blob/master/algorithm/enhancement/CRM.cpp) | CRM | ⭐ |
| 2017 | ArXiv | A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement | [pdf](http://arxiv.org/abs/1711.00591) [code](https://github.com/baidut/BIMEF) | BIMEF | ⭐ |
| 2019 | ICIP | Fast Image Enhancement Based on Maximum and Guided Filters | [pdf](https://ieeexplore.ieee.org/document/8803591) [code](https://github.com/zhouhang95/FIEM) | FIEM | ⭐ |

---

## 评价指标

### 全参考指标 (Full-Reference)
需要一张"标准答案"（正常光照下的参考图像）来做对比。

| 指标 | 缩写 | 含义 | 使用建议 |
|:--:|:--:|------|------|
| 峰值信噪比 | PSNR | 衡量增强结果和真值之间的像素级差异，值越大越好 | 最常用的指标，但和人的主观感受不完全一致 |
| 结构相似性 | SSIM | 从亮度、对比度、结构三个维度衡量两张图的相似度 | 比 PSNR 更接近人眼感知，推荐和 PSNR 搭配使用 |
| 学习感知相似度 | LPIPS | 用预训练深度网络提取特征来比较图像相似度 | 更接近人的感知判断，但计算开销较大。[代码](https://github.com/richzhang/PerceptualSimilarity) |
| 均方误差 | MSE | 像素级差的平方和平均 | 对异常值敏感，较少单独使用 |
| 平均绝对误差 | MAE | 像素级差的绝对值平均 | 比 MSE 更鲁棒 |

### 无参考指标 (No-Reference)
不需要参考图像，直接评价增强结果的"观感质量"。

| 指标 | 缩写 | 含义 | 使用建议 |
|:--:|:--:|------|------|
| 亮度顺序误差 | LOE | 衡量增强前后图像的亮度相对次序保持程度 | 专门为弱光增强设计的指标。[论文](https://ieeexplore.ieee.org/document/6512558) |
| 自然图像质量评估器 | NIQE | 从自然图像的统计规律出发评估质量 | 通用无参考指标，广泛使用。[论文](https://ieeexplore.ieee.org/document/6353522) |
| 手机摄影属性与质量 | SPAQ | 专门针对手机拍摄场景的质量评估 | [代码](https://github.com/h4nwei/SPAQ) |
| 神经图像评估 | NIMA | 用神经网络预测人类对图像的美学评分 | [PyTorch](https://github.com/kentsyx/Neural-IMage-Assessment) [TensorFlow](https://github.com/titu1994/neural-image-assessment) |
| 多尺度图像质量 Transformer | MUSIQ | 用 Transformer 处理多尺度特征评估图像质量 | 目前无参考指标中效果较好的选择。[代码](https://github.com/google-research/google-research/tree/master/musiq) |

> **新手提示**：做实验时，建议至少报告 PSNR + SSIM + NIQE 三项指标。PSNR/SSIM 衡量"多接近真值"，NIQE 衡量"看起来多自然"。如果条件允许加上 LPIPS，它能更好地反映感知质量。

---

## 评测代码

`evaluate.py` 基于官方库实现常用图像质量指标，支持单张图像、单个方法目录和多方法目录三种评测入口：

```bash
# 安装依赖
pip install -r requirements.txt

# 单张图像评测
python evaluate.py -e test_data/enhanced/method_a/scene1.png -g test_data/gt/scene1.png

# 单个方法目录评测
python evaluate.py --enhanced_dir test_data/enhanced/method_a --gt_dir test_data/gt/

# 多方法目录评测
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/

# 只运行不依赖 pyiqa/torch 的基础全参考指标
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/ --fr --nr

# 自定义 CSV 输出位置
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/ --output_csv comparison.csv
```

参数说明：

| 参数 | 说明 |
|------|------|
| `-e`, `--enhanced` | 单张增强图像路径 |
| `-g`, `--gt` | 单张参考图像路径 |
| `--enhanced_dir` | 单个方法的增强结果目录 |
| `--models_root` | 多方法根目录，每个子目录视为一个方法 |
| `--gt_dir` | 参考图像目录，文件名需和增强图像对应 |
| `--fr` | 选择全参考 pyiqa 指标；不接值时关闭 LPIPS 等 pyiqa 全参考指标 |
| `--nr` | 选择无参考 pyiqa 指标；不接值时关闭 NIQE/BRISQUE/PI 等无参考指标 |
| `--device` | 指定 `cuda` 或 `cpu`，默认自动选择 |
| `--resize` | 当增强图和参考图尺寸不一致时，将增强图缩放到参考图尺寸 |
| `--output_csv` | 指定 CSV 输出路径，默认 `metrics_summary.csv` |

| 指标 | 类型 | 实现来源 |
|------|:--:|------|
| PSNR / SSIM / MSE / MAE | 全参考 | `skimage.metrics` (官方库) |
| LPIPS | 全参考 | `pyiqa` (Zhang et al. CVPR 2018) |
| NIQE / BRISQUE / PI | 无参考 | `pyiqa` (官方实现) |
| MUSIQ / NIMA | 无参考 | `pyiqa` 可选 (`--nr ... musiq nima`) |

> **关于 LOE**：由于 scikit-image 和 pyiqa 均未提供 LOE 的官方实现，本脚本不包含 LOE。如需 LOE，请单独使用自定义实现并注明来源。

所有指标均使用 scikit-image 和 pyiqa 官方库，避免手工实现的偏差。适合作为方法对比的标准化评测工具。

### 一键体验

项目内置了测试数据，无需准备图像，开箱即用：

```bash
pip install -r requirements.txt
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/
```

`test_data/` 结构如下：

```
test_data/
├── gt/                    ← 正常光照参考图像
├── lowlight/              ← 原始弱光图像
└── enhanced/
    ├── method_a/          ← 方法 A 的增强结果
    └── method_b/          ← 方法 B 的增强结果
```

---

## 综述与基准测试

| 年份 | 发表 | 论文 | 链接 | 说明 |
|:--:|------|------|:--:|------|
| 2022 | ArXiv | Low-Light Image and Video Enhancement: A Comprehensive Survey and Beyond | [pdf](http://arxiv.org/abs/2212.10772) [code](https://github.com/shenzheng2000/llie_survey) | 较新的综述，适合快速了解领域全貌 |

---

## 相关研究方向

| 年份 | 发表 | 论文 | 链接 | 方法名 | 标签 |
|:--:|------|------|:--:|------|------|
| 2015 | ACM TOG | Automatic Photo Adjustment Using Deep Neural Networks | [web](https://sites.google.com/site/homepagezhichengyan/home/dl_img_adjust) [code](https://github.com/stephenyan1984/DeepPhotoStyle_TensorFlow) | DeepPhoto | 照片增强 |
| 2018 | CVPR | Distort-and-Recover: Color Enhancement using Deep Reinforcement Learning | [web](https://sites.google.com/view/distort-and-recover/) [pdf](https://doi.org/10.1109/CVPR.2018.00621) | Distort-and-Recover | 照片增强 |
| 2021 | TMM | Recurrent exposure generation for low-light face detection | [pdf](https://arxiv.org/abs/2007.10963) [code](https://github.com/sherrycattt/REGDet) | REGDet | 人脸检测 |
| 2022 | ICCP | Robust Scene Inference under Noise-Blur Dual Corruptions | [pdf](https://arxiv.org/abs/2207.11643) [code](https://github.com/bhavyagoyal/noiseblurdual) [web](https://wisionlab.com/projects/noiseblur/) | NoiseBlur | 场景推断 |
| 2024 | AAAI | Aleth-NeRF: Illumination Adaptive NeRF with Concealing Field Assumption | [pdf](https://arxiv.org/abs/2312.09093) [code](https://github.com/cuiziteng/Aleth-NeRF) [web](https://cuiziteng.github.io/Aleth-NeRF/) | Aleth-NeRF | NeRF/3D |

---

## 更多参考资源

- [OpenCE](https://github.com/baidut/OpenCE) — 图像增强算法集合（含多种传统方法实现）
- [image-enhancement-about-Retinex](https://github.com/tiandaoxiaowu/image-enhancement-about-Retinex) — Retinex 相关增强资源汇总
- [Lighting-the-Darkness-in-the-Deep-Learning-Era-Open](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open) — 深度学习时代弱光增强资源列表（含 Zero-DCE 等作者维护）

---

> **维护说明**：本指南基于社区贡献持续更新。如果你发现链接失效、有新的优秀工作推荐、或对内容有任何建议，欢迎通过 [issue](../../issues) 或 [pull request](../../pulls) 反馈。
