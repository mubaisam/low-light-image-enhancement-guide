# Low-Light Image Enhancement: A Beginner's Guide

> A beginner-friendly guide to low-light image enhancement, covering fundamental concepts, classic and deep learning methods, datasets, evaluation metrics, and evaluation code.

This project helps newcomers and graduate students build a systematic understanding of low-light image enhancement. It is not a complete literature database, but a structured learning guide with concept explanations, recommended papers, datasets, metrics, and hands-on evaluation scripts.

Contributions are welcome via [issues](../../issues) or [pull requests](../../pulls).

> **Note**: The main README (中文版) is in Chinese. This English version covers the same structure and key content.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Recommended Starter Papers](#recommended-starter-papers)
- [Datasets](#datasets)
- [Methods Overview](#methods-overview)
- [Evaluation Metrics](#evaluation-metrics)
- [Evaluation Code](#evaluation-code)
- [Surveys & Benchmarks](#surveys--benchmarks)
- [Related Work](#related-work)
- [More References](#more-references)

## Core Concepts

### What is Low-Light Image Enhancement?

Low-Light Image Enhancement (LLIE) refers to algorithms that improve image quality captured under low-light conditions — increasing brightness, revealing details, controlling noise, while maintaining a natural appearance. Applications include **night surveillance, autonomous driving, fluorescence microscopy, and smartphone night photography**.

### Why is it Hard?

- **Low photon counts** → weak signal, low SNR
- **Complex noise** → mixed Poisson-Gaussian noise distributions in dark conditions
- **High dynamic range** → coexisting extremely dark and bright regions
- **Color degradation** → severe color information loss in low light
- **No ground truth** → hard to capture perfectly aligned bright-dark image pairs

### Three Main Paradigms

#### 1. Histogram Equalization (HE)
Redistributes pixel intensity values to stretch dark regions and compress bright ones, improving overall contrast. Simple, fast, requires no training. Prone to over-enhancement and noise amplification. **Start with BPDHE** (brightness-preserving variant).

#### 2. Retinex Theory
Based on Edwin Land's theory: an image = **Illumination × Reflectance**. The reflectance component represents the object's true color/texture (independent of lighting). Methods estimate and correct the illumination component to recover the scene. Physically interpretable but ill-posed. **Start with LIME** (simplest approach: estimate illumination map, then gamma-correct).

#### 3. Deep Learning Methods
Let neural networks learn the "dark → bright" mapping from data. Sub-directions include:
- **End-to-end enhancement**: direct dark-to-bright mapping (SID, MBLLEN)
- **Retinex + Deep Learning**: network-based decomposition (RetinexNet, KinD)
- **GANs**: adversarial training for realism (EnlightenGAN)
- **Unsupervised/Self-supervised**: no paired data needed
- **Diffusion models**: generative enhancement (Diff-Retinex, ExposureDiffusion)
- **Transformers**: global illumination modeling (Retinexformer)

> **Tip**: Most methods boil down to either "learn a mapping function" or "decompose → adjust → reconstruct". Start with RetinexNet and EnlightenGAN.

---

## Recommended Starter Papers

If you only have time for a few papers, read these in order:

| # | Paper | Venue | Why Read | Level |
|:--:|------|:--:|------|:--:|
| 1 | **LIME** — Illumination Map Estimation | 2017 TIP | Clean, elegant traditional method. Best entry point. | ⭐ |
| 2 | **RetinexNet** — Deep Retinex Decomposition | 2018 BMVC | First to combine Retinex with deep learning. | ⭐⭐ |
| 3 | **KinD** — Kindling the Darkness | 2019 ACM MM | Complete decompose-adjust-reconstruct pipeline. | ⭐⭐ |
| 4 | **EnlightenGAN** — Enhancement without Paired Supervision | 2019 TIP | GAN-based, no paired data needed. Influential. | ⭐⭐ |
| 5 | **SID** — Learning to See in the Dark | 2018 CVPR | Landmark work on extreme low-light (~0.1 lux). | ⭐⭐⭐ |
| 6 | **Zero-DCE** — Zero-Reference Deep Curve Estimation | 2020 CVPR | No reference, no paired data. Novel curve-based approach. | ⭐⭐ |
| 7 | **URetinex-Net** — Deep Unfolding Network | 2022 CVPR | Unrolling optimization with deep learning. | ⭐⭐⭐ |
| 8 | **Retinexformer** — One-stage Transformer | 2023 ICCV | Transformer for LLIE. Latest architecture. | ⭐⭐⭐ |

---

## Datasets

Low-light enhancement datasets are usually split by whether paired reference images are available: **supervised paired datasets** and **unpaired/no-reference test sets**.

### Supervised Paired Datasets

These datasets provide low-light images and normal-light reference images. They are suitable for training or for full-reference metrics such as PSNR, SSIM, and LPIPS.

| Dataset | Subset/Version | Description | Link |
|:--:|:--:|------|:--:|
| LOLv1 | - | Classic paired benchmark for low-light image enhancement | [paper](https://arxiv.org/abs/1808.04560) / [project](https://daooshee.github.io/BMVC2018website) |
| LOLv2 | Real | Real captured low/normal-light paired images, closer to real-world scenes | [github](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) |
| LOLv2 | Synthetic | Synthetic low/normal-light paired images for controlled training and comparison | [github](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) |
| MIT-Adobe FiveK | - | 5,000 RAW photos with expert retouching results; often used for photo enhancement and exposure correction | [dataset](https://data.csail.mit.edu/graphics/fivek/) |
| LSRW | Huawei | Large-scale real-world paired low/normal-light data captured with Huawei devices | [paper](https://arxiv.org/abs/2106.14501) / [github](https://github.com/JianghaiSCU/R2RNet) |
| LSRW | Nikon | Large-scale real-world paired low/normal-light data captured with Nikon cameras | [paper](https://arxiv.org/abs/2106.14501) / [github](https://github.com/JianghaiSCU/R2RNet) |

### Unpaired / No-Reference Test Sets

These datasets usually contain low-light images without strictly paired normal-light references. They are commonly used for unsupervised methods, zero-reference methods, and cross-dataset generalization tests.

| Dataset | Description | Link |
|:--:|------|:--:|
| VV | Challenging low-light/exposure images for visual quality comparison | [github](https://github.com/baidut/BIMEF) |
| NPE | Naturalness Preserved Enhancement test set | [github](https://github.com/baidut/BIMEF) |
| LIME | Low-light test images used by the LIME paper | [github](https://github.com/baidut/BIMEF) |
| MEF | Multi-exposure fusion related low-light test images | [github](https://github.com/baidut/BIMEF) |
| DICM | Low-light images captured by commercial digital cameras | [github](https://github.com/baidut/BIMEF) |

---

## Methods Overview

For the complete paper list with links, codes, and difficulty ratings, see the [main Chinese README](./README.md#方法总览). The methods are organized into:

- **HE-based** — Simple, fast, no training needed. Good for real-time/embedded scenarios.
- **Retinex-based** — Physically interpretable. Traditional (LIME, MF) or deep (RetinexNet, KinD, Retinexformer).
- **Learning-based** — State-of-the-art performance. From CNNs to GANs to Diffusion to Transformers.
- **Other methods** — Camera response models, multi-exposure fusion, filtering-based.

---

## Evaluation Metrics

### Full-Reference (require ground truth)

| Metric | What it measures | Notes |
|:--:|------|------|
| **PSNR** | Pixel-level difference | Most common; doesn't always match perception |
| **SSIM** | Structural similarity (luminance, contrast, structure) | More perceptually aligned than PSNR |
| **LPIPS** | Deep feature similarity | Best perceptual alignment. [Code](https://github.com/richzhang/PerceptualSimilarity) |
| **MSE / MAE** | Pixel error (squared / absolute) | Simple baselines |

### No-Reference (no ground truth needed)

| Metric | What it measures | Notes |
|:--:|------|------|
| **LOE** | Lightness order preservation | Specifically designed for LLIE |
| **NIQE** | Naturalness via statistical regularities | Widely used general-purpose metric |
| **MUSIQ** | Multi-scale quality assessment with Transformer | Strong current no-reference metric |
| **NIMA** | Neural aesthetic scoring | Human preference prediction |
| **SPAQ** | Smartphone photography quality | Mobile-specific |

> **Tip**: Always report at least PSNR + SSIM + NIQE. Add LPIPS for perceptual quality.

Python implementations of these metrics are provided by `evaluate.py` in the project root.

---

## Evaluation Code

`evaluate.py` uses official libraries to compute common image-quality metrics. It supports single-image, one-method folder, and multi-method folder evaluation:

```bash
# Install dependencies
pip install -r requirements.txt

# Single image evaluation
python evaluate.py -e test_data/enhanced/method_a/scene1.png -g test_data/gt/scene1.png

# One-method folder evaluation
python evaluate.py --enhanced_dir test_data/enhanced/method_a --gt_dir test_data/gt/

# Multi-method folder evaluation
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/

# Run only basic full-reference metrics without pyiqa/torch
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/ --fr --nr

# Custom CSV output
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/ --output_csv comparison.csv
```

Arguments:

| Argument | Description |
|------|------|
| `-e`, `--enhanced` | Single enhanced image path |
| `-g`, `--gt` | Single reference image path |
| `--enhanced_dir` | Enhanced-result folder for one method |
| `--models_root` | Multi-method root folder; each subfolder is treated as one method |
| `--gt_dir` | Reference image folder; filenames should match enhanced images |
| `--fr` | Select full-reference pyiqa metrics; pass no values to disable LPIPS and other pyiqa full-reference metrics |
| `--nr` | Select no-reference pyiqa metrics; pass no values to disable NIQE/BRISQUE/PI and other no-reference metrics |
| `--device` | Use `cuda` or `cpu`; default is auto |
| `--resize` | Resize enhanced images to the reference image size when shapes differ |
| `--output_csv` | CSV output path; default is `metrics_summary.csv` |

| Metric | Type | Implementation |
|------|:--:|------|
| PSNR / SSIM / MSE / MAE | Full-ref | `skimage.metrics` (official) |
| LPIPS | Full-ref | `pyiqa` (Zhang et al. CVPR 2018) |
| NIQE / BRISQUE / PI | No-ref | `pyiqa` (official) |
| MUSIQ / NIMA | No-ref | `pyiqa` optional (`--nr ... musiq nima`) |

> **About LOE**: LOE is not included because neither scikit-image nor pyiqa provides an official implementation. If LOE is needed, use a custom implementation and clearly mark it as such.

All metrics use scikit-image and pyiqa official libraries for standardized, reproducible evaluation.

### Quick Demo

The project includes built-in test data. No image preparation needed:

```bash
pip install -r requirements.txt

# Compare method_a vs method_b on 3 test images
python evaluate.py --models_root test_data/enhanced/ --gt_dir test_data/gt/
```

`test_data/` structure:

```
test_data/
├── gt/                    ← Ground truth (normal light)
├── lowlight/              ← Original low-light images
└── enhanced/
    ├── method_a/          ← Results from method A
    └── method_b/          ← Results from method B
```

---

## Surveys & Benchmarks

| Year | Venue | Paper | Links |
|:--:|------|------|:--:|
| 2022 | ArXiv | Low-Light Image and Video Enhancement: A Comprehensive Survey and Beyond | [pdf](http://arxiv.org/abs/2212.10772) [code](https://github.com/shenzheng2000/llie_survey) |

---

## Related Work

| Year | Venue | Paper | Links | Topic |
|:--:|------|------|:--:|------|
| 2015 | ACM TOG | Automatic Photo Adjustment Using Deep Neural Networks | [web](https://sites.google.com/site/homepagezhichengyan/home/dl_img_adjust) [code](https://github.com/stephenyan1984/DeepPhotoStyle_TensorFlow) | Photo Enhancement |
| 2018 | CVPR | Distort-and-Recover: Color Enhancement using Deep RL | [web](https://sites.google.com/view/distort-and-recover/) [pdf](https://doi.org/10.1109/CVPR.2018.00621) | Photo Enhancement |
| 2021 | TMM | Recurrent exposure generation for low-light face detection | [pdf](https://arxiv.org/abs/2007.10963) [code](https://github.com/sherrycattt/REGDet) | Face Detection |
| 2022 | ICCP | Robust Scene Inference under Noise-Blur Dual Corruptions | [pdf](https://arxiv.org/abs/2207.11643) [code](https://github.com/bhavyagoyal/noiseblurdual) | Scene Inference |
| 2024 | AAAI | Aleth-NeRF: Illumination Adaptive NeRF | [pdf](https://arxiv.org/abs/2312.09093) [code](https://github.com/cuiziteng/Aleth-NeRF) | NeRF/3D |

---

## More References

- [OpenCE](https://github.com/baidut/OpenCE) — Collection of image enhancement algorithms
- [image-enhancement-about-Retinex](https://github.com/tiandaoxiaowu/image-enhancement-about-Retinex) — Retinex resources
- [Lighting-the-Darkness-in-the-Deep-Learning-Era-Open](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open) — Deep learning era LLIE resources

---

> **Maintenance**: This guide is community-maintained. For broken links, new work, or suggestions, please use [issues](../../issues) or [pull requests](../../pulls).
