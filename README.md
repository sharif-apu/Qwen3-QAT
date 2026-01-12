# Technical Assignment 1: LLM Quantization & Optimization

## Executive Summary

The Qwen/Qwen3-0.6B model has been compressed up to **75% (358.45 MB)** with 4-bit quantization. The quantization error has been successfully recovered with quantization-aware training (QAT), achieving **0.4154 MMLU score**, surpassing the baseline FP16 model (0.4036). Additionally, several PTQ methods (AWQ, RTN, NF4, GPTQ) and mixed-precision quantization techniques have been explored.

![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/model_compression_teaser.png)

## 📋 Table of Contents

- [Environments and Setup](#environments-and-setup)
- [Performance Summary](#performance-summary)
- [Methodology](#methodology)
  - [Step 0: Pre-Experiment Analysis](#step-0-pre-experiment-analysis)
  - [Step 1: Baseline (RTN)](#step-1-baseline-rtn)
  - [Step 2: Advanced Quantization](#step-2-advanced-quantization)
  - [Step 3: Mixed Precision Quantization](#step-3-mixed-precision-quantization)
  - [Step 4: Activation Quantization](#step-4-activation-quantization-wa)
- [Edge Deployment](#edge-deployment)
- [References](#references)

---

## 🛠️ Environments and Setup

### Virtual Environment
```bash
python3.12 -m venv ~/envTest
```

### Additional Package Installation
```bash
pip install -r requirements.txt
```

### Configuration

| Component | Specification |
|-----------|---------------|
| GPU | RTX 3060 (12GB) |
| CPU | Ryzen 3 3200G |
| RAM | 32GB |
| CUDA | 12.1 |
| Python | 3.12 |

### Experiment Setup

**MODEL_ID:**
```
Qwen/Qwen3-0.6B
```

**TASK_NAME:**
```
mmlu
```

**Parameters:**
- LIMIT: 0.1
- MAX_SEQ_LEN: 512
- Batch Size: 1
- Calibration/Training Dataset: WikiText
- Samples: 4090 (Training), 128 (Calibration)

---

## 📊 Performance Summary

| Method | Size (MB) | Compression | MMLU Score | Δ Score | Weight Bits | Act. Bits | Notes |
|--------|-----------|-------------|------------|---------|-------------|-----------|-------|
| Original (FP16) | 1433.5 | - | 0.4035 | - | 16 | 16 | Baseline |
| RTN | 803.5 | 43.95% | 0.3021 | -0.1014 | 4 | 16 | Best RTN |
| NF4 | 803.5 | 43.95% | 0.3280 | -0.0755 | 4 | 16 | Best PTQ |
| QAT (W-only) | 803.5 | 43.95% | 0.3937 | -0.0098 | 4 | 16 | Excl. lm_head/emb |
| QAT Mixed (W4/W8) | 506.82 | 64.60% | 0.4182 | +0.0147 | 4-8 | 16 | ⭐ Best overall |
| QAT (W4A4) | 358.45 | 75.00% | 0.4154 | +0.0119 | 4 | 4 | ⭐ Highest compression |
| RTN (W4A4) | 803.5 | 43.95% | 0.2497 | -0.1538 | 4 | 4 | With act. quant. |

---

## 🔬 Methodology

### Step 0: Pre-Experiment Analysis

The Qwen/Qwen3-0.6B model has been analyzed layer by layer to identify layer-wise properties, including parameters, size, and compression effects.

**To reproduce analysis:**
```bash
python ta1_step0_model_analysis.py
```

#### Model Configuration Analysis

![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/summary_dashboard.png)

| Config | Emb | LM Head | Linear | Total (MB) | Core (MB) | Total Comp | Total Red% | Core Comp | Core Red% |
|--------|-----|---------|--------|------------|-----------|------------|------------|-----------|-----------|
| baseline_fp16 | FP16 | FP16 | FP16 | 1433.62 | 840.12 | 1.00× | 0.0% | 1.00× | 0.0% |
| all_4bit | 4-bit | 4-bit | 4-bit | 358.45 | 210.07 | 4.00× | 75.0% | 4.00× | 75.0% |
| mixed_4_2 | 4-bit | 4-bit | 2-bit | 253.44 | 105.07 | 5.66× | 82.3% | 8.00× | 87.5% |
| all_8bit | 8-bit | 8-bit | 8-bit | 716.84 | 420.09 | 2.00× | 50.0% | 2.00× | 50.0% |
| mixed_8_2 | 8-bit | 8-bit | 2-bit | 401.82 | 105.07 | 3.57× | 72.0% | 8.00× | 87.5% |
| mixed_8_4 | 8-bit | 8-bit | 4-bit | 506.82 | 210.07 | 2.83× | 64.6% | 4.00× | 75.0% |
| mixed_fp16_4 | FP16 | FP16 | 4-bit | 803.57 | 210.07 | 1.78× | 43.9% | 4.00× | 75.0% |

---

### Step 1: Baseline (RTN)

Round-to-Nearest (RTN) post-training quantization has been applied to weights (excluding embedding and lm_head) using the quantization formula:

**Quantization Formula:**
```
W_q = Round(W / Δ) · Δ
```

RTN supports multiple granularities (per-tensor, per-channel, group-wise) with both symmetric and asymmetric quantization modes.

#### RTN Results

| Method | Granularity | Mode | Full Size (MB) | Compression | Core Size (MB) | Core Comp | MMLU Score | Δ Score |
|--------|-------------|------|----------------|-------------|----------------|-----------|------------|---------|
| Original | - | - | 1433.5 | - | 840 | 1.00× | 0.4035 | - |
| RTN_Tensor | Tensor | Asym | 803.5 | 43.95% | 210 | 4.00× | 0.2490 | -0.1545 |
| RTN_Channel | Channel | Asym | 803.5 | 43.95% | 210 | 4.00× | 0.2692 | -0.1343 |
| RTN_Group | Group-128 | Asym | 803.5 | 43.95% | 210 | 4.00× | 0.3021 | -0.1014 |
| RTN_Tensor_sym | Tensor | Sym | 803.5 | 43.95% | 210 | 4.00× | 0.2343 | -0.1692 |
| RTN_Channel_sym | Channel | Sym | 803.5 | 43.95% | 210 | 4.00× | 0.2811 | -0.1224 |
| RTN_Group_sym | Group-128 | Sym | 803.5 | 43.95% | 210 | 4.00× | 0.2671 | -0.1364 |

✅ **Verdict:** RTN_Group (Asymmetric, Group-128): 0.3021 MMLU - Best RTN method

**To reproduce:**
```bash
python ta1_step1_baseline.py
```

---

### Step 2: Advanced Quantization

For Steps 2 and 3, Quantization-Aware Training (QAT) was chosen due to:

1. **Proven track record:** QAT works better for quantization-sensitive small models like Qwen3-0.6B
2. **Time management:** Small LLMs on RTX 3060 take 20-30 minutes per epoch on 3000-4000 samples. QAT typically reaches local minima within 1-10 epochs (~3-5 hours per experiment)
3. **Reliable results:** Qwen3 is relatively new; widely-used libraries (AutoGPTQ, bitsandbytes, etc.) do not provide official support and have version compatibility issues. Although we could patch libraries or implement algorithms ourselves, results may be inconsistent due to hyperparameter search/implementation constraints
4. **Results from Step 1:** Results from Step 1 provide clear insights into which configurations (granularities) work better for the given model. These findings are directly used for static weight-only quantization

#### Quantization Method Comparison

| Method | MMLU Accuracy | Accuracy Drop | Relative Drop (%) | Model Size (MB) | Compression Ratio |
|--------|---------------|---------------|-------------------|-----------------|-------------------|
| Original (FP16) | 0.4035 | - | - | 840 | 1.00× (baseline) |
| RTN_G128 | 0.2629 | -0.1406 | 34.85% | 210 | 4.00× |
| GPTQ* | 0.3077 | -0.0958 | 23.74% | 210 | 4.00× |
| AWQ* | 0.3112 | -0.0923 | 22.87% | 210 | 4.00× |
| NF4* | 0.3280 | -0.0755 | 18.71% | 210 | 4.00× |
| QAT** | 0.3937 | -0.0098 | 2.43% | 210 | 4.00× |

*Manually implemented. Need verifications.

**Didn't use gradient accumulation. Results could be improved.

Loss during QAT
![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/Screenshot%202026-01-13%20at%202.45.34%E2%80%AFAM.png)

**To produce PTQ results:**
```bash
python ta1_step2_ptq.py
```

**To perform QAT:**
```bash
python ta1_step2_qat.py
```

---

### Step 3: Mixed Precision Quantization

To perform mixed-precision (INT) QAT, we first analyzed lm_head/embedding impact on the network. It was found that lm_head/embedding only affects 0.70% accuracy drop while providing significant compression. Such results provide a clear indication to further compress the model.
![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/lm0_summary.png)
![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/lm5_output_distributions.png)

**To execute lm_head-embedding quantization analysis:**
```bash
python ta1_step3_lmhead_analysis.py
```

#### Strategy

![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/mp_qat.png)
To search for architecture and find optimal architecture, the following strategy was taken, inspired by neural architecture search algorithms:

1. First, heuristically find combinations of LM head-embedding and linear layers
2. Force the best trade-off model (analyzed layer by layer) to adopt lower bit-widths
3. Train optimal architecture

Layer-by-layer bit selection requires substantial compute and time. Due to time constraints, the second stage was skipped (implemented and can be found in `ta1_step3_mp_qatkd` file), and the best two architectures from stage 1 were trained. Knowledge Distillation (KD) was integrated into the pipeline; however, due to memory constraints, it was skipped.

To speed up experiments, we selected 4-bit and 8-bit quantization due to their optimal balance between model compression and accuracy preservation:

- **8-bit quantization** provides near-lossless performance while achieving 4× memory reduction and faster inference compared to 32-bit floating-point, making it the de facto standard for production deployment
- **4-bit quantization** pushes compression further with 8× memory savings, enabling deployment on resource-constrained edge devices, though it requires careful calibration to maintain acceptable accuracy

#### Results Before QAT

| Config | Accuracy | Acc Drop | Δ (%) | Core Comp | Total Comp | Core Size (MB) | Total Size (MB) |
|--------|----------|----------|-------|-----------|------------|----------------|-----------------|
| baseline_fp16 | 0.4035 | - | - | 1.00× | 1.00× | 840.12 | 1433.62 |
| all_8bit | 0.4021 | -0.0014 | -0.35% | 2.00× | 2.00× | 420.09 | 716.84 |
| all_4bit | 0.2895 | -0.1140 | -28.25% | 4.00× | 4.00× | 210.07 | 358.45 |
| mixed_8_2 | 0.2385 | -0.1650 | -40.89% | 8.00× | 3.57× | 105.07 | 401.82 |
| mixed_4_2 | 0.2357 | -0.1678 | -41.58% | 8.00× | 5.66× | 105.07 | 253.44 |

#### Final Results After QAT

| Config | Accuracy | Acc Drop | Δ (%) | Core Comp | Total Comp | Core Size (MB) | Total Size (MB) |
|--------|----------|----------|-------|-----------|------------|----------------|-----------------|
| baseline_fp16 | 0.4035 | - | - | 1.00× | 1.00× | 840.12 | 1433.62 |
| QAT_mixed_4_8 | 0.4182 | +0.0147 | +3.64% | 4.00× | 2.83× | 210.07 | 506.82 |
| QAT_all_4bit | 0.4154 | +0.0119 | +2.95% | 4.00× | 4.00× | 210.07 | 358.45 |

**To perform mixed-precision QAT:**
```bash
python ta1_step3_mp_qatkd.py
```

Default 4/8 bit quantization without KD. To turn on stage one, set `RUN_STAGE1 = True`

![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/Screenshot%202026-01-13%20at%202.45.11%E2%80%AFAM.png)


---

### Step 4: Activation Quantization [W+A]

To further investigate activation quantization, a dynamic quantization approach was implemented that converts floating-point activations to 8-bit integers using calibrated scale and zero-point parameters. The process involves two stages:

1. **Calibration:** 128 representative samples from WikiText-2 are processed to collect activation statistics and determine optimal quantization ranges via percentile-based clipping
2. **Runtime quantization:** Activations are mapped to discrete integer representations

While this enables efficient computation on INT8-accelerated hardware, experiments reveal that activation quantization introduces calibration and inference overhead with potential accuracy degradation, depending on bit-width, calibration quality, and symmetric versus asymmetric quantization schemes.

![Alt Text](https://github.com/sharif-apu/nota_ta_260111/blob/main/plots/wa_summary_dashboard.png)


#### Technical Challenges in LLMs

**Extreme Outlier Activations:**

LLMs exhibit severe outlier features in specific channels (particularly in attention and feed-forward layers) where a few values dominate the dynamic range, causing catastrophic quantization errors. These outliers emerge systematically across tokens and layers, making uniform quantization schemes ineffective [1].

**Dynamic Range Variability:**

Unlike static weights, activations exhibit token-dependent and layer-dependent dynamic ranges that vary significantly across different input contexts. This requires adaptive per-layer or per-token quantization parameters that increase computational overhead [2].

**Limited Practical Benefits:**

Activation quantization provides no model size reduction since only runtime computations are affected. Offers speedup benefits exclusively on INT8-accelerated hardware while introducing calibration overhead and implementation complexity [4].

---

## 🚀 Edge Deployment

### Required Conversion Steps

Deploying quantized models to edge hardware requires hardware-aware conversions that vary by manufacturer and device constraints. Developed models (PyTorch/TensorFlow) are typically exported to intermediate representations like ONNX for platform independence, then converted to target-specific formats:

- `.trt` for NVIDIA Jetson
- `.vino` for Intel OpenVINO
- `.tflite` for mobile devices

End-to-end toolchains like Torch-TensorRT streamline this process.

As an investigation into exporting of Qwen3-0.6B, an export function (to ONNX) has been developed to run it. Please execute the following command:

```bash
python eval_export.py --checkpoint /path/to/checkpoint --export_onnx
```

This script can also be used to evaluate trained weights. [Click Here](https://drive.google.com/drive/folders/1M_uPIZ2ka-aGHN_ED8kRK0zbroYVpEAk?usp=sharing) to download the checkpoint.

The deployment pipeline involves four critical steps:

#### 1. Exporting:

- PyTorch model to edge-compatible intermediate formats (ONNX, TensorFlow Lite, GGML)
- Preserving quantization parameters (scales, zero-points, group sizes)

#### 2. Converting:

- Fake quantization to true integer-only operations
- Weights stored as INT4/INT8 with hardware-aligned quantization granularity

#### 3. Optimizing:

Graph optimizations including:
- Operator fusion (Linear-Activation, Conv-BatchNorm)
- Constant folding
- Dead code elimination
- Attention mechanism simplification

Reduces memory bandwidth and computational overhead.

#### 4. Compiling:

For target hardware accelerators:
- ARM NEON
- Qualcomm Hexagon DSP
- Apple Neural Engine

Using optimized kernel libraries (XNNPACK, QNNPACK) or vendor-specific SDKs. Proper memory allocation strategies for constrained environments.

---

### Technical Challenges

Edge deployment faces significant obstacles:

#### Custom Kernel Requirements:

Group-wise quantization (group_size=128) and mixed-precision operations (INT4 weights with INT8 activations) lack native support on standard mobile frameworks. Necessitates platform-specific implementations [1].

#### Hardware Compatibility Issues:

- Most existing edge hardware does not support native INT4 compute units
- Forces emulation via INT8 which negates compression benefits
- Certain operations (e.g., complex attention mechanisms, layer normalization) require excessive CPU/GPU cycles on specific hardware
- Necessitates layer pruning or replacement with equivalent optimized operations [2]

#### Asymmetric Quantization Overhead:

Zero-point operations add computational overhead on hardware optimized for symmetric schemes.

#### Deployment Constraints:

- Inability to perform on-device calibration
- Requires pre-computed parameters that must generalize across diverse inputs
- Framework fragmentation across TFLite/ONNX Runtime/PyTorch Mobile with varying quantization support
- Critical trade-off: fake quantization's FP16 weights must convert to true INT4 storage, but runtime dequantization overhead may increase end-to-end latency beyond theoretical speedup gains [3][4]

---

### Recommended Strategy

For practical edge deployment:

✅ Prioritize weight-only quantization to eliminate activation calibration overhead

✅ Target symmetric quantization for broader hardware compatibility

✅ Validate early on target hardware to identify unsupported operations

✅ Implement fallback mechanisms for custom ops

✅ Profile layer-wise execution to prune or replace computationally expensive operations

✅ Benchmark end-to-end latency, including dequantization costs rather than relying on theoretical compression ratios

---

## 📚 References

[1] Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS.

[2] Xiao, G., et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML.

[3] Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR.

[4] Gholami, A., et al. (2022). "A Survey of Quantization Methods for Efficient Neural Network Inference." Low-Power Computer Vision.

[5] Frantar, E., & Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR.

[6] Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys.
