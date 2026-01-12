import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset
import gc, time, os, json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from quantization_utils import (
    set_seed, cleanup_memory, load_fresh_model, evaluate_accuracy, quantize_tensor_fake,
    MODEL_ID, DEVICE
)


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

SEED = 42
WEIGHT_BITS = 4
ACTIVATION_BITS = 8
GROUP_SIZE = 128
WEIGHT_SYMMETRIC = False  # ✅ Asymmetric (reference default)
ACTIVATION_SYMMETRIC = True
EVAL_LIMIT = 0.1
EVAL_BATCH_SIZE = 1
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQ_LENGTH = 256
OUTPUT_DIR = "w_vs_wa_comparison"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


set_seed(SEED)

# ==========================================
# 🧹 UTILITIES
# ==========================================

def get_model_stats(model):
    """
    ✅ MATCHES REFERENCE: Only counts layers with quant_bit_width != 16
    """
    total_bits_full = 0
    target_bits = 0
    target_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            bits = 16  # Embeddings always 16-bit
            count = module.weight.numel()
            total_bits_full += count * bits
        elif isinstance(module, nn.Linear):
            if "lm_head" in name:
                bits = 16  # lm_head always 16-bit
                count = module.weight.numel()
                total_bits_full += count * bits
            else:
                bits = getattr(module, 'quant_bit_width', 16)
                count = module.weight.numel()
                total_bits_full += count * bits
                target_bits += count * bits
                target_params += count

    full_mb = total_bits_full / 8 / 1024**2
    target_mb = target_bits / 8 / 1024**2
    
    if target_params > 0:
        avg_target_bits = target_bits / target_params
        target_ratio = 16.0 / avg_target_bits
    else:
        target_ratio = 1.0
        
    return {
        'total_mb': full_mb,
        'quantized_mb': target_mb,
        'ratio': target_ratio,
        'avg_bits': avg_target_bits if target_params > 0 else 16.0,
        'total_params': target_params
    }



# ==========================================
# 🔧 WEIGHT QUANTIZATION (EXACT REFERENCE MATCH)
# ==========================================
def quantize_tensor_fake(w, n_bit=4, granularity="per_group", group_size=128, sym=False):
    """
    ✅ EXACT COPY from reference script
    """
    original_shape = w.shape
    
    # 1. Reshape based on Granularity
    if granularity == "per_tensor":
        w_reshaped = w.flatten().reshape(1, -1)
    elif granularity == "per_channel":
        w_reshaped = w.reshape(w.shape[0], -1)
    elif granularity == "per_group":
        if w.numel() % group_size != 0:
            pad = group_size - (w.numel() % group_size)
            w = torch.nn.functional.pad(w.flatten(), (0, pad))
        w_reshaped = w.reshape(-1, group_size)
    
    # 2. Quantization Logic (Sym/Asym)
    if sym:
        max_val = w_reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        max_int = 2**(n_bit - 1) - 1
        scale = max_val / max_int
        w_q = torch.clamp(torch.round(w_reshaped / scale), -max_int, max_int)
        w_fake = w_q * scale
    else:
        min_val = w_reshaped.amin(dim=-1, keepdim=True)
        max_val = w_reshaped.amax(dim=-1, keepdim=True)
        scale = (max_val - min_val) / (2**n_bit - 1)
        scale = scale.clamp(min=1e-5)
        zero_point = torch.round(-min_val / scale)
        w_q = torch.clamp(torch.round(w_reshaped / scale + zero_point), 0, 2**n_bit - 1)
        w_fake = (w_q - zero_point) * scale
    
    # 3. Restore Shape
    if granularity == "per_group":
        w_fake = w_fake.reshape(-1)
        if w_fake.numel() > original_shape.numel():
            w_fake = w_fake[:original_shape.numel()]
    return w_fake.reshape(original_shape)

def apply_weight_quantization(model, n_bits=4, group_size=128, symmetric=False):
    """
    ✅ CORRECTED: Exact match to reference run_rtn()
    - Only quantizes Linear layers (excluding lm_head)
    - Does NOT touch Embedding layers
    """
    print(f"   [W-Quant] Applying {n_bits}-bit weight quantization (sym={symmetric})...")
    quantized_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            # ✅ Quantize Linear layers (except lm_head)
            module.quant_bit_width = n_bits
            with torch.no_grad():
                module.weight.data = quantize_tensor_fake(
                    module.weight.data,
                    n_bit=n_bits,
                    granularity="per_group",
                    group_size=group_size,
                    sym=symmetric
                )
            quantized_count += 1
        # ✅ IMPORTANT: Do NOT touch Embedding layers (reference behavior)
    
    print(f"   [W-Quant] ✅ Quantized {quantized_count} layers")
    return model

# ==========================================
# 🔧 ACTIVATION QUANTIZATION (CALIBRATION-BASED)
# ==========================================
class ActivationQuantizer(nn.Module):
    def __init__(self, n_bits=8, symmetric=True, percentile=0.99):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.percentile = percentile
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('clip_min', torch.tensor(-6.0))
        self.register_buffer('clip_max', torch.tensor(6.0))
        self.calibrated = False
        self.collecting_stats = False
        self.activation_buffer = []
        if symmetric:
            self.max_int = 2 ** (n_bits - 1) - 1
            self.min_int = -(2 ** (n_bits - 1))
        else:
            self.max_int = 2 ** n_bits - 1
            self.min_int = 0
    
    def collect_stats(self, x):
        if self.collecting_stats and len(self.activation_buffer) < 100:
            sample = x.detach().cpu().float().flatten()[::100]
            self.activation_buffer.append(sample)
    
    def calibrate(self):
        if not self.activation_buffer:
            return
        all_acts = torch.cat(self.activation_buffer)
        if self.symmetric:
            abs_max = torch.quantile(all_acts.abs(), self.percentile)
            self.clip_min = -abs_max
            self.clip_max = abs_max
            self.scale = abs_max / self.max_int if abs_max > 1e-8 else torch.tensor(1.0)
            self.zero_point = torch.tensor(0.0)
        else:
            self.clip_min = torch.quantile(all_acts, 1 - self.percentile)
            self.clip_max = torch.quantile(all_acts, self.percentile)
            range_val = self.clip_max - self.clip_min
            if range_val > 1e-8:
                self.scale = range_val / (self.max_int - self.min_int)
                self.zero_point = self.min_int - torch.round(self.clip_min / self.scale)
                self.zero_point = torch.clamp(self.zero_point, self.min_int, self.max_int)
            else:
                self.scale = torch.tensor(1.0)
                self.zero_point = torch.tensor(0.0)
        self.calibrated = True
        self.activation_buffer = []
    
    def forward(self, x):
        if self.collecting_stats:
            self.collect_stats(x)
            return x
        if not self.calibrated:
            return x
        x_clipped = torch.clamp(x, self.clip_min.to(x.device), self.clip_max.to(x.device))
        scale = self.scale.to(x.device)
        zero_point = self.zero_point.to(x.device)
        if self.symmetric:
            if scale < 1e-8:
                return x_clipped
            x_int = torch.clamp(torch.round(x_clipped / scale), self.min_int, self.max_int)
            return x_int * scale
        else:
            if scale < 1e-8:
                return x_clipped
            x_int = torch.clamp(torch.round(x_clipped / scale + zero_point), self.min_int, self.max_int)
            return (x_int - zero_point) * scale

class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, act_bits=8, act_symmetric=True):
        super().__init__()
        self.linear = linear_layer
        self.act_quantizer = ActivationQuantizer(act_bits, act_symmetric)
    
    def forward(self, x):
        x_quant = self.act_quantizer(x)
        return self.linear(x_quant)

def apply_activation_quantization(model, act_bits=8, act_symmetric=True):
    print(f"   [A-Quant] Applying {act_bits}-bit activation quantization...")
    converted_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent = model
            if '.' in name:
                *parent_names, attr_name = name.split('.')
                for pname in parent_names:
                    parent = getattr(parent, pname)
            else:
                attr_name = name
            quant_layer = QuantizedLinear(module, act_bits, act_symmetric)
            setattr(parent, attr_name, quant_layer)
            converted_count += 1
    print(f"   [A-Quant] ✅ Wrapped {converted_count} layers")
    return model

def calibrate_activations(model, tokenizer, num_samples=128, seq_length=256):
    print(f"   [Calibration] Collecting statistics from {num_samples} samples...")
    for module in model.modules():
        if isinstance(module, ActivationQuantizer):
            module.collecting_stats = True
            module.activation_buffer = []
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    model.eval()
    with torch.no_grad():
        count = 0
        for sample in tqdm(dataset, desc="Calibrating", total=num_samples):
            if count >= num_samples:
                break
            if len(sample['text']) < 50:
                continue
            tokens = tokenizer(sample['text'], return_tensors="pt", max_length=seq_length, 
                             truncation=True, padding="max_length")
            input_ids = tokens['input_ids'].to(DEVICE)
            attention_mask = tokens['attention_mask'].to(DEVICE)
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                count += 1
            except Exception as e:
                continue
    print(f"   [Calibration] Computing quantization parameters...")
    calibrated_count = 0
    for name, module in model.named_modules():
        if isinstance(module, ActivationQuantizer):
            module.collecting_stats = False
            module.calibrate()
            if module.calibrated:
                calibrated_count += 1
    print(f"   [Calibration] ✅ Calibrated {calibrated_count} quantizers")
    return model

# ==========================================
# 📊 PLOTTING FUNCTIONS
# ==========================================
def plot_accuracy_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    experiments = results['experiments']
    names = [exp['name'] for exp in experiments]
    accuracies = [exp['accuracy'] for exp in experiments]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{acc:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('MMLU Accuracy: Weight-Only vs Weight+Activation', fontsize=16, fontweight='bold')
    ax.set_ylim([0, max(accuracies) * 1.15])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '1_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 1_accuracy_comparison.png")

def plot_accuracy_drop(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    experiments = results['experiments'][1:]
    names = [exp['name'] for exp in experiments]
    drops = [exp['accuracy_drop_%'] for exp in experiments]
    colors = ['#e74c3c' if drop > 0 else '#2ecc71' for drop in drops]
    bars = ax.bar(names, drops, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{drop:.2f}%',
                ha='center', va='bottom' if drop > 0 else 'top', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Drop from Baseline', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '2_accuracy_drop.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 2_accuracy_drop.png")

def plot_model_size_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    experiments = results['experiments']
    names = [exp['name'] for exp in experiments]
    sizes = [exp['model_size_mb'] for exp in experiments]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(names, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{size:.1f} MB',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Size (MB)', fontsize=14, fontweight='bold')
    ax.set_title('Model Size Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '3_model_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 3_model_size.png")

def plot_compression_ratio(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    experiments = results['experiments']
    names = [exp['name'] for exp in experiments]
    ratios = [exp['compression_ratio'] for exp in experiments]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(names, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{ratio:.2f}x',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Compression Ratio (Higher is Better)', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '4_compression_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 4_compression_ratio.png")

def plot_time_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    experiments = results['experiments']
    names = [exp['name'] for exp in experiments]
    
    # Quantization time
    ax = axes[0]
    quant_times = [exp.get('quantization_time_seconds', 0) for exp in experiments]
    colors = ['#95a5a6', '#e74c3c', '#3498db']
    bars = ax.bar(names, quant_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, time_val in zip(bars, quant_times):
        if time_val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{time_val:.1f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Quantization Time', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Evaluation time
    ax = axes[1]
    eval_times = [exp['eval_time_seconds'] for exp in experiments]
    bars = ax.bar(names, eval_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, time_val in zip(bars, eval_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{time_val:.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Time', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '5_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 5_time_comparison.png")

def plot_accuracy_vs_size_tradeoff(results):
    fig, ax = plt.subplots(figsize=(10, 8))
    experiments = results['experiments']
    
    for exp in experiments:
        acc = exp['accuracy']
        size = exp['model_size_mb']
        name = exp['name']
        
        if 'Baseline' in name:
            color, marker, size_mult = '#2ecc71', 'o', 300
        elif 'Weight-Only' in name:
            color, marker, size_mult = '#e74c3c', 's', 250
        else:
            color, marker, size_mult = '#3498db', '^', 250
        
        ax.scatter(size, acc, s=size_mult, c=color, marker=marker, alpha=0.7, 
                  edgecolors='black', linewidth=2, label=name)
        ax.annotate(name, (size, acc), xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Model Size (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs Model Size Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '6_accuracy_vs_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 6_accuracy_vs_size.png")

def create_summary_dashboard(results):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    experiments = results['experiments']
    
    # Plot 1: Accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    names = [exp['name'] for exp in experiments]
    accuracies = [exp['accuracy'] for exp in experiments]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax1.bar(names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{acc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('MMLU Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Accuracy drop
    ax2 = fig.add_subplot(gs[0, 2])
    drops = [exp.get('accuracy_drop_%', 0) for exp in experiments[1:]]
    names_drop = [exp['name'] for exp in experiments[1:]]
    colors_drop = ['#e74c3c' if d > 0 else '#2ecc71' for d in drops]
    bars = ax2.barh(names_drop, drops, color=colors_drop, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, drop in zip(bars, drops):
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2., f' {drop:.2f}%',
                ha='left' if drop > 0 else 'right', va='center', fontsize=10, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Drop (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Drop', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Model size
    ax3 = fig.add_subplot(gs[1, 0])
    sizes = [exp['model_size_mb'] for exp in experiments]
    bars = ax3.bar(names, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, size in zip(bars, sizes):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{size:.0f}MB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Size (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Size', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Compression ratio
    ax4 = fig.add_subplot(gs[1, 1])
    ratios = [exp['compression_ratio'] for exp in experiments]
    bars = ax4.bar(names, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, ratio in zip(bars, ratios):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{ratio:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Compression', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Key metrics table
    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.axis('off')
    baseline_exp, w_exp, wa_exp = experiments[0], experiments[1], experiments[2]
    acc_diff = w_exp['accuracy'] - wa_exp['accuracy']
    
    table_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['Baseline Acc', f"{baseline_exp['accuracy']:.4f}"],
        ['W-Only Acc', f"{w_exp['accuracy']:.4f}"],
        ['W+A Acc', f"{wa_exp['accuracy']:.4f}"],
        ['', ''],
        ['W vs W+A Diff', f'{acc_diff:+.4f}'],
        ['Diff (%)', f'{acc_diff*100:+.2f}%'],
        ['', ''],
        ['Winner', 'W-Only' if acc_diff > 0 else 'W+A' if acc_diff < -0.001 else 'Tie'],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(table_data)):
        cell = table[(i, 0)]
        if i in [0, 9]:
            cell.set_facecolor('#3498db')
            cell.set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        elif i in [1, 5, 8]:
            cell.set_facecolor('#ecf0f1')
            table[(i, 1)].set_facecolor('#ecf0f1')
    
    # Plot 6: Conclusion
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.axis('off')
    
    if acc_diff > 0.001:
        conclusion = f"✅ WEIGHT-ONLY QUANTIZATION IS BETTER\n\n"
        conclusion += f"Accuracy advantage: +{acc_diff*100:.2f}%\n\n"
        conclusion += "Key insights:\n"
        conclusion += "• Activation quantization adds overhead\n"
        conclusion += "• No accuracy benefit observed\n"
        conclusion += "• Simpler deployment with W-only"
        color = '#2ecc71'
    elif acc_diff < -0.001:
        conclusion = f"✅ WEIGHT+ACTIVATION QUANTIZATION IS BETTER\n\n"
        conclusion += f"Accuracy advantage: +{-acc_diff*100:.2f}%\n\n"
        conclusion += "Key insights:\n"
        conclusion += "• Activation quantization helps accuracy\n"
        conclusion += "• Worth the calibration overhead\n"
        conclusion += "• Better for INT8-accelerated hardware"
        color = '#3498db'
    else:
        conclusion = f"⚖️ NO SIGNIFICANT DIFFERENCE\n\n"
        conclusion += f"Accuracy difference: {acc_diff*100:+.2f}%\n\n"
        conclusion += "Recommendation:\n"
        conclusion += "• Use W-only for simplicity\n"
        conclusion += "• Use W+A if targeting INT8 hardware"
        color = '#f39c12'
    
    ax6.text(0.5, 0.5, conclusion, transform=ax6.transAxes, fontsize=13,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=1),
            family='monospace', fontweight='bold')
    
    plt.suptitle('Weight-Only vs Weight+Activation Quantization Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(PLOT_DIR, '0_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 0_summary_dashboard.png")

def generate_all_plots(results):
    print("\n" + "="*100)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*100)
    create_summary_dashboard(results)
    plot_accuracy_comparison(results)
    plot_accuracy_drop(results)
    plot_model_size_comparison(results)
    plot_compression_ratio(results)
    plot_time_comparison(results)
    plot_accuracy_vs_size_tradeoff(results)
    print(f"\n   ✅ All plots saved to: {PLOT_DIR}/")

# ==========================================
# 🎯 MAIN COMPARISON EXPERIMENT
# ==========================================
def run_w_vs_wa_comparison():
    print("\n" + "="*100)
    print("🎯 QUANTIZATION COMPARISON: W-Only vs W+A")
    print("="*100)
    print(f"Model: {MODEL_ID}")
    print(f"Weight: {WEIGHT_BITS}-bit {'Symmetric' if WEIGHT_SYMMETRIC else 'Asymmetric'} (group_size={GROUP_SIZE})")
    print(f"Activation: {ACTIVATION_BITS}-bit {'Symmetric' if ACTIVATION_SYMMETRIC else 'Asymmetric'}")
    print(f"Evaluation: MMLU ({EVAL_LIMIT*100}% of dataset)")
    print("="*100)
    
    results = {
        'config': {
            'model_id': MODEL_ID, 'weight_bits': WEIGHT_BITS, 'weight_symmetric': WEIGHT_SYMMETRIC,
            'activation_bits': ACTIVATION_BITS, 'activation_symmetric': ACTIVATION_SYMMETRIC,
            'group_size': GROUP_SIZE, 'eval_limit': EVAL_LIMIT, 'calibration_samples': CALIBRATION_SAMPLES
        },
        'experiments': []
    }
    
    # Baseline
    print("\n" + "="*100)
    print("EXPERIMENT 0: BASELINE (FP16)")
    print("="*100)
    model_baseline, tokenizer = load_fresh_model()
    acc_baseline = evaluate_accuracy(model_baseline, tokenizer, "Baseline FP16")
    stats_baseline = get_model_stats(model_baseline)
    results['experiments'].append({
        'name': 'Baseline (FP16)', 'type': 'baseline', 'accuracy': acc_baseline,
        'accuracy_drop_%': 0.0, 'model_size_mb': stats_baseline['total_mb'],
        'compression_ratio': 1.0, 'avg_bits': 16.0, 
    })
    print(f"\n   📊 Baseline: Acc={acc_baseline:.4f}, Size={stats_baseline['total_mb']:.2f}MB")
    cleanup_memory(model_baseline)
    
    # Weight-Only
    print("\n" + "="*100)
    print("EXPERIMENT 1: WEIGHT-ONLY QUANTIZATION (W)")
    print("="*100)
    model_w, _ = load_fresh_model()
    start_time = time.time()
    model_w = apply_weight_quantization(model_w, WEIGHT_BITS, GROUP_SIZE, WEIGHT_SYMMETRIC)
    quant_time_w = time.time() - start_time
    acc_w = evaluate_accuracy(model_w, tokenizer, "Weight-Only (W)")
    stats_w = get_model_stats(model_w)
    results['experiments'].append({
        'name': 'Weight-Only (W)', 'type': 'weight_only', 'weight_bits': WEIGHT_BITS,
        'weight_symmetric': WEIGHT_SYMMETRIC, 'activation_bits': 16, 'accuracy': acc_w,
        'accuracy_drop_%': (acc_baseline - acc_w) * 100, 'model_size_mb': stats_w['quantized_mb'],
        'compression_ratio': stats_w['ratio'], 'avg_bits': stats_w['avg_bits'],
        'quantization_time_seconds': quant_time_w,
    })
    print(f"\n   📊 W-Only: Acc={acc_w:.4f} (drop: {(acc_baseline-acc_w)*100:.2f}%), "
          f"Size={stats_w['quantized_mb']:.2f}MB, Compression={stats_w['ratio']:.2f}x")
    cleanup_memory(model_w)
    
    # Weight+Activation
    print("\n" + "="*100)
    print("EXPERIMENT 2: WEIGHT + ACTIVATION QUANTIZATION (W+A)")
    print("="*100)
    model_wa, _ = load_fresh_model()
    start_time = time.time()
    model_wa = apply_weight_quantization(model_wa, WEIGHT_BITS, GROUP_SIZE, WEIGHT_SYMMETRIC)
    model_wa = apply_activation_quantization(model_wa, ACTIVATION_BITS, ACTIVATION_SYMMETRIC)
    model_wa = calibrate_activations(model_wa, tokenizer, CALIBRATION_SAMPLES, CALIBRATION_SEQ_LENGTH)
    quant_time_wa = time.time() - start_time
    acc_wa = evaluate_accuracy(model_wa, tokenizer, "Weight+Activation (W+A)")
    stats_wa = get_model_stats(model_wa)
    results['experiments'].append({
        'name': 'Weight + Activation (W+A)', 'type': 'weight_activation', 'weight_bits': WEIGHT_BITS,
        'weight_symmetric': WEIGHT_SYMMETRIC, 'activation_bits': ACTIVATION_BITS,
        'activation_symmetric': ACTIVATION_SYMMETRIC, 'accuracy': acc_wa,
        'accuracy_drop_%': (acc_baseline - acc_wa) * 100, 'model_size_mb': stats_wa['quantized_mb'],
        'compression_ratio': stats_wa['ratio'], 'avg_bits': stats_wa['avg_bits'],
        'quantization_time_seconds': quant_time_wa, 'calibration_samples': CALIBRATION_SAMPLES,
    })
    print(f"\n   📊 W+A: Acc={acc_wa:.4f} (drop: {(acc_baseline-acc_wa)*100:.2f}%), "
          f"Size={stats_wa['quantized_mb']:.2f}MB, Compression={stats_wa['ratio']:.2f}x")
    cleanup_memory(model_wa)
    
    # Analysis
    print("\n" + "="*100)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*100)
    baseline_exp, w_exp, wa_exp = results['experiments'][0], results['experiments'][1], results['experiments'][2]
    acc_diff = (w_exp['accuracy'] - wa_exp['accuracy']) * 100
    #time_overhead = (wa_exp['eval_time_seconds'] / w_exp['eval_time_seconds'] - 1) * 100
    #quant_time_overhead = (wa_exp['quantization_time_seconds'] / w_exp['quantization_time_seconds'] - 1) * 100
    
    results['comparison'] = {
        'accuracy_difference_%': acc_diff,
        #'eval_time_overhead_%': time_overhead,
        #'quantization_time_overhead_%': quant_time_overhead,
        'model_size_same': w_exp['model_size_mb'] == wa_exp['model_size_mb'],
    }
    
    print(f"\n{'Metric':<40} {'Baseline':<15} {'W-Only':<15} {'W+A':<15}")
    print("-"*85)
    print(f"{'Accuracy':<40} {baseline_exp['accuracy']:<15.4f} {w_exp['accuracy']:<15.4f} {wa_exp['accuracy']:<15.4f}")
    print(f"{'Accuracy Drop (%)':<40} {'-':<15} {w_exp['accuracy_drop_%']:<15.2f} {wa_exp['accuracy_drop_%']:<15.2f}")
    print(f"{'Model Size (MB)':<40} {baseline_exp['model_size_mb']:<15.2f} {w_exp['model_size_mb']:<15.2f} {wa_exp['model_size_mb']:<15.2f}")
    #print(f"{'Compression Ratio':<40} {baseline_exp['compression_ratio']:<15.2f}x {w_exp['compression_ratio']:<15.2f}x {wa_exp['compression_ratio']:<15.2f}x")
    #print(f"{'Quantization Time (s)':<40} {'-':<15} {w_exp['quantization_time_seconds']:<15.1f} {wa_exp['quantization_time_seconds']:<15.1f}")
    #print(f"{'Evaluation Time (s)':<40} {baseline_exp['eval_time_seconds']:<15.1f} {w_exp['eval_time_seconds']:<15.1f} {wa_exp['eval_time_seconds']:<15.1f}")
    
    print("\n" + "="*100)
    print("🔍 KEY FINDINGS:")
    print("="*100)
    print(f"1. Accuracy difference (W-only vs W+A): {acc_diff:+.2f}%")
    if abs(acc_diff) < 0.5:
        print(f"   → Activation quantization has MINIMAL impact")
    elif acc_diff > 0:
        print(f"   → W-only is BETTER than W+A")
    else:
        print(f"   → W+A is slightly better than W-only")
    print(f"\n2. Model size: {'SAME' if results['comparison']['model_size_same'] else 'DIFFERENT'}")
    print(f"   → Activation quantization does NOT reduce model size")
    #print(f"\n3. Evaluation time overhead: {time_overhead:+.1f}%")
    #print(f"4. Quantization time overhead: {quant_time_overhead:+.1f}%")
    
    # Generate plots
    generate_all_plots(results)
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    return results

# ==========================================
# 🚀 MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*100)
    print("🚀 STARTING W vs W+A QUANTIZATION COMPARISON")
    print("="*100)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE} | Model: {MODEL_ID}")
    print("="*100)
    start_time = time.time()
    try:
        results = run_w_vs_wa_comparison()
        print(f"\n✅ Experiment completed! Total time: {(time.time() - start_time)/60:.1f} minutes")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "="*100)
    print("🎉 DONE!")
    print("="*100)
