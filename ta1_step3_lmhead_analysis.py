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
import matplotlib.pyplot as plt
import seaborn as sns

from quantization_utils import (
    set_seed, cleanup_memory, load_fresh_model,
    get_model_stats, evaluate_accuracy, quantize_tensor_fake,
    MODEL_ID, DEVICE
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

SEED = 42
set_seed(SEED)
WEIGHT_BITS = 4
GROUP_SIZE = 128
EVAL_LIMIT = 0.1
EVAL_BATCH_SIZE = 1
ANALYSIS_SAMPLES = 50
ANALYSIS_SEQ_LENGTH = 256
OUTPUT_DIR = "lmhead_embedding_analysis"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ==========================================
# 🔧 JSON SERIALIZATION HELPER
# ==========================================
def convert_to_json_serializable(obj):
    """Convert PyTorch tensors and numpy arrays to JSON-serializable types"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def get_model_stats(model):
    total_bits_full = target_bits = target_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            bits = getattr(module, 'quant_bit_width', 16)
            count = module.weight.numel()
            total_bits_full += count * bits
            if bits != 16:
                target_bits += count * bits
                target_params += count
    full_mb = total_bits_full / 8 / 1024**2
    target_mb = target_bits / 8 / 1024**2
    target_ratio = 16.0 / (target_bits / target_params) if target_params > 0 else 1.0
    return full_mb, target_mb, target_ratio

# def evaluate_accuracy(model, tokenizer, desc="Evaluation"):
#     print(f"   [Eval] {desc}...")
#     start_time = time.time()
#     lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=EVAL_BATCH_SIZE)
#     results = simple_evaluate(model=lm_obj, tasks=["mmlu"], limit=EVAL_LIMIT, device=DEVICE, log_samples=False)
#     acc = results["results"]["mmlu"]["acc,none"]
#     elapsed = time.time() - start_time
#     print(f"   [Eval] ✅ {desc}: {acc:.4f} ({elapsed:.1f}s)")
#     del lm_obj
#     cleanup_memory()
#     return acc, elapsed

# ==========================================
# 🔧 QUANTIZATION (REFERENCE METHOD)
# ==========================================
# def quantize_tensor_fake(w, n_bit=4, granularity="per_group", group_size=128, sym=False):
#     original_shape = w.shape
#     if granularity == "per_tensor":
#         w_reshaped = w.flatten().reshape(1, -1)
#     elif granularity == "per_channel":
#         w_reshaped = w.reshape(w.shape[0], -1)
#     elif granularity == "per_group":
#         if w.numel() % group_size != 0:
#             pad = group_size - (w.numel() % group_size)
#             w = torch.nn.functional.pad(w.flatten(), (0, pad))
#         w_reshaped = w.reshape(-1, group_size)
    
#     if sym:
#         max_val = w_reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
#         max_int = 2**(n_bit - 1) - 1
#         scale = max_val / max_int
#         w_q = torch.clamp(torch.round(w_reshaped / scale), -max_int, max_int)
#         w_fake = w_q * scale
#     else:
#         min_val = w_reshaped.amin(dim=-1, keepdim=True)
#         max_val = w_reshaped.amax(dim=-1, keepdim=True)
#         scale = (max_val - min_val) / (2**n_bit - 1)
#         scale = scale.clamp(min=1e-5)
#         zero_point = torch.round(-min_val / scale)
#         w_q = torch.clamp(torch.round(w_reshaped / scale + zero_point), 0, 2**n_bit - 1)
#         w_fake = (w_q - zero_point) * scale
    
#     if granularity == "per_group":
#         w_fake = w_fake.reshape(-1)
#         if w_fake.numel() > original_shape.numel():
#             w_fake = w_fake[:original_shape.numel()]
#     return w_fake.reshape(original_shape)

def apply_weight_quantization(model, n_bits=4, group_size=128, skip_lmhead_embedding=False):
    mode = "Skip LM Head/Embedding" if skip_lmhead_embedding else "Quantize All"
    print(f"   [W-Quant] Applying {n_bits}-bit weight quantization ({mode})...")
    quantized_count = skipped_count = 0
    skipped_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = skip_lmhead_embedding and "lm_head" in name
            if not should_skip:
                module.quant_bit_width = n_bits
                with torch.no_grad():
                    module.weight.data = quantize_tensor_fake(
                        module.weight.data, n_bit=n_bits, granularity="per_group",
                        group_size=group_size, sym=False
                    )
                quantized_count += 1
            else:
                module.quant_bit_width = 16
                skipped_count += 1
                skipped_layers.append(name)
        elif isinstance(module, nn.Embedding):
            if skip_lmhead_embedding:
                module.quant_bit_width = 16
                skipped_count += 1
                skipped_layers.append(name)
            else:
                module.quant_bit_width = n_bits
                with torch.no_grad():
                    module.weight.data = quantize_tensor_fake(
                        module.weight.data, n_bit=n_bits, granularity="per_group",
                        group_size=group_size, sym=False
                    )
                quantized_count += 1
    
    print(f"   [W-Quant] ✅ Quantized {quantized_count} layers")
    if skipped_count > 0:
        print(f"   [W-Quant] ⏭️  Skipped {skipped_count} layers: {skipped_layers[:3]}...")
    return model

# ==========================================
# 📊 ANALYSIS FUNCTIONS
# ==========================================
def analyze_weight_distribution(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = module.weight.data.float().cpu()
            stats[name] = {
                'type': 'Embedding' if isinstance(module, nn.Embedding) else 'Linear',
                'shape': list(weight.shape),
                'mean': float(weight.mean().item()),
                'std': float(weight.std().item()),
                'min': float(weight.min().item()),
                'max': float(weight.max().item()),
                'abs_max': float(weight.abs().max().item()),
                'outlier_ratio': float((weight.abs() > 3 * weight.std()).float().mean().item()),
                'dynamic_range': float((weight.max() - weight.min()).item()),
                'kurtosis': float(((weight - weight.mean()) ** 4).mean().item() / (weight.std() ** 4 + 1e-8)),
                'weight_values': weight.flatten().numpy()
            }
    return stats

def analyze_quantization_error(model_fp16, model_quant):
    errors = {}
    fp16_modules = {name: module for name, module in model_fp16.named_modules()}
    quant_modules = {name: module for name, module in model_quant.named_modules()}
    
    for name in fp16_modules:
        if name not in quant_modules:
            continue
        fp16_module = fp16_modules[name]
        quant_module = quant_modules[name]
        if not isinstance(fp16_module, (nn.Linear, nn.Embedding)):
            continue
        
        fp16_weight = fp16_module.weight.data.float().cpu()
        quant_weight = quant_module.weight.data.float().cpu()
        error = (fp16_weight - quant_weight).abs()
        relative_error = error / (fp16_weight.abs() + 1e-8)
        
        errors[name] = {
            'type': 'Embedding' if isinstance(fp16_module, nn.Embedding) else 'Linear',
            'mae': float(error.mean().item()),
            'mse': float((error ** 2).mean().item()),
            'max_error': float(error.max().item()),
            'relative_mae': float(relative_error.mean().item()),
            'snr_db': float(10 * torch.log10((fp16_weight ** 2).mean() / ((error ** 2).mean() + 1e-8)).item()),
        }
    return errors

def analyze_output_distribution(model, tokenizer, num_samples=50):
    print(f"   [Analysis] Analyzing output distribution on {num_samples} samples...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    model.eval()
    all_logits, all_entropy, all_top1_confidence = [], [], []
    
    with torch.no_grad():
        count = 0
        for sample in tqdm(dataset, desc="Analyzing outputs", total=num_samples):
            if count >= num_samples:
                break
            if len(sample['text']) < 50:
                continue
            tokens = tokenizer(sample['text'], return_tensors="pt", max_length=ANALYSIS_SEQ_LENGTH, truncation=True)
            input_ids = tokens['input_ids'].to(DEVICE)
            try:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :].float().cpu()
                all_logits.append(logits)
                probs = F.softmax(logits, dim=-1)
                all_entropy.append(-(probs * torch.log(probs + 1e-10)).sum().item())
                all_top1_confidence.append(probs.max().item())
                count += 1
            except:
                continue
    
    all_logits = torch.stack(all_logits)
    return {
        'logits': {
            'mean': float(all_logits.mean().item()),
            'std': float(all_logits.std().item()),
            'dynamic_range': float((all_logits.max() - all_logits.min()).item()),
            'values': all_logits.numpy(),
        },
        'entropy': {
            'mean': float(np.mean(all_entropy)),
            'std': float(np.std(all_entropy)),
            'values': all_entropy,
        },
        'confidence': {
            'top1_mean': float(np.mean(all_top1_confidence)),
            'top1_values': all_top1_confidence,
        }
    }

def analyze_embedding_usage(model, tokenizer, num_samples=50):
    print(f"   [Analysis] Analyzing embedding usage on {num_samples} samples...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    vocab_size = model.get_input_embeddings().weight.shape[0]
    token_counts = torch.zeros(vocab_size)
    
    count = 0
    for sample in tqdm(dataset, desc="Analyzing embeddings", total=num_samples):
        if count >= num_samples or len(sample['text']) < 50:
            if count >= num_samples:
                break
            continue
        tokens = tokenizer(sample['text'], return_tensors="pt", max_length=ANALYSIS_SEQ_LENGTH, truncation=True)
        for token_id in tokens['input_ids'][0]:
            token_counts[token_id] += 1
        count += 1
    
    used_tokens = (token_counts > 0).sum().item()
    return {
        'vocab_size': int(vocab_size),
        'used_tokens': int(used_tokens),
        'unused_tokens': int(vocab_size - used_tokens),
        'usage_ratio': float(used_tokens / vocab_size),
        'top_10_tokens': [int(x) for x in token_counts.topk(10).indices.tolist()],
        'top_10_counts': [float(x) for x in token_counts.topk(10).values.tolist()],
    }

# ==========================================
# 📊 PLOTTING FUNCTIONS
# ==========================================
def create_bar_plot(ax, names, values, colors, ylabel, title, show_values=True):
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    if show_values:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}' if val < 1 else f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

def plot_accuracy_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    experiments = results['experiments']
    names = [exp['name'] for exp in experiments]
    accuracies = [exp['accuracy'] for exp in experiments]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    create_bar_plot(ax, names, accuracies, colors, 'Accuracy', 'MMLU Accuracy Comparison')
    ax.set_ylim([0, max(accuracies) * 1.15])
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
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{drop:.2f}%', ha='center', va='bottom' if drop > 0 else 'top', 
                fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Drop from Baseline', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '2_accuracy_drop.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 2_accuracy_drop.png")

def plot_weight_distributions(baseline_stats):
    lmhead_stats = {k: v for k, v in baseline_stats.items() if 'lm_head' in k.lower()}
    embed_stats = {k: v for k, v in baseline_stats.items() if 'embed' in k.lower() and 'lm_head' not in k.lower()}
    other_stats = {k: v for k, v in baseline_stats.items() if 'lm_head' not in k.lower() and 'embed' not in k.lower()}
    
    layers_to_plot = {}
    if lmhead_stats:
        layers_to_plot['LM Head'] = list(lmhead_stats.values())[0]
    if embed_stats:
        layers_to_plot['Embedding'] = list(embed_stats.values())[0]
    if other_stats:
        layers_to_plot['Transformer Layer'] = list(other_stats.values())[len(other_stats)//2]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, stats) in enumerate(layers_to_plot.items()):
        ax = axes[idx]
        weights = stats['weight_values']
        sample_size = min(10000, len(weights))
        weights_sample = np.random.choice(weights, sample_size, replace=False)
        ax.hist(weights_sample, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
        ax.axvline(stats['mean'] + stats['std'], color='orange', linestyle='--', linewidth=2, label=f"Std: {stats['std']:.4f}")
        ax.axvline(stats['mean'] - stats['std'], color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}\nOutliers: {stats["outlier_ratio"]*100:.2f}%', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '3_weight_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 3_weight_distributions.png")

def plot_quantization_errors(quant_errors_all, quant_errors_skip):
    lmhead_name = [k for k in quant_errors_all.keys() if 'lm_head' in k.lower()]
    embed_name = [k for k in quant_errors_all.keys() if 'embed' in k.lower() and 'lm_head' not in k.lower()]
    if not lmhead_name and not embed_name:
        print("   ⚠️  No lm_head/embedding layers found for error plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    layers, snr_all, snr_skip, mae_all, mae_skip = [], [], [], [], []
    
    if lmhead_name:
        layers.append('LM Head')
        snr_all.append(quant_errors_all[lmhead_name[0]]['snr_db'])
        snr_skip.append(quant_errors_skip.get(lmhead_name[0], {}).get('snr_db', 0))
        mae_all.append(quant_errors_all[lmhead_name[0]]['relative_mae'] * 100)
        mae_skip.append(quant_errors_skip.get(lmhead_name[0], {}).get('relative_mae', 0) * 100)
    if embed_name:
        layers.append('Embedding')
        snr_all.append(quant_errors_all[embed_name[0]]['snr_db'])
        snr_skip.append(quant_errors_skip.get(embed_name[0], {}).get('snr_db', 0))
        mae_all.append(quant_errors_all[embed_name[0]]['relative_mae'] * 100)
        mae_skip.append(quant_errors_skip.get(embed_name[0], {}).get('relative_mae', 0) * 100)
    
    x = np.arange(len(layers))
    width = 0.35
    
    # SNR plot
    ax = axes[0]
    bars1 = ax.bar(x - width/2, snr_all, width, label='Quantize All', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, snr_skip, width, label='Skip LM/Emb', color='#2ecc71', alpha=0.8)
    ax.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Quantization SNR\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    
    # MAE plot
    ax = axes[1]
    bars1 = ax.bar(x - width/2, mae_all, width, label='Quantize All', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, mae_skip, width, label='Skip LM/Emb', color='#2ecc71', alpha=0.8)
    ax.set_ylabel('Relative MAE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quantization Error\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '4_quantization_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 4_quantization_errors.png")

def plot_output_distributions(baseline_out, all_out, skip_out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Logits distribution
    ax = axes[0, 0]
    for logits, label, color in [(baseline_out['logits']['values'].flatten(), 'Baseline', '#2ecc71'),
                                   (all_out['logits']['values'].flatten(), 'Quantize All', '#e74c3c'),
                                   (skip_out['logits']['values'].flatten(), 'Skip LM/Emb', '#3498db')]:
        sample = np.random.choice(logits, min(5000, len(logits)), replace=False)
        ax.hist(sample, bins=50, alpha=0.5, label=label, color=color)
    ax.set_xlabel('Logit Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Logits Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Entropy comparison
    ax = axes[0, 1]
    entropy_data = [baseline_out['entropy']['values'], all_out['entropy']['values'], skip_out['entropy']['values']]
    bp = ax.boxplot(entropy_data, labels=['Baseline', 'Quantize All', 'Skip LM/Emb'], patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c', '#3498db']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Entropy\n(Lower = More Confident)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Top-1 Confidence
    ax = axes[1, 0]
    confidence_data = [baseline_out['confidence']['top1_values'], all_out['confidence']['top1_values'], 
                       skip_out['confidence']['top1_values']]
    bp = ax.boxplot(confidence_data, labels=['Baseline', 'Quantize All', 'Skip LM/Emb'], patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c', '#3498db']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Top-1 Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence\n(Higher = More Confident)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Normalized metrics
    ax = axes[1, 1]
    metrics = ['Logit Range', 'Entropy', 'Top-1 Conf']
    baseline_vals = [baseline_out['logits']['dynamic_range'], baseline_out['entropy']['mean'], 
                     baseline_out['confidence']['top1_mean']]
    all_vals = [all_out['logits']['dynamic_range'], all_out['entropy']['mean'], all_out['confidence']['top1_mean']]
    skip_vals = [skip_out['logits']['dynamic_range'], skip_out['entropy']['mean'], skip_out['confidence']['top1_mean']]
    all_norm = [a/b for a, b in zip(all_vals, baseline_vals)]
    skip_norm = [s/b for s, b in zip(skip_vals, baseline_vals)]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, all_norm, width, label='Quantize All', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, skip_norm, width, label='Skip LM/Emb', color='#3498db', alpha=0.8)
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', linewidth=2, label='Baseline')
    ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
    ax.set_title('Metrics Relative to Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '5_output_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 5_output_distributions.png")

def plot_embedding_usage(embedding_stats):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    sizes = [embedding_stats['used_tokens'], embedding_stats['unused_tokens']]
    labels = [f"Used\n({embedding_stats['used_tokens']:,})", f"Unused\n({embedding_stats['unused_tokens']:,})"]
    ax.pie(sizes, explode=(0.05, 0), labels=labels, colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title(f'Embedding Usage\n(Vocab Size: {embedding_stats["vocab_size"]:,})', fontsize=14, fontweight='bold')
    
    ax = axes[1]
    top_tokens = embedding_stats['top_10_tokens'][:10]
    top_counts = embedding_stats['top_10_counts'][:10]
    bars = ax.barh(range(len(top_tokens)), top_counts, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels([f"Token {t}" for t in top_tokens])
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Most Frequent Tokens', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, top_counts)):
        ax.text(count, i, f' {int(count)}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '6_embedding_usage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 6_embedding_usage.png")

def plot_layer_statistics_comparison(baseline_stats):
    lmhead_stats = {k: v for k, v in baseline_stats.items() if 'lm_head' in k.lower()}
    embed_stats = {k: v for k, v in baseline_stats.items() if 'embed' in k.lower() and 'lm_head' not in k.lower()}
    other_stats = {k: v for k, v in baseline_stats.items() if 'lm_head' not in k.lower() and 'embed' not in k.lower()}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    categories, std_vals, outlier_vals, dynamic_range_vals, kurtosis_vals = [], [], [], [], []
    
    if lmhead_stats:
        categories.append('LM Head')
        lmhead_data = list(lmhead_stats.values())[0]
        std_vals.append(lmhead_data['std'])
        outlier_vals.append(lmhead_data['outlier_ratio'] * 100)
        dynamic_range_vals.append(lmhead_data['dynamic_range'])
        kurtosis_vals.append(lmhead_data['kurtosis'])
    if embed_stats:
        categories.append('Embedding')
        embed_data = list(embed_stats.values())[0]
        std_vals.append(embed_data['std'])
        outlier_vals.append(embed_data['outlier_ratio'] * 100)
        dynamic_range_vals.append(embed_data['dynamic_range'])
        kurtosis_vals.append(embed_data['kurtosis'])
    if other_stats:
        categories.append('Transformer\nLayers (avg)')
        std_vals.append(np.mean([v['std'] for v in other_stats.values()]))
        outlier_vals.append(np.mean([v['outlier_ratio'] for v in other_stats.values()]) * 100)
        dynamic_range_vals.append(np.mean([v['dynamic_range'] for v in other_stats.values()]))
        kurtosis_vals.append(np.mean([v['kurtosis'] for v in other_stats.values()]))
    
    colors = ['#e74c3c', '#f39c12', '#3498db']
    
    for ax, vals, ylabel, title in [(axes[0,0], std_vals, 'Standard Deviation', 'Weight Standard Deviation'),
                                      (axes[0,1], outlier_vals, 'Outlier Ratio (%)', 'Outlier Ratio (|w| > 3σ)'),
                                      (axes[1,0], dynamic_range_vals, 'Dynamic Range', 'Weight Dynamic Range'),
                                      (axes[1,1], kurtosis_vals, 'Kurtosis', 'Weight Kurtosis')]:
        bars = ax.bar(categories, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if ylabel == 'Kurtosis':
            ax.axhline(y=3, color='red', linestyle='--', linewidth=2, label='Normal Distribution')
            ax.legend()
        for bar, val in zip(bars, vals):
            fmt = f'{val:.4f}' if ylabel == 'Standard Deviation' or ylabel == 'Dynamic Range' else f'{val:.2f}'
            if ylabel == 'Outlier Ratio (%)':
                fmt += '%'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), fmt, 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '7_layer_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 7_layer_statistics.png")

def create_summary_plot(results):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    experiments = results['experiments']
    
    # Accuracy comparison
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
    
    # Accuracy drop
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
    
    # Model size
    ax3 = fig.add_subplot(gs[1, 0])
    sizes = [exp.get('model_size_mb', 0) for exp in experiments]
    if sizes[0] == 0:
        sizes[0] = sizes[1] * 4
    bars = ax3.bar(names, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, size in zip(bars, sizes):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{size:.0f}MB', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Size (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Size', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Compression ratio
    ax4 = fig.add_subplot(gs[1, 1])
    ratios = [exp.get('compression_ratio', 1.0) for exp in experiments]
    bars = ax4.bar(names, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, ratio in zip(bars, ratios):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{ratio:.2f}x', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Compression', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Key metrics table
    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.axis('off')
    baseline_acc, all_acc, skip_acc = experiments[0]['accuracy'], experiments[1]['accuracy'], experiments[2]['accuracy']
    diff = all_acc - skip_acc
    table_data = [
        ['Metric', 'Value'], ['', ''],
        ['Baseline Accuracy', f'{baseline_acc:.4f}'],
        ['Quantize All Acc', f'{all_acc:.4f}'],
        ['Skip LM/Emb Acc', f'{skip_acc:.4f}'],
        ['', ''], ['Difference', f'{diff:+.4f}'],
        ['Difference (%)', f'{diff*100:+.2f}%'],
        ['', ''], ['Winner', 'Quantize All' if diff > 0 else 'Skip LM/Emb'],
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
    
    # Conclusion
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.axis('off')
    if diff > 0.001:
        conclusion = f"✅ QUANTIZING LM HEAD/EMBEDDING IS BETTER\nAccuracy improvement: +{diff*100:.2f}%\n\n"
        conclusion += "Likely reasons:\n• Regularization effect\n• Outlier smoothing\n• Logit dampening"
        color = '#2ecc71'
    elif diff < -0.001:
        conclusion = f"❌ SKIPPING LM HEAD/EMBEDDING IS BETTER\nAccuracy improvement: +{-diff*100:.2f}%\n\n"
        conclusion += "Likely reasons:\n• Sensitive layers\n• Error propagation\n• Well-calibrated model"
        color = '#e74c3c'
    else:
        conclusion = f"⚖️ NO SIGNIFICANT DIFFERENCE\nAccuracy difference: {diff*100:+.2f}%\n\nEither approach is acceptable."
        color = '#f39c12'
    ax6.text(0.5, 0.5, conclusion, transform=ax6.transAxes, fontsize=13, verticalalignment='center',
             horizontalalignment='center', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=1),
             family='monospace', fontweight='bold')
    
    plt.suptitle('LM Head & Embedding Quantization Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(PLOT_DIR, '0_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 0_summary.png")

# ==========================================
# 🎯 MAIN ANALYSIS
# ==========================================
def run_lmhead_embedding_analysis():
    print("\n" + "="*100)
    print("🔬 ROOT CAUSE ANALYSIS: LM HEAD & EMBEDDING QUANTIZATION")
    print("="*100)
    print(f"Model: {MODEL_ID} | Weight: {WEIGHT_BITS}-bit | Group Size: {GROUP_SIZE} | Samples: {ANALYSIS_SAMPLES}")
    print("="*100)
    
    results = {
        'config': {'model_id': MODEL_ID, 'weight_bits': WEIGHT_BITS, 'group_size': GROUP_SIZE, 
                   'analysis_samples': ANALYSIS_SAMPLES, 'method': 'reference_vectorized'},
        'experiments': [], 'analysis': {}
    }
    
    # Baseline
    print("\n" + "="*100)
    print("EXPERIMENT 0: BASELINE (FP16)")
    print("="*100)
    model_baseline, tokenizer = load_fresh_model()
    acc_baseline = evaluate_accuracy(model_baseline, tokenizer, "Baseline FP16")
    full_mb, _, _ = get_model_stats(model_baseline)
    
    print("\n   [Analysis] Analyzing baseline...")
    baseline_weight_stats = analyze_weight_distribution(model_baseline)
    baseline_output_stats = analyze_output_distribution(model_baseline, tokenizer, ANALYSIS_SAMPLES)
    embedding_stats = analyze_embedding_usage(model_baseline, tokenizer, ANALYSIS_SAMPLES)
    
    results['experiments'].append({
        'name': 'Baseline (FP16)', 'accuracy': acc_baseline,
        'model_size_mb': full_mb, 'compression_ratio': 1.0,
    })
    results['analysis']['baseline'] = {
        'weight_stats': {k: {key: val for key, val in v.items() if key != 'weight_values'} 
                        for k, v in baseline_weight_stats.items()},
        'output_stats': {k: {key: val for key, val in v.items() if key != 'values'} 
                        for k, v in baseline_output_stats.items()},
        'embedding_stats': embedding_stats,
    }
    
    # Quantize All
    print("\n" + "="*100)
    print("EXPERIMENT 1: QUANTIZE ALL")
    print("="*100)
    model_all, _ = load_fresh_model()
    model_all = apply_weight_quantization(model_all, WEIGHT_BITS, GROUP_SIZE, skip_lmhead_embedding=False)
    acc_all = evaluate_accuracy(model_all, tokenizer, "Quantize All")
    full_mb, target_mb, ratio = get_model_stats(model_all)
    
    print("\n   [Analysis] Analyzing quantization errors...")
    quant_errors_all = analyze_quantization_error(model_baseline, model_all)
    output_stats_all = analyze_output_distribution(model_all, tokenizer, ANALYSIS_SAMPLES)
    
    results['experiments'].append({
        'name': 'Quantize All', 'accuracy': acc_all, 'accuracy_drop_%': (acc_baseline - acc_all) * 100,
        'model_size_mb': target_mb, 'compression_ratio': ratio, 
    })
    results['analysis']['quantize_all'] = {
        'quantization_errors': {k: {key: val for key, val in v.items()} for k, v in quant_errors_all.items()},
        'output_stats': {k: {key: val for key, val in v.items() if key != 'values'} for k, v in output_stats_all.items()},
    }
    cleanup_memory(model_all)
    
    # Skip LM Head/Embedding
    print("\n" + "="*100)
    print("EXPERIMENT 2: SKIP LM HEAD & EMBEDDING")
    print("="*100)
    model_skip, _ = load_fresh_model()
    model_skip = apply_weight_quantization(model_skip, WEIGHT_BITS, GROUP_SIZE, skip_lmhead_embedding=True)
    acc_skip = evaluate_accuracy(model_skip, tokenizer, "Skip LM Head/Embedding")
    full_mb, target_mb, ratio = get_model_stats(model_skip)
    
    print("\n   [Analysis] Analyzing quantization errors...")
    quant_errors_skip = analyze_quantization_error(model_baseline, model_skip)
    output_stats_skip = analyze_output_distribution(model_skip, tokenizer, ANALYSIS_SAMPLES)
    
    results['experiments'].append({
        'name': 'Skip LM Head/Embedding', 'accuracy': acc_skip, 'accuracy_drop_%': (acc_baseline - acc_skip) * 100,
        'model_size_mb': target_mb, 'compression_ratio': ratio, 
    })
    results['analysis']['skip_lmhead_embedding'] = {
        'quantization_errors': {k: {key: val for key, val in v.items()} for k, v in quant_errors_skip.items()},
        'output_stats': {k: {key: val for key, val in v.items() if key != 'values'} for k, v in output_stats_skip.items()},
    }
    cleanup_memory(model_skip)
    
    # Generate Plots
    print("\n" + "="*100)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*100)
    create_summary_plot(results)
    plot_accuracy_comparison(results)
    plot_accuracy_drop(results)
    plot_weight_distributions(baseline_weight_stats)
    plot_quantization_errors(quant_errors_all, quant_errors_skip)
    plot_output_distributions(baseline_output_stats, output_stats_all, output_stats_skip)
    plot_embedding_usage(embedding_stats)
    plot_layer_statistics_comparison(baseline_weight_stats)
    print(f"\n   ✅ All plots saved to: {PLOT_DIR}/")
    cleanup_memory(model_baseline)
    
    # Summary
    print("\n" + "="*100)
    print("📊 ANALYSIS SUMMARY")
    print("="*100)
    acc_diff = acc_all - acc_skip
    print(f"\n{'Metric':<50} {'Baseline':<15} {'Quantize All':<15} {'Skip LM/Emb':<15}")
    print("-"*95)
    print(f"{'Accuracy':<50} {acc_baseline:<15.4f} {acc_all:<15.4f} {acc_skip:<15.4f}")
    print(f"{'Accuracy Drop (%)':<50} {'-':<15} {(acc_baseline-acc_all)*100:<15.2f} {(acc_baseline-acc_skip)*100:<15.2f}")
    
    print("\n" + "="*100)
    print("🎯 CONCLUSION")
    print("="*100)
    if acc_diff > 0.001:
        print(f"\n✅ Quantizing LM Head/Embedding IMPROVES accuracy by {acc_diff*100:.2f}%")
        print("\n📝 Recommendation: Quantize ALL layers (including lm_head/embedding)")
    elif acc_diff < -0.001:
        print(f"\n❌ Skipping LM Head/Embedding is BETTER by {-acc_diff*100:.2f}%")
        print("\n📝 Recommendation: Skip lm_head/embedding quantization")
    else:
        print(f"\n⚖️  No significant difference ({acc_diff*100:+.2f}%)")
        print("\n📝 Recommendation: Either approach is acceptable")
    print(f"\n📊 Check visualizations in: {PLOT_DIR}/")
    
    # Save results - ✅ FIXED: Convert tensors to JSON-serializable format
    output_file = os.path.join(OUTPUT_DIR, "analysis_results.json")
    with open(output_file, 'w') as f:
        json_safe_results = convert_to_json_serializable(results)
        json.dump(json_safe_results, f, indent=2)
    print(f"✅ Results saved to: {output_file}")
    return results

# ==========================================
# 🚀 MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*100)
    print("🚀 STARTING LM HEAD & EMBEDDING QUANTIZATION ANALYSIS")
    print("="*100)
    start_time = time.time()
    try:
        results = run_lmhead_embedding_analysis()
        print(f"\n✅ Experiment completed! Total time: {(time.time() - start_time)/60:.1f} minutes")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "="*100)
    print("🎉 DONE!")
    print("="*100)
