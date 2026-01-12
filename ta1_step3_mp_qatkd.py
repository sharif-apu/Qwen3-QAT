import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset
import gc, time, os, copy, json
from tqdm import tqdm
from datetime import datetime
from quantization_utils import ( set_seed, load_fresh_model, evaluate_accuracy, quantize_tensor_fake,
MODEL_ID, DEVICE
)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

SEED = 42

# ==========================================
# 🔥 MAIN CONTROL SWITCHES
# ==========================================
RUN_STAGE1 = False
ENABLE_LAYER_WISE_SENSITIVITY = False
ENABLE_EMBEDDING_LMHEAD_QUANTIZATION = True  # Set to False to keep FP16

# 🔥 BIT-WIDTH CONTROLS
QAT_EMBEDDING_BITS = 8  # Embedding bit-width (2, 4, 8, or 16 for FP16)
QAT_LMHEAD_BITS = 8     # LM Head bit-width (2, 4, 8, or 16 for FP16)
QAT_LINEAR_BITS = 4     # Linear layers bit-width (2, 4, 8)

# Evaluation Settings
EVAL_LIMIT = 0.1
EVAL_BATCH_SIZE = 1

# Stage 1: Config Search Settings
STAGE1_SEARCH_CONFIGS = [
    {'name': 'all_4bit', 'emb': 4, 'lm': 4, 'linear': 4},
    {'name': 'mixed_4_2', 'emb': 4, 'lm': 4, 'linear': 2},
    {'name': 'all_8bit', 'emb': 8, 'lm': 8, 'linear': 8},
    {'name': 'mixed_8_2', 'emb': 8, 'lm': 8, 'linear': 2},
]

# Stage 2: PTQ Settings
STAGE2_EVAL_LIMIT = 0.005
STAGE2_LAYER_EVAL_LIMIT = 0.002
STAGE2_BIT_OPTIONS = [2, 4, 8]  # Bit widths to test
STAGE2_SENSITIVITY_THRESHOLDS = {
    'high': 5.0,    # >5% drop → 8-bit
    'medium': 2.0   # 2-5% drop → 4-bit, <2% → 2-bit
}

# 🔥 QAT Settings
QAT_GROUP_SIZE = 128
QAT_SYMMETRIC = False  # ✅ Asymmetric (reference default)

# 🔥 Training - MEMORY OPTIMIZED
QAT_NUM_EPOCHS = 5
QAT_BATCH_SIZE = 1
QAT_GRADIENT_ACCUMULATION_STEPS = 2  # ✅ Accumulate gradients
QAT_LEARNING_RATE = 5e-5
QAT_NUM_TRAIN_SAMPLES = 4096
QAT_MAX_SEQ_LENGTH = 512
QAT_USE_GRADIENT_CHECKPOINTING = True  # ✅ Enable gradient checkpointing

# Checkpoint & Logging
CHECKPOINT_DIR = "qat_checkpoints_mp"
SAVE_BEST_MODEL = True
SAVE_EVERY_N_EPOCHS = 2
TENSORBOARD_DIR = "runs/qat_training"
LOG_WEIGHT_DISTRIBUTIONS = True
LOG_GRADIENT_FLOW = True
LOG_EVERY_N_STEPS = 1000

# ==========================================
# 🔥 KNOWLEDGE DISTILLATION CONFIGURATION
# ==========================================
ENABLE_KD = False  # 🔥 Master switch for Knowledge Distillation
KD_TEMPERATURE = 4.0  # Temperature for softening probabilities
KD_ALPHA = 0.5  # Balance between KD loss and task loss (0.0-1.0)
KD_LOSS_TYPE = "kl_div"  # Options: "kl_div", "mse", "cosine"

# Advanced KD options
KD_LAYER_WISE = False  # Enable layer-wise feature distillation
KD_LAYER_ALPHA = 0.3  # Weight for intermediate layer distillation
KD_ATTENTION_DISTILL = False  # Distill attention maps
KD_ATTENTION_ALPHA = 0.2  # Weight for attention distillation

# Directories
for d in ["stage1_config_search", "stage2_sensitivity", "stage3_qat_training", "final_quantized_model", CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)


set_seed(SEED)

# ==========================================
# 🧹 ENHANCED MEMORY MANAGEMENT
# ==========================================
def cleanup_memory(model=None, lm_obj=None, optimizer=None, scheduler=None, dataset=None, dataloader=None):
    """✅ Comprehensive memory cleanup"""
    if model is not None:
        del model
    if lm_obj is not None:
        del lm_obj
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    if dataset is not None:
        del dataset
    if dataloader is not None:
        del dataloader
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_stats(label=""):
    """✅ Print current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   [Memory {label}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")



def get_model_stats(model):
    """✅ Calculate model statistics"""
    total_bits_full = 0
    target_bits = 0
    target_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            bits = getattr(module, 'quant_bit_width', 16)
            count = module.weight.numel()
            total_bits_full += count * bits
        elif isinstance(module, nn.Linear):
            if "lm_head" in name:
                bits = getattr(module, 'quant_bit_width', 16)
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
        avg_target_bits = 16.0
        
    return {
        'total_mb': full_mb,
        'quantized_mb': target_mb,
        'ratio': target_ratio,
        'avg_bits': avg_target_bits,
        'total_params': target_params
    }



# ==========================================
# 🔥 FAKE QUANTIZATION FOR QAT
# ==========================================
class FakeQuantize(nn.Module):
    """✅ Fake quantization with STE"""
    def __init__(self, n_bits=4, group_size=128, symmetric=False):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
    
    def forward(self, x):
        if not self.training:
            return quantize_tensor_fake(x, self.n_bits, "per_group", self.group_size, self.symmetric)
        else:
            with torch.no_grad():
                x_quant = quantize_tensor_fake(x, self.n_bits, "per_group", self.group_size, self.symmetric)
            return x + (x_quant - x).detach()

# ==========================================
# 🔥 QAT LINEAR & EMBEDDING
# ==========================================
class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, n_bits=4, group_size=128, symmetric=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_quantizer = FakeQuantize(n_bits=n_bits, group_size=group_size, symmetric=symmetric)
        self.n_bits = n_bits
        self.symmetric = symmetric
        
    def forward(self, x):
        w_quant = self.weight_quantizer(self.linear.weight)
        return F.linear(x, w_quant, self.linear.bias)
    
    @staticmethod
    def from_linear(linear_layer, n_bits=4, group_size=128, symmetric=False):
        qat_linear = QATLinear(
            linear_layer.in_features, linear_layer.out_features,
            bias=linear_layer.bias is not None,
            n_bits=n_bits, group_size=group_size, symmetric=symmetric
        )
        qat_linear.linear.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            qat_linear.linear.bias.data = linear_layer.bias.data.clone()
        return qat_linear

class QATEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_bits=4, group_size=128, symmetric=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.weight_quantizer = FakeQuantize(n_bits=n_bits, group_size=group_size, symmetric=symmetric)
        self.n_bits = n_bits
        self.symmetric = symmetric
        
    def forward(self, input):
        w_quant = self.weight_quantizer(self.embedding.weight)
        return F.embedding(input, w_quant, self.embedding.padding_idx, 
                          self.embedding.max_norm, self.embedding.norm_type,
                          self.embedding.scale_grad_by_freq, self.embedding.sparse)
    
    @staticmethod
    def from_embedding(embedding_layer, n_bits=4, group_size=128, symmetric=False):
        qat_embedding = QATEmbedding(
            embedding_layer.num_embeddings, embedding_layer.embedding_dim,
            n_bits=n_bits, group_size=group_size, symmetric=symmetric
        )
        qat_embedding.embedding.weight.data = embedding_layer.weight.data.clone()
        if embedding_layer.padding_idx is not None:
            qat_embedding.embedding.padding_idx = embedding_layer.padding_idx
        return qat_embedding

# ==========================================
# 🔥 CONVERT TO QAT
# ==========================================
def convert_to_qat(model, linear_bits=4, emb_bits=4, lm_bits=4, group_size=128, symmetric=False):
    """✅ Convert model to QAT"""
    quant_type = "Symmetric" if symmetric else "Asymmetric"
    print(f"   [QAT] Converting model to QAT mode (group-{group_size}, {quant_type})...")
    print(f"   [QAT] Bit-widths: Embeddings={emb_bits}bit, LM Head={lm_bits}bit, Linear={linear_bits}bit")
    
    skip_lm_head = (lm_bits == 16) or not ENABLE_EMBEDDING_LMHEAD_QUANTIZATION
    skip_embeddings = (emb_bits == 16) or not ENABLE_EMBEDDING_LMHEAD_QUANTIZATION
    
    converted_count = 0
    skipped_count = 0
    module_list = list(model.named_modules())
    
    for name, module in module_list:
        if isinstance(module, nn.Linear):
            if "lm_head" in name:
                if skip_lm_head:
                    print(f"   [QAT] Skipping LM Head: {name} (keeping FP16)")
                    skipped_count += 1
                    continue
                bits = lm_bits
            else:
                bits = linear_bits
            
            if '.' in name:
                *parent_names, attr_name = name.split('.')
                parent = model
                for pname in parent_names:
                    parent = getattr(parent, pname)
            else:
                parent = model
                attr_name = name
            
            qat_layer = QATLinear.from_linear(
                module, n_bits=bits, group_size=group_size, symmetric=symmetric
            )
            setattr(parent, attr_name, qat_layer)
            converted_count += 1
            del module
        
        elif isinstance(module, nn.Embedding):
            if skip_embeddings:
                print(f"   [QAT] Skipping Embedding: {name} (keeping FP16)")
                skipped_count += 1
                continue
            
            if '.' in name:
                *parent_names, attr_name = name.split('.')
                parent = model
                for pname in parent_names:
                    parent = getattr(parent, pname)
            else:
                parent = model
                attr_name = name
            
            qat_layer = QATEmbedding.from_embedding(
                module, n_bits=emb_bits, group_size=group_size, symmetric=symmetric
            )
            setattr(parent, attr_name, qat_layer)
            converted_count += 1
            del module
    
    print(f"   [QAT] Converted {converted_count} layers to QAT")
    if skipped_count > 0:
        print(f"   [QAT] Skipped {skipped_count} layers (kept FP16)")
    
    if QAT_USE_GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
        print(f"   [QAT] Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    del module_list
    cleanup_memory()
    print_memory_stats("after QAT conversion")
    
    return model

# ==========================================
# 🔥 FINALIZE QAT
# ==========================================
def finalize_qat(model):
    """✅ Convert QAT layers to quantized weights"""
    print("\n   [QAT] Finalizing QAT - converting to quantized weights...")
    
    finalized_count = 0
    module_list = list(model.named_modules())
    
    for name, module in module_list:
        if isinstance(module, QATLinear):
            if '.' in name:
                *parent_names, attr_name = name.split('.')
                parent = model
                for pname in parent_names:
                    parent = getattr(parent, pname)
            else:
                parent = model
                attr_name = name
            
            new_linear = nn.Linear(
                module.linear.in_features, module.linear.out_features,
                bias=module.linear.bias is not None
            )
            
            with torch.no_grad():
                new_linear.weight.data = quantize_tensor_fake(
                    module.linear.weight.data,
                    n_bit=module.n_bits,
                    granularity="per_group",
                    group_size=module.weight_quantizer.group_size,
                    sym=module.symmetric
                )
                if module.linear.bias is not None:
                    new_linear.bias.data = module.linear.bias.data.clone()
            
            new_linear.quant_bit_width = module.n_bits
            setattr(parent, attr_name, new_linear)
            finalized_count += 1
            del module
        
        elif isinstance(module, QATEmbedding):
            if '.' in name:
                *parent_names, attr_name = name.split('.')
                parent = model
                for pname in parent_names:
                    parent = getattr(parent, pname)
            else:
                parent = model
                attr_name = name
            
            new_embedding = nn.Embedding(module.embedding.num_embeddings, module.embedding.embedding_dim)
            
            with torch.no_grad():
                new_embedding.weight.data = quantize_tensor_fake(
                    module.embedding.weight.data,
                    n_bit=module.n_bits,
                    granularity="per_group",
                    group_size=module.weight_quantizer.group_size,
                    sym=module.symmetric
                )
            
            if module.embedding.padding_idx is not None:
                new_embedding.padding_idx = module.embedding.padding_idx
            
            new_embedding.quant_bit_width = module.n_bits
            setattr(parent, attr_name, new_embedding)
            finalized_count += 1
            del module
    
    print(f"   [QAT] Finalized {finalized_count} layers")
    del module_list
    cleanup_memory()
    
    return model

# ==========================================
# 🔥 PTQ QUANTIZATION
# ==========================================
def apply_ptq_quantization(model, linear_bits=4, emb_bits=4, lm_bits=4, group_size=128, symmetric=False):
    """✅ Apply PTQ quantization"""
    quant_type = "Symmetric" if symmetric else "Asymmetric"
    print(f"   [PTQ] Applying quantization (Group-{group_size}, {quant_type})...")
    print(f"   [PTQ] Bit-widths: Embeddings={emb_bits}bit, LM Head={lm_bits}bit, Linear={linear_bits}bit")
    
    skip_lm_head = (lm_bits == 16) or not ENABLE_EMBEDDING_LMHEAD_QUANTIZATION
    skip_embeddings = (emb_bits == 16) or not ENABLE_EMBEDDING_LMHEAD_QUANTIZATION
    
    quantized_count = 0
    skipped_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lm_head" in name:
                if skip_lm_head:
                    module.quant_bit_width = 16
                    skipped_count += 1
                    continue
                bits = lm_bits
            else:
                bits = linear_bits
            
            with torch.no_grad():
                module.weight.data = quantize_tensor_fake(
                    module.weight.data,
                    n_bit=bits,
                    granularity="per_group",
                    group_size=group_size,
                    sym=symmetric
                )
                module.quant_bit_width = bits
                quantized_count += 1
        
        elif isinstance(module, nn.Embedding):
            if skip_embeddings:
                module.quant_bit_width = 16
                skipped_count += 1
                continue
            
            with torch.no_grad():
                module.weight.data = quantize_tensor_fake(
                    module.weight.data,
                    n_bit=emb_bits,
                    granularity="per_group",
                    group_size=group_size,
                    sym=symmetric
                )
                module.quant_bit_width = emb_bits
                quantized_count += 1
    
    print(f"   [PTQ] Quantized {quantized_count} layers")
    if skipped_count > 0:
        print(f"   [PTQ] Skipped {skipped_count} layers (kept FP16)")
    
    cleanup_memory()
    print_memory_stats("after PTQ")
    
    return model



# ==========================================
# 🔥 STAGE 1: CONFIG SEARCH
# ==========================================
def stage1_config_search(model, tokenizer, baseline_acc):
    print("\n" + "="*100)
    print("STAGE 1: CONFIGURATION SEARCH (PTQ)")
    print("="*100)
    print(f"Testing {len(STAGE1_SEARCH_CONFIGS)} different quantization configurations")
    print("="*100)
    
    results = []
    best_config = None
    best_score = -1
    
    for idx, config_spec in enumerate(STAGE1_SEARCH_CONFIGS):
        print(f"\n[Config {idx+1}/{len(STAGE1_SEARCH_CONFIGS)}] Testing: {config_spec['name']}")
        print(f"   Embeddings: {config_spec['emb']}bit, LM Head: {config_spec['lm']}bit, Linear: {config_spec['linear']}bit")
        
        model_test = copy.deepcopy(model)
        
        for name, module in model_test.named_modules():
            if isinstance(module, nn.Embedding):
                with torch.no_grad():
                    module.weight.data = quantize_tensor_fake(
                        module.weight.data, config_spec['emb'], "per_group", QAT_GROUP_SIZE, False
                    )
                    module.quant_bit_width = config_spec['emb']
        
        for name, module in model_test.named_modules():
            if isinstance(module, nn.Linear):
                bits = config_spec['lm'] if "lm_head" in name else config_spec['linear']
                with torch.no_grad():
                    module.weight.data = quantize_tensor_fake(
                        module.weight.data, bits, "per_group", QAT_GROUP_SIZE, False
                    )
                    module.quant_bit_width = bits
        
        acc_test = evaluate_accuracy(model_test, tokenizer, config_spec['name'])
        stats_test = get_model_stats(model_test)
        
        acc_drop = baseline_acc - acc_test
        score = acc_test * stats_test['ratio']
        
        result = {
            'name': config_spec['name'],
            'config': config_spec,
            'accuracy': acc_test,
            'accuracy_drop': acc_drop,
            'accuracy_drop_%': acc_drop * 100,
            'compression_ratio': stats_test['ratio'],
            'model_size_mb': stats_test['quantized_mb'],
            'avg_bits': stats_test['avg_bits'],
            'score': score
        }
        results.append(result)
        
        print(f"   Results: Acc={acc_test:.4f} (drop: {acc_drop*100:.2f}%), Ratio={stats_test['ratio']:.2f}x, Size={stats_test['quantized_mb']:.1f}MB, Score={score:.2f}")
        
        if score > best_score:
            best_score = score
            best_config = config_spec
            print(f"   ✨ New best configuration!")
        
        cleanup_memory(model_test)
        print_memory_stats(f"after config {idx+1}")
    
    print(f"\n{'='*100}\nSTAGE 1 SUMMARY\n{'='*100}")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"\nAll Configurations (sorted by score):")
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"  {r['name']:15s}: Acc={r['accuracy']:.4f}, Drop={r['accuracy_drop_%']:5.2f}%, "
              f"Ratio={r['compression_ratio']:.2f}x, Size={r['model_size_mb']:5.1f}MB, Score={r['score']:.2f}")
    
    print(f"\n🏆 Best Configuration: {best_config['name']}")
    print(f"   Embeddings: {best_config['emb']}bit, LM Head: {best_config['lm']}bit, Linear: {best_config['linear']}bit")
    print(f"{'='*100}")
    
    stage1_results = {
        'baseline_accuracy': baseline_acc,
        'configurations': results,
        'best_config': best_config,
        'best_score': best_score
    }
    
    with open("stage1_config_search/stage1_results.json", 'w') as f:
        json.dump(stage1_results, f, indent=2)
    
    print(f"\n✅ Stage 1 results saved to stage1_config_search/")
    return best_config, stage1_results

# ==========================================
# STAGE 2: SENSITIVITY ANALYSIS
# ==========================================

def stage2_layer_sensitivity_analysis(model, tokenizer, baseline_acc, base_config):
    """✅ Layer-wise sensitivity analysis starting from base config (Stage 1 or predefined)"""
    print("\n" + "="*100)
    print("STAGE 2: LAYER-WISE SENSITIVITY ANALYSIS")
    print("="*100)
    print(f"   Starting from base config: Emb={base_config['emb']}bit, LM={base_config['lm']}bit, Linear={base_config['linear']}bit")
    
    # Get all quantizable layers (excluding embeddings and lm_head)
    layers_to_test = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            layers_to_test.append({
                'name': name, 
                'module': module, 
                'type': 'linear', 
                'params': module.weight.numel()
            })
    
    print(f"   Found {len(layers_to_test)} linear layers to analyze (excluding embeddings and lm_head)")
    print(f"   Testing bit widths: {STAGE2_BIT_OPTIONS}\n")
    
    layer_results = {}
    recommended_bits = {}
    
    # Initialize with base config
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            recommended_bits[name] = base_config['emb']
        elif isinstance(module, nn.Linear):
            if "lm_head" in name:
                recommended_bits[name] = base_config['lm']
            else:
                recommended_bits[name] = base_config['linear']
    
    progress_bar = tqdm(layers_to_test, desc="Analyzing layers")
    
    for layer_info in progress_bar:
        layer_name = layer_info['name']
        progress_bar.set_description(f"Testing {layer_name[:40]}")
        
        bit_accuracies = {}
        
        for n_bits in STAGE2_BIT_OPTIONS:
            # Create test config
            test_config = recommended_bits.copy()
            test_config[layer_name] = n_bits
            
            # Apply PTQ with test config
            model_test = copy.deepcopy(model)
            model_test = apply_ptq_quantization(
                model_test, 
                group_size=QAT_GROUP_SIZE, 
                symmetric=QAT_SYMMETRIC, 
                layer_bit_config=test_config
            )
            
            acc = evaluate_accuracy(model_test, tokenizer, desc=f"{layer_name}@{n_bits}bit", limit=STAGE2_EVAL_LIMIT)
            bit_accuracies[n_bits] = acc
            
            del model_test
            cleanup_memory()
        
        # Calculate sensitivity (difference between 8-bit and 2-bit)
        sensitivity = (bit_accuracies[max(STAGE2_BIT_OPTIONS)] - bit_accuracies[min(STAGE2_BIT_OPTIONS)]) * 100
        
        # Assign bits based on sensitivity
        if sensitivity > STAGE2_SENSITIVITY_THRESHOLDS['high']:
            assigned_bits = 8
        elif sensitivity > STAGE2_SENSITIVITY_THRESHOLDS['medium']:
            assigned_bits = 4
        else:
            assigned_bits = 2
        
        layer_results[layer_name] = {
            'type': layer_info['type'],
            'params': layer_info['params'],
            'accuracies': bit_accuracies,
            'sensitivity': sensitivity,
            'assigned_bits': assigned_bits
        }
        
        recommended_bits[layer_name] = assigned_bits
        progress_bar.set_postfix({'sensitivity': f'{sensitivity:.2f}%', 'assigned': f'{assigned_bits}bit'})
    
    progress_bar.close()
    
    # Summary
    print("\n" + "="*100)
    print("LAYER SENSITIVITY SUMMARY (Top 10 Most Sensitive)")
    print("="*100)
    
    sorted_layers = sorted(layer_results.items(), key=lambda x: x[1]['sensitivity'], reverse=True)[:10]
    
    for layer_name, info in sorted_layers:
        status = "🔴" if info['sensitivity'] > 5.0 else "🟡" if info['sensitivity'] > 2.0 else "🟢"
        print(f"{status} {layer_name[:60]:<60} | Sens: {info['sensitivity']:5.2f}% | Assigned: {info['assigned_bits']}-bit")
    
    # Statistics
    linear_params = sum(info['params'] for info in layer_results.values())
    linear_bits = sum(recommended_bits[name] * info['params'] for name, info in layer_results.items())
    
    # Count all params including embeddings and lm_head
    total_params = 0
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            params = module.weight.numel()
            bits = recommended_bits.get(name, 16)
            total_params += params
            total_bits += params * bits
    
    avg_bits = total_bits / total_params if total_params > 0 else 16.0
    
    bit_dist = {2: 0, 4: 0, 8: 0}
    for name, bits in recommended_bits.items():
        if bits in bit_dist:
            bit_dist[bits] += 1
    
    print(f"\n   Average bits (all layers): {avg_bits:.2f}")
    print(f"   Linear layer distribution: 2-bit={bit_dist[2]}, 4-bit={bit_dist[4]}, 8-bit={bit_dist[8]}")
    
    # Test recommended config
    print("\n   Testing recommended configuration...")
    model_recommended = copy.deepcopy(model)
    model_recommended = apply_ptq_quantization(
        model_recommended, 
        group_size=QAT_GROUP_SIZE, 
        symmetric=QAT_SYMMETRIC, 
        layer_bit_config=recommended_bits
    )
    
    acc_recommended = evaluate_accuracy(model_recommended, tokenizer, desc="Recommended Config")
    stats_recommended = get_model_stats(model_recommended)
    
    print(f"\n   Baseline:    {baseline_acc:.4f}")
    print(f"   Recommended: {acc_recommended:.4f} (degradation: {(baseline_acc - acc_recommended)*100:.2f}%)")
    print(f"   Size:        {stats_recommended['quantized_mb']:.2f} MB")
    print(f"   Avg bits:    {stats_recommended['avg_bits']:.2f}")
    print(f"   Ratio:       {stats_recommended['ratio']:.2f}x")
    print("="*100)
    
    # Save results
    results = {
        'baseline_accuracy': baseline_acc,
        'base_config': base_config,
        'recommended_accuracy': acc_recommended,
        'recommended_bits': recommended_bits,
        'layer_results': {k: {**v, 'accuracies': {int(k2): float(v2) for k2, v2 in v['accuracies'].items()}} for k, v in layer_results.items()},
        'avg_bits': float(avg_bits),
        'bit_distribution': bit_dist,
        'compression_ratio': float(stats_recommended['ratio'])
    }
    
    with open('stage2_sensitivity/layer_sensitivity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('stage2_sensitivity/recommended_bit_config.json', 'w') as f:
        json.dump(recommended_bits, f, indent=2)
    
    cleanup_memory(model_recommended)
    
    return recommended_bits, layer_results, acc_recommended, stats_recommended


def stage2_sensitivity_analysis(model, tokenizer, baseline_acc, stage1_config=None):
    print("\n" + "="*100)
    print(f"STAGE 2: SENSITIVITY ANALYSIS ({'LAYER-WISE' if ENABLE_LAYER_WISE_SENSITIVITY else 'FAST MODE'})")
    print("="*100)
    
    if stage1_config and RUN_STAGE1:
        print(f"   Using Stage 1 best config: Emb={stage1_config['emb']}bit, LM={stage1_config['lm']}bit, Linear={stage1_config['linear']}bit")
        linear_bits = stage1_config['linear']
        emb_bits = stage1_config['emb']
        lm_bits = stage1_config['lm']
    else:
        linear_bits = QAT_LINEAR_BITS
        emb_bits = QAT_EMBEDDING_BITS
        lm_bits = QAT_LMHEAD_BITS
    
    model_ptq = copy.deepcopy(model)
    model_ptq = apply_ptq_quantization(
        model_ptq, 
        linear_bits=linear_bits,
        emb_bits=emb_bits,
        lm_bits=lm_bits,
        group_size=QAT_GROUP_SIZE, 
        symmetric=QAT_SYMMETRIC
    )
    
    acc_ptq = evaluate_accuracy(model_ptq, tokenizer, "PTQ Baseline")
    stats_ptq = get_model_stats(model_ptq)
    
    print(f"\n{'='*100}\nSTAGE 2 SUMMARY\n{'='*100}")
    print(f"Baseline:  Acc={baseline_acc:.4f}, Size={stats_ptq['total_mb']:.2f}MB")
    print(f"PTQ:       Acc={acc_ptq:.4f}, Size={stats_ptq['quantized_mb']:.2f}MB, Ratio={stats_ptq['ratio']:.2f}x")
    print(f"Degradation: {(baseline_acc - acc_ptq) * 100:.2f}%, Avg bits: {stats_ptq['avg_bits']:.2f}")
    print(f"{'='*100}")
    
    cleanup_memory(model_ptq)
    print_memory_stats("after Stage 2")
    
    results = {
        'baseline_accuracy': baseline_acc,
        'ptq_accuracy': acc_ptq,
        'compression_ratio': stats_ptq['ratio'],
        'avg_bits': stats_ptq['avg_bits'],
        'used_stage1_config': RUN_STAGE1 and stage1_config is not None
    }
    
    with open("stage2_sensitivity/stage2_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return acc_ptq, stats_ptq, results

# ==========================================
# TENSORBOARD LOGGING
# ==========================================
def log_weight_distributions(writer, model, step):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QATLinear)):
            weights = module.linear.weight.data if isinstance(module, QATLinear) else module.weight.data
            prefix = f"weights_qat/{name}" if isinstance(module, QATLinear) else f"weights/{name}"
            writer.add_histogram(f"{prefix}/distribution", weights, step)
            writer.add_scalar(f"{prefix}/mean", weights.mean().item(), step)
            writer.add_scalar(f"{prefix}/std", weights.std().item(), step)

def log_gradient_flow(writer, model, step):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            writer.add_scalar(f"gradients/{name}/norm", param_norm, step)
    writer.add_scalar("gradients/total_norm", total_norm ** 0.5, step)

def log_quantization_error(writer, model, step):
    for name, module in model.named_modules():
        if isinstance(module, (QATLinear, QATEmbedding)):
            with torch.no_grad():
                if isinstance(module, QATLinear):
                    original = module.linear.weight.data
                else:
                    original = module.embedding.weight.data
                quantized = quantize_tensor_fake(
                    original, module.n_bits, "per_group",
                    module.weight_quantizer.group_size, module.symmetric
                )
                error = (original - quantized).abs()
                writer.add_scalar(f"quantization_error/{name}/mean", error.mean().item(), step)
                writer.add_scalar(f"quantization_error/{name}/max", error.max().item(), step)

# ==========================================
# 🔥 KNOWLEDGE DISTILLATION LOSSES
# ==========================================
class KnowledgeDistillationLoss(nn.Module):
    """✅ KD loss with multiple strategies"""
    def __init__(self, temperature=4.0, alpha=0.5, loss_type="kl_div"):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type
        
    def forward(self, student_logits, teacher_logits, labels=None, task_loss=None):
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        
        if self.loss_type == "kl_div":
            student_log_probs = F.log_softmax(student_logits_flat / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits_flat / self.temperature, dim=-1)
            
            kd_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
        elif self.loss_type == "mse":
            kd_loss = F.mse_loss(student_logits_flat, teacher_logits_flat)
            
        elif self.loss_type == "cosine":
            student_norm = F.normalize(student_logits_flat, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_logits_flat, p=2, dim=-1)
            kd_loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=-1).mean()
        
        else:
            raise ValueError(f"Unknown KD loss type: {self.loss_type}")
        
        if task_loss is not None:
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss
            return total_loss, kd_loss, task_loss
        else:
            return kd_loss

class LayerWiseDistillationLoss(nn.Module):
    """✅ Layer-wise feature distillation"""
    def __init__(self, alpha=0.3, loss_type="mse"):
        super().__init__()
        self.alpha = alpha
        self.loss_type = loss_type
        
    def forward(self, student_hidden_states, teacher_hidden_states):
        if len(student_hidden_states) != len(teacher_hidden_states):
            indices = torch.linspace(0, len(teacher_hidden_states) - 1, len(student_hidden_states)).long()
            teacher_hidden_states = [teacher_hidden_states[i] for i in indices]
        
        layer_losses = []
        for s_hidden, t_hidden in zip(student_hidden_states, teacher_hidden_states):
            if self.loss_type == "mse":
                loss = F.mse_loss(s_hidden, t_hidden)
            elif self.loss_type == "cosine":
                s_norm = F.normalize(s_hidden, p=2, dim=-1)
                t_norm = F.normalize(t_hidden, p=2, dim=-1)
                loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=-1).mean()
            else:
                raise ValueError(f"Unknown layer loss type: {self.loss_type}")
            
            layer_losses.append(loss)
        
        return sum(layer_losses) / len(layer_losses)

class AttentionDistillationLoss(nn.Module):
    """✅ Attention map distillation"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, student_attentions, teacher_attentions):
        if len(student_attentions) != len(teacher_attentions):
            indices = torch.linspace(0, len(teacher_attentions) - 1, len(student_attentions)).long()
            teacher_attentions = [teacher_attentions[i] for i in indices]
        
        attn_losses = []
        for s_attn, t_attn in zip(student_attentions, teacher_attentions):
            s_attn_mean = s_attn.mean(dim=1)
            t_attn_mean = t_attn.mean(dim=1)
            loss = F.mse_loss(s_attn_mean, t_attn_mean)
            attn_losses.append(loss)
        
        return sum(attn_losses) / len(attn_losses)

# ==========================================
# 🔥 TEACHER MODEL WRAPPER
# ==========================================
class TeacherModelWrapper:
    """✅ Teacher model wrapper"""
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def get_outputs(self, input_ids, attention_mask, return_hidden_states=False, return_attentions=False):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
            output_attentions=return_attentions
        )
        
        result = {'logits': outputs.logits}
        
        if return_hidden_states and hasattr(outputs, 'hidden_states'):
            result['hidden_states'] = outputs.hidden_states
        
        if return_attentions and hasattr(outputs, 'attentions'):
            result['attentions'] = outputs.attentions
        
        return result
    
    def cleanup(self):
        del self.model
        cleanup_memory()

# ==========================================
# QAT DATASET
# ==========================================
class QATDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, num_samples=4096):
        self.tokenizer, self.max_length, self.samples = tokenizer, max_length, []
        print(f"   [QAT] Loading training data...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for sample in tqdm(dataset, desc="Preparing dataset", total=num_samples):
            if len(sample['text']) > 100:
                tokens = tokenizer(sample['text'], return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
                self.samples.append({'input_ids': tokens['input_ids'].squeeze(0), 'attention_mask': tokens['attention_mask'].squeeze(0)})
                if len(self.samples) >= num_samples: break
        
        del dataset
        cleanup_memory()
        print(f"   [QAT] Loaded {len(self.samples)} samples")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def save_checkpoint(model, tokenizer, epoch, acc, loss, metadata, is_best=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    name = f"best_model_epoch{epoch}_acc{acc:.4f}.pt" if is_best else f"checkpoint_epoch{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    path = os.path.join(CHECKPOINT_DIR, name)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'accuracy': acc, 'loss': loss, 'metadata': metadata}, path)
    print(f"   [Checkpoint] Saved: {path}")
    return path

# ==========================================
# 🔥 STAGE 3: QAT TRAINING WITH KD
# ==========================================
def stage3_qat_training_with_kd(model, tokenizer, baseline_acc, acc_before_qat, stats_before):
    """✅ QAT Training with Knowledge Distillation"""
    print("\n" + "="*100)
    print(f"STAGE 3: QAT TRAINING {'WITH KNOWLEDGE DISTILLATION' if ENABLE_KD else '(NO KD)'}")
    print("="*100)
    print(f"   Quantization: Emb={QAT_EMBEDDING_BITS}bit, LM={QAT_LMHEAD_BITS}bit, Linear={QAT_LINEAR_BITS}bit")
    print(f"   KD Enabled: {ENABLE_KD}")
    if ENABLE_KD:
        print(f"   KD Temperature: {KD_TEMPERATURE}, Alpha: {KD_ALPHA}, Loss: {KD_LOSS_TYPE}")
        print(f"   Layer-wise KD: {KD_LAYER_WISE}, Attention KD: {KD_ATTENTION_DISTILL}")
    print("="*100)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"lin{QAT_LINEAR_BITS}b_emb{QAT_EMBEDDING_BITS}b_lm{QAT_LMHEAD_BITS}b{'_kd' if ENABLE_KD else ''}_{timestamp}"
    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, run_name))
    
    # ✅ Load teacher model if KD enabled
    if ENABLE_KD:
        print("\n   [KD] Loading teacher model...")
        teacher_model, _ = load_fresh_model()
        teacher_wrapper = TeacherModelWrapper(teacher_model, DEVICE)
        print_memory_stats("after teacher load")
    else:
        teacher_wrapper = None
    
    # Convert student to QAT
    model = convert_to_qat(
        model,
        linear_bits=QAT_LINEAR_BITS,
        emb_bits=QAT_EMBEDDING_BITS,
        lm_bits=QAT_LMHEAD_BITS,
        group_size=QAT_GROUP_SIZE,
        symmetric=QAT_SYMMETRIC
    ).to(DEVICE)
    
    metadata = {
        'model_id': MODEL_ID,
        'embedding_bits': QAT_EMBEDDING_BITS,
        'lmhead_bits': QAT_LMHEAD_BITS,
        'linear_bits': QAT_LINEAR_BITS,
        'kd_enabled': ENABLE_KD,
        'kd_temperature': KD_TEMPERATURE if ENABLE_KD else None,
        'kd_alpha': KD_ALPHA if ENABLE_KD else None,
        'kd_loss_type': KD_LOSS_TYPE if ENABLE_KD else None,
        'kd_layer_wise': KD_LAYER_WISE if ENABLE_KD else None,
        'kd_attention_distill': KD_ATTENTION_DISTILL if ENABLE_KD else None,
    }
    writer.add_text("hyperparameters", json.dumps(metadata, indent=2), 0)
    
    # Initialize KD losses
    if ENABLE_KD:
        kd_loss_fn = KnowledgeDistillationLoss(
            temperature=KD_TEMPERATURE,
            alpha=KD_ALPHA,
            loss_type=KD_LOSS_TYPE
        )
        
        if KD_LAYER_WISE:
            layer_kd_loss_fn = LayerWiseDistillationLoss(alpha=KD_LAYER_ALPHA)
        
        if KD_ATTENTION_DISTILL:
            attn_kd_loss_fn = AttentionDistillationLoss(alpha=KD_ATTENTION_ALPHA)
    
    # Pre-training evaluation
    model_eval_pre = finalize_qat(copy.deepcopy(model))
    acc_pre_qat = evaluate_accuracy(model_eval_pre, tokenizer, "Pre-QAT")
    writer.add_scalar("accuracy/pre_training", acc_pre_qat, 0)
    cleanup_memory(model_eval_pre)
    
    # Setup training
    train_dataset = QATDataset(tokenizer, QAT_MAX_SEQ_LENGTH, QAT_NUM_TRAIN_SAMPLES)
    train_loader = DataLoader(train_dataset, batch_size=QAT_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    optimizer = AdamW(model.parameters(), lr=QAT_LEARNING_RATE, weight_decay=0.01)
    
    num_training_steps = len(train_loader) * QAT_NUM_EPOCHS // QAT_GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)
    
    epoch_results = [{'epoch': 0, 'loss': None, 'accuracy': acc_pre_qat, 'is_pre_training': True}]
    best_accuracy, best_epoch, global_step = acc_pre_qat, 0, 0
    
    # ✅ Training loop with KD
    for epoch in range(QAT_NUM_EPOCHS):
        print(f"\n{'='*100}\nEPOCH {epoch+1}/{QAT_NUM_EPOCHS}\n{'='*100}")
        print_memory_stats(f"start of epoch {epoch+1}")
        
        model.train()
        epoch_loss = 0
        epoch_kd_loss = 0
        epoch_task_loss = 0
        epoch_layer_loss = 0
        epoch_attn_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # ✅ Student forward pass
            student_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=KD_LAYER_WISE if ENABLE_KD else False,
                output_attentions=KD_ATTENTION_DISTILL if ENABLE_KD else False
            )
            
            task_loss = student_outputs.loss
            
            # ✅ Knowledge Distillation
            if ENABLE_KD and teacher_wrapper is not None:
                # Get teacher outputs
                teacher_outputs = teacher_wrapper.get_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden_states=KD_LAYER_WISE,
                    return_attentions=KD_ATTENTION_DISTILL
                )
                
                # Logit distillation
                total_loss, kd_loss, task_loss_component = kd_loss_fn(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs['logits'],
                    labels=input_ids,
                    task_loss=task_loss
                )
                
                epoch_kd_loss += kd_loss.item()
                epoch_task_loss += task_loss_component.item()
                
                # ✅ Layer-wise distillation (SAFE)
                if KD_LAYER_WISE:
                    student_hidden = getattr(student_outputs, 'hidden_states', None)
                    teacher_hidden = teacher_outputs.get('hidden_states', None)
                    
                    if student_hidden is not None and teacher_hidden is not None:
                        layer_loss = layer_kd_loss_fn(student_hidden, teacher_hidden)
                        total_loss = total_loss + KD_LAYER_ALPHA * layer_loss
                        epoch_layer_loss += layer_loss.item()
                    elif step == 0:
                        print(f"   [Warning] Hidden states not available, skipping layer-wise KD")
                
                # ✅ Attention distillation (SAFE)
                if KD_ATTENTION_DISTILL:
                    student_attn = getattr(student_outputs, 'attentions', None)
                    teacher_attn = teacher_outputs.get('attentions', None)
                    
                    if student_attn is not None and teacher_attn is not None:
                        attn_loss = attn_kd_loss_fn(student_attn, teacher_attn)
                        total_loss = total_loss + KD_ATTENTION_ALPHA * attn_loss
                        epoch_attn_loss += attn_loss.item()
                    elif step == 0:
                        print(f"   [Warning] Attention maps not available, skipping attention KD")
                
                loss = total_loss
            else:
                loss = task_loss
            
            # Backward pass with gradient accumulation
            loss = loss / QAT_GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            epoch_loss += loss.item() * QAT_GRADIENT_ACCUMULATION_STEPS
            
            # Update weights
            if (step + 1) % QAT_GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % LOG_EVERY_N_STEPS == 0:
                    writer.add_scalar("training/total_loss", loss.item() * QAT_GRADIENT_ACCUMULATION_STEPS, global_step)
                    if ENABLE_KD:
                        writer.add_scalar("training/kd_loss", kd_loss.item(), global_step)
                        writer.add_scalar("training/task_loss", task_loss_component.item(), global_step)
                        if KD_LAYER_WISE and epoch_layer_loss > 0:
                            writer.add_scalar("training/layer_kd_loss", layer_loss.item(), global_step)
                        if KD_ATTENTION_DISTILL and epoch_attn_loss > 0:
                            writer.add_scalar("training/attention_kd_loss", attn_loss.item(), global_step)
                    
                    writer.add_scalar("training/learning_rate", scheduler.get_last_lr()[0], global_step)
                    if LOG_GRADIENT_FLOW:
                        log_gradient_flow(writer, model, global_step)
                    log_quantization_error(writer, model, global_step)
            
            # Free memory
            del input_ids, attention_mask, student_outputs, loss
            if ENABLE_KD:
                del teacher_outputs
            
            if step % 50 == 0:
                cleanup_memory()
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        if ENABLE_KD:
            avg_kd_loss = epoch_kd_loss / len(train_loader)
            avg_task_loss = epoch_task_loss / len(train_loader)
            print(f"\n   Epoch {epoch+1} Losses:")
            print(f"      Total: {avg_epoch_loss:.4f} | KD: {avg_kd_loss:.4f} | Task: {avg_task_loss:.4f}")
            if KD_LAYER_WISE and epoch_layer_loss > 0:
                avg_layer_loss = epoch_layer_loss / len(train_loader)
                print(f"      Layer KD: {avg_layer_loss:.4f}")
            if KD_ATTENTION_DISTILL and epoch_attn_loss > 0:
                avg_attn_loss = epoch_attn_loss / len(train_loader)
                print(f"      Attention KD: {avg_attn_loss:.4f}")
        
        if LOG_WEIGHT_DISTRIBUTIONS:
            log_weight_distributions(writer, model, global_step)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            model_eval = finalize_qat(copy.deepcopy(model))
            epoch_acc = evaluate_accuracy(model_eval, tokenizer, f"Epoch {epoch+1}")
        
        writer.add_scalar("epoch/loss", avg_epoch_loss, epoch + 1)
        writer.add_scalar("epoch/accuracy", epoch_acc, epoch + 1)
        
        is_best = epoch_acc > best_accuracy
        if is_best:
            print(f"\n   🎉 New best: {epoch_acc:.4f} (prev: {best_accuracy:.4f})")
            best_accuracy, best_epoch = epoch_acc, epoch + 1
            writer.add_scalar("best/accuracy", best_accuracy, epoch + 1)
            if SAVE_BEST_MODEL:
                save_checkpoint(model_eval, tokenizer, epoch + 1, epoch_acc, avg_epoch_loss, metadata, True)
        
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model_eval, tokenizer, epoch + 1, epoch_acc, avg_epoch_loss, metadata, False)
        
        cleanup_memory(model_eval)
        print_memory_stats(f"end of epoch {epoch+1}")
        
        epoch_results.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'kd_loss': avg_kd_loss if ENABLE_KD else None,
            'task_loss': avg_task_loss if ENABLE_KD else None,
            'accuracy': epoch_acc,
            'is_best': is_best
        })
    
    # Cleanup teacher
    if ENABLE_KD and teacher_wrapper is not None:
        teacher_wrapper.cleanup()
        del teacher_wrapper
        cleanup_memory()
    
    cleanup_memory(optimizer=optimizer, scheduler=scheduler, dataset=train_dataset, dataloader=train_loader)
    
    # Final evaluation
    model = finalize_qat(model).eval()
    acc_final = evaluate_accuracy(model, tokenizer, f"Final QAT{'+KD' if ENABLE_KD else ''}")
    stats_final = get_model_stats(model)
    
    print(f"\n{'='*100}\nSTAGE 3 SUMMARY\n{'='*100}")
    print(f"Baseline: {baseline_acc:.4f} | PTQ: {acc_before_qat:.4f} | QAT{'+KD' if ENABLE_KD else ''}: {acc_final:.4f} | Best: {best_accuracy:.4f}")
    print(f"Recovery: {(acc_final - acc_before_qat)*100:+.2f}% | Ratio: {stats_final['ratio']:.2f}x | Size: {stats_final['quantized_mb']:.2f}MB")
    print(f"{'='*100}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'baseline_accuracy': baseline_acc,
        'final_accuracy': acc_final,
        'best_accuracy': best_accuracy,
        'metadata': metadata
    }, f"final_quantized_model/qat{'_kd' if ENABLE_KD else ''}_model_final.pt")
    
    results = {
        'baseline_accuracy': baseline_acc,
        'before_qat_accuracy': acc_before_qat,
        'after_qat_accuracy': acc_final,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'qat_recovery_%': (acc_final - acc_before_qat) * 100,
        'compression_ratio': stats_final['ratio'],
        'model_size_mb': stats_final['quantized_mb'],
        'kd_enabled': ENABLE_KD,
        'epoch_results': epoch_results
    }
    
    with open(f"stage3_qat_training/qat{'_kd' if ENABLE_KD else ''}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    print(f"   ✅ TensorBoard: {os.path.join(TENSORBOARD_DIR, run_name)}")
    print_memory_stats("after Stage 3")
    
    return model, acc_final, results

# ==========================================
# 🔥 MAIN PIPELINE
# ==========================================
def run_complete_pipeline():
    start_time = datetime.now()
    print(f"\n{'='*100}\n🚀 QUANTIZATION PIPELINE\n{'='*100}")
    print(f"Model: {MODEL_ID} | Device: {DEVICE}")
    print(f"KD Enabled: {ENABLE_KD}")
    if ENABLE_KD:
        print(f"KD Config: T={KD_TEMPERATURE}, α={KD_ALPHA}, Loss={KD_LOSS_TYPE}")
        print(f"Layer-wise: {KD_LAYER_WISE}, Attention: {KD_ATTENTION_DISTILL}")
    print(f"{'='*100}")
    
    # Baseline
    model_original, tokenizer = load_fresh_model()
    baseline_acc = evaluate_accuracy(model_original, tokenizer, "Baseline FP16")
    stats_original = get_model_stats(model_original)
    print(f"\n✅ Baseline: {baseline_acc:.4f}, {stats_original['total_mb']:.2f}MB")
    
    # Stage 1 (Optional)
    stage1_config = None
    if RUN_STAGE1:
        stage1_config, stage1_results = stage1_config_search(model_original, tokenizer, baseline_acc)
    
    # Stage 2
    acc_ptq, stats_ptq, stage2_results = stage2_sensitivity_analysis(model_original, tokenizer, baseline_acc, stage1_config)
    
    cleanup_memory(model_original)
    print_memory_stats("after Stage 2 cleanup")
    
    # Stage 3 with KD
    model_qat, _ = load_fresh_model()
    final_model, final_acc, qat_results = stage3_qat_training_with_kd(model_qat, tokenizer, baseline_acc, acc_ptq, stats_ptq)
    
    # Final Summary
    duration = datetime.now() - start_time
    print(f"\n{'='*100}\n🎉 PIPELINE COMPLETE\n{'='*100}")
    print(f"Baseline: {baseline_acc:.4f} | PTQ: {acc_ptq:.4f} | QAT{'+KD' if ENABLE_KD else ''} Best: {qat_results['best_accuracy']:.4f} | Final: {final_acc:.4f}")
    print(f"Duration: {duration} | Compression: {qat_results['compression_ratio']:.2f}x | Size: {qat_results['model_size_mb']:.2f}MB")
    print(f"{'='*100}")
    print_memory_stats("final")
    
    summary = {
        'model_id': MODEL_ID,
        'duration_seconds': duration.total_seconds(),
        'baseline_accuracy': baseline_acc,
        'kd_enabled': ENABLE_KD,
        'stage1_run': RUN_STAGE1,
        'stage2': {
            'accuracy': acc_ptq, 
            'degradation_%': (baseline_acc - acc_ptq) * 100,
            'compression_ratio': stats_ptq['ratio']
        },
        'stage3': {
            'final_accuracy': final_acc,
            'best_accuracy': qat_results['best_accuracy'],
            'best_epoch': qat_results['best_epoch'],
            'compression_ratio': qat_results['compression_ratio'],
            'model_size_mb': qat_results['model_size_mb'],
            'recovery_%': qat_results['qat_recovery_%']
        }
    }
    
    with open(f"final_quantized_model/pipeline_summary{'_kd' if ENABLE_KD else ''}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💡 TensorBoard: tensorboard --logdir={TENSORBOARD_DIR}")
    return final_model, summary

# ==========================================
# 🚀 MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*100}\nSTARTING QUANTIZATION PIPELINE\n{'='*100}")
    print(f"🔥 CONTROL FLAGS:")
    print(f"   Stage 1 (Config Search):            {'ON' if RUN_STAGE1 else 'OFF'}")
    print(f"   Stage 2 (Layer-wise Sensitivity):   {'ON' if ENABLE_LAYER_WISE_SENSITIVITY else 'OFF'}")
    print(f"   Embedding/LMHead Quantization:      {'ON' if ENABLE_EMBEDDING_LMHEAD_QUANTIZATION else 'OFF (FP16)'}")
    print(f"   Knowledge Distillation:             {'ON' if ENABLE_KD else 'OFF'}")
    print(f"\n⚙️  QAT BIT-WIDTHS:")
    print(f"   Embeddings:  {QAT_EMBEDDING_BITS}-bit")
    print(f"   LM Head:     {QAT_LMHEAD_BITS}-bit")
    print(f"   Linear:      {QAT_LINEAR_BITS}-bit")
    print(f"   Group Size:  {QAT_GROUP_SIZE}")
    print(f"   Mode:        {'Symmetric' if QAT_SYMMETRIC else 'Asymmetric'}")
    print(f"\n⚙️  MEMORY OPTIMIZATION:")
    print(f"   Gradient Accumulation:  {QAT_GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   Gradient Checkpointing: {'ON' if QAT_USE_GRADIENT_CHECKPOINTING else 'OFF'}")
    if ENABLE_KD:
        print(f"\n⚙️  KNOWLEDGE DISTILLATION:")
        print(f"   Temperature:     {KD_TEMPERATURE}")
        print(f"   Alpha (KD/Task): {KD_ALPHA}")
        print(f"   Loss Type:       {KD_LOSS_TYPE}")
        print(f"   Layer-wise KD:   {'ON' if KD_LAYER_WISE else 'OFF'}")
        print(f"   Attention KD:    {'ON' if KD_ATTENTION_DISTILL else 'OFF'}")
    print(f"\n⚙️  TRAINING: {QAT_NUM_EPOCHS} epochs, {QAT_NUM_TRAIN_SAMPLES} samples")
    print(f"⚙️  EVALUATION: {EVAL_LIMIT*100}% of MMLU dataset")
    print(f"{'='*100}")
    
    print_memory_stats("initial")
    
    try:
        final_model, pipeline_summary = run_complete_pipeline()
        
        # ✅ Final cleanup
        cleanup_memory(final_model)
        
        print("\n" + "="*100)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*100)
        print("\n📁 Output Files:")
        print(f"   • Final Model: final_quantized_model/qat{'_kd' if ENABLE_KD else ''}_model_final.pt")
        print(f"   • Pipeline Summary: final_quantized_model/pipeline_summary{'_kd' if ENABLE_KD else ''}.json")
        print(f"   • Stage 3 Results: stage3_qat_training/qat{'_kd' if ENABLE_KD else ''}_results.json")
        if RUN_STAGE1:
            print(f"   • Stage 1 Results: stage1_config_search/stage1_results.json")
        print(f"   • Stage 2 Results: stage2_sensitivity/stage2_results.json")
        print(f"   • Checkpoints: {CHECKPOINT_DIR}/")
        print(f"   • TensorBoard Logs: {TENSORBOARD_DIR}/")
        
        print("\n📊 Final Results:")
        print(f"   Baseline Accuracy:    {pipeline_summary['baseline_accuracy']:.4f}")
        print(f"   PTQ Accuracy:         {pipeline_summary['stage2']['accuracy']:.4f}")
        print(f"   Final QAT Accuracy:   {pipeline_summary['stage3']['final_accuracy']:.4f}")
        print(f"   Best QAT Accuracy:    {pipeline_summary['stage3']['best_accuracy']:.4f}")
        print(f"   Compression Ratio:    {pipeline_summary['stage3']['compression_ratio']:.2f}x")
        print(f"   Model Size:           {pipeline_summary['stage3']['model_size_mb']:.2f} MB")
        print(f"   Accuracy Recovery:    {pipeline_summary['stage3']['recovery_%']:+.2f}%")
        print(f"   Total Duration:       {pipeline_summary['duration_seconds']/60:.1f} minutes")
        
        print("\n💡 Next Steps:")
        print(f"   1. View training progress: tensorboard --logdir={TENSORBOARD_DIR}")
        print(f"   2. Load best model from: {CHECKPOINT_DIR}/best_model_*.pt")
        print(f"   3. Deploy final model: final_quantized_model/qat{'_kd' if ENABLE_KD else ''}_model_final.pt")
        print("="*100)
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with exception:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting:")
        print("   1. Check CUDA memory: nvidia-smi")
        print("   2. Reduce batch size or gradient accumulation steps")
        print("   3. Enable gradient checkpointing")
        print("   4. Disable layer-wise or attention KD")
        print("   5. Check logs in stage*/ directories")
        
        cleanup_memory()
        raise
    
    finally:
        # ✅ Always cleanup at the end
        cleanup_memory()
        print_memory_stats("final cleanup")

        
