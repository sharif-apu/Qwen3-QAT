import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import gc

# ==========================================
# ⚙️ COMMON CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🌱 SEED SETTING
# ==========================================
def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 🧹 MEMORY MANAGEMENT
# ==========================================
def cleanup_memory(model=None, lm_obj=None):
    """Clean up memory by deleting objects and clearing cache"""
    if model is not None:
        del model
    if lm_obj is not None:
        del lm_obj
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 📥 MODEL LOADING
# ==========================================
def load_fresh_model():
    """Load a fresh model and tokenizer"""
    print(f"   [System] Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ==========================================
# 📊 MODEL STATISTICS (Version 1 - Simple)
# ==========================================
def get_model_stats(model):
    """
    Calculate model statistics including size and compression ratio.
    Used in: ta1_step2_ptq.py, ta1_step4_comp.py
    """
    total_bits_full = 0
    target_bits = 0
    target_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            bits = 16
            count = module.weight.numel()
            total_bits_full += count * bits
        elif isinstance(module, nn.Linear):
            if "lm_head" in name:
                bits = 16
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
        
    return full_mb, target_mb, target_ratio

# ==========================================
# 📊 MODEL STATISTICS (Version 2 - Detailed)
# ==========================================
def get_model_stats_detailed(model):
    """
    Calculate detailed model statistics.
    Used in: ta1_step2_qat.py, ta1_step3_mp_qatkd.py
    Returns dict with total_mb, quantized_mb, ratio, avg_bits, total_params
    """
    total_bits_full = 0
    target_bits = 0
    target_params = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            bits = 16
            count = module.weight.numel()
            total_bits_full += count * bits
            total_params += count
        elif isinstance(module, nn.Linear):
            count = module.weight.numel()
            total_params += count
            
            if "lm_head" in name:
                bits = 16
                total_bits_full += count * bits
            else:
                bits = getattr(module, 'quant_bit_width', 16)
                total_bits_full += count * bits
                target_bits += count * bits
                target_params += count

    full_mb = total_bits_full / 8 / 1024**2
    target_mb = target_bits / 8 / 1024**2
    
    if target_params > 0:
        avg_target_bits = target_bits / target_params
        target_ratio = 16.0 / avg_target_bits
    else:
        avg_target_bits = 16
        target_ratio = 1.0
    
    total_mb = total_params * 16 / 8 / 1024**2
        
    return {
        'total_mb': total_mb,
        'quantized_mb': full_mb,
        'target_mb': target_mb,
        'ratio': target_ratio,
        'avg_bits': avg_target_bits,
        'total_params': total_params,
        'target_params': target_params
    }

# ==========================================
# ✅ QUANTIZATION VERIFICATION
# ==========================================
def verify_quantization(model, original_model):
    """Verify that quantization actually changed the weights"""
    print("\n   [Verification] Checking if weights actually changed...")
    
    differences = []
    for (name1, module1), (name2, module2) in zip(model.named_modules(), original_model.named_modules()):
        if isinstance(module1, nn.Linear) and "lm_head" not in name1 and "embed" not in name1.lower():
            diff = (module1.weight.data - module2.weight.data).abs().mean().item()
            differences.append(diff)
    
    avg_diff = np.mean(differences) if differences else 0
    print(f"   [Verification] Average weight difference: {avg_diff:.6f}")
    
    if avg_diff < 1e-6:
        print("   ⚠️  WARNING: Weights barely changed! Quantization may not be working!")
        return False
    else:
        print(f"   ✅ Weights changed significantly - quantization is working!")
        return True

# ==========================================
# 📈 EVALUATION
# ==========================================
def evaluate_accuracy(model, tokenizer, desc="Evaluation", eval_limit=0.1, eval_batch_size=1):
    """Evaluate model accuracy on MMLU"""
    print(f"   [Eval] Starting MMLU Evaluation ({desc})...")
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
    results = simple_evaluate(
        model=lm_obj,
        tasks=["mmlu"],
        limit=eval_limit,
        device=DEVICE
    )
    acc = results["results"]["mmlu"]["acc,none"]
    del lm_obj
    cleanup_memory()
    return acc

# ==========================================
# 🔧 QUANTIZATION - Core Function
# ==========================================
def quantize_tensor_fake(w, n_bit=4, granularity="per_group", group_size=128, sym=False):
    """
    Fake quantization function (exact match across multiple scripts).
    Used in: ta1_step3_lmhead_analysis.py, ta1_step3_mp_qatkd.py, ta1_step4_comp.py
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