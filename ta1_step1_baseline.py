import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset
import gc
from quantization_utils import set_seed, cleanup_memory, load_fresh_model, DEVICE, evaluate_accuracy
# ==========================================
# ⚙️ Configuration
# ==========================================
SEED = 42
set_seed(SEED)
CALIB_SEQ_LEN = 128
# ==========================================
# 📊 Layer Analysis Function
# ==========================================
def analyze_model_layers(model):
    """
    Analyze and display detailed information about all layers in the model
    """
    print("\n" + "="*140)
    print("📊 MODEL LAYER ANALYSIS")
    print("="*140)
    
    layer_info = []
    total_params = 0
    total_size_mb = 0
    
    # Collect layer information
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            layer_type = type(module).__name__
            
            # Get weight information
            if hasattr(module, 'weight'):
                weight = module.weight
                shape = tuple(weight.shape)
                num_params = weight.numel()
                dtype = weight.dtype
                
                # Calculate size in MB (assuming bfloat16 = 2 bytes)
                if dtype == torch.bfloat16 or dtype == torch.float16:
                    bytes_per_param = 2
                elif dtype == torch.float32:
                    bytes_per_param = 4
                elif dtype == torch.int8:
                    bytes_per_param = 1
                else:
                    bytes_per_param = 2  # default
                
                size_mb = (num_params * bytes_per_param) / (1024**2)
                
                # Get quantization bit width if available
                bit_width = getattr(module, 'quant_bit_width', 16)
                
                layer_info.append({
                    'name': name,
                    'type': layer_type,
                    'shape': shape,
                    'params': num_params,
                    'size_mb': size_mb,
                    'dtype': str(dtype),
                    'bits': bit_width
                })
                
                total_params += num_params
                total_size_mb += size_mb
    
    # Display table header
    print(f"{'#':<4} | {'Layer Name':<50} | {'Type':<10} | {'Shape':<25} | {'Params':<12} | {'Size (MB)':<10} | {'Dtype':<12} | {'Bits':<5}")
    print("-" * 140)
    
    # Display each layer
    for idx, info in enumerate(layer_info, 1):
        shape_str = str(info['shape'])
        params_str = f"{info['params']:,}"
        print(f"{idx:<4} | {info['name']:<50} | {info['type']:<10} | {shape_str:<25} | {params_str:<12} | {info['size_mb']:<10.2f} | {info['dtype']:<12} | {info['bits']:<5}")
    
    # Display summary
    print("-" * 140)
    print(f"{'TOTAL':<4} | {'':<50} | {'':<10} | {'':<25} | {total_params:<12,} | {total_size_mb:<10.2f} | {'':<12} | {'':<5}")
    print("="*140)
    
    # Display layer type summary
    print("\n📈 LAYER TYPE SUMMARY:")
    print("-" * 80)
    type_summary = {}
    for info in layer_info:
        layer_type = info['type']
        if layer_type not in type_summary:
            type_summary[layer_type] = {'count': 0, 'params': 0, 'size_mb': 0}
        type_summary[layer_type]['count'] += 1
        type_summary[layer_type]['params'] += info['params']
        type_summary[layer_type]['size_mb'] += info['size_mb']
    
    print(f"{'Layer Type':<15} | {'Count':<8} | {'Total Params':<15} | {'Total Size (MB)':<15}")
    print("-" * 80)
    for layer_type, stats in type_summary.items():
        print(f"{layer_type:<15} | {stats['count']:<8} | {stats['params']:<15,} | {stats['size_mb']:<15.2f}")
    print("="*80)
    
    return layer_info, total_params, total_size_mb

# ==========================================
# 📊 Helper: Statistics
# ==========================================
def get_model_stats(model):
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
# 📚 Calibration Data
# ==========================================
def get_calibration_data(tokenizer):
    print("      -> Loading Wikitext-2 for calibration...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_buffer = ""
        for sample in dataset:
            text = sample['text']
            if len(text) > 50: 
                text_buffer += text + "\n"
                if len(text_buffer) > CALIB_SEQ_LEN * 4:
                    break
        
        tokens = tokenizer(text_buffer, return_tensors="pt").input_ids
        if tokens.size(1) > CALIB_SEQ_LEN:
            tokens = tokens[:, :CALIB_SEQ_LEN]
        return tokens.to(DEVICE)
    except Exception as e:
        print(f"      [Warning] Failed to load Wikitext ({e}). Fallback to dummy data.")
        text = "Artificial intelligence is the study of intelligent agents. " * 100
        return tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

# ==========================================
# 🧮 Core: Fake Quantization (Flexible)
# ==========================================
def quantize_tensor_fake(w, n_bit, granularity="per_group", group_size=128, sym=True):
    original_shape = w.shape
    
    # 1. Reshape based on Granularity
    if granularity == "per_tensor":
        # ✅ FIX: Flatten to [1, N] so we get 1 scale for the whole tensor
        w_reshaped = w.flatten().reshape(1, -1)
        
    elif granularity == "per_channel":
        # [Out, In] -> Scale per Out (dim 0)
        w_reshaped = w.reshape(w.shape[0], -1)
        
    elif granularity == "per_group":
        # [N_Groups, Group_Size]
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


# ==========================================
# 🛠️ Experiment Methods
# ==========================================

def run_rtn(model, granularity="per_group", sym=False):
    """
    Step 1: RTN with configurable Granularity and Symmetry
    """
    print(f"   [Method] RTN ({granularity}, Sym={sym})...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            with torch.no_grad():
                module.weight.data = quantize_tensor_fake(
                    module.weight.data, 
                    n_bit=4, 
                    granularity=granularity,
                    group_size=128, 
                    sym=sym
                )
                module.quant_bit_width = 4
    return model



# ==========================================
# 🚀 MAIN EXPERIMENT LOOP
# ==========================================
if __name__ == "__main__":
    # ==========================================
    # 🔍 STEP 0: ANALYZE MODEL ARCHITECTURE
    # ==========================================
    print("\n" + "🔍 LOADING MODEL FOR INITIAL ANALYSIS...")
    model, tokenizer = load_fresh_model()
    
    # Analyze and display all layers
    layer_info, total_params, total_size_mb = analyze_model_layers(model)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Total Size: {total_size_mb:.2f} MB")
    print(f"   Device: {DEVICE}")
    
    # Clean up before starting experiments
    cleanup_memory(model)

    
    # ==========================================
    # 🧪 START EXPERIMENTS
    # ==========================================
    results = {}
    
    # 1. Original
    print("\n=== EXP 0: ORIGINAL ===")
    model, tokenizer = load_fresh_model()
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['Original'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)
    
    # 2. RTN - Per Tensor (Baseline 1)
    print("\n=== EXP 1.1: RTN (Per-Tensor) ===")
    model, tokenizer = load_fresh_model()
    run_rtn(model, granularity="per_tensor", sym=False)
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['RTN_Tensor'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)

    # 3. RTN - Per Channel (Baseline 2)
    print("\n=== EXP 1.2: RTN (Per-Channel) ===")
    model, tokenizer = load_fresh_model()
    run_rtn(model, granularity="per_channel", sym=False)
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['RTN_Channel'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)

    # 4. RTN - Per Group (Baseline 3 - The Strongest RTN)
    print("\n=== EXP 1.3: RTN (Per-Group 128) ===")
    model, tokenizer = load_fresh_model()
    run_rtn(model, granularity="per_group", sym=False)
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['RTN_Group'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)

     # 2. RTN - Per Tensor (Baseline 1)
    print("\n=== EXP 1.4: RTN (Per-Tensor) symmetric ===")
    model, tokenizer = load_fresh_model()
    run_rtn(model, granularity="per_tensor", sym=True)
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['RTN_Tensor_sym'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)

    # 3. RTN - Per Channel (Baseline 2)
    print("\n=== EXP 1.5: RTN (Per-Channel) symmetric===")
    model, tokenizer = load_fresh_model()
    run_rtn(model, granularity="per_channel", sym=True)
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['RTN_Channel_sym'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)

    # 4. RTN - Per Group (Baseline 3 - The Strongest RTN)
    print("\n=== EXP 1.6: RTN (Per-Group 128) symmetric===")
    model, tokenizer = load_fresh_model()
    run_rtn(model, granularity="per_group", sym=True)
    acc = evaluate_accuracy(model, tokenizer)
    mb, tgt_mb, ratio = get_model_stats(model)
    results['RTN_Group_sym'] = {'acc': acc, 'tgt_mb': tgt_mb, 'ratio': ratio}
    cleanup_memory(model)
    

    # --- FINAL REPORT ---
    print("\n" + "="*120)
    print(f"{'Method':<15} | {'Granularity':<12} | {'Acc':<8} | {'Tgt Size':<12} | {'Ratio':<6} | {'Notes'}")
    print("-" * 120)
    
    meta = {
        'Original': ('-', 'None'),
        'RTN_Tensor': ('Tensor', 'Asym'),
        'RTN_Channel': ('Channel', 'Asym'),
        'RTN_Group': ('Group-128', 'Asym'),
        'RTN_Tensor_sym': ('Tensor', 'Sym'),
        'RTN_Channel_sym': ('Channel', 'Sym'),
        'RTN_Group_sym': ('Group-128', 'Sym'),
        # 'AWQ_Lite': ('Group-128', 'Weighted MSE'),
        # 'GPTQ_Lite': ('Group-128', 'Hessian MSE')
    }
    
    for k, v in results.items():
        gran, note = meta[k]
        print(f"{k:<15} | {gran:<12} | {v['acc']:.4f}   | {v['tgt_mb']:.2f} MB     | {v['ratio']:<6.2f} | {note}")
    print("="*120)
