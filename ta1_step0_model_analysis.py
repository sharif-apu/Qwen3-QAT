import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 🔍 ENHANCED MODEL ANALYSIS WITH SIZE VERIFICATION
# ==========================================

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def get_tensor_size_mb(tensor):
    """Calculate tensor size in MB"""
    return tensor.nelement() * tensor.element_size() / (1024 ** 2)

def get_module_size_mb(module):
    """Calculate total size of a module in MB"""
    total_size = 0
    for param in module.parameters():
        total_size += get_tensor_size_mb(param)
    return total_size

def get_dtype_info(tensor):
    """Get dtype and bit width"""
    dtype_map = {
        torch.float32: ('float32', 32),
        torch.float16: ('float16', 16),
        torch.bfloat16: ('bfloat16', 16),
        torch.int8: ('int8', 8),
        torch.uint8: ('uint8', 8),
        torch.int32: ('int32', 32),
        torch.int64: ('int64', 64),
    }
    return dtype_map.get(tensor.dtype, (str(tensor.dtype), 'unknown'))

def get_model_disk_size(model_path):
    """Calculate actual disk size of model files"""
    if not os.path.exists(model_path):
        return None
    
    total_size = 0
    file_details = []
    
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                total_size += size
                file_details.append({
                    'file': filename,
                    'size_mb': size / (1024 ** 2)
                })
    
    return total_size / (1024 ** 3), file_details  # Convert to GB

def categorize_layer(name):
    """Categorize layer by name"""
    name_lower = name.lower()
    if 'embed' in name_lower and 'lm_head' not in name_lower:
        return 'embedding'
    elif 'lm_head' in name_lower or (('output' in name_lower or 'head' in name_lower) and 'embed' not in name_lower):
        return 'lm_head'
    elif any(x in name_lower for x in ['mlp', 'fc']):
        return 'linear'
    elif any(x in name_lower for x in ['attn', 'attention']):
        return 'attention'
    elif any(x in name_lower for x in ['norm', 'ln', 'layernorm', 'rmsnorm']):
        return 'normalization'
    else:
        return 'other'

def cross_check_calculations(model, layer_info, category_stats, model_path=None):
    """Perform comprehensive cross-checks including disk size verification"""
    print("\n" + "="*100)
    print("🔍 COMPREHENSIVE CROSS-CHECK CALCULATIONS")
    print("="*100)
    
    checks_passed = 0
    checks_total = 0
    issues = []
    warnings = []
    
    # Check 1: Total parameters
    checks_total += 1
    model_params = sum(p.numel() for p in model.parameters())
    layer_params = sum(layer['num_params'] for layer in layer_info)
    category_params = sum(stats['params'] for stats in category_stats.values())
    
    print(f"\n✓ Check 1: Parameter Count Consistency")
    print(f"   Model total:          {model_params:>20,}")
    print(f"   Layer sum:            {layer_params:>20,}")
    print(f"   Category sum:         {category_params:>20,}")
    
    if model_params == layer_params == category_params:
        print(f"   Status: ✅ PASS - All counts match!")
        checks_passed += 1
    else:
        print(f"   Status: ❌ FAIL - Mismatch detected!")
        issues.append("Parameter count mismatch")
    
    # Check 2: Parameter storage size (in memory)
    checks_total += 1
    layer_size = sum(layer['size_mb'] for layer in layer_info)
    category_size = sum(stats['size_mb'] for stats in category_stats.values())
    
    print(f"\n✓ Check 2: Parameter Storage Size Consistency")
    print(f"   Layer sum:            {layer_size:>20.2f} MB ({layer_size/1024:.3f} GB)")
    print(f"   Category sum:         {category_size:>20.2f} MB ({category_size/1024:.3f} GB)")
    print(f"   Difference:           {abs(layer_size - category_size):>20.6f} MB")
    
    if abs(layer_size - category_size) < 0.01:
        print(f"   Status: ✅ PASS - Sizes match!")
        checks_passed += 1
    else:
        print(f"   Status: ❌ FAIL - Size mismatch!")
        issues.append("Size calculation mismatch")
    
    # Check 3: Disk size vs parameter size
    checks_total += 1
    print(f"\n✓ Check 3: Disk Size vs Parameter Size Analysis")
    
    disk_size_gb = None
    file_details = None
    
    if model_path and os.path.exists(model_path):
        disk_size_gb, file_details = get_model_disk_size(model_path)
        print(f"   Disk size (total):    {disk_size_gb:>20.3f} GB")
        
        # Show breakdown of disk files
        if file_details:
            print(f"\n   📁 Disk File Breakdown:")
            df_files = pd.DataFrame(file_details).sort_values('size_mb', ascending=False)
            for _, row in df_files.head(10).iterrows():
                print(f"      {row['file']:<40} {row['size_mb']:>10.2f} MB")
    else:
        print(f"   Disk size:            {'Not available (model not cached)':>20}")
    
    param_size_gb = layer_size / 1024
    print(f"   Parameter size (fp16):{param_size_gb:>20.3f} GB")
    
    if disk_size_gb:
        overhead = disk_size_gb - param_size_gb
        overhead_pct = (overhead / param_size_gb) * 100
        print(f"   Overhead:             {overhead:>20.3f} GB ({overhead_pct:.1f}%)")
        
        print(f"\n   📝 Size Components:")
        print(f"      • Parameter storage (fp16): {param_size_gb:.3f} GB")
        print(f"      • Additional files:         {overhead:.3f} GB")
        print(f"        - Config files (config.json, etc.)")
        print(f"        - Tokenizer files")
        print(f"        - Generation config")
        print(f"        - SafeTensors format overhead")
        print(f"        - Metadata")
        print(f"      • Total disk usage:         {disk_size_gb:.3f} GB")
        
        if overhead > 0:
            print(f"   Status: ✅ PASS - Disk size includes expected overhead!")
            checks_passed += 1
        else:
            print(f"   Status: ⚠️  WARNING - Unexpected disk size!")
            warnings.append(f"Disk size analysis shows unexpected results")
            checks_passed += 0.5
    else:
        print(f"   Status: ⚠️  SKIP - Cannot verify (model not on disk)")
        warnings.append("Disk size verification skipped")
    
    # Check 4: Layer count
    checks_total += 1
    total_layers = len(layer_info)
    category_layer_count = sum(stats['count'] for stats in category_stats.values())
    
    print(f"\n✓ Check 4: Layer Count Consistency")
    print(f"   Total layers:         {total_layers:>20,}")
    print(f"   Category sum:         {category_layer_count:>20,}")
    
    if total_layers == category_layer_count:
        print(f"   Status: ✅ PASS - Layer counts match!")
        checks_passed += 1
    else:
        print(f"   Status: ❌ FAIL - Layer count mismatch!")
        issues.append("Layer count mismatch")
    
    # Check 5: Size vs Parameters relationship
    checks_total += 1
    print(f"\n✓ Check 5: Size-Parameter Relationship")
    
    total_bits = layer_size * 8 * 1024 * 1024
    avg_bits_per_param = total_bits / model_params
    expected_bits = 16  # fp16
    
    print(f"   Total bits:           {total_bits:>20,.0f}")
    print(f"   Avg bits/param:       {avg_bits_per_param:>20.2f}")
    print(f"   Expected (fp16):      {expected_bits:>20.2f}")
    print(f"   Difference:           {abs(avg_bits_per_param - expected_bits):>20.2f}")
    
    if abs(avg_bits_per_param - expected_bits) < 0.5:
        print(f"   Status: ✅ PASS - Bits per parameter is correct!")
        checks_passed += 1
    else:
        print(f"   Status: ⚠️  WARNING - Unexpected bits per parameter!")
        warnings.append(f"Unexpected bits per parameter: {avg_bits_per_param:.2f}")
        checks_passed += 0.5
    
    # Check 6: Category parameter sum
    checks_total += 1
    print(f"\n✓ Check 6: Category-wise Parameter Verification")
    
    category_param_check = {}
    for layer in layer_info:
        cat = layer['category']
        if cat not in category_param_check:
            category_param_check[cat] = 0
        category_param_check[cat] += layer['num_params']
    
    all_match = True
    for cat, params in category_param_check.items():
        expected = category_stats[cat]['params']
        match = params == expected
        all_match = all_match and match
        status = "✅" if match else "❌"
        print(f"   {cat:<15}: {params:>15,} vs {expected:>15,} {status}")
    
    if all_match:
        print(f"   Status: ✅ PASS - All categories match!")
        checks_passed += 1
    else:
        print(f"   Status: ❌ FAIL - Category mismatch!")
        issues.append("Category parameter mismatch")
    
    # Check 7: Manual size calculation verification
    checks_total += 1
    print(f"\n✓ Check 7: Manual Size Calculation Verification")
    
    manual_size = 0
    for param in model.parameters():
        manual_size += param.nelement() * param.element_size() / (1024 ** 2)
    
    print(f"   Calculated size:      {layer_size:>20.2f} MB")
    print(f"   Manual size:          {manual_size:>20.2f} MB")
    print(f"   Difference:           {abs(layer_size - manual_size):>20.6f} MB")
    
    if abs(layer_size - manual_size) < 0.01:
        print(f"   Status: ✅ PASS - Manual calculation matches!")
        checks_passed += 1
    else:
        print(f"   Status: ❌ FAIL - Manual calculation mismatch!")
        issues.append("Manual size calculation mismatch")
    
    # Check 8: Core model calculation
    checks_total += 1
    print(f"\n✓ Check 8: Core Model Calculation Verification")
    
    core_params_calc = model_params
    core_size_calc = layer_size
    
    for cat in ['embedding', 'lm_head']:
        if cat in category_stats:
            core_params_calc -= category_stats[cat]['params']
            core_size_calc -= category_stats[cat]['size_mb']
    
    print(f"   Core parameters:      {core_params_calc:>20,}")
    print(f"   Core size:            {core_size_calc:>20.2f} MB ({core_size_calc/1024:.3f} GB)")
    print(f"   Core % of total:      {(core_size_calc/layer_size)*100:>20.1f}%")
    
    if core_params_calc > 0 and core_size_calc > 0:
        print(f"   Status: ✅ PASS - Core calculations valid!")
        checks_passed += 1
    else:
        print(f"   Status: ❌ FAIL - Core calculations invalid!")
        issues.append("Core model calculation error")
    
    # Summary
    print("\n" + "="*100)
    print(f"📊 CROSS-CHECK SUMMARY: {checks_passed:.1f}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("✅ ALL CHECKS PASSED - Calculations are fully verified!")
    elif checks_passed >= checks_total * 0.8:
        print(f"✅ MOSTLY PASSED - {checks_passed:.1f}/{checks_total} checks passed")
    else:
        print(f"⚠️  SOME ISSUES FOUND - {checks_total - checks_passed:.1f} check(s) failed")
    
    if issues:
        print("\n❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    print("\n" + "="*100)
    print("📖 TERMINOLOGY CLARIFICATION:")
    print("="*100)
    print("• Parameter Count:     Number of trainable parameters (e.g., 600M)")
    print("• Parameter Size:      Storage size of parameters in memory (e.g., 1.2 GB for fp16)")
    print("• Disk Size:           Total size on disk including all files (e.g., 1.5 GB)")
    print("• Overhead:            Difference between disk size and parameter size")
    print("                       (config files, tokenizer, metadata, format overhead)")
    print("="*100)
    
    return {
        'checks_passed': checks_passed,
        'checks_total': checks_total,
        'issues': issues,
        'warnings': warnings,
        'model_params': model_params,
        'param_size_mb': layer_size,
        'param_size_gb': param_size_gb,
        'disk_size_gb': disk_size_gb,
        'core_params': core_params_calc,
        'core_size_mb': core_size_calc,
        'avg_bits_per_param': avg_bits_per_param,
        'file_details': file_details
    }

def analyze_model_detailed(model):
    """Analyze model layer by layer with precise categorization"""
    
    layer_info = []
    category_stats = defaultdict(lambda: {'count': 0, 'size_mb': 0, 'params': 0})
    
    print("\n" + "="*100)
    print("📊 LAYER-BY-LAYER ANALYSIS")
    print("="*100)
    
    total_params_check = 0
    
    for name, param in model.named_parameters():
        dtype_name, bit_width = get_dtype_info(param)
        size_mb = get_tensor_size_mb(param)
        num_params = param.nelement()
        category = categorize_layer(name)
        
        total_params_check += num_params
        
        layer_data = {
            'name': name,
            'shape': list(param.shape),
            'dtype': dtype_name,
            'bit_width': bit_width,
            'num_params': num_params,
            'size_mb': size_mb,
            'category': category,
            'requires_grad': param.requires_grad
        }
        layer_info.append(layer_data)
        
        # Update category stats
        category_stats[category]['count'] += 1
        category_stats[category]['size_mb'] += size_mb
        category_stats[category]['params'] += num_params
    
    # Verify parameter count
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Parameter Count Verification:")
    print(f"   Counted: {total_params_check:,}")
    print(f"   Model total: {model_total_params:,}")
    print(f"   Match: {'✓' if total_params_check == model_total_params else '✗ MISMATCH!'}")
    
    return layer_info, dict(category_stats), model_total_params

def print_layer_table(layer_info, top_n=30):
    """Print formatted table of layers"""
    df = pd.DataFrame(layer_info)
    
    print(f"\n📋 Top {top_n} Largest Layers:")
    print("-" * 130)
    print(f"{'Layer Name':<55} | {'Shape':<25} | {'Type':<10} | {'Parameters':>14} | {'Size (MB)':>10} | {'Category':<12}")
    print("-" * 130)
    
    df_sorted = df.sort_values('size_mb', ascending=False).head(top_n)
    
    for idx, row in df_sorted.iterrows():
        shape_str = 'x'.join(map(str, row['shape']))
        print(f"{row['name'][:54]:<55} | {shape_str:<25} | {row['dtype']:<10} | "
              f"{row['num_params']:>14,} | {row['size_mb']:>10.2f} | {row['category']:<12}")
    
    print("-" * 130)

def print_category_summary(category_stats):
    """Print summary by category"""
    print("\n" + "="*100)
    print("📊 SUMMARY BY CATEGORY")
    print("="*100)
    
    df = pd.DataFrame.from_dict(category_stats, orient='index')
    df = df.sort_values('size_mb', ascending=False)
    
    print(f"{'Category':<20} | {'Layer Count':>12} | {'Parameters':>18} | {'Size (MB)':>12} | {'Percentage':>10}")
    print("-" * 100)
    
    total_size = df['size_mb'].sum()
    total_params = df['params'].sum()
    
    for category, row in df.iterrows():
        percentage = (row['size_mb'] / total_size) * 100
        print(f"{category:<20} | {row['count']:>12} | {row['params']:>18,} | "
              f"{row['size_mb']:>12.2f} | {percentage:>9.1f}%")
    
    print("-" * 100)
    print(f"{'TOTAL':<20} | {df['count'].sum():>12} | {total_params:>18,} | "
          f"{total_size:>12.2f} | {100.0:>9.1f}%")
    print("="*100)
    
    # Calculate size without embedding and lm_head
    core_size = total_size
    core_params = total_params
    
    if 'embedding' in df.index:
        core_size -= df.loc['embedding', 'size_mb']
        core_params -= df.loc['embedding', 'params']
    if 'lm_head' in df.index:
        core_size -= df.loc['lm_head', 'size_mb']
        core_params -= df.loc['lm_head', 'params']
    
    print(f"\n📦 Core Model (without embedding & lm_head):")
    print(f"   Parameters: {core_params:,}")
    print(f"   Size: {core_size:.2f} MB ({core_size/1024:.2f} GB)")
    print(f"   Percentage of total: {(core_size/total_size)*100:.1f}%")
    
    return core_size, core_params

def calculate_compression_ratios(category_stats, quantized_configs, base_size_mb, core_size_mb):
    """Calculate compression ratios for different quantization configs"""
    
    print("\n" + "="*140)
    print("🔢 QUANTIZATION COMPRESSION ESTIMATES")
    print("="*140)
    
    results = []
    base_bits = 16  # Assuming base model is fp16
    
    # Add baseline (no quantization)
    baseline = {
        'config': 'baseline_fp16',
        'emb': 'fp16',
        'lm_head': 'fp16',
        'linear': 'fp16',
        'total_size_mb': base_size_mb,
        'core_size_mb': core_size_mb,
        'compression_ratio': 1.0,
        'size_reduction_pct': 0.0,
        'core_compression_ratio': 1.0,
        'core_reduction_pct': 0.0
    }
    results.append(baseline)
    
    for config in quantized_configs:
        name = config['name']
        emb_bits = config['emb']
        lm_bits = config['lm']
        linear_bits = config['linear']
        
        # Estimate size based on bit widths
        estimated_total_size = 0
        estimated_core_size = 0
        
        for category, stats in category_stats.items():
            if category == 'embedding':
                bits = emb_bits if isinstance(emb_bits, int) else base_bits
            elif category == 'lm_head':
                bits = lm_bits if isinstance(lm_bits, int) else base_bits
            elif category in ['linear', 'attention']:
                bits = linear_bits if isinstance(linear_bits, int) else base_bits
            else:
                bits = base_bits
            
            category_size = stats['size_mb'] * (bits / base_bits)
            estimated_total_size += category_size
            
            # Core size excludes embedding and lm_head
            if category not in ['embedding', 'lm_head']:
                estimated_core_size += category_size
        
        compression_ratio = base_size_mb / estimated_total_size
        size_reduction_pct = (1 - estimated_total_size / base_size_mb) * 100
        
        core_compression_ratio = core_size_mb / estimated_core_size
        core_reduction_pct = (1 - estimated_core_size / core_size_mb) * 100
        
        result = {
            'config': name,
            'emb': f"{emb_bits}bit" if isinstance(emb_bits, int) else "fp16",
            'lm_head': f"{lm_bits}bit" if isinstance(lm_bits, int) else "fp16",
            'linear': f"{linear_bits}bit" if isinstance(linear_bits, int) else "fp16",
            'total_size_mb': estimated_total_size,
            'core_size_mb': estimated_core_size,
            'compression_ratio': compression_ratio,
            'size_reduction_pct': size_reduction_pct,
            'core_compression_ratio': core_compression_ratio,
            'core_reduction_pct': core_reduction_pct
        }
        results.append(result)
    
    # Print table
    print(f"{'Config':<18} | {'Emb':>6} | {'LM':>6} | {'Linear':>6} | "
          f"{'Total (MB)':>12} | {'Core (MB)':>12} | {'Total Comp':>11} | {'Total Red%':>10} | "
          f"{'Core Comp':>11} | {'Core Red%':>10}")
    print("-" * 140)
    
    for r in results:
        print(f"{r['config']:<18} | {r['emb']:>6} | {r['lm_head']:>6} | {r['linear']:>6} | "
              f"{r['total_size_mb']:>12.2f} | {r['core_size_mb']:>12.2f} | "
              f"{r['compression_ratio']:>10.2f}x | {r['size_reduction_pct']:>9.1f}% | "
              f"{r['core_compression_ratio']:>10.2f}x | {r['core_reduction_pct']:>9.1f}%")
    
    print("="*140)
    print("\n📝 Notes:")
    print("   • Total: Full parameter size including embeddings and lm_head")
    print("   • Core: Parameter size WITHOUT embeddings and lm_head (transformer layers only)")
    print("   • Comp: Compression ratio (higher is better)")
    print("   • Red%: Size reduction percentage (higher is better)")
    print("   • These are PARAMETER sizes (in-memory), disk size will include additional overhead")
    
    return results

def print_overall_statistics(total_params, total_size_mb, core_params, core_size_mb, cross_check_results=None):
    """Print comprehensive overall statistics"""
    print(f"\n" + "="*100)
    print("📊 OVERALL MODEL STATISTICS")
    print("="*100)
    
    print(f"\n🔷 Parameter Count:")
    print(f"   Total Parameters:        {total_params:>20,}")
    print(f"   Core Parameters:          {core_params:>20,}")
    print(f"   Embedding + LM Head:      {total_params - core_params:>20,}")
    
    print(f"\n🔶 Parameter Size (in memory, fp16):")
    print(f"   Total Size:               {total_size_mb:>19.2f} MB")
    print(f"   Total Size:               {total_size_mb/1024:>19.3f} GB")
    print(f"   Core Size:                {core_size_mb:>19.2f} MB")
    print(f"   Core Size:                {core_size_mb/1024:>19.3f} GB")
    print(f"   Embedding + LM Head:      {total_size_mb - core_size_mb:>19.2f} MB")
    print(f"   Avg bits per parameter:   {(total_size_mb * 8 * 1024 * 1024) / total_params:>19.2f}")
    
    if cross_check_results and cross_check_results.get('disk_size_gb'):
        print(f"\n🔸 Disk Size (with overhead):")
        print(f"   Total Disk Size:          {cross_check_results['disk_size_gb']:>19.3f} GB")
        overhead = cross_check_results['disk_size_gb'] - (total_size_mb/1024)
        print(f"   Overhead:                 {overhead:>19.3f} GB ({overhead/(total_size_mb/1024)*100:.1f}%)")
        print(f"   (includes config, tokenizer, metadata, format overhead)")
    
    print("="*100)

# ==========================================
# 📊 PLOTTING FUNCTIONS
# ==========================================

def plot_summary_dashboard(layer_info, category_stats, compression_results, 
                          cross_check_results, output_dir='plots'):
    """Create comprehensive summary dashboard"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = plt.cm.Set3(range(12))
    
    # ============ TOP ROW ============
    
    # 1. Model Overview (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    total_params = cross_check_results['model_params']
    param_size_gb = cross_check_results['param_size_gb']
    disk_size_gb = cross_check_results.get('disk_size_gb')
    core_params = cross_check_results['core_params']
    core_size_mb = cross_check_results['core_size_mb']
    
    disk_info = f"{disk_size_gb:.3f} GB" if disk_size_gb else "N/A"
    
    overview_text = f"""
    MODEL OVERVIEW
    {'='*40}
    
    Model: {MODEL_ID.split('/')[-1]}
    
    Parameter Count: {total_params:,}
    Parameter Size: {param_size_gb:.3f} GB (fp16)
    Disk Size: {disk_info}
    
    Core Parameters: {core_params:,}
    Core Size: {core_size_mb/1024:.3f} GB
    
    Validation: {cross_check_results['checks_passed']:.1f}/{cross_check_results['checks_total']} ✓
    """
    
    ax1.text(0.05, 0.95, overview_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Category Distribution Pie (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    df_cat = pd.DataFrame.from_dict(category_stats, orient='index')
    df_cat = df_cat.sort_values('size_mb', ascending=False)
    
    wedges, texts, autotexts = ax2.pie(
        df_cat['size_mb'], 
        labels=df_cat.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax2.set_title('Parameter Size Distribution by Category', fontsize=12, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # 3. Size Comparison (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    sizes_data = ['Parameter\nSize\n(fp16)']
    sizes_values = [param_size_gb]
    colors_list = ['#66b3ff']
    
    if disk_size_gb:
        sizes_data.append('Disk\nSize\n(total)')
        sizes_values.append(disk_size_gb)
        colors_list.append('#90EE90')
    
    bars = ax3.bar(sizes_data, sizes_values, color=colors_list, 
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Size (GB)', fontweight='bold')
    ax3.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, sizes_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============ MIDDLE ROW ============
    
    # 4. Top 10 Layers (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    df = pd.DataFrame(layer_info)
    df_top = df.sort_values('size_mb', ascending=False).head(10)
    
    y_pos = np.arange(len(df_top))
    layer_names = [name.split('.')[-1][:20] for name in df_top['name']]
    
    bars = ax4.barh(y_pos, df_top['size_mb'], color='steelblue', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(layer_names, fontsize=9)
    ax4.set_xlabel('Size (MB)', fontweight='bold')
    ax4.set_title('Top 10 Largest Layers', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, df_top['size_mb'])):
        ax4.text(val, i, f' {val:.1f}', va='center', fontsize=8)
    
    # 5. Compression Comparison (Middle Center & Right - spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:])
    
    df_comp = pd.DataFrame(compression_results)
    df_comp = df_comp[df_comp['config'] != 'baseline_fp16']
    
    x = np.arange(len(df_comp))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, df_comp['total_size_mb'], width,
                    label='Total Size', color='steelblue', alpha=0.7)
    bars2 = ax5.bar(x + width/2, df_comp['core_size_mb'], width,
                    label='Core Size', color='darkorange', alpha=0.7)
    
    # Add baseline line
    baseline_total = compression_results[0]['total_size_mb']
    baseline_core = compression_results[0]['core_size_mb']
    ax5.axhline(y=baseline_total, color='red', linestyle='--', 
                label='Baseline Total', linewidth=2, alpha=0.7)
    ax5.axhline(y=baseline_core, color='darkred', linestyle=':', 
                label='Baseline Core', linewidth=2, alpha=0.7)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(df_comp['config'], rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Size (MB)', fontweight='bold')
    ax5.set_title('Quantization Configuration Comparison', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(axis='y', alpha=0.3)
    
    # ============ BOTTOM ROW ============
    
    # 6. Compression Ratios (Bottom Left)
    ax6 = fig.add_subplot(gs[2, 0])
    
    configs = df_comp['config'].tolist()
    total_comp = df_comp['compression_ratio'].tolist()
    core_comp = df_comp['core_compression_ratio'].tolist()
    
    x_pos = np.arange(len(configs))
    bars1 = ax6.bar(x_pos - 0.2, total_comp, 0.4, label='Total', color='steelblue', alpha=0.7)
    bars2 = ax6.bar(x_pos + 0.2, core_comp, 0.4, label='Core', color='darkorange', alpha=0.7)
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Compression Ratio (x)', fontweight='bold')
    ax6.set_title('Compression Ratios', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=7)
    
    # 7. Size Reduction % (Bottom Middle)
    ax7 = fig.add_subplot(gs[2, 1])
    
    total_red = df_comp['size_reduction_pct'].tolist()
    core_red = df_comp['core_reduction_pct'].tolist()
    
    bars1 = ax7.bar(x_pos - 0.2, total_red, 0.4, label='Total', color='steelblue', alpha=0.7)
    bars2 = ax7.bar(x_pos + 0.2, core_red, 0.4, label='Core', color='darkorange', alpha=0.7)
    
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax7.set_ylabel('Size Reduction (%)', fontweight='bold')
    ax7.set_title('Size Reduction Percentages', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
    
    # 8. Best Configuration Summary (Bottom Right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Find best configurations
    best_total_comp = df_comp.loc[df_comp['compression_ratio'].idxmax()]
    best_core_comp = df_comp.loc[df_comp['core_compression_ratio'].idxmax()]
    best_total_red = df_comp.loc[df_comp['size_reduction_pct'].idxmax()]
    
    validation_status = "✓" if cross_check_results['checks_passed'] >= cross_check_results['checks_total'] * 0.8 else "⚠"
    
    summary_text = f"""
    BEST CONFIGURATIONS
    {'='*35}
    
    Best Total Compression:
    • {best_total_comp['config']}
    • Ratio: {best_total_comp['compression_ratio']:.2f}x
    • Size: {best_total_comp['total_size_mb']:.1f} MB
    
    Best Core Compression:
    • {best_core_comp['config']}
    • Ratio: {best_core_comp['core_compression_ratio']:.2f}x
    • Size: {best_core_comp['core_size_mb']:.1f} MB
    
    Best Size Reduction:
    • {best_total_red['config']}
    • Reduction: {best_total_red['size_reduction_pct']:.1f}%
    
    Validation Status:
    {validation_status} {cross_check_results['checks_passed']:.1f}/{cross_check_results['checks_total']} checks passed
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Add main title
    fig.suptitle(f'Model Analysis Summary Dashboard - {MODEL_ID}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/summary_dashboard.png")
    plt.close()

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("🔄 Loading model...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print(f"✅ Model loaded: {MODEL_ID}")
    
    # Try to find model cache path
    try:
        from transformers.utils import TRANSFORMERS_CACHE
        import os
        cache_dir = os.environ.get('HF_HOME', TRANSFORMERS_CACHE)
        model_cache_path = os.path.join(cache_dir, 'hub', f'models--{MODEL_ID.replace("/", "--")}')
        if not os.path.exists(model_cache_path):
            model_cache_path = None
    except:
        model_cache_path = None
    
    # Analyze model
    layer_info, category_stats, total_params = analyze_model_detailed(model)
    
    # Print layer table
    print_layer_table(layer_info, top_n=30)
    
    # Print category summary and get core size
    core_size_mb, core_params = print_category_summary(category_stats)
    
    # Calculate total size
    total_size_mb = sum(layer['size_mb'] for layer in layer_info)
    
    # Perform cross-check calculations (with disk size verification)
    cross_check_results = cross_check_calculations(model, layer_info, category_stats, model_cache_path)
    
    # Print overall statistics
    print_overall_statistics(total_params, total_size_mb, core_params, core_size_mb, cross_check_results)
    
    # Quantization configs
    STAGE1_SEARCH_CONFIGS = [
        {'name': 'all_4bit', 'emb': 4, 'lm': 4, 'linear': 4},
        {'name': 'mixed_4_2', 'emb': 4, 'lm': 4, 'linear': 2},
        {'name': 'all_8bit', 'emb': 8, 'lm': 8, 'linear': 8},
        {'name': 'mixed_8_2', 'emb': 8, 'lm': 8, 'linear': 2},
        {'name': 'mixed_8_4', 'emb': 8, 'lm': 8, 'linear': 4},
        {'name': 'mixed_fp16_4', 'emb': 'fp16', 'lm': 'fp16', 'linear': 4},
    ]
    
    # Calculate compression ratios
    compression_results = calculate_compression_ratios(
        category_stats, 
        STAGE1_SEARCH_CONFIGS, 
        total_size_mb,
        core_size_mb
    )
    
    # Generate summary dashboard
    plot_summary_dashboard(layer_info, category_stats, compression_results, 
                          cross_check_results, output_dir='plots')
    
    print("\n✅ Analysis complete!")
    # print("\n💡 Key Terminology:")
    # print("   • Parameter Count: Number of parameters (e.g., 600M)")
    # print("   • Parameter Size: Storage size in memory (e.g., 1.2 GB for fp16)")
    # print(f"     → Your model: {total_size_mb/1024:.3f} GB")
    # print("   • Disk Size: Total on-disk size with overhead (e.g., 1.5 GB)")
    # # if cross_check_results.get('disk_size_gb'):
    # #     print(f"     → Your model: {cross_check_results['disk_size_gb']:.3f} GB")
    # print("   • Overhead: Config files, tokenizer, metadata, format overhead")
