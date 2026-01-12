import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # 🔥 NEW
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset
import gc
from tqdm import tqdm
import json
from datetime import datetime
import os
import copy

from quantization_utils import (
    set_seed, cleanup_memory, load_fresh_model,
    get_model_stats_detailed as get_model_stats,  # Use detailed version
    verify_quantization, evaluate_accuracy,
    quantize_tensor_fake,
    MODEL_ID, DEVICE
)
# ==========================================
# ⚙️ Configuration
# ==========================================
SEED = 42
set_seed(SEED)
# QAT Hyperparameters - ASYMMETRIC QUANTIZATION
QAT_N_BITS = 4
QAT_GROUP_SIZE = 128
QAT_SYMMETRIC = False  # 🔥 ASYMMETRIC
QAT_NUM_EPOCHS = 1
QAT_BATCH_SIZE = 1
QAT_LEARNING_RATE = 5e-5
QAT_NUM_TRAIN_SAMPLES = 4096
QAT_MAX_SEQ_LENGTH = 512

# Evaluation settings
EVAL_BATCH_SIZE = 1
EVAL_LIMIT = 0.1

# 🔥 Checkpoint settings
CHECKPOINT_DIR = "qat_checkpoints"
SAVE_BEST_MODEL = True
SAVE_EVERY_N_EPOCHS = 2

# 🔥 TensorBoard settings
TENSORBOARD_DIR = "runs/qat_training"
LOG_WEIGHT_DISTRIBUTIONS = True  # Log weight histograms
LOG_GRADIENT_FLOW = True  # Log gradient statistics
LOG_EVERY_N_STEPS = 1000  # Log training metrics every N steps





# ==========================================
# 📊 Statistics & Verification
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
# 🔥 Fake Quantization Module - ASYMMETRIC
# ==========================================
class FakeQuantize(nn.Module):
    """
    Fake quantization with ASYMMETRIC support
    """
    def __init__(self, n_bits=4, group_size=128, symmetric=False):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
        
        if symmetric:
            self.max_int = 2 ** (n_bits - 1) - 1
            self.min_int = -(2 ** (n_bits - 1))
        else:
            self.max_int = 2 ** n_bits - 1
            self.min_int = 0
        
    def forward(self, x):
        if not self.training:
            return self.quantize_dequantize(x)
        else:
            return self.quantize_dequantize_ste(x)
    
    def quantize_dequantize(self, x):
        original_shape = x.shape
        original_dtype = x.dtype
        
        x_flat = x.flatten().float()
        n_elements = x_flat.numel()
        
        if n_elements % self.group_size != 0:
            pad_size = self.group_size - (n_elements % self.group_size)
            x_flat = F.pad(x_flat, (0, pad_size), value=0)
        
        x_groups = x_flat.reshape(-1, self.group_size)
        n_groups = x_groups.shape[0]
        
        quantized_groups = torch.zeros_like(x_groups)
        
        for i in range(n_groups):
            group = x_groups[i]
            
            if self.symmetric:
                max_val = group.abs().max()
                if max_val < 1e-8:
                    quantized_groups[i] = group
                    continue
                
                scale = max_val / self.max_int
                quantized_int = torch.clamp(
                    torch.round(group / scale),
                    self.min_int, self.max_int
                )
                quantized_groups[i] = quantized_int * scale
            else:
                min_val = group.min()
                max_val = group.max()
                
                if (max_val - min_val) < 1e-8:
                    quantized_groups[i] = group
                    continue
                
                scale = (max_val - min_val) / (self.max_int - self.min_int)
                zero_point = self.min_int - torch.round(min_val / scale)
                zero_point = torch.clamp(zero_point, self.min_int, self.max_int)
                
                quantized_int = torch.clamp(
                    torch.round(group / scale + zero_point),
                    self.min_int, self.max_int
                )
                quantized_groups[i] = (quantized_int - zero_point) * scale
        
        result = quantized_groups.flatten()[:n_elements].reshape(original_shape)
        return result.to(original_dtype)
    
    def quantize_dequantize_ste(self, x):
        x_quant = self.quantize_dequantize(x)
        return x + (x_quant - x).detach()


# ==========================================
# 🔥 QAT-aware Linear Layer
# ==========================================
class QATLinear(nn.Module):
    """
    Linear layer with fake quantization for QAT
    """
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
            linear_layer.in_features,
            linear_layer.out_features,
            bias=linear_layer.bias is not None,
            n_bits=n_bits,
            group_size=group_size,
            symmetric=symmetric
        )
        qat_linear.linear.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            qat_linear.linear.bias.data = linear_layer.bias.data.clone()
        return qat_linear


# ==========================================
# 🔥 Convert Model to QAT
# ==========================================
def convert_to_qat(model, n_bits=4, group_size=128, symmetric=False, skip_lm_head=True):
    quant_type = "Symmetric" if symmetric else "Asymmetric"
    print(f"   [QAT] Converting model to QAT mode ({n_bits}-bit, group-{group_size}, {quant_type})...")
    
    converted_count = 0
    module_list = list(model.named_modules())
    
    for name, module in module_list:
        if isinstance(module, nn.Linear):
            if skip_lm_head and "lm_head" in name:
                continue
            if "embed" in name.lower():
                continue
            
            if '.' in name:
                *parent_names, attr_name = name.split('.')
                parent = model
                for pname in parent_names:
                    parent = getattr(parent, pname)
            else:
                parent = model
                attr_name = name
            
            qat_layer = QATLinear.from_linear(
                module, 
                n_bits=n_bits, 
                group_size=group_size,
                symmetric=symmetric
            )
            setattr(parent, attr_name, qat_layer)
            converted_count += 1
    
    print(f"   [QAT] Converted {converted_count} Linear layers to QATLinear")
    return model


# ==========================================
# 🔥 Finalize QAT
# ==========================================
def finalize_qat(model):
    """
    After QAT training, convert QATLinear back to regular Linear
    """
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
                module.linear.in_features,
                module.linear.out_features,
                bias=module.linear.bias is not None
            )
            
            with torch.no_grad():
                new_linear.weight.data = module.weight_quantizer.quantize_dequantize(
                    module.linear.weight.data
                )
                if module.linear.bias is not None:
                    new_linear.bias.data = module.linear.bias.data.clone()
            
            new_linear.quant_bit_width = module.n_bits
            
            setattr(parent, attr_name, new_linear)
            finalized_count += 1
    
    print(f"   [QAT] Finalized {finalized_count} layers")
    return model

def temporarily_finalize_for_eval(model):
    """
    Temporarily finalize model for fast evaluation
    """
    print("   [Eval] Temporarily finalizing model for fast evaluation...")
    model_eval = finalize_qat(model)
    return model_eval


# ==========================================
# 🔥 PTQ Quantization - ASYMMETRIC
# ==========================================
def quantize_tensor_asymmetric(weight, n_bits=4, group_size=128):
    """
    ASYMMETRIC quantization with per-group granularity
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    
    weight = weight.float()
    weight_flat = weight.flatten()
    n_elements = weight_flat.numel()
    
    if n_elements % group_size != 0:
        pad_size = group_size - (n_elements % group_size)
        weight_flat = torch.nn.functional.pad(weight_flat, (0, pad_size), value=0)
    
    weight_groups = weight_flat.reshape(-1, group_size)
    n_groups = weight_groups.shape[0]
    
    quantized_groups = torch.zeros_like(weight_groups)
    
    max_int = 2 ** n_bits - 1
    min_int = 0
    
    for i in range(n_groups):
        group = weight_groups[i]
        
        min_val = group.min()
        max_val = group.max()
        
        if (max_val - min_val) < 1e-8:
            quantized_groups[i] = group
            continue
        
        scale = (max_val - min_val) / (max_int - min_int)
        zero_point = min_int - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, min_int, max_int)
        
        quantized_int = torch.clamp(
            torch.round(group / scale + zero_point),
            min_int, max_int
        )
        
        quantized_groups[i] = (quantized_int - zero_point) * scale
    
    result = quantized_groups.flatten()[:n_elements].reshape(original_shape)
    
    return result.to(original_dtype)

def quantize_tensor_symmetric(weight, n_bits=4, group_size=128):
    """
    Symmetric quantization
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    
    weight = weight.float()
    weight_flat = weight.flatten()
    n_elements = weight_flat.numel()
    
    if n_elements % group_size != 0:
        pad_size = group_size - (n_elements % group_size)
        weight_flat = torch.nn.functional.pad(weight_flat, (0, pad_size), value=0)
    
    weight_groups = weight_flat.reshape(-1, group_size)
    n_groups = weight_groups.shape[0]
    
    quantized_groups = torch.zeros_like(weight_groups)
    
    max_int = 2 ** (n_bits - 1) - 1
    
    for i in range(n_groups):
        group = weight_groups[i]
        
        max_val = group.abs().max()
        if max_val < 1e-8:
            quantized_groups[i] = group
            continue
        
        scale = max_val / max_int
        
        quantized_int = torch.clamp(
            torch.round(group / scale),
            -max_int, max_int
        )
        
        quantized_groups[i] = quantized_int * scale
    
    result = quantized_groups.flatten()[:n_elements].reshape(original_shape)
    
    return result.to(original_dtype)

def apply_ptq_quantization(model, n_bits=4, group_size=128, symmetric=False):
    """Apply PTQ quantization to model"""
    quant_type = "Symmetric" if symmetric else "Asymmetric"
    print(f"   [PTQ] Applying {n_bits}-bit quantization (Group-{group_size}, {quant_type})...")
    
    quantized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name and "embed" not in name.lower():
            with torch.no_grad():
                if symmetric:
                    quantized = quantize_tensor_symmetric(
                        module.weight.data, 
                        n_bits=n_bits, 
                        group_size=group_size
                    )
                else:
                    quantized = quantize_tensor_asymmetric(
                        module.weight.data, 
                        n_bits=n_bits, 
                        group_size=group_size
                    )
                module.weight.data = quantized
                module.quant_bit_width = n_bits
                quantized_count += 1
    
    print(f"   [PTQ] Quantized {quantized_count} layers")
    return model


# ==========================================
# 🔥 CHECKPOINT MANAGEMENT
# ==========================================
def save_qat_checkpoint(model, tokenizer, epoch, accuracy, loss, metadata, checkpoint_dir, is_best=False):
    """
    Save QAT model checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_best:
        checkpoint_name = f"best_model_epoch{epoch}_acc{accuracy:.4f}.pt"
    else:
        checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Save model state dict and metadata
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'loss': loss,
        'metadata': metadata,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save metadata as JSON for easy inspection
    metadata_path = os.path.join(checkpoint_dir, checkpoint_name.replace('.pt', '_metadata.json'))
    with open(metadata_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'accuracy': accuracy,
            'loss': loss,
            'metadata': metadata,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"   [Checkpoint] Saved: {checkpoint_path}")
    
    return checkpoint_path

def load_qat_checkpoint(checkpoint_path, device="cuda"):
    """
    Load QAT model from checkpoint
    """
    print(f"\n   [Checkpoint] Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    metadata = checkpoint['metadata']
    model_id = metadata['model_id']
    
    print(f"   [Checkpoint] Model: {model_id}")
    print(f"   [Checkpoint] Epoch: {checkpoint['epoch']}")
    print(f"   [Checkpoint] Accuracy: {checkpoint['accuracy']:.4f}")
    print(f"   [Checkpoint] Quantization: {metadata['n_bits']}-bit, Group-{metadata['group_size']}")
    
    # Load fresh base model
    print(f"   [Checkpoint] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load state dict
    print(f"   [Checkpoint] Loading weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"   [Checkpoint] ✅ Model loaded successfully!")
    
    return model, tokenizer, checkpoint

def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint in the directory
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    best_checkpoint = None
    best_accuracy = -1
    
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("best_model") and filename.endswith(".pt"):
            # Extract accuracy from filename
            try:
                acc_str = filename.split("_acc")[1].split(".pt")[0]
                accuracy = float(acc_str)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_checkpoint = os.path.join(checkpoint_dir, filename)
            except:
                continue
    
    return best_checkpoint


# ==========================================
# 🔥 QAT Training Dataset
# ==========================================
class QATDataset(Dataset):
    """
    Dataset for QAT fine-tuning
    """
    def __init__(self, tokenizer, max_length=512, num_samples=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"   [QAT] Loading training data...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        for sample in tqdm(dataset, desc="Preparing QAT dataset"):
            text = sample['text']
            if len(text) > 100:
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                )
                self.samples.append({
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0)
                })
                
                if len(self.samples) >= num_samples:
                    break
        
        print(f"   [QAT] Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 🔥 TENSORBOARD LOGGING UTILITIES
# ==========================================
def log_weight_distributions(writer, model, global_step):
    """
    Log weight distributions to TensorBoard
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QATLinear)):
            if isinstance(module, QATLinear):
                weights = module.linear.weight.data
                tag_prefix = f"weights_qat/{name}"
            else:
                weights = module.weight.data
                tag_prefix = f"weights/{name}"
            
            # Log histogram
            writer.add_histogram(f"{tag_prefix}/distribution", weights, global_step)
            
            # Log statistics
            writer.add_scalar(f"{tag_prefix}/mean", weights.mean().item(), global_step)
            writer.add_scalar(f"{tag_prefix}/std", weights.std().item(), global_step)
            writer.add_scalar(f"{tag_prefix}/min", weights.min().item(), global_step)
            writer.add_scalar(f"{tag_prefix}/max", weights.max().item(), global_step)

def log_gradient_flow(writer, model, global_step):
    """
    Log gradient statistics to TensorBoard
    """
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            # Log per-layer gradient norms
            writer.add_scalar(f"gradients/{name}/norm", param_norm, global_step)
            
            # Log gradient histograms for key layers
            if "weight" in name and ("attn" in name or "mlp" in name):
                writer.add_histogram(f"gradients/{name}/distribution", param.grad.data, global_step)
    
    total_norm = total_norm ** 0.5
    writer.add_scalar("gradients/total_norm", total_norm, global_step)

def log_quantization_error(writer, model, global_step):
    """
    Log quantization error for QAT layers
    """
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            with torch.no_grad():
                original_weight = module.linear.weight.data
                quantized_weight = module.weight_quantizer.quantize_dequantize(original_weight)
                
                # Calculate quantization error
                error = (original_weight - quantized_weight).abs()
                
                writer.add_scalar(f"quantization_error/{name}/mean", error.mean().item(), global_step)
                writer.add_scalar(f"quantization_error/{name}/max", error.max().item(), global_step)
                writer.add_histogram(f"quantization_error/{name}/distribution", error, global_step)


# ==========================================
# 🔥 QAT Training Function with TensorBoard
# ==========================================
def run_qat_training_with_eval(
    model,
    tokenizer,
    n_bits=4,
    group_size=128,
    symmetric=False,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    num_train_samples=1000,
    max_seq_length=512,
    device="cuda",
    checkpoint_dir="qat_checkpoints",
    save_best=True,
    save_every_n_epochs=10,
    tensorboard_dir="runs/qat_training",  # 🔥 NEW
    log_every_n_steps=50  # 🔥 NEW
):
    """
    Perform QAT training with TensorBoard logging
    """
    quant_type = "Symmetric" if symmetric else "Asymmetric"
    print("\n" + "="*80)
    print(f"=== QAT PREPARATION & TRAINING ({n_bits}-bit, Group-{group_size}, {quant_type}) ===")
    print("="*80)
    
    # 🔥 Create TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{quant_type.lower()}_{n_bits}bit_g{group_size}_{timestamp}"
    writer = SummaryWriter(os.path.join(tensorboard_dir, run_name))
    
    print(f"\n   📊 TensorBoard logging to: {os.path.join(tensorboard_dir, run_name)}")
    print(f"   💡 To view: tensorboard --logdir={tensorboard_dir}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Convert model to QAT mode
    model = convert_to_qat(model, n_bits=n_bits, group_size=group_size, symmetric=symmetric)
    model.to(device)
    
    # Metadata for checkpoints
    metadata = {
        'model_id': MODEL_ID,
        'n_bits': n_bits,
        'group_size': group_size,
        'symmetric': symmetric,
        'quantization_type': quant_type,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_train_samples': num_train_samples
    }
    
    # 🔥 Log hyperparameters to TensorBoard
    writer.add_text("hyperparameters", json.dumps(metadata, indent=2), 0)
    
    # ========================================
    # EVALUATE BEFORE TRAINING
    # ========================================
    print("\n" + "-"*80)
    print("PRE-TRAINING EVALUATION (After QAT Preparation)")
    print("-"*80)
    
    model_eval = temporarily_finalize_for_eval(model)
    acc_before_training = evaluate_accuracy(model_eval, tokenizer, desc="QAT Pre-Training")
    stats_before = get_model_stats(model_eval)
    
    # 🔥 Log pre-training metrics
    writer.add_scalar("accuracy/pre_training", acc_before_training, 0)
    writer.add_scalar("model_size/mb", stats_before['quantized_mb'], 0)
    
    del model_eval
    cleanup_memory()
    
    print(f"\n   📊 Pre-Training Results:")
    print(f"      - Accuracy: {acc_before_training:.4f}")
    print(f"      - Model Size: {stats_before['quantized_mb']:.2f} MB")
    print("-"*80)
    
    # Prepare dataset
    train_dataset = QATDataset(
        tokenizer, 
        max_length=max_seq_length, 
        num_samples=num_train_samples
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # 🔥 Log model graph (first batch)
    try:
        dummy_batch = next(iter(train_loader))
        dummy_input = dummy_batch['input_ids'][:1].to(device)
        writer.add_graph(model, dummy_input)
        print("   ✅ Model graph logged to TensorBoard")
    except Exception as e:
        print(f"   ⚠️  Could not log model graph: {e}")
    
    # Training loop
    print(f"\n   [QAT] Starting training for {num_epochs} epochs...")
    print(f"   [QAT] Checkpoints will be saved to: {checkpoint_dir}")
    
    epoch_results = []
    best_accuracy = acc_before_training
    best_epoch = 0
    global_step = 0
    
    # Add pre-training result
    epoch_results.append({
        'epoch': 0,
        'loss': None,
        'accuracy': acc_before_training,
        'is_pre_training': True
    })
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 🔥 Log training metrics
            if global_step % log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / (batch_idx + 1)
                
                writer.add_scalar("training/loss", loss.item(), global_step)
                writer.add_scalar("training/avg_loss", avg_loss, global_step)
                writer.add_scalar("training/learning_rate", current_lr, global_step)
                
                # Log gradient flow
                if LOG_GRADIENT_FLOW:
                    log_gradient_flow(writer, model, global_step)
                
                # Log quantization error
                log_quantization_error(writer, model, global_step)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # 🔥 Log weight distributions at end of epoch
        if LOG_WEIGHT_DISTRIBUTIONS:
            log_weight_distributions(writer, model, global_step)
        
        # Evaluation phase
        print(f"\n   [Epoch {epoch+1}] Training Loss: {avg_epoch_loss:.4f}")
        print(f"   [Epoch {epoch+1}] Evaluating model accuracy...")
        
        model_eval = temporarily_finalize_for_eval(model)
        epoch_acc = evaluate_accuracy(model_eval, tokenizer, desc=f"Epoch {epoch+1}")
        
        # 🔥 Log epoch metrics
        writer.add_scalar("epoch/loss", avg_epoch_loss, epoch + 1)
        writer.add_scalar("epoch/accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("epoch/accuracy_improvement", epoch_acc - acc_before_training, epoch + 1)
        
        # 🔥 Save checkpoint if this is the best model
        is_best = epoch_acc > best_accuracy
        if is_best:
            print(f"\n   🎉 New best accuracy: {epoch_acc:.4f} (previous: {best_accuracy:.4f})")
            best_accuracy = epoch_acc
            best_epoch = epoch + 1
            
            # 🔥 Log best model metrics
            writer.add_scalar("best/accuracy", best_accuracy, epoch + 1)
            writer.add_scalar("best/epoch", best_epoch, epoch + 1)
            
            if save_best:
                save_qat_checkpoint(
                    model_eval, 
                    tokenizer, 
                    epoch + 1, 
                    epoch_acc, 
                    avg_epoch_loss, 
                    metadata, 
                    checkpoint_dir, 
                    is_best=True
                )
        
        # 🔥 Save periodic checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            save_qat_checkpoint(
                model_eval, 
                tokenizer, 
                epoch + 1, 
                epoch_acc, 
                avg_epoch_loss, 
                metadata, 
                checkpoint_dir, 
                is_best=False
            )
        
        del model_eval
        cleanup_memory()
        
        epoch_results.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'accuracy': epoch_acc,
            'is_pre_training': False,
            'is_best': is_best
        })
        
        print(f"   [Epoch {epoch+1}] ✅ Accuracy: {epoch_acc:.4f}")
    
    # 🔥 Create accuracy comparison chart
    epochs = [r['epoch'] for r in epoch_results]
    accuracies = [r['accuracy'] for r in epoch_results]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, accuracies, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=acc_before_training, color='r', linestyle='--', label='Pre-training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'QAT Training Progress ({quant_type}, {n_bits}-bit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    writer.add_figure("training/accuracy_progress", fig, num_epochs)
    plt.close(fig)
    
    print("\n" + "="*80)
    print("QAT TRAINING COMPLETE")
    print(f"   🏆 Best Accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")
    print("="*80)
    
    # Finalize QAT
    model = finalize_qat(model)
    model.eval()
    
    # 🔥 Save final model
    save_qat_checkpoint(
        model, 
        tokenizer, 
        num_epochs, 
        epoch_results[-1]['accuracy'], 
        epoch_results[-1]['loss'], 
        metadata, 
        checkpoint_dir, 
        is_best=False
    )
    
    # 🔥 Close TensorBoard writer
    writer.close()
    print(f"\n   ✅ TensorBoard logs saved to: {os.path.join(tensorboard_dir, run_name)}")
    
    return model, epoch_results, best_accuracy, best_epoch


# ==========================================
# 🚀 MAIN EXPERIMENT
# ==========================================
if __name__ == "__main__":
    results = {}
    start_time = datetime.now()
    
    quant_type = "Symmetric" if QAT_SYMMETRIC else "Asymmetric"
    
    print("\n" + "="*80)
    print("=== QAT QUANTIZATION EXPERIMENT WITH TENSORBOARD ===")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Quantization: {QAT_N_BITS}-bit, Group-{QAT_GROUP_SIZE}, {quant_type}")
    print(f"Training: {QAT_NUM_EPOCHS} epochs, {QAT_NUM_TRAIN_SAMPLES} samples")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"TensorBoard: {TENSORBOARD_DIR}")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # ========================================
    # STEP 1: Evaluate Original FP16 Model
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: EVALUATE ORIGINAL FP16 MODEL")
    print("="*80)
    
    model_original, tokenizer = load_fresh_model()
    
    acc_original = evaluate_accuracy(model_original, tokenizer, desc="Original FP16")
    stats_original = get_model_stats(model_original)
    
    results['original'] = {
        'accuracy': acc_original,
        'size_mb': stats_original['total_mb'],
        'quantized_mb': stats_original['quantized_mb'],
        'avg_bits': stats_original['avg_bits']
    }
    
    print(f"\n   ✅ Original Model Results:")
    print(f"      - Accuracy: {acc_original:.4f}")
    print(f"      - Model Size: {stats_original['total_mb']:.2f} MB")
    
    # ========================================
    # STEP 2: Apply PTQ Quantization
    # ========================================
    print("\n" + "="*80)
    print(f"STEP 2: APPLY PTQ QUANTIZATION (BASELINE - {quant_type})")
    print("="*80)
    
    model_ptq, _ = load_fresh_model()
    model_ptq = apply_ptq_quantization(
        model_ptq, 
        n_bits=QAT_N_BITS, 
        group_size=QAT_GROUP_SIZE,
        symmetric=QAT_SYMMETRIC
    )
    
    verify_quantization(model_ptq, model_original)
    
    acc_ptq = evaluate_accuracy(model_ptq, tokenizer, desc=f"PTQ Quantized ({quant_type})")
    stats_ptq = get_model_stats(model_ptq)
    
    results['ptq'] = {
        'accuracy': acc_ptq,
        'size_mb': stats_ptq['quantized_mb'],
        'ratio': stats_ptq['ratio'],
        'avg_bits': stats_ptq['avg_bits'],
        'degradation': (acc_original - acc_ptq) * 100
    }
    
    print(f"\n   ✅ PTQ Quantized Model Results:")
    print(f"      - Accuracy: {acc_ptq:.4f}")
    print(f"      - Degradation: {results['ptq']['degradation']:.2f}%")
    
    cleanup_memory(model_ptq)
    
    # ========================================
    # STEP 3: QAT Training with TensorBoard
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: QAT TRAINING WITH TENSORBOARD")
    print("="*80)
    
    model_qat, _ = load_fresh_model()
    
    model_qat, epoch_results, best_accuracy, best_epoch = run_qat_training_with_eval(
        model=model_qat,
        tokenizer=tokenizer,
        n_bits=QAT_N_BITS,
        group_size=QAT_GROUP_SIZE,
        symmetric=QAT_SYMMETRIC,
        num_epochs=QAT_NUM_EPOCHS,
        batch_size=QAT_BATCH_SIZE,
        learning_rate=QAT_LEARNING_RATE,
        num_train_samples=QAT_NUM_TRAIN_SAMPLES,
        max_seq_length=QAT_MAX_SEQ_LENGTH,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        save_best=SAVE_BEST_MODEL,
        save_every_n_epochs=SAVE_EVERY_N_EPOCHS,
        tensorboard_dir=TENSORBOARD_DIR,  # 🔥 NEW
        log_every_n_steps=LOG_EVERY_N_STEPS  # 🔥 NEW
    )
    
    results['qat_epochs'] = epoch_results
    results['qat_best'] = {
        'accuracy': best_accuracy,
        'epoch': best_epoch,
        'degradation': (acc_original - best_accuracy) * 100
    }
    
    # ========================================
    # STEP 4: Load and Evaluate Best Model
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: LOAD AND EVALUATE BEST CHECKPOINT")
    print("="*80)
    
    # Find best checkpoint
    best_checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
    
    if best_checkpoint_path:
        print(f"\n   [Info] Found best checkpoint: {best_checkpoint_path}")
        
        # Clean up current model
        cleanup_memory(model_qat)
        
        # Load best checkpoint
        model_best, tokenizer_best, checkpoint_info = load_qat_checkpoint(
            best_checkpoint_path, 
            device=DEVICE
        )
        
        # Evaluate loaded model
        print("\n   [Eval] Evaluating loaded best model...")
        acc_best_loaded = evaluate_accuracy(model_best, tokenizer_best, desc="Best Checkpoint Loaded")
        
        print(f"\n   ✅ Best Checkpoint Evaluation:")
        print(f"      - Checkpoint Accuracy: {checkpoint_info['accuracy']:.4f}")
        print(f"      - Re-evaluated Accuracy: {acc_best_loaded:.4f}")
        print(f"      - Epoch: {checkpoint_info['epoch']}")
        
        results['best_checkpoint_loaded'] = {
            'accuracy': acc_best_loaded,
            'checkpoint_accuracy': checkpoint_info['accuracy'],
            'epoch': checkpoint_info['epoch']
        }
        
        model_qat = model_best
    else:
        print("\n   ⚠️  No best checkpoint found, using final model")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*120)
    print(f"FINAL SUMMARY - QAT QUANTIZATION EXPERIMENT ({quant_type})")
    print("="*120)
    
    print(f"\n{'Stage':<35} | {'Accuracy':<10} | {'Degradation':<12} | {'Size (MB)':<12} | {'Status'}")
    print("-" * 120)
    
    print(f"{'1. Original (FP16)':<35} | {acc_original:.4f}     | {'N/A':<12} | {stats_original['total_mb']:>8.2f}    | ✅ Baseline")
    
    ptq_status = "✅ GOOD" if results['ptq']['degradation'] < 5 else "⚠️  HIGH"
    print(f"{'2. PTQ Quantized':<35} | {acc_ptq:.4f}     | {results['ptq']['degradation']:>6.2f}%     | {stats_ptq['quantized_mb']:>8.2f}    | {ptq_status}")
    
    print(f"{'3. QAT Best (Epoch ' + str(best_epoch) + ')':<35} | {best_accuracy:.4f}     | {results['qat_best']['degradation']:>6.2f}%     | {stats_ptq['quantized_mb']:>8.2f}    | 🏆 BEST")
    
    print("="*120)
    
    # Save results
    results['metadata'] = {
        'model_id': MODEL_ID,
        'n_bits': QAT_N_BITS,
        'group_size': QAT_GROUP_SIZE,
        'symmetric': QAT_SYMMETRIC,
        'quantization_type': quant_type,
        'num_epochs': QAT_NUM_EPOCHS,
        'checkpoint_dir': CHECKPOINT_DIR,
        'tensorboard_dir': TENSORBOARD_DIR,
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': str(duration)
    }
    
    filename = f'qat_experiment_results_{quant_type.lower()}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to '{filename}'")
    print(f"✅ Checkpoints saved to '{CHECKPOINT_DIR}'")
    print(f"✅ TensorBoard logs saved to '{TENSORBOARD_DIR}'")
    print(f"✅ Best model: Epoch {best_epoch}, Accuracy {best_accuracy:.4f}")
    print(f"\n💡 To view TensorBoard: tensorboard --logdir={TENSORBOARD_DIR}")
    print("="*120)
    
    cleanup_memory(model_original)
    cleanup_memory(model_qat)
    
    print("\n✅ QAT Experiment Complete!")
