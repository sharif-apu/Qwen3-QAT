import torch
import torch.nn as nn
import argparse
import json
import os
import gc
from pathlib import Path

import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# ============================================================
# GLOBAL CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCH_SIZE = 1
EVAL_LIMIT = 0.1
DEFAULT_OPSET = 17

# ============================================================
# Disable problematic attention paths (CRITICAL)
# ============================================================
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ============================================================
# Utility
# ============================================================
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def should_quantize_layer(name):
    skip = [
        "lm_head", "embed", "embedding", "wte", "wpe"
    ]
    name = name.lower()
    return not any(s in name for s in skip)

# ============================================================
# Load QAT Checkpoint
# ============================================================
def load_qat_checkpoint(ckpt_path, device):
    """✅ Modified to load your QAT model (8-bit embeddings/lm_head)"""
    print(f"\n📂 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    meta = ckpt["metadata"]
    model_id = meta["model_id"]
    
    # model: embeddings=8bit, lm_head=8bit, linear=original (4-bit)
    emb_bits = meta.get("embedding_bits", 8)
    lm_bits = meta.get("lmhead_bits", 8)
    linear_bits = meta.get("linear_bits", 4)
    group_size = meta.get("group_size", 128)

    print(f"   Model       : {model_id}")
    print(f"   Epoch       : {ckpt.get('epoch', 'N/A')}")
    print(f"   Quant       : Emb={emb_bits}bit, LM={lm_bits}bit, Linear={linear_bits}bit (group {group_size})")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,     # ONNX-safe
        device_map=device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Attach quant metadata
    q, s = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            m.quant_bit_width = emb_bits
            if emb_bits < 16:
                q += 1
            else:
                s += 1
        elif isinstance(m, nn.Linear):
            if "lm_head" in name:
                m.quant_bit_width = lm_bits
                if lm_bits < 16:
                    q += 1
                else:
                    s += 1
            else:
                m.quant_bit_width = linear_bits  # Original precision
                if linear_bits < 16:
                    q += 1
                else:
                    s += 1

    print(f"   Quantized layers : {q}")
    print(f"   Skipped layers   : {s}")

    model.eval()
    return model, tokenizer, ckpt

# ============================================================
# Model Statistics
# ============================================================
def model_stats(model):
    total_params = 0
    total_bits = 0
    quantized = 0

    for m in model.modules():
        if isinstance(m, nn.Embedding):
            n = m.weight.numel()
            total_params += n
            bits = getattr(m, "quant_bit_width", 16)
            total_bits += n * bits
            if bits < 16:
                quantized += n

        elif isinstance(m, nn.Linear):
            n = m.weight.numel()
            total_params += n
            bits = getattr(m, "quant_bit_width", 16)
            total_bits += n * bits
            if bits < 16:
                quantized += n

    size_mb = total_bits / 8 / 1024**2
    orig_mb = total_params * 16 / 8 / 1024**2

    return {
        "size_mb": size_mb,
        "original_mb": orig_mb,
        "compression_ratio": orig_mb / size_mb,
        "avg_bits": total_bits / total_params,
        "quantized_params": quantized,
        "total_params": total_params,
    }

# ============================================================
# Evaluation (Optional)
# ============================================================
def evaluate(model, tokenizer):
    print("\n🔍 Evaluating accuracy (MMLU subset)")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=EVAL_BATCH_SIZE
    )
    res = simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        limit=EVAL_LIMIT,
        device=DEVICE
    )
    acc = res["results"]["mmlu"]["acc,none"]
    del lm
    cleanup()
    return acc

# ============================================================
# ONNX Wrapper
# ============================================================
class CausalLMONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]
        return logits

# ============================================================
# Export to ONNX (IR)
# ============================================================
def export_onnx(model, tokenizer, out_path, seq_len, opset):
    print("\n🔄 Exporting QAT model to ONNX")

    device = next(model.parameters()).device

    model_cpu = model.cpu().float().eval()
    if hasattr(model_cpu.config, "use_cache"):
        model_cpu.config.use_cache = False

    wrapper = CausalLMONNX(model_cpu).eval()

    dummy_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), dtype=torch.long
    )
    dummy_mask = torch.ones_like(dummy_ids)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        out_path,
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        do_constant_folding=True,
    )

    # ✅ Correct validation for large models
    onnx.checker.check_model(out_path)

    size = os.path.getsize(out_path) / 1024**2
    print(f"   ✅ ONNX exported & validated ({size:.2f} MB)")

    model.to(device)
    return size

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=0.1)
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--onnx_seq_len", type=int, default=128)
    parser.add_argument("--onnx_opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    global DEVICE, EVAL_BATCH_SIZE, EVAL_LIMIT
    DEVICE = args.device
    EVAL_BATCH_SIZE = args.batch_size
    EVAL_LIMIT = args.limit

    model, tokenizer, ckpt = load_qat_checkpoint(
        args.checkpoint, DEVICE
    )

    stats = model_stats(model)

    acc = None
    if not args.skip_eval:
        acc = evaluate(model, tokenizer)

    onnx_size = None
    if args.export_onnx:
        out_dir = Path(args.checkpoint).with_suffix("").as_posix() + "_onnx"
        os.makedirs(out_dir, exist_ok=True)
        onnx_path = os.path.join(out_dir, "model.onnx")
        onnx_size = export_onnx(
            model,
            tokenizer,
            onnx_path,
            args.onnx_seq_len,
            args.onnx_opset,
        )

    results = {
        "model": ckpt["metadata"]["model_id"],
        "epoch": ckpt["epoch"],
        "accuracy": acc,
        "stats": stats,
        "onnx_size_mb": onnx_size,
    }

    out_json = args.checkpoint.replace(".pt", "_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done. Results saved to {out_json}")

if __name__ == "__main__":
    main()
