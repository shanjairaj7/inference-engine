"""Qwen2 weights + paged KV cache, executed by the fused C++ forward (csrc/kernels.cpp)."""
import json
import os
import sys

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
_ext = None


def kernels():
    global _ext
    if _ext is None:
        # ninja ships inside the venv; make sure the JIT build can find it
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        _ext = load(name="ie_kernels", sources=[os.path.join(_here, "csrc", "kernels.cpp")],
                    extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
                    extra_ldflags=["-fopenmp"], verbose=False)
    return _ext


class Qwen2:
    def __init__(self, path: str, num_blocks: int, block_size: int, max_pos: int = 8192):
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg = cfg
        self.nh, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.D = cfg["hidden_size"] // self.nh
        self.eps = cfg["rms_norm_eps"]
        self.block_size = block_size
        sf = safe_open(os.path.join(path, "model.safetensors"), "pt")   # stream tensors, low peak RAM
        g = lambda k: sf.get_tensor(f"model.{k}").float().contiguous()
        self.embed = g("embed_tokens.weight")
        self.final_norm = g("norm.weight")
        self.weights = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"layers.{i}."
            wqkv = torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"]).contiguous()
            bqkv = torch.cat([g(p + f"self_attn.{n}_proj.bias") for n in "qkv"]).contiguous()
            wgu = torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]).contiguous()
            self.weights += [g(p + "input_layernorm.weight"), wqkv, bqkv, g(p + "self_attn.o_proj.weight"),
                             g(p + "post_attention_layernorm.weight"), wgu, g(p + "mlp.down_proj.weight")]
        del sf
        L = cfg["num_hidden_layers"]
        shape = (num_blocks, block_size, self.nkv, self.D)
        self.k_cache = [torch.zeros(shape) for _ in range(L)]
        self.v_cache = [torch.zeros(shape) for _ in range(L)]
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.D, 2, dtype=torch.float32) / self.D))
        f = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
        self.cos_sin = torch.cat([f.cos(), f.sin()], dim=1).contiguous()
        self.ext = kernels()

    @torch.inference_mode()
    def forward(self, batch):
        """batch: [(seq, n_new)]. Returns logits for sequences whose last token is computed
        this step, and the batch indices they belong to."""
        bs = self.block_size
        ids, pos, slots, info, logit_rows, logit_seqs = [], [], [], [], [], []
        max_blocks = max(len(s.block_table) for s, _ in batch)
        btab = torch.zeros((len(batch), max_blocks), dtype=torch.int32)
        for b, (seq, n) in enumerate(batch):
            start = seq.num_computed
            info.append((len(ids), n, start + n))
            ids += seq.token_ids[start:start + n]
            r = range(start, start + n)
            pos += r
            bt = seq.block_table
            slots += [bt[p // bs] * bs + p % bs for p in r]
            btab[b, :len(bt)] = torch.tensor(bt, dtype=torch.int32)
            if start + n == len(seq.token_ids):
                logit_rows.append(len(ids) - 1)
                logit_seqs.append(b)
        logits = self.ext.forward(
            torch.tensor(ids), torch.tensor(pos), torch.tensor(slots), torch.tensor(info, dtype=torch.int32),
            btab, torch.tensor(logit_rows, dtype=torch.long), self.embed, self.weights, self.final_norm,
            self.k_cache, self.v_cache, self.cos_sin, self.nh, self.nkv, self.D, bs, self.eps)
        return logits, logit_seqs
