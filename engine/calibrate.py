"""SmoothQuant calibration: per-input-channel max|activation| for every linear layer.

Activation outliers (a few channels with huge magnitude) wreck per-token int8 activation
quantization. SmoothQuant moves that range into the weights: X' = X / s, W' = W * s with
s_k = max|X_k|^a / max|W_k|^(1-a). The 1/s is folded into whatever produces X (the RMSNorm
weight, v_proj, or up_proj), so inference does no extra work."""
import os

import torch


def collect_act_max(model_path: str, text: str, n_tokens: int = 4096, ctx: int = 512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
    ids = tok.encode(text)[:n_tokens]
    stats, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        for name, mod in [("qkv", layer.self_attn.q_proj), ("o", layer.self_attn.o_proj),
                          ("gu", layer.mlp.gate_proj), ("down", layer.mlp.down_proj)]:
            key = f"{i}.{name}"

            def hook(_m, inp, _out, key=key):
                m = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0)
                stats[key] = torch.maximum(stats[key], m) if key in stats else m
            hooks.append(mod.register_forward_hook(hook))
    hooks.append(model.lm_head.register_forward_hook(
        lambda _m, inp, _o: stats.__setitem__("lm", torch.maximum(stats["lm"], inp[0].abs().reshape(-1, inp[0].shape[-1]).amax(0)) if "lm" in stats else inp[0].abs().reshape(-1, inp[0].shape[-1]).amax(0))))
    with torch.inference_mode():
        for s in range(0, len(ids), ctx):
            model(torch.tensor([ids[s:s + ctx]]))
    for h in hooks:
        h.remove()
    del model
    return stats


def load_act_max(model_path: str, calib_file: str | None = None):
    cache = os.path.join(model_path, "act_max.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    calib_file = calib_file or os.environ.get("IE_CALIB_FILE")
    if not calib_file:
        raise FileNotFoundError("int8 needs calibration: set IE_CALIB_FILE to a text file (e.g. wikitext-2 train)")
    stats = collect_act_max(model_path, open(calib_file).read())
    torch.save(stats, cache)
    return stats


def smooth_scale(act_max, w, alpha: float, gqa=None):
    """w: [out, in] weight consuming the activation. gqa=(nkv, head_dim) shares the scale
    across the q heads that read the same v head (needed to fold 1/s into v_proj)."""
    a, wm = act_max.clone(), w.abs().amax(0)
    if gqa:
        nkv, D = gqa
        a = a.view(nkv, -1, D).amax(1, keepdim=True).expand(nkv, a.numel() // (nkv * D), D).reshape(-1)
        wm = wm.view(nkv, -1, D).amax(1, keepdim=True).expand(nkv, wm.numel() // (nkv * D), D).reshape(-1)
    return ((a.clamp(min=1e-5) ** alpha) / (wm.clamp(min=1e-5) ** (1 - alpha))).clamp(min=1e-5)
