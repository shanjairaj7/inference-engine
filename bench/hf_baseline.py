"""Baseline: HF transformers model.generate (static batching, left padding, greedy)."""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from workload import make_prompts, chat_ids

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/ubuntu/models/qwen2.5-0.5b")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--dtype", default="float32")
a = ap.parse_args()
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
ids = chat_ids(tok, make_prompts(a.n))
print("prompt lens", min(map(len, ids)), max(map(len, ids)))
with torch.inference_mode():
    model.generate(torch.tensor([ids[0]]), max_new_tokens=4, do_sample=False)  # warmup
    t0 = time.perf_counter(); total = 0
    for i in range(0, a.n, a.batch):
        chunk = ids[i:i + a.batch]
        enc = tok.pad({"input_ids": chunk}, return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=a.gen, min_new_tokens=a.gen, do_sample=False, repetition_penalty=1.0)
        total += (out.shape[1] - enc["input_ids"].shape[1]) * len(chunk)
    dt = time.perf_counter() - t0
print(f"HF {a.dtype} n={a.n} batch={a.batch} gen={a.gen}: {total} tokens in {dt:.1f}s -> {total/dt:.2f} tok/s")
