"""Greedy outputs of the engine must match HF transformers token-for-token
(also exercises chunked prefill, prefix-cache hits, and preemption)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.llm import LLMEngine
from engine.scheduler import SamplingParams
from workload import make_prompts, chat_ids

M = os.environ.get("MODEL", "/home/ubuntu/models/qwen2.5-0.5b")
GEN = 24
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(M)
ids = chat_ids(tok, make_prompts(6))
ids.append(ids[0] + tok.encode(" And also mention who invented it."))  # shares a long prefix with ids[0]
ref_model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
refs = []
with torch.inference_mode():
    for p in ids:
        out = ref_model.generate(torch.tensor([p]), max_new_tokens=GEN, do_sample=False, repetition_penalty=1.0)
        refs.append(out[0, len(p):].tolist())
del ref_model


def check(name, **kw):
    eng = LLMEngine(M, **kw)
    outs = eng.generate(ids, SamplingParams(max_tokens=GEN))
    ok = 0
    for i, (o, r) in enumerate(zip(outs, refs)):
        n = len(r)
        match = o[:n] == r
        ok += match
        if not match:
            first = next((j for j in range(n) if j >= len(o) or o[j] != r[j]), n)
            print(f"  [{name}] seq {i} diverges at token {first}: {tok.decode(o)!r} vs {tok.decode(r)!r}")
    print(f"{name}: {ok}/{len(refs)} exact matches, stats={eng.stats}")
    del eng
    gc.collect()
    return ok


total = 0
total += check("default", num_blocks=512, max_batched_tokens=512)
total += check("chunked-prefill", num_blocks=512, max_batched_tokens=24)
total += check("tight-memory(preemption)", num_blocks=12, block_size=16, max_batched_tokens=128)
print("ALL", total, "/", 3 * len(refs))
