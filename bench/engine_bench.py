"""Benchmark this engine on the shared workload (all requests submitted at once)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer
from engine.llm import LLMEngine
from engine.scheduler import SamplingParams
from workload import make_prompts, chat_ids

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/ubuntu/models/qwen2.5-0.5b")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--max-seqs", type=int, default=64)
ap.add_argument("--max-batched", type=int, default=512)
ap.add_argument("--blocks", type=int, default=2048)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--quant", default="none", choices=["none", "int8"])
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
ids = chat_ids(tok, make_prompts(a.n))
eng = LLMEngine(a.model, num_blocks=a.blocks, max_num_seqs=a.max_seqs, max_batched_tokens=a.max_batched, threads=a.threads, quant=a.quant)
eng.generate([ids[0]], SamplingParams(max_tokens=4, ignore_eos=True))  # warmup / JIT
eng.blocks.__init__(a.blocks, eng.blocks.block_size)  # fresh cache: no prefix hits from warmup
t0 = time.perf_counter()
outs = eng.generate(ids, SamplingParams(max_tokens=a.gen, ignore_eos=True))
dt = time.perf_counter() - t0
total = sum(map(len, outs))
print(f"engine[{a.quant}] n={a.n} gen={a.gen} max_seqs={a.max_seqs}: {total} tokens in {dt:.1f}s -> {total/dt:.2f} tok/s  stats={eng.stats}")
