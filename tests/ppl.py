"""Perplexity with llama.cpp's llama-perplexity protocol (n_ctx chunks, score the 2nd half),
so numbers are directly comparable with `llama-perplexity -f <file> -c <ctx>`."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from transformers import AutoTokenizer
from engine.llm import LLMEngine
from engine.scheduler import SamplingParams, Sequence

M = os.environ.get("MODEL", "/home/ubuntu/models/qwen2.5-0.5b")
path, ctx, quant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
max_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
tok = AutoTokenizer.from_pretrained(M)
ids = tok.encode(open(path).read())
eng = LLMEngine(M, num_blocks=ctx // 16 + 2, max_batched_tokens=ctx, quant=quant)
nll, cnt = 0.0, 0
bos = tok.convert_tokens_to_ids("<|endoftext|>")
n_chunks = min(len(ids) // ctx, max_chunks)
for c in range(n_chunks):
    chunk = [bos] + ids[c * ctx + 1:(c + 1) * ctx]   # llama-perplexity replaces the first token with BOS
    seq = Sequence(chunk, SamplingParams())
    eng.blocks.allocate_prompt(seq, len(chunk))
    logits, _ = eng.model.forward([(seq, len(chunk))], all_logits=True)
    eng.blocks.release(seq)
    lp = torch.log_softmax(logits[:-1], -1)
    tgt = torch.tensor(chunk[1:])
    first = ctx // 2 - 1
    nll -= lp[first:].gather(1, tgt[first:, None]).sum().item()
    cnt += len(tgt) - first
print(f"engine[{quant}] {os.environ.get('IE_FP32_LINEARS','')} alpha={os.environ.get('IE_SQ_ALPHA','0.65')} chunks={n_chunks} ppl={math.exp(nll/cnt):.4f}")
