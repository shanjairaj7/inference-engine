"""Quality check: perplexity on a fixed text + greedy agreement with fp32, for each engine mode."""
import gc, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.llm import LLMEngine
from engine.scheduler import SamplingParams, Sequence
from workload import make_prompts, chat_ids

M = os.environ.get("MODEL", "/home/ubuntu/models/qwen2.5-0.5b")
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(M)
text = open(os.path.join(M, "LICENSE")).read()[:6000]
ids = tok.encode(text)[:1024]
tgt = torch.tensor(ids[1:])

with torch.inference_mode():
    hf = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    lp = torch.log_softmax(hf(torch.tensor([ids])).logits[0, :-1].float(), -1)
    print(f"HF fp32       ppl={math.exp(-lp.gather(1, tgt[:, None]).mean().item()):.4f}")
    del hf; gc.collect()

prompts = chat_ids(tok, make_prompts(16))
outs = {}
for quant in os.environ.get("MODES", "none,int8").split(","):
    eng = LLMEngine(M, num_blocks=1024, max_batched_tokens=2048, quant=quant)
    seq = Sequence(list(ids), SamplingParams())
    eng.blocks.allocate_prompt(seq, len(ids))
    logits, _ = eng.model.forward([(seq, len(ids))], all_logits=True)
    lp = torch.log_softmax(logits[:-1], -1)
    ppl = math.exp(-lp.gather(1, tgt[:, None]).mean().item())
    eng.blocks.release(seq)
    outs[quant] = eng.generate(prompts, SamplingParams(max_tokens=64, ignore_eos=True))
    base = outs.get("none", outs[quant])
    agree = sum(a == b for x, y in zip(outs[quant], base) for a, b in zip(x, y)) / sum(map(len, outs[quant]))
    # tokens before first divergence, averaged over prompts
    prefix = sum(next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), len(x)) for x, y in zip(outs[quant], base)) / len(prompts)
    print(f"engine {quant:5s} ppl={ppl:.4f}  greedy-match-vs-fp32={agree:.1%}  avg-identical-prefix={prefix:.1f}/64")
    del eng; gc.collect()
print("sample int8:", tok.decode(outs["int8"][0]))
