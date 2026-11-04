"""LLMEngine: the step loop (schedule -> one batched forward -> sample -> update)
plus an asyncio front-end that streams tokens for many concurrent requests."""
import asyncio
import json
import os
import threading

import torch

from .block_manager import BlockManager
from .model import Qwen2
from .scheduler import SamplingParams, Scheduler, Sequence


class LLMEngine:
    def __init__(self, model_path: str, num_blocks: int = 2048, block_size: int = 16,
                 max_num_seqs: int = 64, max_batched_tokens: int = 512, threads: int | None = None,
                 quant: str = "none"):
        if threads:
            torch.set_num_threads(threads)
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        gen = json.load(open(os.path.join(model_path, "generation_config.json")))
        eos = gen.get("eos_token_id", cfg["eos_token_id"])
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.model = Qwen2(model_path, num_blocks, block_size, quant=quant)
        self.blocks = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.blocks, max_num_seqs, max_batched_tokens, eos_id=-1)
        self.scheduler.is_eos = lambda t: t in self.eos_ids
        self.stats = {"steps": 0, "prompt_tokens": 0, "cached_tokens": 0, "gen_tokens": 0}

    def add_request(self, prompt_ids, params: SamplingParams | None = None, on_token=None) -> Sequence:
        seq = Sequence(list(prompt_ids), params or SamplingParams(), on_token=on_token)
        self.scheduler.add(seq)
        return seq

    def _sample(self, logits, seqs):
        temps = torch.tensor([s.params.temperature for s in seqs])
        greedy = logits.argmax(-1)
        if bool((temps > 0).any()):
            probs = torch.softmax(logits / temps.clamp(min=1e-5)[:, None], -1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            greedy = torch.where(temps > 0, sampled, greedy)
        return greedy.tolist()

    def step(self):
        batch = self.scheduler.schedule()
        if not batch:
            return
        logits, rows = self.model.forward(batch)
        toks = self._sample(logits, [batch[i][0] for i in rows])
        next_tokens = [None] * len(batch)
        for i, t in zip(rows, toks):
            next_tokens[i] = t
        for seq, n in batch:
            if seq.num_computed < seq.num_prompt:
                self.stats["prompt_tokens"] += n
        self.stats["steps"] += 1
        self.stats["gen_tokens"] += len(toks)
        self.scheduler.postprocess(batch, next_tokens)

    def generate(self, prompts, params: SamplingParams | None = None):
        seqs = [self.add_request(p, params) for p in prompts]
        self.stats["cached_tokens"] = 0
        while self.scheduler.has_work():
            self.step()
        self.stats["cached_tokens"] = sum(s.num_cached for s in seqs)
        return [s.output_ids for s in seqs]


class AsyncLLMEngine:
    """Many coroutines call `generate`; a single background thread runs the engine loop and
    batches whatever is in flight. Tokens are streamed back through per-request queues."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._lock = threading.Lock()
        self._wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            self._wake.wait()
            with self._lock:
                if not self.engine.scheduler.has_work():
                    self._wake.clear()
                    continue
                self.engine.step()

    async def generate(self, prompt_ids, params: SamplingParams | None = None):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        cb = lambda seq, tok, done: loop.call_soon_threadsafe(q.put_nowait, (tok, done))
        with self._lock:
            self.engine.add_request(prompt_ids, params, on_token=cb)
        self._wake.set()
        while True:
            tok, done = await q.get()
            yield tok
            if done:
                return
