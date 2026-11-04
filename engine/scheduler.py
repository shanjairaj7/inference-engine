"""Continuous-batching scheduler with chunked prefill and preemption-by-recompute."""
import itertools
from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager

_ids = itertools.count()


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0
    ignore_eos: bool = False


@dataclass
class Sequence:
    token_ids: list
    params: SamplingParams
    seq_id: int = field(default_factory=lambda: next(_ids))
    num_prompt: int = 0
    num_computed: int = 0       # tokens whose KV is in the cache
    num_cached: int = 0         # prompt tokens served from the prefix cache
    num_hashed_blocks: int = 0
    block_table: list = field(default_factory=list)
    finished: bool = False
    on_token: object = None     # callback(seq, token, finished)

    def __post_init__(self):
        self.num_prompt = len(self.token_ids)

    @property
    def output_ids(self):
        return self.token_ids[self.num_prompt:]


class Scheduler:
    def __init__(self, blocks: BlockManager, max_num_seqs: int, max_batched_tokens: int, eos_id: int):
        self.blocks = blocks
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.eos_id = eos_id
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.is_eos = lambda t: t == eos_id

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence):
        self.blocks.release(seq)
        seq.num_computed = seq.num_hashed_blocks = 0
        self.running.remove(seq)
        self.waiting.appendleft(seq)

    def schedule(self):
        """Returns [(seq, num_new_tokens)] for one forward pass."""
        budget, batch = self.max_batched_tokens, []
        # 1) running sequences first (decode = 1 token, or the next chunk of a long prefill)
        for seq in list(self.running):
            if seq not in self.running:
                continue
            n = min(len(seq.token_ids) - seq.num_computed, budget)
            if n <= 0:
                break
            while not self.blocks.ensure_capacity(seq, seq.num_computed + n):
                victim = self.running[-1]          # newest request loses its blocks
                self._preempt(victim)
                if victim is seq:
                    break
            if seq in self.running:
                batch.append((seq, n))
                budget -= n
        # 2) admit waiting requests (FCFS) while memory, seq slots and token budget allow
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            n = self.blocks.allocate_prompt(seq, budget)
            if n == 0:
                break
            self.waiting.popleft()
            self.running.append(seq)
            batch.append((seq, n))
            budget -= n
        return batch

    def postprocess(self, batch, next_tokens):
        """next_tokens[i] is the sampled token for batch[i] if that seq produced logits, else None."""
        for (seq, n), tok in zip(batch, next_tokens):
            seq.num_computed += n
            self.blocks.commit_full_blocks(seq)
            if tok is None:
                continue
            seq.token_ids.append(tok)
            done = (len(seq.output_ids) >= seq.params.max_tokens
                    or (not seq.params.ignore_eos and self.is_eos(tok)))
            if done:
                seq.finished = True
                self.blocks.release(seq)
                self.running.remove(seq)
            if seq.on_token:
                seq.on_token(seq, tok, done)
