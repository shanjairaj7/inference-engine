"""Paged KV-cache block allocator with hash-chained prefix caching.

Each full block of tokens is identified by hash((parent_hash, tokens)), so a hit on block i
implies blocks 0..i-1 matched too (the tree described in the README). Blocks are ref-counted:
requests sharing a prefix share the physical blocks. A block whose ref count drops to 0 goes
back on the free list but keeps its hash, so later requests can still hit it until the
allocator actually reuses it (LRU eviction). Blocks are released tail-first, so the tail of a
finished request is evicted before its (more shareable) prefix.
"""
from collections import OrderedDict


def block_hash(parent: int, tokens) -> int:
    return hash((parent, tuple(tokens)))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.ref = [0] * num_blocks
        self.hash_of = [None] * num_blocks
        self.cached: dict[int, int] = {}                # hash -> block id
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free)

    def _take_fresh(self) -> int:
        bid, _ = self.free.popitem(last=False)          # least recently freed
        if self.hash_of[bid] is not None:
            self.cached.pop(self.hash_of[bid], None)
            self.hash_of[bid] = None
        self.ref[bid] = 1
        return bid

    def _prefix_hits(self, tokens):
        """Block ids of the longest cached prefix of full blocks."""
        bs, parent, hits = self.block_size, 0, []
        for i in range(len(tokens) // bs):
            h = block_hash(parent, tokens[i * bs:(i + 1) * bs])
            bid = self.cached.get(h)
            if bid is None:
                break
            hits.append(bid)
            parent = h
        return hits

    def allocate_prompt(self, seq, budget: int) -> int:
        """Attach cached prefix blocks and allocate fresh blocks for the first prefill chunk
        (at most `budget` tokens). Returns the chunk size, or 0 (changing nothing) if memory
        is short."""
        hits = self._prefix_hits(seq.token_ids)
        # always recompute at least the last prompt token so we get logits for it
        while hits and len(hits) * self.block_size >= len(seq.token_ids):
            hits.pop()
        cached = len(hits) * self.block_size
        n = min(len(seq.token_ids) - cached, budget)
        need_fresh = -(-(cached + n) // self.block_size) - len(hits)
        if need_fresh + sum(1 for b in hits if self.ref[b] == 0) > self.num_free:
            return 0
        for b in hits:
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1
        seq.block_table = hits + [self._take_fresh() for _ in range(need_fresh)]
        seq.num_computed = seq.num_cached = cached
        seq.num_hashed_blocks = len(hits)
        return n

    def ensure_capacity(self, seq, upto_tokens: int) -> bool:
        need = -(-upto_tokens // self.block_size) - len(seq.block_table)
        if need > self.num_free:
            return False
        seq.block_table += [self._take_fresh() for _ in range(max(need, 0))]
        return True

    def commit_full_blocks(self, seq):
        """Register hashes for blocks whose KV has been fully computed."""
        bs = self.block_size
        n_full = seq.num_computed // bs
        parent = self.hash_of[seq.block_table[seq.num_hashed_blocks - 1]] if seq.num_hashed_blocks else 0
        for i in range(seq.num_hashed_blocks, n_full):
            h = block_hash(parent, seq.token_ids[i * bs:(i + 1) * bs])
            bid = seq.block_table[i]
            if h not in self.cached and self.hash_of[bid] is None:
                self.cached[h] = bid
                self.hash_of[bid] = h
            parent = h
        seq.num_hashed_blocks = max(seq.num_hashed_blocks, n_full)

    def release(self, seq):
        for bid in reversed(seq.block_table):
            self.ref[bid] -= 1
            if self.ref[bid] == 0:
                self.free[bid] = None
        seq.block_table = []
