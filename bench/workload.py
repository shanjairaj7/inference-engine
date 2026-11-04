"""Shared benchmark workload so every engine runs the exact same requests."""
import random

TOPICS = [
    "the history of the printing press", "how transformers process tokens", "the water cycle",
    "why the sky is blue", "the rules of chess", "how vaccines train the immune system",
    "the life cycle of a star", "how a CPU executes instructions", "the causes of World War I",
    "how compound interest works", "the basics of photosynthesis", "how airplanes generate lift",
    "what a hash table is", "the plot of Hamlet", "how GPS determines location", "the theory of evolution",
]


def make_prompts(n: int, seed: int = 0):
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        t = TOPICS[i % len(TOPICS)]
        extra = " ".join(rng.choice(["Be detailed.", "Use simple words.", "Give examples.", "Explain step by step."]) for _ in range(rng.randint(1, 6)))
        prompts.append(f"Write a long explanation of {t}. {extra}")
    return prompts


def chat_ids(tok, prompts):
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        out.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
    # transformers>=5 may return BatchEncoding
    return [list(x["input_ids"]) if hasattr(x, "keys") else list(x) for x in out]
