# inference-engine

Basic inference engine logic using arrays and simulated GPU blocks for learning.

Each incoming set of tensors/tokens are divided into set of tokens to be put into blocks of the GPU allocated for the KV cache
With each set of tokens having 4 tokens (in this example), if the tokens are filled they are hashed and assigned to a GPU block as the pre-fill cache
For the final set of tokens that need to be decoded (either it fills the full 4 tokens or partially), they are run using the `run` function and assigned a GPU block. This simulates the decode step of actual transformer models.
As this final set of tokens gets filled to the 4 token limit they are hashed and cached. They are not hashed and therefore cached until they fully fill a GPU block. So that its useful for future requests to make use of full tokens, rather than partial ones.
This process keeps going until the max completion tokens limit is reached and the `[EOS]` token is returned by the LLM.
Once the LLM returns `EOS` meaning its done generating the completions, the GPU blocks that were used are put back into the free GPU blocks list.
The GPU blocks are allocated to be used in a reverse order. For example if the usage of Gpu blocks was [0,1,2,3] the free gpu blocks through which the code finds and allocates tokens to the available GPUs will reverse the order to be `[3,2,1,0]`. This is for 2 reasons:
  - Each GPU block tokens are hashed with the previous hash as the key. Think of it as a tree and each token block in the set of tokens are connected and allocated to GPU blocks and therefore hashed with every previous hash value so that the previous tokens can be mapped. This means that using the first GPU block `0` in the `[0,1,2,3]` if not reversed, would take out the first set of tokens from the tree and disconnect the next order of tokens which are supposed to be connected to have the proper meaning of the tokens.
  - Also so that the most latest block of tokens are taken out and the initial token blocks are kept to allow for future requests to hit the tree of connections which will be preserved and there is a possible set of tokens that could be the generated tokens for incoming requests that hit cache of the first token blocks. For example if blocks are removed from the bottom, and a new request with the exact same tokens as `2` comes in, then the `[2,1,0]` can be used as possible next generation tokens that might be generated.
The current code handles one request. Ideally this simulation should simulate async calls, background processes running as workers (each request) and process multiple requests, do continuous batching, make new requests wait until there are available GPU blocks, pre-fill tokens on available GPU blocks even if there is no space for decoding.
I'll be updating this with these additions. Writing a actual inference engine that runs the latest open source models.

## CPU engine (`engine/`)

A real engine for Qwen2.5-0.5B on CPU, built on the ideas above:

- `engine/block_manager.py`: paged KV cache with fixed-size blocks, ref counts, a chained-hash prefix cache (`hash(parent_hash, block_tokens)`), only full blocks get cached, and blocks are freed tail-first so shared prefixes survive the longest.
- `engine/scheduler.py`: continuous batching. Each step decodes one token for every running sequence and fills the leftover token budget with chunked prefill. Requests wait when there aren't enough free blocks; under memory pressure the newest sequence is preempted (blocks released, recomputed later).
- `engine/csrc/kernels.cpp`: the whole decoder stack runs in one native call per step ("megakernel" style): RMSNorm, RoPE + KV write, GQA paged attention over block tables, fused SiLU*up, and Fbgemm AVX2 int8 GEMMs with fused dequant/bias epilogues.
- `engine/calibrate.py`: SmoothQuant. Activation maxima are collected once on WikiText-2 train and the scales are folded into RMSNorm / preceding weights, so int8 adds no runtime cost.
- `engine/llm.py`: `LLMEngine` (sync) and `AsyncLLMEngine` (one background loop, a per-request `asyncio.Queue`).

### Results (2 vCPU Xeon with AVX2, 7 GB RAM, no GPU)

Throughput: 32 requests × 128 greedy tokens, chat-templated prompts (`bench/`):

| Engine | tok/s |
|---|---|
| HF transformers fp32 (static batch 32) | 82.7 |
| llama.cpp fp16 | 77.6 |
| llama.cpp Q8_0 | 111.7 |
| this engine fp32 | 98–111 |
| this engine int8 (no SmoothQuant) | 244 |
| **this engine int8 + SmoothQuant (default)** | **279** |

Single request, 128 tokens: llama.cpp Q8_0 26.1 tok/s, this engine int8 48.0 tok/s.

Quality: WikiText-2 test perplexity, ctx 512, first 20 chunks (`tests/ppl.py`, same protocol as `llama-perplexity`):

| Model | PPL |
|---|---|
| this engine fp32 | 16.39 |
| llama.cpp fp16 | 16.55 |
| this engine int8 + SmoothQuant α=0.65 | 16.56 |
| llama.cpp Q8_0 | 16.65 |
| this engine int8, no SmoothQuant | 17.10 |

fp32 greedy output matches HF transformers token-for-token (`tests/test_correctness.py`: 21/21 across default, chunked-prefill and preemption configs).

### Run

```bash
pip install torch transformers safetensors ninja   # plus python3-dev; Fbgemm headers ship with torch
IE_CALIB_FILE=wiki.train.raw python bench/engine_bench.py --quant int8   # calibration is cached in <model>/act_max.pt
python tests/test_correctness.py
```
