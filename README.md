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
