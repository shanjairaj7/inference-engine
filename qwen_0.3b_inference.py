import torch, torch.nn as nn, torch.nn.functional as functional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.requests import Request as FastAPIRequest
from uuid import uuid4
from datetime import datetime

device = torch.device('cpu')

def apply_rope(query, key, cosine, sine):
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)

    def rotate(hidden_states):
        first_half, second_half = hidden_states.chunk(2, dim=-1)
        return hidden_states * cosine + torch.cat((-second_half, first_half), dim=-1) * sine

    return rotate(query), rotate(key)

class QwenModel(nn.Module):
    def __init__(self, loaded_model):
        super().__init__()
        self.validate_config(loaded_model.config)
        self.loaded_model = loaded_model
        self.config = loaded_model.config

    @staticmethod
    def validate_config(config):
        if config.model_type != 'qwen3':
            raise ValueError('Use a dense Qwen3 checkpoint, e.g. Qwen/Qwen3-0.6B')
        if any(kind != 'full_attention' for kind in config.layer_types):
            raise ValueError('This basic runner supports full attention layers only')

    @classmethod
    def load(cls, model_name, device, dtype):
        cls.validate_config(AutoConfig.from_pretrained(model_name))
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map={'': str(device)},
            attn_implementation='sdpa', use_safetensors=True,
        ).eval()
        return cls(loaded_model).eval()

    @torch.inference_mode()
    def forward(self, input_ids, cache, request):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError('Expected one nonempty sequence: [1, new_tokens]')
        num_new_tokens = input_ids.shape[1]
        req = cache.get_request(request.request_id)
        start_position, end_position = req.num_cached_tokens, req.num_cached_tokens + num_new_tokens
        if end_position > cache.max_cached_tokens:
            raise ValueError('Prompt/output exceeds the KV cache capacity')

        # Keep the cached prefix plus this pass's input, once before all layers.
        # Slicing also avoids duplicating the prompt or a retried input.
        req.input_ids = torch.cat((req.input_ids[:, :start_position], input_ids), dim=1)

        transformer = self.loaded_model.model
        hidden_states = transformer.embed_tokens(input_ids)
        position_ids = torch.arange(start_position, end_position, device=hidden_states.device).unsqueeze(0)
        cosine, sine = transformer.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        if start_position > 0 and num_new_tokens > 1:
            causal_mask = torch.arange(end_position, device=hidden_states.device)[None, :] <= position_ids[0, :, None]
            
        for layer_index, layer in enumerate(transformer.layers):
            residual = hidden_states
            normalized_states = layer.input_layernorm(hidden_states)
            attention = layer.self_attn

            head_shape = (1, num_new_tokens, -1, attention.head_dim)
            query = attention.q_norm(attention.q_proj(normalized_states).view(head_shape)).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(normalized_states).view(head_shape)).transpose(1, 2)
            value = attention.v_proj(normalized_states).view(head_shape).transpose(1, 2)

            query, key = apply_rope(query, key, cosine, sine)

            key, value = cache.write_and_read(layer_index, key, value, request)

            if causal_mask is not None and query.shape[1] != key.shape[1]:
                num_query_heads_per_kv_head = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(num_query_heads_per_kv_head, dim=1)
                value = value.repeat_interleave(num_query_heads_per_kv_head, dim=1)

            # print(f"key: {key}")
            # print(f"key shape: {key.shape}")

            attention_output = functional.scaled_dot_product_attention(
                query, key, value, attn_mask=causal_mask,
                is_causal=start_position == 0 and num_new_tokens > 1,
                scale=attention.scaling, enable_gqa=query.shape[1] != key.shape[1],
            )

            attention_output = attention_output.transpose(1, 2).reshape(1, num_new_tokens, -1)
            hidden_states = residual + attention.o_proj(attention_output)
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

        logits = self.loaded_model.lm_head(transformer.norm(hidden_states[:, -1:, :]))

        cache.requests[request.request_id].num_cached_tokens = end_position
        cache.publish_full_blocks(request)
        return logits

@dataclass
class Request:
    request_id: str
    prompt: str
    max_new_tokens: str # how many tokens to generate fully

    input_ids: list[int] = field(default_factory=list)

    block_ids: list[int] = field(default_factory=list)
    shared_logical_blocks: set[int] = field(default_factory=set)

    num_computed_tokens: int = 0
    num_cached_tokens: int = field(default=0)
    num_new_tokens: int = 0

class KVCache:
    def __init__(self,cache_shape,model_config,block_size=4):
        self.cache_shape = cache_shape
        self.block_size = block_size
        self.model_config = model_config
        self.requests = {}

        self.free_blocks = [i for i in range(int(model_config.max_position_embeddings / self.block_size))]
        self.hash_to_blocks = {}
        self.block_reference_count = {block:0 for block in self.free_blocks}

        self.keys = torch.empty(self.cache_shape,device=device,dtype=torch.bfloat16)
        self.values = torch.empty_like(self.keys)

        self.max_cached_tokens = model_config.max_position_embeddings

    def init_request(self,input_ids, request:Request):
        self.requests[request.request_id] = request
        self.requests[request.request_id].input_ids = input_ids

    def free_request_cache(self,request: Request):
        # [2,9,4,5]
        req = self.requests[request.request_id]
        request_block_reference_count = [self.block_reference_count[block] for block in req.block_ids]
        # [3,1,1,2]
        for i, block_reference in enumerate(request_block_reference_count):
            block = req.block_ids[i]
            if block_reference > 1: 
                self.block_reference_count[block] -= 1
            else:
                self.block_reference_count[block] = 0
                self.free_blocks.append(req.block_ids[i])
        del self.requests[request.request_id]


    def write_and_read(self,layer_index,key,value, request: Request):
        current_request = request
        original_request = self.requests[request.request_id]

        # logical_block = key.shape[2] // self.block_size
        # physical_block = self.block_table[logical_block]
        # slot = key.shape[2] % self.block_size

        # start_position = self.num_cached_tokens
        # end_position = key.shape[2] if self.num_cached_tokens == 0 else (self.num_cached_tokens+key.shape[2])

        # print(f"key shape: {key.shape}")
        # print(f"self.keys shape: {self.keys.shape}")
        # print(f"self.input ids shape: {self.input_ids.shape}")
        # print(f"key: {key}")
        num_tokens = key.shape[2]

        tokens = self.requests[request.request_id].input_ids[0].tolist()
        
        print(f"request hashes: {self.hash_to_blocks}")
        
        previous_hash = None
        total_tokens = len(tokens)
        for start in range(0, total_tokens, self.block_size):
            logical_block = start // self.block_size
            block_tokens = tokens[start:start+block_size]

            partial_block = len(block_tokens) < self.block_size
            
            block_hash = hash((previous_hash, tuple(block_tokens)))
            same_hash_block = self.hash_to_blocks.get(block_hash, None)

            if logical_block >= len(original_request.block_ids):
                if same_hash_block is not None and not partial_block:
                    self.requests[request.request_id].block_ids.append(same_hash_block)
                    if same_hash_block in self.free_blocks: self.free_blocks.remove(same_hash_block)
                    self.requests[request.request_id].shared_logical_blocks.add(logical_block)
                    self.block_reference_count[same_hash_block] += 1
                else:
                    chosen_free_block = self.free_blocks[0]
                    for hash_,hash_block in self.hash_to_blocks.copy().items():
                        if hash_block == chosen_free_block:
                            del self.hash_to_blocks[hash_]
                    self.requests[request.request_id].block_ids.append(self.free_blocks.pop(0)) 
                    self.block_reference_count[original_request.block_ids[-1]] += 1

            previous_hash = block_hash

        for token_i in range(key.shape[2]):
            physical_block = None
            # is there a block id already given for this logical block? yes?
            logical_block = (original_request.num_cached_tokens + token_i) // self.block_size # 1
            physical_block = self.requests[request.request_id].block_ids[logical_block]
            slot = (original_request.num_cached_tokens + token_i) % self.block_size

            if logical_block not in original_request.shared_logical_blocks:
                self.keys[layer_index,:,:,physical_block,slot,:].copy_(key[:,:,token_i])
                self.values[layer_index,:,:,physical_block,slot,:].copy_(value[:,:,token_i])
            
            contains_block = self.requests[request.request_id].block_ids.__contains__(physical_block)
            if not contains_block: 
                self.block_reference_count[physical_block] += 1
                current_request.block_ids.append(physical_block)
            current_request.num_cached_tokens = original_request.num_cached_tokens
            current_request.num_computed_tokens = self.requests[request.request_id].input_ids.shape[1]
            # self.current_request.num_cached_tokens += 1

        # self.keys[layer_index,:,:,physical_block,slot,:].copy_(key)
        # self.values[layer_index,:,:,physical_block,slot,:].copy_(value)

        # self.num_cached_tokens = end_position : num_cached_tokens updated during forward pass

        self.requests[current_request.request_id] = current_request

        end = original_request.num_cached_tokens + key.shape[2]

        all_keys = [self.keys[layer_index,:,:,block_id,:] for block_id in current_request.block_ids]
        all_values = [self.values[layer_index,:,:,block_id,:] for block_id in current_request.block_ids]

        return torch.cat(all_keys, dim=2)[:,:,:end,:], torch.cat(all_values, dim=2)[:,:,:end,:]

    def publish_full_blocks(self, request: Request):
        # Only completed tokens have valid keys/values across every layer.
        tokens = request.input_ids[0, :request.num_cached_tokens].tolist()
        previous_hash = None
        full_token_count = len(tokens) - len(tokens) % self.block_size
        for start in range(0, full_token_count, self.block_size):
            block_tokens = tokens[start:start + self.block_size]
            block_hash = hash((previous_hash, tuple(block_tokens)))
            logical_block = start // self.block_size
            # Another request may already have completed the same prefix.
            self.hash_to_blocks.setdefault(block_hash, request.block_ids[logical_block])
            previous_hash = block_hash

    def get_request(self, request_id):
        return self.requests[request_id]

model_name = 'Qwen/Qwen3-0.6B'
block_size=4

model_load_start_time = datetime.now()
tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_load_end_time = datetime.now()
print(f"model {model_name} loaded in : {(model_load_end_time - model_load_start_time).total_seconds():.2f}s\n")

model = QwenModel(qwen_model)
model_config = model.config
max_new_tokens = model_config.max_position_embeddings

# [28, 1, 8, 1024, 4, 128]
model_shape = (
    model_config.num_hidden_layers,
    1,
    model_config.num_key_value_heads,
    int(model_config.max_position_embeddings / block_size),
    block_size,
    model_config.head_dim
)

cache = KVCache(model_shape,model_config,block_size)

def main(request: Request):
    # print(request)
    print(f"requests: {cache.requests}")

    request_id = request.request_id
    user_message = request.prompt

    input_ids = tokenizer.apply_chat_template(
        [{'role':'user','content':user_message}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        tokenize=True
    )['input_ids'].to(device)

    cache.init_request(input_ids, request)
    # print(f"model config: {model_config}")

    eos_token_id = qwen_model.generation_config.eos_token_id

    generation_output = []
    generation_start_time = datetime.now()

    for i in range(max_new_tokens):
        result = model(input_ids,cache, request)
        next_token_id = result[:,-1].argmax().item()
        # print(f"result: {result}")

        if next_token_id in eos_token_id:
            break

        input_ids = torch.tensor([[next_token_id]],device=device)
        generation_output.append(next_token_id)

        decoded_token = tokenizer.decode([next_token_id], skip_special_tensors=True)
        print(decoded_token, end="", flush=True)

    generation_end_time = datetime.now()
    generation_time = generation_end_time - generation_start_time
    tokens_per_second = (len(generation_output) / generation_time.total_seconds())
    print(f"\n\ntokens per second: {tokens_per_second:.2f} tokens/s")

    response = tokenizer.decode(generation_output,skip_special_tensors=True)
    print(f"response: {response}")
    return response, tokens_per_second

# main()

app = FastAPI()



@app.post('/')
def home(message: dict):
    print(f"message is here")
    print(message)

    #     prompt: str
    # max_new_tokens: str # how many tokens to generate fully

    # block_ids: list[int] = []

    # num_computed_tokens: int = 0
    # num_cached_tokens: int = 0
    # num_new_tokens: int = 0

    request = Request(
        request_id=str(uuid4()), 
        prompt=message['prompt']['text'], 
        max_new_tokens=message['prompt']['new_tokens'])
    message, tokens_per_second = main(request)
    
    cache.free_request_cache(request)

    return {'message': message, 'tokens_per_second': f"{tokens_per_second:.2f}"}
