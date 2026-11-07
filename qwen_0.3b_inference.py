import torch, torch.nn as nn, torch.nn.functional as functional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.requests import Request as FastAPIRequest
from uuid import uuid4

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
    def forward(self, input_ids, cache):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError('Expected one nonempty sequence: [1, new_tokens]')
        num_new_tokens = input_ids.shape[1]
        start_position, end_position = cache.num_cached_tokens, cache.num_cached_tokens + num_new_tokens
        if end_position > cache.max_cached_tokens:
            raise ValueError('Prompt/output exceeds the KV cache capacity')

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

            key, value = cache.write_and_read(layer_index, key, value)

            if causal_mask is not None and query.shape[1] != key.shape[1]:
                num_query_heads_per_kv_head = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(num_query_heads_per_kv_head, dim=1)
                value = value.repeat_interleave(num_query_heads_per_kv_head, dim=1)

            print(f"key: {key}")
            print(f"key shape: {key.shape}")

            attention_output = functional.scaled_dot_product_attention(
                query, key, value, attn_mask=causal_mask,
                is_causal=start_position == 0 and num_new_tokens > 1,
                scale=attention.scaling, enable_gqa=query.shape[1] != key.shape[1],
            )

            attention_output = attention_output.transpose(1, 2).reshape(1, num_new_tokens, -1)
            hidden_states = residual + attention.o_proj(attention_output)
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

        logits = self.loaded_model.lm_head(transformer.norm(hidden_states[:, -1:, :]))

        cache.num_cached_tokens = end_position
        return logits

@dataclass
class Request:
    request_id: str
    prompt: str
    max_new_tokens: str # how many tokens to generate fully

    block_ids: list[int] = field(default_factory=list)

    num_computed_tokens: int = 0
    num_cached_tokens: int = field(default=0)
    num_new_tokens: int = 0

class KVCache:
    def __init__(self,cache_shape,input_ids,model_config,request:Request,block_size=4):
        self.cache_shape = cache_shape
        self.block_size = block_size
        self.input_ids = input_ids
        self.model_config = model_config
        self.current_request: Request = request
        self.requests = {}

        self.block_table = [i for i in range(int(model_config.max_position_embeddings / 4))]
        self.free_blocks = self.block_table.copy()

        self.keys = torch.empty(self.cache_shape,device=device,dtype=torch.bfloat16)
        self.values = torch.empty_like(self.keys)

        self.max_cached_tokens = len(input_ids) + model_config.max_position_embeddings
 
    @property
    def num_cached_tokens(self):
        print(f"self.requests: {self.requests}")
        return self.requests[self.current_request.request_id].num_cached_tokens if len(self.requests.items()) > 0 else 0

    @num_cached_tokens.setter
    def num_cached_tokens(self,value):
        self.requests[self.current_request.request_id].num_cached_tokens = value

    def write_and_read(self,layer_index,key,value):
        logical_block = key.shape[2] // self.block_size
        physical_block = self.block_table[logical_block]
        slot = key.shape[2] % self.block_size

        # start_position = self.num_cached_tokens
        # end_position = key.shape[2] if self.num_cached_tokens == 0 else (self.num_cached_tokens+key.shape[2])

        print(f"key shape: {key.shape}")
        print(f"self.keys shape: {self.keys.shape}")
        print(f"self.input ids shape: {self.input_ids.shape}")
        print(f"key: {key}")
        num_tokens = key.shape[2]

        for token_i in range(key.shape[2]):
            print(f"num cached tokens: {self.num_cached_tokens}. token i: {token_i}")
            logical_block = (self.num_cached_tokens + token_i) // self.block_size
            physical_block = self.block_table[logical_block]
            slot = (self.num_cached_tokens + token_i) % self.block_size

            self.keys[layer_index,:,:,physical_block,slot,:].copy_(key[:,:,token_i])
            self.values[layer_index,:,:,physical_block,slot,:].copy_(value[:,:,token_i])
            
            contains_block = self.current_request.block_ids.__contains__(physical_block)
            if not contains_block: self.current_request.block_ids.append(physical_block)
            self.current_request.num_cached_tokens = self.num_cached_tokens
            self.current_request.num_computed_tokens = self.input_ids.shape[1]
            # self.current_request.num_cached_tokens += 1

        # self.keys[layer_index,:,:,physical_block,slot,:].copy_(key)
        # self.values[layer_index,:,:,physical_block,slot,:].copy_(value)

        # self.num_cached_tokens = end_position : num_cached_tokens updated during forward pass


        self.requests[self.current_request.request_id] = self.current_request

        # self.num_cached_tokens = key.shape[2]

        all_keys = [self.keys[layer_index,:,:,block_id,:] for block_id in self.current_request.block_ids]
        all_values = [self.values[layer_index,:,:,block_id,:] for block_id in self.current_request.block_ids]

        return torch.cat(all_keys, dim=2), torch.cat(all_values, dim=2)

    def get_requests(self):
        return self.requests

def main(request: Request):
    model_name = 'Qwen/Qwen3-0.6B'
    block_size=4

    print(request)

    request_id = request.request_id
    user_message = request.prompt

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    model = QwenModel(qwen_model)
    model_config = model.config
    max_new_tokens = model_config.max_position_embeddings

    input_ids = tokenizer.apply_chat_template(
        [{'role':'user','content':user_message}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        tokenize=True
    )['input_ids'].to(device)

    print(f"model config: {model_config}")

    # [28, 1, 8, 1024, 4, 128]
    model_shape = (
        model_config.num_hidden_layers,
        1,
        model_config.num_key_value_heads,
        int(model_config.max_position_embeddings / block_size),
        block_size,
        model_config.head_dim
    )

    eos_token_id = qwen_model.generation_config.eos_token_id

    cache = KVCache(model_shape,input_ids,model_config,request,block_size)
    generation_output = []

    for i in range(max_new_tokens):
        result = model(input_ids,cache)
        next_token_id = result[:,-1].argmax().item()
        print(f"result: {result}")

        if next_token_id in eos_token_id:
            break

        input_ids = torch.tensor([[next_token_id]],device=device)
        generation_output.append(next_token_id)

    response = tokenizer.decode(generation_output,skip_special_tensors=True)
    print(f"response: {response}")

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
    main(request)

    return {'message': message}
