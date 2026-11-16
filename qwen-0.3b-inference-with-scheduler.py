import torch, torch.nn as nn, torch.nn.functional as functional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.requests import Request as FastAPIRequest
from uuid import uuid4
from datetime import datetime
import threading, asyncio, concurrent

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
        print(f"start position: {start_position}")

        # Keep the cached prefix plus this pass's input, once before all layers.
        # Slicing also avoids duplicating the prompt or a retried input.
        req.input_ids = torch.cat((req.input_ids[:, :int(start_position)], input_ids), dim=1)
        print(f"qwen model forward input ids changed: {req.input_ids}")

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
    output_ids: list[int] = field(default_factory=list)

    block_ids: list[int] = field(default_factory=list)
    shared_logical_blocks: set[int] = field(default_factory=set)

    num_computed_tokens: int = 0
    num_cached_tokens: int = field(default=0)
    num_new_tokens: int = 0
    max_new_tokens: int = 0
    is_prefill: bool = True
    is_finished: bool = False

class KVCache:
    def __init__(self,cache_shape,model_config,block_size=4):
        self.cache_shape = cache_shape
        self.block_size = block_size
        self.model_config = model_config
        self.requests = {}

        # 20 is max_num_seqs 
        self.free_blocks = [i for i in range(int((model_config.max_position_embeddings*20) / self.block_size))]
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
        del self.requests[request.request_id], req.block_ids


    def write_and_read(self,layer_index,key,value, request: Request):
        current_request = request
        original_request = self.requests[request.request_id]

        print(f"kv cache write and read layer index: {layer_index}")

        num_tokens = key.shape[2]
        tokens = self.requests[request.request_id].input_ids[0].tolist()
        
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

        self.requests[current_request.request_id] = current_request

        end = original_request.num_cached_tokens + key.shape[2]

        print(f"kv cache write and read. request block ids: {len(current_request.block_ids)}. layer index: {layer_index}.  keys shape: {self.keys.shape}")
        # print(f"kv cache write and read block ids: {current_request.block_ids}")
        all_keys = [self.keys[layer_index,:,:,block_id,:] for block_id in current_request.block_ids]
        all_values = [self.values[layer_index,:,:,block_id,:] for block_id in current_request.block_ids]

        print(f"kv cache write and read end: {end}. num cached tokens: {original_request.num_cached_tokens}. key.shape[2]: {key.shape[2]} key shape: {key.shape}")
        # print(f"kv cache write and read all keys: {torch.cat(all_keys,dim=2)[:,:,:end,:]}")

        print(f"kv cache write and read all keys: {len(torch.cat(all_keys,dim=2)[:,:,:end,:])}")
        print(f"kv cache write and read all values: {len(torch.cat(all_values,dim=2)[:,:,:end,:])}")

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

    def can_allocate(self, req: Request, min_tokens: int):
        print(f"can allocate input ids: {len(req.input_ids)}. can allocate output ids: {len(req.output_ids)}")
        total_tokens = (min_tokens or req.input_ids.shape[1]) + self.block_size
        blocks_needed = (total_tokens + req.max_new_tokens) // self.block_size
        print(f"can allocate blocks needed: {blocks_needed}")
        print(f"can allocate free blocks: {len(self.free_blocks)}")

        if blocks_needed > len(self.free_blocks):
            return -1

        blocks = self.free_blocks[0:int(blocks_needed)]
        return blocks

    def allocate(self, req: Request, blocks: list[int]):
        for b in blocks:
            self.block_reference_count[b] += 1
            self.free_blocks.remove(b)
        req.block_ids = blocks
        return req

    def prefill(self, req: Request, min_tokens: int):
        print(f"prefill input ids length: {req.input_ids.shape[1]}")
        print(f"prefill input ids: {req.input_ids}")
        model(torch.tensor(req.input_ids,device=device), self, req)

class ScheduledRequest(Request):
    scheduled_tokens: int = -1

class Scheduler:
    def __init__(self, cache, model_config):
        self.waiting: dict[Request] = {}
        self.scheduled: dict[ScheduledRequest] = {}
        self.finished: list = []

        self.num_seqs: int = 0
        self.max_num_seqs: int = 20

        self.max_batch_tokens: int = 16384
        self.max_req_batch_tokens: int = int(self.max_batch_tokens / self.max_num_seqs)

        self.cache: KVCache = cache
        self.model_config = model_config
        self.eos_token_id = self.model_config.eos_token_id

    def can_run_request(request: Request):
        pass

    def preempt(self, req: Request):
        total_tokens = req.input_ids.shape[1] + self.cache.block_size
        blocks_needed = (total_tokens + req.max_new_tokens) // self.cache.block_size
        _, freed_blocks = self.cache.free_request_cache(self.scheduled[-1])
        if blocks_needed < len(self.cache.free_blocks):
            return freed_blocks
        return -1

    def add_to_waiting(self, request: Request):
        self.waiting[request.request_id] = request
        if request.request_id in self.scheduled.keys():
            del self.scheduled[request.request_id]
        else: request.is_prefill = False

    def free_request(self, req: Request):
        req.is_finished = True
        self.cache.free_request_cache(req)
        self.num_seqs -= 1
        del self.scheduled[req.request_id]
        self.finished.append(req)
        return req
    
    def step(self, request: Request, waiting: bool = False):
        self.finished = []
        print(f"step request input ids: {len(request.input_ids)}. step request output ids: {len(request.output_ids)}")
        total_input_tokens = request.input_ids.shape[1]

        # prefill
        if self.waiting and self.num_seqs < self.max_num_seqs and request.is_prefill:
            print(f"prefill: {request.request_id}")
            min_batch_tokens = min(total_input_tokens, self.max_req_batch_tokens)
            print(f"prefill min batch tokens: {min_batch_tokens}")
            allocated_blocks = self.cache.can_allocate(request, min_batch_tokens)
            if allocated_blocks == -1:
                allocated_blocks = self.preempt(request)
                if allocated_blocks == -1: 
                    self.add_to_waiting(request)
                    return -1
            print(f"prefill allocated blocks: {len(allocated_blocks)}")
            self.cache.allocate(request, allocated_blocks)
            # request.num_cached_tokens = min_batch_tokens
            self.cache.prefill(request, min_batch_tokens)
            if request.num_cached_tokens == total_input_tokens:
                print(f"prefill all tokens prefilled: {request.request_id}")
                request.is_prefill = False
                self.scheduled[request.request_id] = request
                request.scheduled_tokens = 1

        # decode
        if self.scheduled and self.num_seqs < self.max_num_seqs:
            print(f"decoding: {request.request_id}")
            remaining_tokens =  (len(request.block_ids) * self.cache.block_size) - request.num_cached_tokens
            print(f"{len(request.block_ids)}. {self.cache.block_size}. {request.num_cached_tokens}")
            print(f"remaining tokens: {remaining_tokens}")
            while request.scheduled_tokens > remaining_tokens:
                allocated_blocks = self.cache.can_allocate(request, request.scheduled_tokens)
                if allocated_blocks == -1:
                    print(f"no blocks to allocate: {request.request_id}. prempting...")
                    allocated_blocks = self.preempt(request)
                    print(f"preempted blocks: {allocated_blocks}")
                    if allocated_blocks == -1: 
                        self.add_to_waiting(request)
                        break
                request = self.cache.allocate(request, allocated_blocks)
                remaining_tokens = (len(request.block_ids) * self.cache.block_size) - request.num_cached_tokens
            if request.scheduled_tokens < remaining_tokens:
                print(f"running decode: {request.request_id}. scheduled_tokens: {request.scheduled_tokens}. scheduled tokens type: {type(request.scheduled_tokens)}")
                for s in range(request.scheduled_tokens):
                    output_id, logits = self.run(request, request.scheduled_tokens)
                    request.output_ids.append(output_id)
                    if output_id in self.eos_token_id:
                        print(f"eos token: {request.request_id}. {len(request.output_ids)} tokens generated")
                        self.free_request(request)

        return self.scheduled, self.waiting, request if request.is_finished else -1

    def run(self, request: Request, scheduled_tokens: int = 1):
        print(f"run() - {request.request_id}")
        prompt_length = request.input_ids.shape[1]
        eos_token_id = qwen_model.generation_config.eos_token_id

        # Only the last prefill chunk's logits predict the first generated token.
        generation_limit = cache.max_cached_tokens - prompt_length
        if request.max_new_tokens:
            generation_limit = min(generation_limit, int(request.max_new_tokens))

        generation_output = []
        generation_start_time = datetime.now()
        next_token_id = None

        input_ids = request.input_ids
        if request.output_ids != []: 
            output_ids = torch.tensor([[request.output_ids[-1]]], device=device)
            print(f"step output ids: {output_ids}. last output id: {output_ids[len(output_ids)-1]}")
            input_ids = output_ids
    
        for i in range(scheduled_tokens or generation_limit):
            print(f"running model: {request.request_id}")
            result = model(input_ids, cache, request)
            next_token_id = result[:,-1].argmax().item()
            print(f"result: {result}")
    
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

        response = tokenizer.decode(request.output_ids,skip_special_tensors=True)
        print(f"response: {response}")

        return next_token_id, result.logit
    

    def init(self, request: Request | None):
        # how many tokens for this request?
        print(f"init: {request.request_id}. input ids: {len(request.input_ids)}")
        self.waiting[request.request_id] = request

        # #1 priority self.prefilled , self.waiting
        # have remaining space to prefill remaining tokens + 1 decode token?
        # have space for prefill and decode 1 token? 
        # yes? prefill all and start decode
        # no? prefill selected tokens in remaining space -1 (for decode)

        # prefill_request(request)
            # can prefill all tokens and 1 token space for decode?
            # can prefill selected tokens for remaining space?
            # x/78 tokens prefilled, [6,10,13] blocks prefilled
        
        # are there requests in prefill?
            # how much tokens remaining to prefill
            # (use variable) have remaining space to prefill remaining tokens + 1 decode token?
                # can prefill all tokens?
                    # yes? prefill all tokens - continue to run request
                    # no? prefill selected remaining tokens in available space - 1 (1 decode token space)
                        # return
            # no? prefill request and run if full tokens prefilled
                # space available == (len(tokens) +1) ? have space for prefill all tokens and 1 decode?
                # prefill, run 
                # can_run_request? - have space to prefill and decode 1 token
                # no?
                # how much space for prefill
                    # prefill selected tokens in remaining cache space
                    # add to self.prefill
    
        # updated remaining space after new prefill of self.prefill and self.waiting requests
        # have remaining space for partial prefill or full prefill with 1 decode tokens space?
            # yes: prefill_request(request)
        # no?
            # add to self.waiting
    

model_name = 'Qwen/Qwen3-0.6B'
block_size=4

model_load_start_time = datetime.now()
tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_load_end_time = datetime.now()
print(f"model {model_name} loaded in : {(model_load_end_time - model_load_start_time).total_seconds():.2f}s\n")

model = QwenModel(qwen_model)
model_config = qwen_model.config
max_new_tokens = model_config.max_position_embeddings

# [28, 1, 8, 1024, 4, 128]
model_shape = (
    model_config.num_hidden_layers,
    1,
    model_config.num_key_value_heads,
    int((model_config.max_position_embeddings*20) / block_size),
    block_size,
    model_config.head_dim
)

cache = KVCache(model_shape,model_config,block_size)
scheduler = Scheduler(cache, qwen_model.generation_config)

executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=scheduler.max_num_seqs,
    thread_name_prefix="worker"
)

def handle_finished(future: concurrent.futures.Future):
    result = future.result()
    if result is not None:
        return f"Request finished: {result}"

def dispatch_request(req):
    future = executor.submit(scheduler.step, req)
    future.add_done_callback(handle_finished)

def main(request: Request):
    # print(request)
    print(f"requests: {len(cache.requests)}")

    request_id = request.request_id
    user_message = request.prompt

    input_ids = tokenizer.apply_chat_template(
        [{'role':'user','content':user_message}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        tokenize=True
    )['input_ids'].to(device)

    request.input_ids = input_ids

    cache.init_request(input_ids, request)
    # print(f"model config: {model_config}")

    scheduler.init(request)
    # dispatch_request(request)

    print(f"request start: {request}")

    requests = [*scheduler.scheduled.values(),*scheduler.waiting.values()]
    for r in requests:
        while scheduler.step(r):
            if scheduler.finished != []:
                # find request thread, return to that thread
                pass
            if scheduler.scheduled == [] and scheduler.waiting == []:
                print(f"all requests finished")
                break
        return

    
    return 'scheduler running'

# main()

app = FastAPI()



@app.post('/')
def home(message: dict):
    print(f"message is here")
    print(message)
    print(f"model generation config: {qwen_model.config.max_position_embeddings}")

    request = Request(
        request_id=str(uuid4()), 
        prompt=message['prompt']['text'], 
        max_new_tokens=(message['prompt']['new_tokens'] or qwen_model.config.max_position_embeddings))
    print(f"starting request: {request}")
    message, tokens_per_second = main(request)
    
    cache.free_request_cache(request)

    return {'message': message, 'tokens_per_second': f"{tokens_per_second:.2f}"}
