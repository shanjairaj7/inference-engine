import numpy as np

req_a = ['a','b','c','d','e','f','1','2','3'] # 3 tokens
req_b = ['a','b','c','d','5','3','4', 'w'] # 4 tokens
req_c = ['z','x','y','w','x'] # 2 tokens
req_q = ['e','f','1','2'] # 2 tokens

blocks = ['0_reserve','1','2','3','4']
hashes = {}
free = [1,2,3,4]
usage = []
max_tokens = [3,4,2,2]
waiting = []

def generate(req) -> list:
  print('req:', req[-1])
  return str(np.random.randint(low=0, high=10))

def run(req):
    block_size = 4
    live_req = [req[i:i+block_size] for i in range(0, len(req), block_size)]
    live_req_length = sum(len(block) for block in live_req)# length of all values inside of teh arrays inside of the live_req
    # if gpus are empty then put in waiting
    # waiting.append({'req': req_unique_hash, 'token_block': len(live_req)})
    for token_block_index, token_block in enumerate(live_req):
      chosen_free_gpu_block = free[0] if len(free) > 0 else 'empty'
      req_unique_hash = hash(str(req))
      usage_object = {'req': req_unique_hash, 'block': 4 if len(live_req[-1]) != block_size and chosen_free_gpu_block == 'empty' else  chosen_free_gpu_block, 'current_tokens': live_req_length, 'new_tokens': live_req_length+1}
      def hash_block(block): hash(str(block))
      print('token blocks -', live_req)
      print('curent token block', token_block)
      print('chosen block', chosen_free_gpu_block, free)
      print(len(live_req[-1]), block_size)
      if chosen_free_gpu_block != 'empty' or len(live_req[-1]) != block_size:
        hashed_value = hash(str(token_block)) # add the prefill tokens (filled 4 tokens) into gpu blocks as prefill cache
        hashes[hashed_value] = 4 if len(live_req[-1]) != block_size and chosen_free_gpu_block == 'empty' else  chosen_free_gpu_block # add the used hash and map to the used free block on the gpu
        print('HASHING FOR THIS ->', token_block, free, hashed_value)
        if chosen_free_gpu_block in free:
          free.remove(chosen_free_gpu_block) # now this block is being used, remove this block from the free blocks array
        # if same length is available in (free blocks * block s_size) then just prefill
        # if len(free blocks) * block size < len(token_block) then start generating decode
        print('token block len - ', len(token_block))
        print('block size - ', block_size)
        if len(token_block) < block_size: # there is space to generate tokens/decode, not fully filled
          print('starting now')
          print('live req length', live_req_length, len(req)+6, token_block_index, len(live_req))
          # this is the token block to start generating tokens/decode, keep going until it finishes its full completion tokens
          while (live_req_length != (len(req)+12)) and token_block_index == len(live_req)-1:
            print(live_req_length, (len(req)+6))
            print(live_req)
            print('generating decode now')
            if len(token_block) < block_size:
              live_req[-1].append(generate(live_req)) # generate to the last token block
              live_req_length = sum(len(block) for block in live_req)# length of all values inside of teh arrays inside of the live_req, keep updating this inside the while loop after every addition
            else: # it as filled the free gpu block with 4 tokens
              # now its filled with 4 tokens per block, now hash it and add it to hashes
              # hashed_value = hash(str(token_block))
              # if its here in the while loop then it means there is still tokens to generate/decode, get new gpu block and decode
              chosen_free_gpu_block = free[0] if len(free) > 0 else 'empty'
              if chosen_free_gpu_block != 'empty':
                free.remove(chosen_free_gpu_block)
                live_req.append([generate(live_req)])
                live_req_length = sum(len(block) for block in live_req)# length of all values inside of teh arrays inside of the live_req, keep updating this inside the while loop after every addition
              else:
                # there are no gpu blocks to generate new tokens/decode
                print('all gpus are used -')
                print('current generated tokens -', live_req)
                print('gpu blocks free -', free)
                waiting.append({'req': req_unique_hash, 'token_block': live_req})
                break
              # get new gpu blocks to assign the new decoding
            # at the end of while loop, calculate current full tokens of request, and new token to be generated for the request
            usage_object = {'req': req_unique_hash, 'block': 4 if len(live_req[-1]) == block_size and chosen_free_gpu_block == 'empty' else  chosen_free_gpu_block, 'current_tokens': live_req_length, 'new_tokens': live_req_length+1}
            print('current free -', free)
            try:
              usage_index = [each_usage['req'] == req_unique_hash for each_usage in usage].index(True)
              print('usage index -', usage_index)
              print('usage -', usage)
              usage[usage_index] = usage_object
            except:
              usage.append(usage_object)
          print(live_req, live_req_length, len(req)+6)
      
      print('request with generated tokens/decode -', live_req)
      print('---')
    print(live_req)
    print(hashes)
    print(free)
    print(usage)
run(req_a)
