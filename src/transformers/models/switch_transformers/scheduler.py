import torch
import torch.nn as nn

import torch.distributed as dist

class Scheduler():
    def __init__(self, scheduling_policy="deepspeed", num_experts=8, eq_tokens=150, num_gpus=None):
        if num_gpus is None:
            self.rank = dist.get_rank()
            self.num_gpus = dist.get_world_size()
        else: # We are running unit tests
            self.rank = 0
            self.num_gpus = num_gpus
        self.num_experts = num_experts
        self.eq_tokens = eq_tokens

        match scheduling_policy:
                case "deepspeed":
                    self.scheduler = self.schedule_deepspeed
                case "adnexus":
                    self.scheduler = self.schedule_adnexus
                case "even_split":
                    self.scheduler = self.schedule_even_split
                case "drop":
                    self.scheduler = self.schedule_drop
                case "demeter":
                    self.scheduler = self.schedule_demeter
                case "adfabricus":
                    self.scheduler = self.schedule_adfabricus
                case _:
                    print("SCHEDULING POLICY NOT IMPLEMENTED")
                    exit(1)
    
    # def __call__(self, hidden_states, router_mask, topology):
    #     return self.scheduler(hidden_states, router_mask, topology)

    def __call__(self, meta, topo):
        return self.scheduler(meta, topo)
    
    def schedule_deepspeed(self, hidden_states, router_mask, topology):
        schedule = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        for i in range(self.num_gpus):
            for j in topology[i]:
                tokens = hidden_states[router_mask[:,:,j]]
                schedule[i][j] = (0,tokens.shape[0],tokens)
        return schedule
    
    # TODO need way to update router_prob to 1 to the dropped tokens
    def schedule_drop(self, hidden_states, router_mask, topology):
        amounts = [[0 for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        token_sizes = [hidden_states[router_mask[:,:,idx]].shape[0] for idx in range(self.num_experts)]
        avg = int(sum(token_sizes) / self.num_gpus)
        num_toks_each_gpu = [0 for _ in range(self.num_gpus)]

        schedule = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        for i in range(self.num_gpus):
            for j in topology[i]:
                amt_can_allocate = min(avg-num_toks_each_gpu[i], token_sizes[j])
                if amt_can_allocate == 0:
                    break
                num_toks_each_gpu[i] += amt_can_allocate
                schedule[i][j] = (0, amt_can_allocate, hidden_states[router_mask[:,:,j]][0:amt_can_allocate, :])
        
        return schedule
    
    def schedule_adnexus(self, hidden_states, router_mask, topology):
        expert_inputs = [hidden_states[router_mask[:,:,idx]] for idx in range(self.num_experts)]
        cur_gpu_assignment = [[0 for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        for i, experts in enumerate(topology):
            for expert in experts:
                cur_gpu_assignment[i][expert] = expert_inputs[expert].size(dim=0)
        cur_num_tokens = list(map(lambda arr: sum(arr), cur_gpu_assignment))
        avg = int(sum(cur_num_tokens) / self.num_gpus)

        for i in range(self.num_gpus):
            while cur_num_tokens[i] > avg:
                # Find minimum
                _min = cur_num_tokens[0]
                min_idx = 0
                for j in range(self.num_gpus):
                    if cur_num_tokens[j] < _min:
                        _min = cur_num_tokens[j]
                        min_idx = j
                
                # Check if suitable place to move tokens
                if min_idx == i:
                    break
                if _min + self.eq_tokens > avg:
                    break
                # if _min > avg:
                    # break

                # Get the maximal expert to move
                _max = -1
                max_expert = -1
                for idx, size in enumerate(cur_gpu_assignment[i]):
                    if size > _max:
                        _max = size
                        max_expert = idx

                # Find maximal tokens that can be sent
                # Between how much we want to reduce and the amount the maximal expert has 
                tokens_to_spread = min(_max, cur_num_tokens[i] - avg)
                # Between how much we can reduce thus far and the amount we can add to minimal gpu
                tokens_to_spread = min(tokens_to_spread, avg - cur_num_tokens[min_idx])

                if tokens_to_spread < self.eq_tokens:
                    break

                # Update assignment
                cur_gpu_assignment[i][max_expert] -= tokens_to_spread
                cur_gpu_assignment[min_idx][max_expert] += tokens_to_spread

                cur_num_tokens[i] -= tokens_to_spread
                cur_num_tokens[min_idx] += tokens_to_spread

        # Build up schedule
        schedule = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        for j in range(self.num_experts):
            start = 0
            end = 0
            for i in range(self.num_gpus):
                if cur_gpu_assignment[i][j] == 0:
                    continue
                size = cur_gpu_assignment[i][j]
                end += size
                schedule[i][j] = (start, end, expert_inputs[j][start:end])
                start = end 
                
        return schedule 
    
    def schedule_demeter(self, hidden_states, router_mask, topology):
        expert_inputs = [hidden_states[router_mask[:,:,idx]] for idx in range(self.num_experts)]
        expert_sizes = [expert_inputs[i].shape[0] for i in range(self.num_experts)]
        avg = sum(expert_sizes) // self.num_gpus
        multiplier = 1.15
        avg_multiplier = int(multiplier * avg)

        allocation = [[0 for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        # Step 1: Initial Allocation
        for gpu_idx, experts in enumerate(topology):
            for expert in experts:
                allocation[gpu_idx][expert] = expert_sizes[expert]

        # Step 2: Rebalance
        for i in range(self.num_gpus):
            while sum(allocation[i]) > avg_multiplier:
                # Get largest expert
                max_expert = -1
                max_amount = -1
                for j in range(self.num_experts):
                    if allocation[i][j] > max_amount:
                        max_amount = allocation[i][j]
                        max_expert = j


                # Get the offload GPU for that expert
                offload_gpu = -1
                for j in range(self.num_gpus):
                    if j != i and max_expert in topology[(j+1)%self.num_gpus]:
                        offload_gpu = j
                        break
                
                # Calculate maximal amount that can be shared 
                amount_to_share = min(sum(allocation[i])-avg_multiplier, allocation[i][max_expert])
                amount_to_share = min(amount_to_share, avg-sum(allocation[offload_gpu]))

                # If maximal amount is zero then break
                if amount_to_share <= 0:
                    break

                # Update
                allocation[i][max_expert] -= amount_to_share
                allocation[offload_gpu][max_expert] += amount_to_share

        
        # Building schedule
        schedule = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        for j in range(self.num_experts):
            start = 0
            end = 0
            for i in range(self.num_gpus):
                if allocation[i][j] == 0:
                    continue
                end += allocation[i][j]
                schedule[i][j] = (start,end,expert_inputs[j][start:end])
                start = end 

        return schedule

    def schedule_even_split(self, hidden_states, router_mask, topology):
        expert_inputs = [hidden_states[router_mask[:,:,idx]] for idx in range(self.num_experts)]
        schedule = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        for j in range(self.num_experts):
            if expert_inputs[j].shape[0] == 0:
                continue
            avg = expert_inputs[j].shape[0] // self.num_gpus
            mod = expert_inputs[j].shape[0] % self.num_gpus
            start = 0
            end = 0
            for i in range(self.num_gpus):
                end += avg 
                if i == 0:
                    end += mod
                schedule[i][j] = (start,end,expert_inputs[j][start:end])
                start = end 
        
        return schedule 
    
    def schedule_adfabricus(self, meta, topology):
        # num_gpus x num_experts x num_gpus
        # sender x expert x destination
        schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        gpu_amt = [0 for _ in range(self.num_gpus)]
        
        avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus)

        skipped = []

        # Try to place as according to topology much without breaking apart
        for gpu_idx in range(self.num_gpus):
            for expert_idx in range(self.num_experts):
                num_tokens = meta[gpu_idx][expert_idx].item()
                if num_tokens == 0:
                    continue
                found = False
                for offload_gpu_idx in range(self.num_gpus):
                    if expert_idx in topology[offload_gpu_idx]:
                        if num_tokens + gpu_amt[offload_gpu_idx] <= avg:
                            schedule[gpu_idx][expert_idx][offload_gpu_idx] += num_tokens
                            gpu_amt[offload_gpu_idx] += num_tokens
                            found = True
                            break
                if not found:
                    skipped.append((gpu_idx, expert_idx, num_tokens))
                
        # Now try to saturate the topology
        for i in range(len(skipped)):
            num_tokens = skipped[i][2]
            for offload_gpu_idx in range(self.num_gpus):
                if skipped[i][1] in topology[offload_gpu_idx]:
                    if gpu_amt[offload_gpu_idx] < avg: # Expert already exists so no need for eq tokens
                        tokens_send = min(num_tokens, avg - gpu_amt[offload_gpu_idx])
                        schedule[skipped[i][0]][skipped[i][1]][offload_gpu_idx] += tokens_send
                        gpu_amt[offload_gpu_idx] += tokens_send
                        skipped[i] = (skipped[i][0], skipped[i][1], skipped[i][2]-tokens_send)
        
        # Remove from skipped any that has no more tokens
        temp = []
        for skip in skipped:
            if skip[2] != 0:
                temp.append(skip)
        skipped = temp

        gpu_experts = topology

        # Now will need to split across
        for skip in skipped:
            num_tokens = skip[2]
            # First look for any gpu with already the expert less than avg
            # This is not redundant to the above because a previous skip could have added a new expert to a GPU
            resolved = False
            for offload_gpu_idx in range(self.num_gpus):
                if skip[1] in gpu_experts[offload_gpu_idx]:
                    if gpu_amt[offload_gpu_idx] < avg:
                        tokens_send = min(num_tokens, avg - gpu_amt[offload_gpu_idx])
                        schedule[skip[0]][skip[1]][offload_gpu_idx] += tokens_send
                        gpu_amt[offload_gpu_idx] += tokens_send
                        num_tokens -= tokens_send
                        if num_tokens == 0:
                            resolved = True
                            break
            if resolved:
                continue # Let us rebalance the next skipped token set

            # Then we will look for minimal GPU and if that GPU has atleast eq_tokens space less than avg
            if num_tokens > self.eq_tokens:
                min_gpu_idx = 0
                _min = gpu_amt[0]
                for offload_gpu_idx in range(self.num_gpus):
                    if gpu_amt[offload_gpu_idx] < _min:
                        _min = gpu_amt[offload_gpu_idx]
                        min_gpu_idx = offload_gpu_idx
                if _min + self.eq_tokens <= avg:
                    tokens_send = min(num_tokens, avg - _min)
                    schedule[skip[0]][skip[1]][min_gpu_idx] += tokens_send
                    gpu_experts[min_gpu_idx].append(skip[1])
                    gpu_amt[min_gpu_idx] += tokens_send
                    num_tokens -= tokens_send
                    if num_tokens == 0:
                        resolved = True
            if resolved:
                continue # Let us rebalance the next skipped token set

            # Otherwise we split evenly the tokens across the gpus with the expert already
            # Find all GPUs with the expert
            offloaders = []
            for offload_gpu_idx in range(self.num_gpus):
                if skip[1] in gpu_experts[offload_gpu_idx]:
                    offloaders.append(offload_gpu_idx)
            
            if len(offloaders) > 0:
                # Next we want to add tokens to each to make them balanced
                # This is not just evenly splitting since some GPUs may have more or less tokens
                target = int((sum(map(lambda x: gpu_amt[x], offloaders)) + num_tokens) / len(offloaders))
                # Fill each to the target
                for offload_gpu_idx in offloaders:
                    if gpu_amt[offload_gpu_idx] < target:
                        tokens_send = min(num_tokens, target - gpu_amt[offload_gpu_idx])
                        schedule[skip[0]][skip[1]][offload_gpu_idx] += tokens_send
                        gpu_amt[offload_gpu_idx] += tokens_send
                        num_tokens -= tokens_send
                        if num_tokens == 0:
                            break
                # If in really rare case there are still leftover tokens then just give all to the one with the least
                if num_tokens > 0:
                    min_offload_gpu_idx = offloaders[0]
                    _min = gpu_amt[offloaders[0]]
                    for offload_gpu_idx in offloaders:
                        if gpu_amt[offload_gpu_idx] < _min:
                            _min = gpu_amt[offload_gpu_idx]
                            min_offload_gpu_idx = offload_gpu_idx
                    schedule[skip[0]][skip[1]][min_offload_gpu_idx] += num_tokens
                    gpu_amt[min_offload_gpu_idx] += num_tokens
                    # We are guaranteed finished here
            else:
                # Rare: happens if there is no GPU with the expert
                # Just give it to the GPU with the least 
                min_offload_gpu_idx = 0
                _min = gpu_amt[0]
                for offload_gpu_idx in range(self.num_gpus):
                    if gpu_amt[offload_gpu_idx] < _min:
                        _min = gpu_amt[offload_gpu_idx]
                        min_offload_gpu_idx  = offload_gpu_idx
                schedule[skip[0]][skip[1]][min_offload_gpu_idx] += num_tokens
                gpu_amt[min_offload_gpu_idx] += num_tokens
                gpu_experts[min_offload_gpu_idx].append(skip[1])
                # We are guaranteed finished here

        # TODO can remove this an instead have schedule that is built different
        # # TODO Let us have a way to collect the number of tokens within a GPU expert
        # # When designated to same offload GPU
        # # TODO sort too!
        # # We want to collect all the tokens a specific GPU expert decides to send to another GPU
        # # Because of the multiple parts of the above algorithm, it is possible that there
        # # are multiple entries destined to the same expert.
        # # Furthermore, want to sort it based on incremental order of destination GPU  
        # schedule_cleaned = [[[] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        # for i in range(self.num_gpus):
        #     for j in range(self.num_experts):
        #         for k in range(len(schedule[i][j])):
        #             appened = False
        #             index = 0
        #             for l in range(len(schedule_cleaned[i][j])):
        #                 if schedule_cleaned[i][j][l][0] == schedule[i][j][k][0]:
        #                     schedule_cleaned[i][j][l] = (schedule_cleaned[i][j][l][0], 
        #                         schedule_cleaned[i][j][l][1]+schedule[i][j][k][1])
        #                     appened = True
        #                     break
        #                 elif schedule[i][j][k][0] < schedule_cleaned[i][j][l][0]:
        #                     index = l
        #             if not appened:
        #                 schedule_cleaned[i][j].insert(index, schedule[i][j][k])

        return schedule