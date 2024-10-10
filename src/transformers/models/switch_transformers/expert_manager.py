import torch
import torch.nn as nn
from .configuration_switch_transformers import SwitchTransformersConfig
import torch.distributed as dist

class ExpertManager():
    def __init__(self, experts: nn.ModuleDict, config: SwitchTransformersConfig, expert_class: nn.Module, max_loaded_experts: int):
        self.cpu_experts = experts
        self.num_experts = len(experts)
        self.rank = dist.get_rank()
        self.num_gpus = dist.get_world_size()
        self.max_loaded_experts = max_loaded_experts

        # The experts loaded in GPU memory
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.loaded_experts = [expert_class(config).cuda() for _ in range(self.max_loaded_experts)]

        self.loaded_expert_at_slot = [-1 for _ in range(self.max_loaded_experts)]
        self.is_slot_loaded = [torch.cuda.Event(enable_timing=False) for _ in range(self.max_loaded_experts)]

    
    def expert_parallelise(self):
        # TODO get rid of requirement that ep_size <= max_loaded_experts
        self.topology = [[] for _ in range(self.num_gpus)]
        num_experts_per_gpu = self.num_experts // self.num_gpus
        leftover = self.num_experts % self.num_gpus
        start = 0
        end = 0 # Not inclusive
        for i in range(self.num_gpus):
            end += num_experts_per_gpu
            if leftover > 0:
                end += 1
                leftover -= 1
            
            for j in range(start,end):
                self.topology[i].append(j)

            start = end

        self.load_topology()
    
    def load_topology(self):
        if len(self.topology[self.rank]) > self.max_loaded_experts:
            raise Exception("Cannot load topology given memory requirement, consider increasing max number of loaded experts")
        
        for slot_idx, expert_idx in enumerate(self.topology[self.rank]):
            if self.loaded_expert_at_slot[slot_idx] != expert_idx:
                self.load_expert(expert_idx, slot_idx)
    
    def get_topology(self):
        return self.topology[:] #TODO change to be dynamic based on what is loaded
    
    def load_expert(self, expert_idx: int, load_idx: int):
        with torch.no_grad():
            with torch.cuda.stream(self.stream):
                self.loaded_experts[load_idx].wi.weight.copy_(self.cpu_experts[f"expert_{expert_idx}"].wi.weight)
                self.loaded_experts[load_idx].wo.weight.copy_(self.cpu_experts[f"expert_{expert_idx}"].wo.weight)
                self.loaded_expert_at_slot[load_idx] = expert_idx
                self.is_slot_loaded[load_idx].record()
    
    def execute_expert(self, expert_idx: int, data):
        idx = self.expert_loaded_location(expert_idx)
        if idx == -1:
            raise Exception("Cannot run expert which is not already loaded")
        
        self.is_slot_loaded[idx].synchronize()
        r = self.loaded_experts[idx](data)
        return r
        
    
    # Returns index of expert. Returns -1 if not loaded.
    def expert_loaded_location(self, expert_idx):
        idx = -1
        for i, v in enumerate(self.loaded_expert_at_slot):
            if v == expert_idx:
                idx = i
                break
        return idx 
    
    def is_expert_loaded(self, expert_idx):
        return self.expert_loaded_location(expert_idx) != -1
    
    def execute_job(self, workload: [], straight_exec=False):
        if straight_exec:
            for expert_idx in range(self.num_experts):
                if workload[expert_idx].size(dim=0) == 0:
                    continue
                slot_idx = self.expert_loaded_location(expert_idx)
                workload[expert_idx] = self.loaded_experts[slot_idx](workload[expert_idx])
        else:
            expert_order = []
            num_execs = 0

            # Setup
            for expert_idx in range(self.num_experts):
                if workload[expert_idx].size(dim=0) == 0:
                    continue
                
                num_execs += 1
                if self.is_expert_loaded(expert_idx):
                    expert_order.insert(0, expert_idx)
                else:
                    expert_order.append(expert_idx)
            
            # Start loading as many experts as possible
            for idx, loaded_expert_idx in enumerate(self.loaded_expert_at_slot):
                # Fill if a gap or if loaded expert not used
                if loaded_expert_idx == -1 or loaded_expert_idx not in expert_order:
                    n = self.next_not_loaded_expert(expert_order)
                    if n is None:
                        break

                    self.load_expert(n, idx)

            
            # Begin execution
            for idx, expert_idx in enumerate(expert_order):
                slot_idx = self.expert_loaded_location(expert_idx)
                self.is_slot_loaded[slot_idx].record()
                workload[expert_idx] = self.loaded_experts[slot_idx](workload[expert_idx])
                # Check if anything else needs loading
                n = self.next_not_loaded_expert(expert_order, idx)
                if n is not None:
                    self.load_expert(n, slot_idx)
            
            # Ah oh I don't load it back?
            # TODO and also have feature where I can choose different load back
            
        return workload

        
    def next_not_loaded_expert(self, experts, start=0):
        for idx in range(start,len(experts)):
            if not self.is_expert_loaded(experts[idx]):
                return experts[idx]
        return None

    # If I want to add rebalancing
    # Return a topology
    # def greedy_create_topology(self, expert_freq):
    #     # TODO add penalty for number of experts on each GPU
        
    #     topo = [[] for _ in range(self.num_gpus)]
    #     topo_sum = [0 for _ in range(self.num_gpus)]
    #     for expert_idx, amt in sorted(enumerate(expert_freq), key=lambda x: x[1], reverse=True):
    #         _min = float("inf")
    #         min_idx = -1

    #         for i in range(self.num_gpus):
    #             if topo_sum[i] < _min and len(topo[i]) < self.max_loaded_experts:
    #                 _min = topo_sum[i]
    #                 min_idx = i
            
    #         topo_sum[min_idx] += amt
    #         topo[min_idx].append(expert_idx)

    #     return topo
