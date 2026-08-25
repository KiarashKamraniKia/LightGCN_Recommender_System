import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeGraphRefinement(nn.Module):
    def __init__(self, top_m=10, initial_beta=0.1):
        super(PrototypeGraphRefinement, self).__init__()
        self.top_m = top_m
        
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))

    def _get_topk_affinity(self, assignments):
        affinity = torch.matmul(assignments, assignments.T)
        
        num_nodes = assignments.size(0)
        k = min(self.top_m, num_nodes)
        
        vals, indices = torch.topk(affinity, k=k, dim=-1)
        
        mask = torch.zeros_like(affinity)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(vals))
        
        sparse_affinity = affinity * mask
        return sparse_affinity

    def forward(self, user_assignments, item_assignments, orig_adj):
        num_users = user_assignments.size(0)
        num_items = item_assignments.size(0)
        total_nodes = num_users + num_items
        
        user_proto_adj = self._get_topk_affinity(user_assignments)
        item_proto_adj = self._get_topk_affinity(item_assignments)
        
        A_proto = torch.zeros(
            (total_nodes, total_nodes), 
            device=user_assignments.device, 
            dtype=user_assignments.dtype
        )
        A_proto[:num_users, :num_users] = user_proto_adj
        A_proto[num_users:, num_users:] = item_proto_adj
        
        if orig_adj.is_sparse:
            orig_adj_dense = orig_adj.to_dense()
        else:
            orig_adj_dense = orig_adj
            
        refined_adj = orig_adj_dense + self.beta * A_proto
        
        return refined_adj, A_proto