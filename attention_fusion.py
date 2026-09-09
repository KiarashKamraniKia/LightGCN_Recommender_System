import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossViewAttentionFusion(nn.Module):
    
    def __init__(self, embed_dim: int):
        super(CrossViewAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        self.W_graph = nn.Linear(embed_dim, 1, bias=False)
        self.W_proto = nn.Linear(embed_dim, 1, bias=False)
        self.W_stat = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, h_graph: torch.Tensor, h_proto: torch.Tensor, h_stat: torch.Tensor):
        s_graph = self.W_graph(h_graph)
        s_proto = self.W_proto(h_proto)
        s_stat = self.W_stat(h_stat)
        
        scores = torch.cat([s_graph, s_proto, s_stat], dim=-1)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        alpha_g = attn_weights[:, 0:1]
        alpha_p = attn_weights[:, 1:2]
        alpha_s = attn_weights[:, 2:3]
        
        h = alpha_g * h_graph + alpha_p * h_proto + alpha_s * h_stat
        
        return h, attn_weights
