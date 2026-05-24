import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Dual-Expert Alignment: Bridges the semantic gap between 
    Habit and Exploration spaces using InfoNCE style objective.
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.tau = temperature

    def forward(self, h_habit, h_expl):
        # Normalize representations to hypersphere
        h1 = F.normalize(h_habit, dim=1)
        h2 = F.normalize(h_expl, dim=1)
        
        # Calculate similarity matrix [Batch, Batch]
        logits = torch.matmul(h1, h2.t()) / self.tau
        
        # Positive samples are on the diagonal
        labels = torch.arange(h1.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

class GlobalGraphBuilder:
    """
    Utility for Exploration Expert: Builds the Item-Pattern Bipartite Graph
    used to capture latent collaborative signals.
    """
    def __init__(self, num_items, min_support=5):
        self.num_items = num_items
        self.min_support = min_support

    def build_graph_tensor(self, patterns, device):
        num_patterns = len(patterns) if len(patterns) > 0 else 1
        patterns = patterns if len(patterns) > 0 else [[1]]

        item_nodes, pattern_nodes = [], []
        for p_idx, items in enumerate(patterns):
            for i_id in items:
                if i_id < self.num_items:
                    item_nodes.append(i_id)
                    pattern_nodes.append(p_idx)
        
        i_nodes = torch.LongTensor(item_nodes).to(device)
        p_nodes = torch.LongTensor(pattern_nodes).to(device)
        i2p_index = torch.stack([i_nodes, p_nodes], dim=0)
        p2i_index = torch.stack([p_nodes, i_nodes], dim=0)
        
        # Normalization factors for GCN propagation
        d_item = torch.bincount(i_nodes, minlength=self.num_items + 1).float().clamp(min=1)
        d_pattern = torch.bincount(p_nodes, minlength=num_patterns).float().clamp(min=1)
        edge_weight = torch.pow(d_item[i_nodes], -0.5) * torch.pow(d_pattern[p_nodes], -0.5)
        
        return {'num_patterns': num_patterns, 'i2p_index': i2p_index, 'p2i_index': p2i_index, 'norm': edge_weight}

# Habit Expert Modeling: Temporal Rhythms & Intensity
class FourierTimeEncoder(nn.Module):
    """
    Captures non-monotonic temporal rhythms using sin/cos 
    transformations over elapsed time intervals.
    """
    def __init__(self, dim):
        super(FourierTimeEncoder, self).__init__()
        self.w = nn.Parameter(torch.randn(1, 1, 1, dim // 2)) 
        self.phi = nn.Parameter(torch.randn(1, 1, 1, dim // 2))

    def forward(self, delta_t):
        t = delta_t.unsqueeze(-1) if delta_t.dim() == 3 else delta_t
        phase = t * self.w + self.phi
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

# Exploration Expert Modeling: Pattern-Guided Graph
class GraphEncoder(nn.Module):
    """
    Retrieves collaborative interests by propagating information 
    across the Global Item-Pattern Bipartite Graph.
    """
    def __init__(self, num_items, num_patterns, dim, device):
        super(GraphEncoder, self).__init__()
        self.dim = dim
        self.pattern_emb = nn.Embedding(num_patterns, dim)
        nn.init.xavier_uniform_(self.pattern_emb.weight)
        self.device = device

    def forward(self, item_emb_weight, graph_data):
        i2p = graph_data['i2p_index']
        p2i = graph_data['p2i_index']
        norm = graph_data['norm']
        
        # Item -> Pattern propagation
        msg_i2p = item_emb_weight[i2p[0]] * norm.unsqueeze(-1)
        updated_patterns = torch.zeros(self.pattern_emb.num_embeddings, self.dim, device=self.device)
        updated_patterns.index_add_(0, i2p[1], msg_i2p)
        
        # Pattern -> Item propagation
        msg_p2i = updated_patterns[p2i[0]] * norm.unsqueeze(-1)
        context_delta = torch.zeros_like(item_emb_weight)
        context_delta.index_add_(0, p2i[1], msg_p2i)
        
        return item_emb_weight + context_delta, updated_patterns


class Model(nn.Module):
    def __init__(self, config, numItems):
        super(Model, self).__init__()
        self.dim, self.padIdx, self.device = config.dim, config.padIdx, config.device
        
        # --- Input Layer ---
        self.itemEmb = nn.Embedding(numItems + 1, self.dim, padding_idx=self.padIdx)
        
        # --- Habit Expert Components ---
        self.freq_emb = nn.Embedding(20, self.dim)
        self.time_encoder = FourierTimeEncoder(self.dim)
        self.hawkes_lambda = nn.Embedding(numItems + 1, 1, padding_idx=self.padIdx)
        nn.init.constant_(self.hawkes_lambda.weight, 0.1)
        
        self.w_time = nn.Parameter(torch.tensor(0.1))
        self.w_freq = nn.Parameter(torch.tensor(0.1))
        
        self.attn_W = nn.Linear(self.dim, self.dim)
        self.attn_v = nn.Linear(self.dim, 1, bias=False)
        self.out = nn.Linear(self.dim, numItems + 1)
        
        # --- Exploration Expert Components ---
        numPatterns = getattr(config, 'num_patterns', 100)
        self.graph_encoder = GraphEncoder(numItems + 1, numPatterns, self.dim, self.device)
        self.expl_proj = nn.Linear(self.dim, self.dim)
        
        # --- Fusion Stage: Item-Aware Gating ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, x, decay, delta_t, global_graph_data, freq_seq, userhis, isEval=False):
        # 1. Input Layer: Dense Embeddings
        embs = self.itemEmb(x)
        if not isEval: embs = F.dropout(embs)

        # 2. Habit Expert: Hawkes Intensity + Fourier Encoding
        p_ui = self.time_encoder(delta_t)
        e_freq = self.freq_emb(freq_seq.clamp(0, 19).long())
        embs_enhanced = embs + self.w_time * p_ui + self.w_freq * e_freq

        lambdas = torch.sigmoid(self.hawkes_lambda(x)) 
        hawkes_decay = torch.exp(-lambdas * delta_t) + 0.1
        embs_habit = hawkes_decay * embs_enhanced 

        # Attention Aggregation within Basket
        att_scores = self.attn_v(torch.tanh(self.attn_W(embs_habit))).squeeze(-1)
        att_scores = att_scores.masked_fill((x == self.padIdx), -1e9)
        att_weights = F.softmax(att_scores, dim=2).unsqueeze(-1)
        basket_embs = (embs_habit * att_weights).sum(dim=2)
        h_habit = torch.tanh(basket_embs.sum(1)) 

        # 3. Exploration Expert: Graph-based Collaborative Patterns
        g_items, _ = self.graph_encoder(self.itemEmb.weight, global_graph_data)
        h_expl = torch.tanh(self.expl_proj(h_habit))

        # 4. Fusion and Prediction: Item-Aware Gating
        scores_trans = F.softmax(self.out(h_habit), dim=-1)
        scores_expl = F.softmax(torch.matmul(h_expl, g_items.t()), dim=-1)
        
        user_state = torch.tanh(self.fusion_gate(torch.cat([h_habit, h_expl], dim=-1)))
        alpha = torch.sigmoid(torch.matmul(user_state, self.itemEmb.weight.t()))

        scores_fused = alpha * scores_trans + (1 - alpha) * scores_expl
        return scores_fused, h_habit, h_expl, alpha