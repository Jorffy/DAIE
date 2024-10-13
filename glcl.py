# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


# GLCL
def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj, word_len,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True, nsw=True):
    l1 = nei_con_loss(z1, z2, tau, adj, word_len, hidden_norm, nsw)
    l2 = nei_con_loss(z2, z1, tau, adj, word_len, hidden_norm, nsw)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

def multihead_contrastive_loss(heads, adj, word_len, tau: float = 1.0, nsw=True):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(0, len(heads)-1):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, word_len, tau=tau, nsw=nsw)
    return loss / (len(heads) - 1)

def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def nag_sim_w(x, y, hidden_norm, lam=1):
    # assert len(x) == len(y), "len(x) != len(y)"
    vec1 = x
    vec2 = y
    cos_sim = F.cosine_similarity(vec1, vec2, dim=1)
    s_sim = torch.exp(cos_sim)
    w_sim = F.normalize(lam/s_sim, p=2, dim=0)
    return w_sim

def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, word_len, hidden_norm: bool = True, nsw=True):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    # adj = adj[:word_len,:word_len]
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.as_tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    if nsw:
        # NSW = True
        intra_w = nag_sim_w(z1, z1, hidden_norm, lam=1)
        inter_w = nag_sim_w(z1, z2, hidden_norm, lam=1)

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                intra_w*(intra_view_sim.mul(1-adj)).sum(1) + inter_w*(inter_view_sim.mul(1-adj)).sum(1) +
                (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1) - intra_view_sim.diag())
    else:
        # NSW = False
        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
            intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    loss = loss / nei_count  # divided by the number of positive pairs for each node

    return -torch.log(loss)

# create adj
def dep_adj_m(edge_index, max_len, input_type='texts'):
    if input_type=='texts':
        adj_ms = []
        for dep in edge_index:
            one = torch.ones(max_len)
            adj_m = torch.diag_embed(one)
            for d in dep.T:
                adj_m[d[0],d[1]] = 1
            adj_ms.append(adj_m)
        adj_ms = torch.stack(adj_ms)
    else:
        adj_ms = []
        one = torch.ones(max_len)
        adj_m = torch.diag_embed(one)
        adj_m[1:,0] = adj_m[1:,0] + 1
        adj_m[0,1:] = adj_m[0,1:] + 1
        for i in range(edge_index.size(1)):
            # edge_index[:,i]
            adj_m[edge_index[:,i][0]+1,edge_index[:,i][1]+1] = 1
        adj_ms.append(adj_m)
        adj_ms = torch.stack(adj_ms)
    return adj_ms