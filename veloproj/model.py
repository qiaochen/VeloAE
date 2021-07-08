# -*- coding: utf-8 -*-
"""VeloAutoencoder module.

This module contains the veloAutoencoder and its ablation configurations.

"""
import torch
import numpy as np

from torch import nn
from torch_geometric.nn import GCNConv, Sequential
from torch.nn import functional as F


class Encoder(nn.Module):
    """Encoder
    
    """
    def __init__(self, 
                 in_dim, 
                 z_dim,
                 edge_index,
                 edge_weight,
                 h_dim=256
                ):
        """
        Args:
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            edge_index (LongTensor): shape (2, ?), edge indices
            edge_weight (FloatTensor): shape (?), edge weights.
            h_dim (int): dimensionality of intermediate layers in MLP
            
        """
        super(Encoder, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.fn = nn.Sequential(
            nn.Linear(in_dim, h_dim, bias=True),
            nn.GELU(),
            nn.Linear(h_dim, h_dim, bias=True),
            nn.GELU(),
        )
        self.gc = Sequential( "x, edge_index, edge_weight", 
            [(GCNConv(h_dim, z_dim, cached=False, add_self_loops=True), "x, edge_index, edge_weight -> x"),
              nn.GELU(),
             (GCNConv(z_dim, z_dim, cached=False, add_self_loops=True), "x, edge_index, edge_weight -> x"),
              nn.GELU(),
              nn.Linear(z_dim, z_dim)]
        )
        self.gen = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=True),
        )
        
    def forward(self, x, return_raw=False):
        z = self.fn(x)
        z = self.gc(z, self.edge_index, self.edge_weight)
        if return_raw:
            return self.gen(z), z
        return self.gen(z)
    
        
class Decoder(nn.Module):
    """Decoder
    
    """
    def __init__(self,
                z_col_dim,
                G_rep=None,
                n_genes=None,
                g_rep_dim=None,
                k_dim=32,
                h_dim=256,
                gb_tau=1.0,
                device=None
                ):
        """
        Args:
            z_col_dim (int): size of column vectors in Z.
            G_rep (np.ndarry): representation for genes, e.g. PCA over gene profiles.
            n_genes (int): number of genes.
            g_rep_dim (int): dimensionality of gene representations.
                # Either G_rep or (n_genes, g_rep_dim) should be provided.
                # priority is given to G_rep.
            k_dim (int): dimensionality of keys for attention computation.
            h_dim (int): dimensionality of intermediate layers of MLP.
            gb_tau (float): temperature param of gumbel softmax
            device (torch.device): torch device object.
            
        """
        super(Decoder, self).__init__()
        self.device = device
        if not G_rep is None:
            g_rep_dim = G_rep.shape[-1]
        self.key_Z = nn.Sequential(
            nn.Linear(z_col_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, k_dim)
        )
        self.key_G = nn.Sequential(
            nn.Linear(g_rep_dim, k_dim),
            nn.GELU(),
            nn.Linear(k_dim, k_dim)
        )
        self.G_rep = self._init_G_emb(n_genes, g_rep_dim) if G_rep is None else torch.FloatTensor(G_rep).to(device)
        self.attn = Attention(gb_tau)
        
    def _init_G_emb(self, n_genes, rep_dim):
        embedder = torch.empty(n_genes, rep_dim)
        nn.init.xavier_normal_(embedder)
        return nn.Parameter(embedder).to(self.device)
        
    def forward(self, raw_Z, gen_Z, return_attn=False):
        Z = raw_Z.T
        key = self.key_Z(Z)
        query = self.key_G(self.G_rep)
        X_hat_means, p_attn = self.attn(query, key, gen_Z.T, device=self.device)
        if return_attn:
            return X_hat_means.T, p_attn.T
        return X_hat_means.T
    
    
class Attention(nn.Module):
    """Compute 'Scaled Dot Product Attention'.
    
    """
    def __init__(self, gb_tau=1.0):
        super(Attention, self).__init__()
        self.gb_tau = gb_tau
    
    def forward(self, query, key, value, device=None):
        """
        Args:
            query (torch.FloatTensor): query vectors identifying the gene profiles to be reconstructed.
            key (torch.FloatTensor): key vectors identifying the latent profiles to be attended to.
            value (torch.FloatTensor): Z.
            device (torch.device): torch device object.
            
        Returns:
            FloatTensor: shape (n_genes, n_cells), reconstructed input
            FloatTensor: shape (n_genes, z_dim), gene by attention distribution matrix
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        p_attn = F.gumbel_softmax(scores, tau=self.gb_tau, hard=False, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    
class AblationEncoder(nn.Module):
    """Encoder for Ablation Study
    
    """
    def __init__(self, 
                 in_dim, 
                 z_dim,
                 h_dim=256
                ):
        super(AblationEncoder, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_dim, h_dim, bias=True),
            nn.BatchNorm1d(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, z_dim, bias=True),
            nn.BatchNorm1d(z_dim),
            nn.GELU(),
        )
                
    def forward(self, x):
        z = self.fn(x)
        return z    
        
class AblationDecoder(nn.Module):
    """Decoder for Ablation Study.
    
    """
    def __init__(self,
                z_dim,
                out_dim,
                h_dim=256
                ):
        super(AblationDecoder, self).__init__()
        """
        """
        self.fc = nn.Sequential(
            nn.Linear(z_dim, out_dim),
        )
        
    def forward(self, Z):
        return self.fc(Z)   
    
class AblationCohAgg(nn.Module):
    """Ablation with only Cohort Aggregation.
    
    """
    
    def __init__(self,
               edge_index,
               edge_weight,
               in_dim,
               z_dim,
               h_dim=256,
               device=None
              ):
        """
        Args:
            edge_index (LongTensor): shape (2, ?), edge indices
            edge_weight (FloatTensor): shape (?), edge weights.
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            h_dim (int): dimensionality of intermediate layers in MLP
            device (torch.device): torch device object.
            
        """
        super(AblationCohAgg, self).__init__()
        self.device = device
        self.encoder = Encoder(in_dim, z_dim, edge_index, edge_weight, h_dim=h_dim)
        self.decoder = AblationDecoder(z_dim, in_dim, h_dim)
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, X):
        z = self.encoder(X)
        X_hat = self.decoder(z)
        return self.criterion(X_hat, X)
    
    
class AblationAttComb(nn.Module):
    """Ablation with only Attentive Combination.
    
    """
    def __init__(self,
               z_dim,
               n_genes,
               n_cells,
               h_dim=256,
               k_dim=100,
               G_rep=None,
               g_rep_dim=None,
               gb_tau=1.0,
               device=None
              ):
        """
        
        Args:
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            n_genes (int): number of genes
            n_cells (int): number of cells
            h_dim (int): dimensionality of intermediate layers in MLP
            k_dim (int): dimensionality of keys for attention computation
            G_rep (np.ndarry): representation for genes, e.g. PCA over gene profiles.
            g_rep_dim (int): dimensionality of gene representations.
                # Either G_rep or (n_genes, g_rep_dim) should be provided.
                # priority is given to G_rep.
            gb_tau (float): temperature parameter for gumbel softmax,
            device (torch.device): torch device object.
            
        """
        super(AblationAttComb, self).__init__()
        self.device = device
        self.encoder = AblationEncoder(n_genes, z_dim, h_dim=h_dim)
        self.trans_z = nn.Linear(z_dim, z_dim, bias=True)
        self.decoder = Decoder(n_cells, G_rep, n_genes, g_rep_dim, k_dim, h_dim, gb_tau, device)
        self.criterion = nn.MSELoss(reduction='mean')
        
        
    def forward(self, X):
        z = self.encoder(X)
        gen_z = self.trans_z(z)
        X_hat = self.decoder(z, gen_z, False)
        return self.criterion(X_hat, X)    
    

class VeloAutoencoder(nn.Module):
    """Proposed VeloAutoencoder with both mechanisms.
        
    """
    
    def __init__(self,
               edge_index,
               edge_weight,
               in_dim,
               z_dim,
               n_genes,
               n_cells,
               h_dim=256,
               k_dim=32,
               G_rep=None,
               g_rep_dim=None,
               gb_tau=1.0,
               device=None
              ):
        """
        
        Args:
            edge_index (LongTensor): shape (2, ?), edge indices
            edge_weight (FloatTensor): shape (?), edge weights.
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            n_genes (int): number of genes
            n_cells (int): number of cells
            h_dim (int): dimensionality of intermediate layers in MLP
            k_dim (int): dimensionality of keys for attention computation
            G_rep (np.ndarry): representation for genes, e.g. PCA over gene profiles.
            g_rep_dim (int): dimensionality of gene representations.
                # Either G_rep or (n_genes, g_rep_dim) should be provided.
                # priority is given to G_rep.
            gb_tau (float): temperature parameter for gumbel softmax,
            device (torch.device): torch device object.
            
        """
        super(VeloAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(in_dim, z_dim, edge_index, edge_weight, h_dim=h_dim)
        self.decoder = Decoder(n_cells, G_rep, n_genes, g_rep_dim, k_dim, h_dim, gb_tau, device)
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, X):
        gen_z, raw_z = self.encoder(X, True)
        X_hat = self.decoder(raw_z, gen_z, False)
        return self.criterion(X_hat, X)


def get_mask_pt(x, y=None, perc=[5, 95], device=None):
    """Mask for matrix elements selected for regression 
        (adapt from scVelo)

    Args:
        x (Tensor): Splicing counts projection
        y (Tensor): Unsplicing counts projection
        perc (int): percentile
    return:
        mask (Tensor): bool matrix
    """
    with torch.no_grad():
        xy_norm = torch.clone(x)
        if y is not None:
            y = torch.clone(y)
            xy_norm = xy_norm / torch.clip(torch.max(xy_norm, axis=0).values - torch.min(xy_norm, axis=0).values, 1e-3, None)
            xy_norm += y / torch.clip(torch.max(y, axis=0).values - torch.min(y, axis=0).values, 1e-3, None)
        if isinstance(perc, int):
            mask = xy_norm >= torch.quantile(xy_norm, perc/100, dim=0)
        else:
            lb, ub = torch.quantile(xy_norm, torch.Tensor(perc).to(device)/100, dim=0, keepdim=True)
            mask = (xy_norm <= lb) | (xy_norm >= ub)
    return mask

def prod_sum_obs_pt(A, B):
    """dot product and sum over axis 0 (obs) equivalent to np.sum(A * B, 0)"""
    return torch.einsum("ij, ij -> j", A, B) if A.ndim > 1 else (A * B).sum()
    
def sum_obs_pt(A):
    """summation over axis 0 (obs) equivalent to np.sum(A, 0)"""
    return torch.einsum("ij -> j", A) if A.ndim > 1 else torch.sum(A)    

def leastsq_pt(x, y, fit_offset=False, constraint_positive_offset=False, 
            perc=None, device=None, clamp=None):
    """Solves least squares X*b=Y for b. (adatpt from scVelo)
    
    Args:
        x (Tensor): low-dim splicing projection
        y (Tensor): low-dim unsplicing projection
        fit_offset (bool): whether fit offset
        constraint_positive_offset (bool): whether make non-negative offset
        perc (int or list of int): percentile threshold for points in regression
        device (torch.device): GPU/CPU device object
        clamp (float): normalize and clamp x, y to [-clamp, clamp], for stable fitting in baseline AE

    returns:
        fitted offset, gamma and MSE losses
    """
    if not clamp is None:
        x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
        y = (y - torch.mean(y, dim=0)) / torch.std(y, dim=0)
        x = torch.clamp(x, min=-abs(clamp), max=abs(clamp))
        y = torch.clamp(y, min=-abs(clamp), max=abs(clamp))
        
    if perc is not None:
        if not fit_offset:
            perc = perc[1]
        mask = get_mask_pt(x, y, perc=perc, device=device)
    else:
        mask = None

    xx_ = prod_sum_obs_pt(x, x)
    xy_ = prod_sum_obs_pt(x, y)
    n_obs = x.shape[0] if mask is None else sum_obs_pt(mask)

    if fit_offset:
        
        x_ = sum_obs_pt(x) / n_obs
        y_ = sum_obs_pt(y) / n_obs

        gamma = (xy_ / n_obs - x_ * y_) / (xx_ / n_obs - x_ ** 2)
        offset = y_ - gamma * x_

        # fix negative offsets:
        if constraint_positive_offset:
            idx = offset < 0
            if gamma.ndim > 0:
                gamma[idx] = xy_[idx] / xx_[idx]
            else:
                gamma = xy_ / xx_
            offset = torch.clamp(offset, 0, None)
    else:
        gamma = xy_ / xx_
        offset = torch.zeros(x.shape[1]).to(device) if x.ndim > 1 else 0
    
    # print("n_obs : ", torch.min(n_obs).item(), torch.max(n_obs).item())
    # print("xx: ", torch.max(xx_).item())
    # print("xy: ", torch.max(xy_).item())
    # print("x_: ", torch.max(x_).item())
    # print("y_: ", torch.max(y_).item())
    # print("gamma, offset: ", torch.max(gamma).item(), torch.max(offset).item())
    # print("gamma, offset: ", torch.isinf(gamma).sum().item(), torch.isinf(offset).sum().item())
    # offset_isinf = torch.isinf(offset)
    # print(f"inf gamma: {gamma[offset_isinf]}, xy_: {xy_[offset_isinf]}, xx_: {xx_[offset_isinf]}, x_: {x_[offset_isinf]}, y_: {y_[offset_isinf]},  ")
    
    nans_offset, nans_gamma = torch.isnan(offset), torch.isnan(gamma)
    if torch.any(nans_offset) or torch.any(nans_gamma):
        offset[torch.isnan(offset)], gamma[torch.isnan(gamma)] = 0, 0

    loss = torch.square(y - x * gamma.view(1,-1) - offset)
    if perc is not None:
        loss = loss * mask
    loss = sum_obs_pt(loss) / n_obs
    # print(torch.max(loss).item(), torch.min(loss).item(), torch.mean(loss).item())
    
    return offset, gamma, loss

