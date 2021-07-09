# -*- coding: utf-8 -*-
"""VeloAutoencoder module.

This module contains the veloAutoencoder and its ablation configurations.

"""
import torch
from torch import nn
from torch_geometric.nn import GCNConv, Sequential
from torch.nn import functional as F
import numpy as np


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
    
    def forward(self, query, key, value, k=3, device=None):
        """
        Args:
            query (torch.FloatTensor): query vectors identifying the gene profiles to be reconstructed.
            key (torch.FloatTensor): key vectors identifying the latent profiles to be attended to.
            value (torch.FloatTensor): Z
            
        Returns:
            FloatTensor: shape (n_genes, n_cells), reconstructed input
            FloatTensor: shape (n_genes, z_dim), gene by attention distribution matrix
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        p_attn = F.gumbel_softmax(scores, tau=self.gb_tau, hard=False, dim=-1)
#             p_attn = torch.softmax(scores, dim=-1)
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
            nn.GELU(),
            nn.Linear(h_dim, z_dim, bias=True),
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
               in_dim,
               z_dim,
               n_genes,
               n_cells,
               h_dim=256,
               k_dim=100,
               G_rep=None,
               g_rep_dim=None,
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
            device (torch.device): torch device object.
            
        """
        super(AblationAttComb, self).__init__()
        self.device = device
        self.encoder = AblationEncoder(in_dim, z_dim, h_dim=h_dim)
        self.decoder = Decoder(n_cells, G_rep, n_genes, g_rep_dim, k_dim, h_dim, device)
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, X):
        z = self.encoder(X)
        X_hat = self.decoder(z)
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
