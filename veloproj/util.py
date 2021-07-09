# -*- coding: utf-8 -*-
"""Utility function module.

This module contains util functions.

"""
import torch
import numpy as np
import os, sys
import anndata
import scvelo as scv
import scanpy as sc
import argparse

from matplotlib import pyplot as plt


def get_weight(x, y=None, perc=95, device=None):
    xy_norm = torch.clone(x)
    if y is not None:
        y = torch.clone(y)
        xy_norm = xy_norm / torch.clip(torch.max(xy_norm, axis=0).values - torch.min(xy_norm, axis=0).values, 1e-3, None)
        xy_norm += y / torch.clip(torch.max(y, axis=0).values - torch.min(y, axis=0).values, 1e-3, None)
    if isinstance(perc, int):
        weights = xy_norm >= torch.quantile(xy_norm, perc/100, dim=0, keepdim=True)
    else:
        lb, ub = torch.quantile(xy_norm, torch.Tensor(perc).to(device)/100, dim=0, keepdim=True)
        weights = (xy_norm <= lb) | (xy_norm >= ub)
    return weights

def prod_sum_obs(A, B):
    """dot product and sum over axis 0 (obs) equivalent to np.sum(A * B, 0)"""
    return torch.einsum("ij, ij -> j", A, B) if A.ndim > 1 else (A * B).sum()
    
def sum_obs(A):
    """summation over axis 0 (obs) equivalent to np.sum(A, 0)"""
    return torch.einsum("ij -> j", A) if A.ndim > 1 else torch.sum(A)    

def leastsq_NxN(x, y, fit_offset=False, perc=None, constraint_positive_offset=False, device=None):
    """Solves least squares X*b=Y for b."""
    if perc is not None:
        if not fit_offset:
            perc = perc[1]
        weights = get_weight(x, y, perc=perc, device=device)
        x, y = x * weights, y * weights
    else:
        weights = None


    xx_ = prod_sum_obs(x, x)
    xy_ = prod_sum_obs(x, y)
    n_obs = x.shape[0] if weights is None else sum_obs(weights)
    
    if fit_offset:
        
        x_ = sum_obs(x) / n_obs
        y_ = sum_obs(y) / n_obs
        gamma = (xy_ / n_obs - x_ * y_) / (xx_ / n_obs - x_ ** 2)
        offset = y_ - gamma * x_

        # fix negative offsets:
        if constraint_positive_offset:
            idx = offset < 0
            if gamma.ndim > 0:
                gamma[idx] = xy_[idx] / xx_[idx]
            else:
                gamma = xy_ / xx_
            offset = torch.clip(offset, 0, None)
    else:
        gamma = xy_ / xx_
        offset = torch.zeros(x.shape[1]).to(device) if x.ndim > 1 else 0
    nans_offset, nans_gamma = torch.isnan(offset), torch.isnan(gamma)
    if torch.any(nans_offset) or torch.any(nans_gamma):
        offset[torch.isnan(offset)], gamma[torch.isnan(gamma)] = 0, 0
        
    loss = torch.square(y - x * gamma.view(1,-1) - offset)
    if perc is not None:
        loss = loss * weights
    loss = sum_obs(loss) / n_obs
    return offset, gamma, loss


def get_parser():
    """Get the argument parser
    
    Returns:
        Parser object.
        
    """
    parser = argparse.ArgumentParser(description='VeloAutoencoder Experiment settings')
    parser.add_argument('--data-dir', type=str, default=None, metavar='PATH',
                        help='default directory to adata file')
    parser.add_argument('--model-name', type=str, default="tmp_model.cpt", metavar='PATH',
                        help="""save the trained model with this name in training, or 
                                read the model for velocity projection if the model has
                                already been trained.
                             """
                       )
    parser.add_argument('--exp-name', type=str, default="experiment", metavar='PATH',
                        help='name of the experiment')
    parser.add_argument('--adata', type=str, metavar='PATH', 
                        help="""path to the Anndata file with transcriptom, spliced and unspliced
                                mRNA expressions, the adata should be already preprocessed and with
                                velocity estimated in the original space."""
                       )
    parser.add_argument('--use_x', type=bool, default=True,
                        help="""whether or not to enroll transcriptom reads for training 
                                (default: True)."""
                       )
    parser.add_argument('--use_s', type=bool, default=True,
                        help="""whether or not to enroll spliced mRNA reads for training 
                                (default: True)."""
                       )
    parser.add_argument('--use_u', type=bool, default=True,
                        help="""whether or not to enroll unspliced mRNA reads for training 
                                (default: True)."""
                       )
    parser.add_argument('--refit', type=int, default=1,
                        help="""whether or not refitting veloAE, if False, need to provide
                                a fitted model for velocity projection. (default=1)
                             """
                       )
    parser.add_argument('--output', type=str, default="./",
                        help="Path to output directory (default: ./)"
                       )
    parser.add_argument('--vis-key', type=str, default="X_umap",
                        help="Key to visualization embeddings in adata.obsm (default: X_umap)"
                       )
    parser.add_argument('--z-dim', type=int, default=100,
                        help='dimentionality of the hidden representation Z (default: 100)')
    parser.add_argument('--g-rep-dim', type=int, default=100,
                        help='dimentionality of gene representation (default: 256)')
    parser.add_argument('--h-dim', type=int, default=256,
                        help='dimentionality of intermedeate layers of MLP (default: 256)')
    parser.add_argument('--k-dim', type=int, default=50,
                        help='dimentionality of attention keys/queries (default: 50)')
    parser.add_argument('--conv-thred', type=float, default=1e-6,
                        help='convergence threshold of early-stopping (default: 1e-6)')
    parser.add_argument('--n-epochs', type=int, default=20000, metavar='N',
                        help='number of epochs to train (default: 20000)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay strength (default 0.0)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how frequntly logging training status (default: 100)')
    parser.add_argument('--device', type=str, default="cpu",
                        help='specify device: e.g., cuda:0, cpu (default: cpu)')
    return parser


def init_model(adata, args, device):
    """Initialize a model
    
    Args:
        adata (Anndata): Anndata object.
        args (ArgumentParser): ArgumentParser instance.
        device (torch.device): device instance
        
    Returns:
        nn.Module: model instance
    """
    from sklearn.decomposition import PCA
    n_cells, n_genes = adata.X.shape
    G_embeddings = PCA(n_components=args.g_rep_dim).fit_transform(adata.X.T.toarray())
    model = get_veloAE(
                     adata, 
                     args.z_dim, 
                     n_genes, 
                     n_cells, 
                     args.h_dim, 
                     args.k_dim, 
                     G_embeddings=G_embeddings, 
                     g_rep_dim=args.g_rep_dim,
                     device=device
                    )
    return model


def fit_model(args, adata, model, inputs):
    """Fit a velo autoencoder
    
    Args:
        args (ArgumentParser): ArgumentParser object
        adata (Anndata): Anndata object
        model (nn.Module): VeloAE instance
        inputs (list of tensors): inputs for training VeloAE, e.g., [x, s, u]
    
    Returns:
        nn.Module: Fitted model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  weight_decay=args.weight_decay)
    
    model.train()
    i, losses = 0, [sys.maxsize]
    while i < args.n_epochs:
        i += 1
        loss = train_step_AE(inputs, model, optimizer)                
        losses.append(loss)
        if i % args.log_interval == 0:
            print("Train Epoch: {:2d}/{:2d} \tLoss: {:.6f}"
                  .format(i, args.n_epochs, losses[-1]))

    plt.plot(losses[1:])
    plt.savefig(os.path.join(args.output, "training_loss.png"))
    torch.save(model.state_dict(), os.path.join(args.output, args.model_name))
    return model

def do_projection(model, 
                  adata,
                  args,
                  tensor_x, 
                  tensor_s, 
                  tensor_u, 
                  tensor_v
                 ):
    """Project everything into the low-dimensional space
    
    Args:
        model (nn.Module): trained Model instance.
        adata (Anndata): Anndata instance in the raw dimension.
        args (ArgumentParser): ArgumentParser instance.
        tensor_x (FloatTensor): transcriptom expressions.
        tensor_s (FloatTensor): spliced mRNA expressions.
        tensor_u (FloatTensor): unspliced mRNA expressions.
        tensor_v (FloatTensor): Velocity in the raw dimensional space.
        
    Returns:
        Anndata: Anndata object in the latent space.
    
    """
    x = model.encoder(tensor_x).detach().cpu().numpy()
    s = model.encoder(tensor_s).detach().cpu().numpy()
    u = model.encoder(tensor_u).detach().cpu().numpy()
    v = model.encoder(tensor_s + tensor_v).detach().cpu().numpy() - s
    
    new_adata = anndata.AnnData(x)
    new_adata.layers['spliced'] = s
    new_adata.layers['unspliced'] = u
    new_adata.layers['velocity'] = v
    new_adata.obs.index = adata.obs.index.copy()
    
    for key in adata.obs:
        new_adata.obs[key] = adata.obs[key].copy()
    for key in adata.obsm:
        new_adata.obsm[key] = adata.obsm[key].copy()
    for key in adata.obsp:
        new_adata.obsp[key] = adata.obsp[key].copy()
    for clr in [key for key in adata.uns if key.split("_")[-1] == 'colors' ]:
        new_adata.uns[clr] = adata.uns[clr]
    
    scv.pp.moments(new_adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity_graph(new_adata, vkey='velocity')
    scv.pl.velocity_embedding_stream(new_adata, vkey="velocity", basis=args.vis_key,
                                    title="Project Original Velocity into Low-Dim Space",
                                    save='un_colored_velo_projection.png'
                                    )
    return new_adata


def init_adata(args):
    """Initialize Anndata object
    
    Args:
        args (ArgumentParser): ArgumentParser instance
        
    Returns:
        Anndata: preprocessed Anndata instance
    """
    adata = sc.read(args.adata)
    scv.utils.show_proportions(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, vkey='stc_velocity', mode="stochastic")
    return adata

    
def new_adata(adata, x, s, u, v=None, 
              copy_moments=True, 
              new_v_key="new_velocity", 
              X_emb_key="X_umap",
              g_basis="SU",
             ):
    """Copy a new Anndata object while keeping some original information.
    
    Args:
        adata (Anndata): Anndata object
        x (np.ndarray): new transcriptome.
        s (np.ndarray): new spliced mRNA expression.
        u (np.ndarray): new unspliced mRNA expression.
        v (np.ndarray): new velocity.
        copy_moments (bool): whether to copy the moments.
        X_emb_key (str): key string of the embedding of X for visualization.
    
    Returns:
        Anndata: a new Anndata object
    
    """
    from sklearn.decomposition import PCA
    new_adata = anndata.AnnData(x)
    new_adata.layers['spliced'] = s
    new_adata.layers['unspliced'] = u
    if not v is None:
        new_adata.layers[new_v_key] = v
        
    new_adata.obs.index = adata.obs.index.copy()
    
    for key in adata.obs:
        new_adata.obs[key] = adata.obs[key].copy()
    
    basis_dict = {"X":adata.X, "S":adata.layers['spliced'], "U":adata.layers['unspliced']}
    new_adata.obsm['X_pca'] = PCA(n_components=100).fit_transform(np.hstack([basis_dict[k].toarray() for k in g_basis]))
    scv.pp.moments(new_adata, n_pcs=30, n_neighbors=30)
        
    for clr in [key for key in adata.uns if key.split("_")[-1] == 'colors' ]:
        new_adata.uns[clr] = adata.uns[clr]
        
#     new_adata.uns['neighbors'] = adata.uns['neighbors'].copy()
    new_adata.obsm[X_emb_key] = adata.obsm[X_emb_key].copy()
    return new_adata


def train_step_AE(Xs, model, optimizer, SV, xyids=[-2, -1], device=None, lreg_weight=3):
    """Conduct a train step.
    
    Args:
        Xs (list[FloatTensor]): inputs for Autoencoder
        model (nn.Module): Instance of Autoencoder class
        optimizer (nn.optim.Optimizer): instance of pytorch Optimizer class
    
    Returns:
        float: loss of this step.
        
    """
    optimizer.zero_grad()
    loss = 0
    for X in Xs:
        loss += model(X)
    ae_loss = loss.item()
    s, u = model.encoder(Xs[xyids[0]]), model.encoder(Xs[xyids[1]])
    _, gamma, vloss = leastsq_NxN(
                              s, u,
                              fit_offset=True, 
                              perc=[5, 95],
                              device=device)
#     v = model.encoder(SV + Xs[xyids[0]])  - s
#     vloss = torch.square((u - gamma * s) - v).sum()
    vloss = torch.sum(vloss) * lreg_weight
    loss += vloss
    
    loss.backward()
    optimizer.step()
    return loss.item(), ae_loss, vloss.item()


def sklearn_decompose(method, X, S, U, V):
    """General interface using sklearn.decomposition.XXX method
    
    Args:
        method (sklearn.decomposition class): e.g., instance of sklearn.decomposition.PCA
        X (np.ndarray): High-dimensional transcriptom.
        S (np.ndarray): High-dimensional spliced mRNA expression.
        U (np.ndarray): High-dimensional unspliced mRNA expression.
        V (np.ndarray): High-dimensional cell velocity estimation.
        
    Returns:
        np.ndarray: decomposed low-dimensional representations for X, S, U and V
    
    """
    n_cells = X.shape[0]
    X_orig = np.concatenate([
                    X, 
                    S, 
                    U
                   ], axis=0)
    
    method.fit(X_orig)
    x = method.transform(X)
    s = method.transform(S)
    u = method.transform(U)
    v = method.transform(S + V) - s
    return x, s, u, v
    
    
def get_baseline_AE(in_dim, z_dim, h_dim):
    """Instantiate a Baseline Autoencoder.
    
    Args:
        in_dim (int): dimensionality of input.
        z_dim (int): dimensionality of low-dimensional space.
        h_dim (int): dimensionality of intermediate layers in MLP.
            
    Returns:
        nn.Module: AE instance
    
    
    """    
    from .baseline import AutoEncoder
    model = AutoEncoder(
                in_dim,
                z_dim,
                h_dim
                )
    return model


def get_ablation_CohAgg(
                edge_index,
                edge_weight,
                in_dim,
                z_dim,
                h_dim,
                device):
    """Get Ablation Cohort Aggregation instance
    
    Args:
        edge_index (LongTensor): shape (2, ?), edge indices
        edge_weight (FloatTensor): shape (?), edge weights.
        in_dim (int): dimensionality of the input
        z_dim (int): dimensionality of the low-dimensional space
        h_dim (int): dimensionality of intermediate layers in MLP
        device (torch.device): torch device object.

    Returns:
        nn.Module: model instance
    """
    from .model import AblationCohAgg
    model = AblationCohAgg(
        edge_index,
        edge_weight,
        in_dim,
        z_dim,
        h_dim,
        device
    )
    return model.to(device)


def get_ablation_attcomb(
                        z_dim,
                        n_genes,
                        n_cells,
                        h_dim,
                        k_dim,
                        G_rep,
                        g_rep_dim,
                        device):
    """Instantiate an AttenComb configuration for ablation study.
    
    Args:
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
    
    Returns:
        nn.Module: model instance
    """
    from .model import AblationAttComb
    model = AblationAttComb(
        n_genes,
        z_dim,
        n_genes,
        n_cells,
        h_dim,
        k_dim,
        G_rep,
        g_rep_dim,
        device
    )
    return model.to(device)


def get_veloAE(
             adata, 
             z_dim, 
             n_genes, 
             n_cells, 
             h_dim, 
             k_dim,
             G_embeddings=None, 
             g_rep_dim=100,
             gb_tau=1.0,
             g_basis="SU",
             device=None):
    """Instantiate a VeloAE object.
    
    Args:
        adata (Anndata): Anndata object
        z_dim (int): dimensionality of the low-dimensional space
        n_genes (int): number of genes
        n_cells (int): number of cells
        h_dim (int): dimensionality of intermediate layers in MLP
        k_dim (int): dimensionality of keys for attention computation
        G_embeddings (np.ndarry): representation for genes, e.g. PCA over gene profiles.
        g_rep_dim (int): dimensionality of gene representations.
            # Either G_rep or (n_genes, g_rep_dim) should be provided.
            # priority is given to G_rep.
        device (torch.device): torch device object.
    
    Returns:
        nn.Module: model instance
    """
    from .model import VeloAutoencoder
    from sklearn.decomposition import PCA
    adata = adata.copy()
    basis_dict = {"X":adata.X, "S":adata.layers['spliced'], "U":adata.layers['unspliced']}
    # more informative neighborhood construction
    adata.obsm['X_pca'] = PCA(n_components=100).fit_transform(np.hstack([basis_dict[k].toarray() for k in g_basis]))
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    conn = adata.obsp['connectivities']
    nb_indices = adata.uns['neighbors']['indices']
    xs, ys = np.repeat(range(n_cells), nb_indices.shape[1]-1), nb_indices[:, 1:].flatten()
    edge_weight = torch.FloatTensor(conn[xs,ys]).view(-1).to(device)
    edge_index = torch.LongTensor(np.vstack([xs.reshape(1,-1), xs.reshape(1, -1)])).to(device)
    model = VeloAutoencoder(
                edge_index,
                edge_weight,
                n_genes,
                z_dim,
                n_genes,
                n_cells,
                h_dim=h_dim,
                k_dim=k_dim,
                G_rep=G_embeddings,
                g_rep_dim=g_rep_dim,
                gb_tau=gb_tau,
                device=device
                )
    return model.to(device)