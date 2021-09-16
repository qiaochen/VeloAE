# -*- coding: utf-8 -*-
"""Evaluation utility functions.

This module contains util functions for computing evaluation scores.

"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: Group-wise aggregation scores.
        float: score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s }
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg


def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe

    Returns:
        list: Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]


def cross_boundary_scvelo_probs(adata, k_cluster, cluster_edges, k_trans_g, return_raw=False):
    """Compute Cross-Boundary Confidence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_trans_g (str): key to the transition graph computed using velocity.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    
    scores = {}
    all_scores = {}
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        type_score = [trans_probs.toarray()[:, nodes].mean() 
                      for trans_probs, nodes in zip(adata.uns[k_trans_g][sel], boundary_nodes) 
                      if len(nodes) > 0]
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    if return_raw:
        return all_scores    
    return scores, np.mean([sc for sc in scores.values()])


def cross_boundary_coh(adata, k_cluster, k_velocity, cluster_edges, return_raw=False):
    """Cross-Boundary Velocity Coherence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        
        velocities = adata.layers[k_velocity]
        v_us = velocities[sel]
        type_score = [cosine_similarity(v_us[[ith]], velocities[nodes]).mean()
                      for ith, nodes in enumerate(boundary_nodes) 
                      if len(nodes) > 0]
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def in_cluster_scvelo_coh(adata, k_cluster, k_confidence, return_raw=False):
    """In-Cluster Confidence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_confidence (str): key to the column of cell velocity confidence in adata.obs.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
    
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        type_score = adata.obs[k_confidence][sel].values.tolist()
        scores[cat] = np.mean(type_score)
        all_scores[cat] = type_score
        
    if return_raw:
        return all_scores
    
    return scores, np.mean([s for _, s in scores.items()])

def cross_boundary_correctness(adata, 
                               k_cluster, 
                               k_velocity, 
                               cluster_edges, 
                               return_raw=False, 
                               x_emb="X_umap"):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding for visualization.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    
    x_emb = adata.obsm[x_emb]
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
        
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])


def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)
        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])


def evaluate(adata, cluster_edges, k_cluster, k_velocity, x_emb="X_umap", verbose=True):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): Anndata object.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        x_emb (str): key to x embedding for visualization.
        
    Returns:
        dict: aggregated metric scores.
    
    """
    
    trans_probs = cross_boundary_scvelo_probs(adata, k_cluster, cluster_edges, "{}_graph".format(k_velocity), True)
    crs_bdr_coh = cross_boundary_coh(adata, k_cluster, k_velocity, cluster_edges, True)
    crs_bdr_crc = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, True)
    ic_scvelo_coh = in_cluster_scvelo_coh(adata, k_cluster, "{}_confidence".format(k_velocity), True)
    
    if verbose:
        print("# Cross-Boundary Transition Score (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(trans_probs)))
        print("# Cross-Boundary Velocity Coherence (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_coh)))
        print("# Cross-Boundary Direction Correctness (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_crc)))
        print("# In-cluster Coherence\n{}\nTotal Mean: {}".format(*summary_scores(ic_coh)))
        print("# In-cluster Confidence\n{}\nTotal Mean: {}".format(*summary_scores(ic_scvelo_coh)))
    
    return {
        "Cross-Boundary Transition Score (A->B)": trans_probs,
        "Cross-Boundary Velocity Coherence (A->B)": crs_bdr_coh,
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
        "In-cluster Confidence": ic_scvelo_coh
       }