# -*- coding: utf-8 -*-
"""Baseline Autoencoder.

This module contains the baseline autoencoder.

"""

import torch
import numpy as np

from torch import nn



class Encoder(nn.Module):
    """Encoder
    
    """
    def __init__(self, 
                 in_dim, 
                 z_dim,
                 h_dim=256,
                 batchnorm=False
                ):
        """
        Args:
            in_dim (int): dimensionality of input.
            z_dim (int): dimensionality of low-dimensional space.
            h_dim (int): dimensionality of intermediate layers in MLP.
            
        """
        super(Encoder, self).__init__()
        if batchnorm:
            self.fn = nn.Sequential(
                nn.Linear(in_dim, h_dim, bias=True),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Linear(h_dim, z_dim, bias=True),
                nn.LayerNorm(z_dim),
                nn.GELU(),            
            )
        else:
            self.fn = nn.Sequential(
                nn.Linear(in_dim, h_dim, bias=True),
                nn.GELU(),
                nn.Linear(h_dim, z_dim, bias=True),
                nn.GELU(),            
            )
        
        
    def forward(self, x):
        z = self.fn(x)
        return z

        
class Decoder(nn.Module):
    """Decoder
    
    """
    def __init__(self,
                z_dim,
                out_dim,
                h_dim=256
                ):
        """
        Args:
            z_dim (int): dimensionality of low-dimensional space.
            out_dim (int): dimensionality of output.
            h_dim (int): dimensionality of intermediate layers in MLP.
            
        """
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, out_dim)
        )
        
    def forward(self, Z):
        return self.fc(Z)


class AutoEncoder(nn.Module):
    """Baseline AutoEncoder.
    
    """
    def __init__(self,
               in_dim,
               z_dim,
               h_dim=256,
               batchnorm=False
              ):
        """
        Args:
            in_dim (int): dimensionality of input.
            z_dim (int): dimensionality of low-dimensional space.
            h_dim (int): dimensionality of intermediate layers in MLP.
            
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_dim, z_dim, h_dim=h_dim, batchnorm=batchnorm)
        self.decoder = Decoder(z_dim, in_dim, h_dim=h_dim)
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, X):
        z = self.encoder(X)
        X_hat = self.decoder(z)
        return self.criterion(X_hat, X)


def get_mask_np(x, y=None, perc=[5, 95]):
    """Mask for matrix elements selected for regression 
        (adapt from scVelo)

    Args:
        x (ndarray): Splicing counts projection
        y (ndarray): Unsplicing counts projection
        perc (int): percentile
    return:
        mask (ndarray): bool matrix
    """
    xy_norm = x.copy()
    if y is not None:
        y = y.copy()
        xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0, keepdims=True) - np.min(xy_norm, axis=0, keepdims=True), 1e-3, None)
        xy_norm += y / np.clip(np.max(y, axis=0, keepdims=True) - np.min(y, axis=0, keepdims=True), 1e-3, None)
    if isinstance(perc, int):
        mask = xy_norm >= np.quantile(xy_norm, perc/100, axis=0)
    else:
        lb, ub = np.quantile(xy_norm, np.array(perc)/100, axis=0, keepdims=True)
        mask = (xy_norm <= lb) | (xy_norm >= ub)
    return mask

def prod_sum_obs_np(A, B):
    """dot product and sum over axis 0 (obs) equivalent to np.sum(A * B, 0)"""
    return np.einsum("ij, ij -> j", A, B) if A.ndim > 1 else (A * B).sum()
    
def sum_obs_np(A):
    """summation over axis 0 (obs) equivalent to np.sum(A, 0)"""
    return np.einsum("ij -> j", A) if A.ndim > 1 else np.sum(A)    

def leastsq_np(x, y, fit_offset=False, constraint_positive_offset=False, 
            perc=None, norm=False):
    """Solves least squares X*b=Y for b. (adatpt from scVelo)
    
    Args:
        x (Tensor): low-dim splicing projection
        y (Tensor): low-dim unsplicing projection
        fit_offset (bool): whether fit offset
        constraint_positive_offset (bool): whether make non-negative offset
        perc (int or list of int): percentile threshold for points in regression

    returns:
        fitted offset, gamma and MSE losses
    """
    if norm:
        x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
        y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)
        x = np.clip(x, -1, 1)
        y = np.clip(y, -1, 1)
    
    if perc is not None:
        if not fit_offset:
            perc = perc[1]
        mask = get_mask_np(x, y, perc=perc)
        x, y = x * mask, y * mask
    else:
        mask = None

    xx_ = prod_sum_obs_np(x, x)
    xy_ = prod_sum_obs_np(x, y)
    n_obs = x.shape[0] if mask is None else sum_obs_np(mask)
    
    if fit_offset:
        
        x_ = sum_obs_np(x) / n_obs
        y_ = sum_obs_np(y) / n_obs
        gamma = (xy_ / n_obs - x_ * y_) / (xx_ / n_obs - x_ ** 2)
        offset = y_ - gamma * x_

        # fix negative offsets:
        if constraint_positive_offset:
            idx = offset < 0
            if gamma.ndim > 0:
                gamma[idx] = xy_[idx] / xx_[idx]
            else:
                gamma = xy_ / xx_
            offset = np.clip(offset, 0, None)
    else:
        gamma = xy_ / xx_
        offset = np.zeros(x.shape[1]) if x.ndim > 1 else 0
    nans_offset, nans_gamma = np.isnan(offset), np.isnan(gamma)
    if np.any(nans_offset) or np.any(nans_gamma):
        offset[np.isnan(offset)], gamma[np.isnan(gamma)] = 0, 0
        
    loss = (y - x * gamma.reshape(1,-1) - offset)**2
    if perc is not None:
        loss = loss * mask
    loss = sum_obs_np(loss) / n_obs
    return offset, gamma, loss        

