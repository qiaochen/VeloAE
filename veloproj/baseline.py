# -*- coding: utf-8 -*-
"""Baseline Autoencoder.

This module contains the baseline autoencoder.

"""

import torch

from torch import nn


class Encoder(nn.Module):
    """Encoder
    
    """
    def __init__(self, 
                 in_dim, 
                 z_dim,
                 h_dim=256
                ):
        """
        Args:
            in_dim (int): dimensionality of input.
            z_dim (int): dimensionality of low-dimensional space.
            h_dim (int): dimensionality of intermediate layers in MLP.
            
        """
        super(Encoder, self).__init__()
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
               h_dim=256
              ):
        """
        Args:
            in_dim (int): dimensionality of input.
            z_dim (int): dimensionality of low-dimensional space.
            h_dim (int): dimensionality of intermediate layers in MLP.
            
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_dim, z_dim, h_dim=h_dim)
        self.decoder = Decoder(z_dim, in_dim, h_dim=h_dim)
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, X):
        z = self.encoder(X)
        X_hat = self.decoder(z)
        return self.criterion(X_hat, X)




