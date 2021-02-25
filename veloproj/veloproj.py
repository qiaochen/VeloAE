# -*- coding: utf-8 -*-
"""Command line usage support

"""
from .util import get_parser, new_adata, init_model, init_adata, fit_model, do_projection
import torch
import os
import numpy as np


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    device = torch.device(args.device)
    
    if not os.path.exists(args.adata):
        raise
        
    adata = init_adata(args)
    spliced = adata.layers['Ms']
    unspliced = adata.layers['Mu']
    tensor_s = torch.FloatTensor(spliced).to(device)
    tensor_u = torch.FloatTensor(unspliced).to(device)
    tensor_x = torch.FloatTensor(adata.X.toarray()).to(device)
    tensor_v = torch.FloatTensor(adata.layers['stc_velocity']).to(device)
    
    model = init_model(adata, args, device)
    inputs = []
    if args.use_x:
        inputs.append(tensor_x)
    if args.use_s:
        inputs.append(tensor_s)
    if args.use_u:
        inputs.append(tensor_u)        
    
    if args.refit == 1:
        model = fit_model(args, adata, model, inputs)
    else:
        if not os.path.exists(args.model_name):
            raise
            
        model.load_state_dict(torch.load(args.model_name))
        model = model.to(device)
        
    new_adata = do_projection(model,
                              adata,
                              args,
                              tensor_x, 
                              tensor_s, 
                              tensor_u, 
                              tensor_v)
    
    new_adata.write(os.path.join(args.output, "projection.h5ad"))
        