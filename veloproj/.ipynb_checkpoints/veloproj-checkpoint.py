# -*- coding: utf-8 -*-
"""Command line usage support

"""
from .util import get_parser, new_adata, init_model, init_adata, fit_model, do_projection
import torch
import os, logging, timeit
import numpy as np


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    start_time = timeit.default_timer()
    logger.info("Start processing...")
    
    parser = get_parser()
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    device = torch.device(args.device)
    
    if not os.path.exists(args.adata):
        raise Exception(f"Did not find Anndata based on input path: {args.adata}. \nPlease specify a correct path.")
    
    logger.info("Initializing data...")
    adata = init_adata(args)
    logger.info("Finished data preprocessing.")
    logger.info(f"{timeit.default_timer() - start_time:.2}s passed.")
    
    spliced = adata.layers['Ms']
    unspliced = adata.layers['Mu']
    tensor_s = torch.FloatTensor(spliced).to(device)
    tensor_u = torch.FloatTensor(unspliced).to(device)
    tensor_x = torch.FloatTensor(adata.X.toarray()).to(device)
    tensor_v = torch.FloatTensor(adata.layers['stc_velocity']).to(device)
    
    logger.info("Initializing model...")
    model = init_model(adata, args, device)
    logger.info("Finished model initialization...")
    logger.info(f"{timeit.default_timer() - start_time:.2}s passed.")
    
    inputs = [tensor_s, tensor_u]
    xyids = [0, 1]
    if args.use_x:
        inputs.append(tensor_x)
    
    if args.refit == 1:
        logger.info("Fitting model...")
        model = fit_model(args, adata, model, inputs, xyids, device)
        logger.info("Finished model fitting.")
        logger.info(f"{(timeit.default_timer() - start_time)/60:.2}min passed.")
    else:
        if not os.path.exists(args.model_name):
            raise Exception(f"Did not find a valid model file following input path: {args.model_name}. \nPlease specify a trained model, when --refiting is turned off (0).")
            
        logger.info("Loading model...")
        model.load_state_dict(torch.load(args.model_name))
        model = model.to(device)
        
    logger.info("Do projection...")
    ld_adata = do_projection(model,
                             adata,
                             args,
                             tensor_x, 
                             tensor_s, 
                             tensor_u, 
                             tensor_v,
                             device=device)
    logger.info("Finished projection...")
    logger.info(f"{(timeit.default_timer() - start_time)/60:.2}min passed.")
    
    ld_adata.write(os.path.join(args.output, args.ld_adata))
    logger.info(f'Low-dimensional results saved in {os.path.join(args.output, args.ld_adata)}')
        
