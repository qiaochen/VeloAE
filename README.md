# VeloRep
Low-dimensional Projection of Single Cell Velocity with AutoEncoder

## Install

#### Dependencies:

- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- anndata>=0.7.4
- scanpy>=1.5.1
- scvelo>=0.2.2
- [pytorch>=1.7](https://pytorch.org/get-started/locally/)
- [pytorch-geometric>=1.6.3](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


#### Install using pip

```
pip install git+https://github.com/qiaochen/VeloRep
```

## Usage

### Command line usage

Basic execution command and arguments:
#### Project velocity into low-dimensional space using a trained model.

```
veloproj --refit 0 --adata /data/groups/yhhuang/scNT/neuron_splicing_lite.h5ad --device cuda:3 --model-name ./notebooks/scNTseq/scNT_model.cpt
```
- Arguments:
    - --refit: if 0, do not fit a new model, should specify an existing model for projection.
    - --adata: path to the Anndata with X_umap, transcriptom, spliced and unspliced mRNA expressions.
    - --device: gpu or cpu (e.g., cuda:2). Fitting using GPU is much faster than cpu.
    - --model-name: path to the existing model.
    
- Output:
    - A new low-dimensional Anndata object with default name "projection.h5ad" in the output folder (default ./)
    - An uncolored figure showing the low-dimensional velocity field, stored in folder './figures'.

#### Fit a new veloAE model and project velocity into low-dimensional space.

```
veloproj --lr 8e-7 --refit 1 --adata /data/groups/yhhuang/scNT/neuron_splicing_lite.h5ad --device cuda:3 --model-name scNT_model.cpt 
```
- Arguments:
    - --lr: learning rate, (should be tunned to minimize loss robustly)
    - --refit: if 1, fit a new model
    - --adata: path to the Anndata with X_umap, transcriptom, spliced and unspliced mRNA expressions.
    - --device: gpu or cpu. Fitting using GPU is much faster than cpu.
    - --model-name: path for storing the trained model.
    
- Output:
    - A trained model with "model-name" in the output folder (default ./).
    - A plot showing the training loss curve in the output folder.
    - A new low-dimensional Anndata instance with default name "projection.h5ad" in the output folder (default ./)
    - An uncolored figure showing the low-dimensional velocity field, stored in folder './figures'.

Use command line help to investigate more arguments.
```
veloproj -h
```

