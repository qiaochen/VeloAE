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
veloproj --refit 0 --adata notebooks/dentategyrus/data/DentateGyrus/10X43_1.h5ad --device cuda:3 --model-name ./notebooks/dentategyrus/dentategyrus_model.cpt
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
veloproj --lr 1e-5 --nb_g_src X --gumbsoft_tau 5 --refit 1 --adata notebooks/dentategyrus/data/DentateGyrus/10X43_1.h5ad --device cuda:3 --model-name dentategyrus_model.cpt
```
- Arguments:
    - --lr: learning rate, (tunning it if the model does not learn given the default configuration)
    - --nb_g_src: expression matrix for generating neighborhood graph of cells, can be 'X' (transcriptome), 'S' (spliced), and 'U' (unspliced)
    - --gumbsoft_tau: temperature parameter of gumbel softmax function, a smaller value (e.g., 1) makes attention sparse and training more challenging, while a larger value (e.g., 10) makes attention more evenly distributed and loss more smoothly converge to lower value.
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

