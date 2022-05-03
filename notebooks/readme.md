## New feartures experimented:
New Feature: Transfer knowledge from a given velocity estimates in the high-dimentional space into latent space.

In notebooks `retina` and `oligodendrocyte`, we experimented a new feature to enhance veloAE with signals from a given velocity estimation, e.g., a velocity matrix inferred with scVelo stochastic mode in the high-dimensional space.

Briefly, now we can have two versions of velocities in the low-dimensional space of veloAE:

1. a latent velocity of steady-state estimation using projected Spliced and Unspliced reads in the low-dimensional space;
2. a projected velocity by passing the given velocity from the raw space through the encoder of veloAE;

During training, we may choose to add an auxiliary loss/constraint to make the two versions of velocites to be close, hence transfering the knowledge from, e.g., scvelo to veloAE.

In
- `retina`, we apply this auxiliary loss to all cell types, making veloAE function as denoising the output of scvelo stochastic mode.
- `Oligodendrocyte`, we apply this auxiliary loss to only the `NFOLs`  cell type, encouraging veloAE to retain the velocity for the designated cell type. This new feature provides an interface for injecting only part of the knowledge (e.g., velocites of cell clusters we are more confident) from previous estimates that we are more certain about.






## Retrieving the Datasets

### Online available datasets
- DentateGyrus: <https://github.com/theislab/scvelo_notebooks/raw/master/data/DentateGyrus/10X43_1.h5ad>
- Human Bone Marrow: <https://ndownloader.figshare.com/files/27686835>
- Pancreas: <https://ndownloader.figshare.com/files/25060877>
- Reprogramming: <https://ndownloader.figshare.com/files/25503773>

### Datasets available by requesting
- Mouse Bone Marrow: [Dr. Peter Kharchenko](https://www.nature.com/articles/s41586-018-0414-6)
- Intestinal Epithelium: [Dr. Gioele La Manno](https://www.nature.com/articles/s41586-018-0414-6)
- scEUseq: [Dr. Nico Battich](https://www.science.org/doi/10.1126/science.aax3072)
- scNTseq: [Dr. Qi Qiu](https://www.nature.com/articles/s41592-020-0935-4)
- Human Erythroid: [Dr. Melania Barile](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02414-y)
- Mouse Erythroid: [Dr. Melania Barile](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02414-y)
