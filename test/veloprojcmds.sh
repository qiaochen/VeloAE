# -*- coding: utf-8 -*-
"""Training cmds for datasets

"""


veloproj --seed $1 --n_raw_gene $2 --lr 1e-6 --refit 1 --adata /data/groups/yhhuang/cqiao/endocrinogenesis_day15.5.h5ad --model-name pancreas_model_$1_$2.cpt     --exp-name pancreas_$1_$2        --ld_adata ld_pancreas_$1_$2.h5      --device cuda:3
veloproj --seed $1 --n_raw_gene $2 --lr 1e-5 --refit 1 --adata ../notebooks/dentategyrus/data/DentateGyrus/10X43_1.h5ad --model-name dentategyrus_model_$1_$2.cpt --exp-name dentategyrus_$1_$2    --ld_adata ld_dentategyrus_$1_$2.h5  --device cuda:0 --gumbsoft_tau 8 --nb_g_src X
veloproj --seed $1 --n_raw_gene $2 --lr 1e-6 --refit 1 --adata /data/groups/yhhuang/scNT/neuron_splicing_lite.h5ad      --model-name scNT_model_$1_$2.cpt         --exp-name scNT_$1_$2            --ld_adata ld_scNT_$1_$2.h5          --device cuda:3 --gumbsoft_tau 5
veloproj --seed $1 --n_raw_gene $2 --lr 1e-6 --refit 1 --adata /data/users/cqiao/notebooks/data/organoids.h5ad          --model-name scEU_model_$1_$2.cpt         --exp-name scEU_$1_$2            --ld_adata ld_scEU_$1_$2.h5          --device cuda:0 
veloproj --seed $1 --n_raw_gene $2 --lr 1e-5 --refit 1 --adata /data/groups/yhhuang/cqiao/Melania/Erythroid_mouse.h5    --model-name Erythroid_mouse_$1_$2.cpt    --exp-name Erythroid_mouse_$1_$2 --ld_adata ld_Ery_mouse_$1_$2.h5     --device cuda:3
veloproj --seed $1 --n_raw_gene $2 --lr 1e-5 --refit 1 --adata /data/groups/yhhuang/cqiao/Melania/Erythroid_human.h5    --model-name Erythroid_human_$1_$2.cpt    --exp-name Erythroid_human_$1_$2 --ld_adata ld_Ery_human_$1_$2.h5     --device cuda:0







