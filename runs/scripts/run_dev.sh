module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu_dev \
wandb.project=cluster-test \
training.max_clip_norm=null,0.5 \
training.alpha=null,1.0 \
common.self_attn_num_heads=null,1,2 \
common.non_linearity=GELU,SiLU \
common.variant=aggr,bca,mha \