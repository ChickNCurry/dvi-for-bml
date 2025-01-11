module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu \
wandb.project=cluster \
common.variant=aggr,bca,mha \
common.non_linearity=GELU,SiLU \
common.self_attn_num_heads=null,1,2 \
training.max_clip_norm=null,0.5 \
training.alpha=null,1.0 \