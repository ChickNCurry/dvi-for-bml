module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu \
wandb.project=cluster-sine \
benchmark._target_=metalearning_benchmarks.sinusoid1d_benchmark.Sinusoid1D \
training.seed=1,2,3 \
training.max_clip_norm=1.0,0.1 \
common.self_attn_num_heads=null,1 \
common.non_linearity=GELU,SiLU \
common.variant=aggr,bca,mha \

# training.alpha=null,1.0 \
# common.self_attn_num_heads=1 \