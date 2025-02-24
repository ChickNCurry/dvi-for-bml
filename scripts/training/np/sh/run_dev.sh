module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_np.py --multirun \
hydra=gpu_dev \
wandb.project=cluster-np \
training.seed=1 \
training.max_clip_norm=null \
model.self_attn_num_heads=null \
model.model_variant=lnp \
model.context_variant=bca \
training.trainer_variant=data \