module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_np.py --multirun \
hydra=gpu \
wandb.project=cluster-np \
training.seed=0 \
training.max_clip_norm=null,1.0 \
model.self_attn_num_heads=null,1 \
model.model_variant=cnp,lnp \
model.context_variant=mean,bca \
training.trainer_variant=data,target,context \