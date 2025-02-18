module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_np.py --multirun \
hydra=gpu \
wandb.project=cluster-np \
training.seed=1 \
training.max_clip_norm=null,1.0 \
model.self_attn_num_heads=null,1 \
model.model_variant=lnp,cnp \
model.context_variant=bca,mean \
training.trainer_variant=data,target,context \